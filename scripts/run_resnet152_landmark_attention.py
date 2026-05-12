import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_landmark_dataloader
from src.evaluation.evaluator import evaluate_and_show
from src.models import get_model
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


DEFAULT_SOURCE_CKPT = PROJECT_ROOT / "checkpoints" / "resnet152_rot30_2019Nov14_12.47"


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("DDP needs CUDA GPUs. Run without torchrun for CPU/single process.")

    cuda_count = torch.cuda.device_count()
    if local_rank >= cuda_count:
        raise RuntimeError(
            f"DDP launched local_rank={local_rank}, but torch sees only {cuda_count} CUDA device(s). "
            "On Kaggle, switch Accelerator to GPU T4 x2 or set --nproc_per_node to the visible GPU count."
        )

    torch.cuda.set_device(local_rank)
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    if rank == 0:
        print(f"--- DDP backend: {backend}")
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint does not contain a valid state dict.")


def resolve_data_path(data_path):
    required_files = {"train.csv", "val.csv", "test.csv"}
    raw_path = Path(data_path)
    candidates = [raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path]
    if Path("/kaggle/input").exists():
        candidates.append(Path("/kaggle/input"))

    for candidate in candidates:
        if candidate.is_dir():
            files = {p.name for p in candidate.iterdir() if p.is_file()}
            if required_files.issubset(files):
                return str(candidate)

            for current_dir, _, files in os.walk(candidate):
                if required_files.issubset(set(files)):
                    print(f"--> Data path not exact; using discovered split folder: {current_dir}")
                    return current_dir

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find train.csv, val.csv, test.csv. "
        f"Searched under: {searched}"
    )


def load_model_checkpoint(model, checkpoint_path, device, strict=True):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    state_dict = {
        key.replace("module.", "", 1).replace("_orig_mod.", "", 1): value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"--> Loaded attention checkpoint: {checkpoint_path}")
    if missing:
        print(f"--> Missing keys: {len(missing)}")
    if unexpected:
        print(f"--> Unexpected keys: {len(unexpected)}")
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate ResNet152 + landmark-guided attention."
    )
    parser.add_argument("--config", type=str, default="resnet152_landmark_attention")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--source-ckpt", type=str, default=str(DEFAULT_SOURCE_CKPT))
    parser.add_argument(
        "--attention-ckpt",
        type=str,
        default=None,
        help="Checkpoint produced by this attention model. Required for meaningful eval-only attention accuracy.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Folder containing train.csv, val.csv, test.csv. Overrides config/env data_path.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--logit-fusion",
        choices=["attention", "source", "sum"],
        default=None,
        help="Override config model.logit_fusion.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--strict", action="store_true", help="Strict load for --attention-ckpt.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@record
def main():
    distributed, rank, world_size, local_rank = setup_distributed()
    try:
        args = parse_args()
        config = load_config(args.config, args.env)

        config["model"]["name"] = "resnet152_landmark_attention"
        config["data"]["image_size"] = 224
        config["data"]["channels"] = 3
        config["data"]["normalize"] = False

        if args.batch_size is not None:
            config["data"]["batch_size"] = args.batch_size
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if config["training"].get("epochs") is None:
            config["training"]["epochs"] = 60
            if is_main_process():
                print("--> training.epochs was empty; fallback to 60 epochs.")
        if args.logit_fusion is not None:
            config["model"]["logit_fusion"] = args.logit_fusion
        if args.output_dir is not None:
            config["output_dir"] = args.output_dir

        # Source-only eval is just a sanity check for the original checkpoint head.
        # It should not require downloading/initializing CLIP text embeddings.
        if args.eval_only and config["model"].get("logit_fusion") == "source":
            config["model"]["use_clip_dictionary"] = False

        set_seed(config["seed"].get("random_seed", 42) + rank)
        ddp_cfg = config.get("ddp", {})
        device = (
            torch.device(f"cuda:{local_rank}")
            if distributed
            else resolve_device(args.device)
        )
        if is_main_process():
            print(f"--> Using device: {device}")
            if distributed:
                print(f"--> DDP enabled: world_size={world_size}")

        data_path = args.data_path or config.get("paths", {}).get(
            "data_path",
            config.get("data_path", "dataset/fer13-split"),
        )
        data_path = resolve_data_path(data_path)
        output_dir = config.get("paths", {}).get("output_dir", config.get("output_dir", "outputs"))
        if args.output_dir is not None:
            output_dir = args.output_dir
        if is_main_process():
            print(f"--> Data path: {data_path}")
            print(f"--> Output dir: {output_dir}")

        train_loader, val_loader, test_loader = build_landmark_dataloader(
            config,
            data_path,
            distributed=distributed,
            world_size=world_size,
        )
        model = get_model(name=config["model"]["name"], config=config)

        if args.attention_ckpt:
            load_model_checkpoint(model, args.attention_ckpt, device="cpu", strict=args.strict)
        else:
            model.load_pretrained_backbones(args.source_ckpt, device="cpu")
            if config["model"].get("freeze_backbone_epochs", 0) > 0:
                model.freeze_backbones()

        if (
            config["model"].get("logit_fusion") == "source"
            and hasattr(model, "res_backbone")
            and not model.res_backbone.has_source_classifier
        ):
            raise RuntimeError(
                "--logit-fusion source was requested, but the source checkpoint did not expose "
                "a compatible fc.weight/fc.bias classifier."
            )

        model = model.to(device)
        if distributed:
            if ddp_cfg.get("sync_batchnorm", False):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=ddp_cfg.get("find_unused_parameters", False),
                broadcast_buffers=ddp_cfg.get("broadcast_buffers", True),
            )

        if args.eval_only:
            if not is_main_process():
                return
            eval_model = model.module if hasattr(model, "module") else model
            if not args.attention_ckpt and config["model"].get("logit_fusion", "attention") == "attention":
                print(
                    "[WARN] Eval-only with no --attention-ckpt loads only the ResNet152 backbone; "
                    "the new attention classifier is still randomly initialized."
                )
                print("[WARN] Use --logit-fusion source to sanity-check the original checkpoint fc, if it exists.")

            eval_dir = Path(output_dir) / "evaluation_resnet152_landmark_attention"
            eval_dir.mkdir(parents=True, exist_ok=True)
            evaluate_and_show(eval_model, test_loader, os.path.join(data_path, "test.csv"), device, str(eval_dir))
            return

        criterion = build_loss(config).to(device)
        optimizer = build_optimizer(model=model, config=config)
        scheduler = build_scheduler(optimizer=optimizer, config=config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = Path(output_dir) / "checkpoints" / "resnet152_landmark_attention"
        if is_main_process():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()
        best_path = ckpt_dir / f"resnet152_landmark_attention_{timestamp}_best.pth"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            run_name=f"resnet152_landmark_attention_{timestamp}",
            save_dir=str(best_path),
        )
        trainer.fit()

        if distributed:
            dist.barrier()
        if not is_main_process():
            return

        print("\n" + "=" * 50)
        print("Evaluate best attention checkpoint on test set")
        print("=" * 50)
        eval_model = model.module if hasattr(model, "module") else model
        best_ckpt = safe_torch_load(best_path, map_location=device)
        eval_model.load_state_dict(best_ckpt["model_state_dict"])

        eval_dir = Path(output_dir) / "evaluation_resnet152_landmark_attention"
        eval_dir.mkdir(parents=True, exist_ok=True)
        evaluate_and_show(eval_model, test_loader, os.path.join(data_path, "test.csv"), device, str(eval_dir))
        print(f"--> Best checkpoint: {best_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
