import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloader
from src.evaluation.evaluator import evaluate_and_show
from src.models import get_model
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


DEFAULT_SOURCE_CKPT = PROJECT_ROOT / "checkpoints" / "resnet152_rot30_2019Nov14_12.47"


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
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if data_path.is_dir():
        files = {p.name for p in data_path.iterdir() if p.is_file()}
        if required_files.issubset(files):
            return str(data_path)

        for current_dir, _, files in os.walk(data_path):
            if required_files.issubset(set(files)):
                print(f"--> Data path not exact; using discovered split folder: {current_dir}")
                return current_dir

    raise FileNotFoundError(f"Could not find train.csv, val.csv, test.csv under: {data_path}")


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
        description="Train/evaluate ResNet152 + facial-region attention."
    )
    parser.add_argument("--config", type=str, default="resnet152_region_attention")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--source-ckpt", type=str, default=str(DEFAULT_SOURCE_CKPT))
    parser.add_argument(
        "--attention-ckpt",
        type=str,
        default=None,
        help="Checkpoint produced by this attention model. Required for meaningful eval-only attention accuracy.",
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


def main():
    args = parse_args()
    config = load_config(args.config, args.env)

    config["model"]["name"] = "resnet152_region_attention"
    config["data"]["image_size"] = 224
    config["data"]["channels"] = 3
    config["data"]["normalize"] = False
    config["logging"]["use_wandb"] = False

    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.logit_fusion is not None:
        config["model"]["logit_fusion"] = args.logit_fusion
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    # Source-only eval is just a sanity check for the original checkpoint head.
    # It should not require downloading/initializing CLIP text embeddings.
    if args.eval_only and config["model"].get("logit_fusion") == "source":
        config["model"]["use_clip_dictionary"] = False

    set_seed(config["seed"].get("random_seed", 42))
    device = resolve_device(args.device)
    print(f"--> Using device: {device}")

    data_path = config.get("paths", {}).get("data_path", config.get("data_path", "dataset/fer13-split"))
    data_path = resolve_data_path(data_path)
    output_dir = config.get("paths", {}).get("output_dir", config.get("output_dir", "outputs"))
    if args.output_dir is not None:
        output_dir = args.output_dir
    print(f"--> Data path: {data_path}")
    print(f"--> Output dir: {output_dir}")

    train_loader, val_loader, test_loader = build_dataloader(config, data_path)
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

    if args.eval_only:
        if not args.attention_ckpt and config["model"].get("logit_fusion", "attention") == "attention":
            print(
                "[WARN] Eval-only with no --attention-ckpt loads only the ResNet152 backbone; "
                "the new attention classifier is still randomly initialized."
            )
            print("[WARN] Use --logit-fusion source to sanity-check the original checkpoint fc, if it exists.")

        eval_dir = Path(output_dir) / "evaluation_resnet152_region_attention"
        eval_dir.mkdir(parents=True, exist_ok=True)
        evaluate_and_show(model, test_loader, os.path.join(data_path, "test.csv"), device, str(eval_dir))
        return

    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(output_dir) / "checkpoints" / "resnet152_region_attention"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"resnet152_region_attention_{timestamp}_best.pth"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        run_name=f"resnet152_region_attention_{timestamp}",
        save_dir=str(best_path),
    )
    trainer.fit()

    print("\n" + "=" * 50)
    print("Evaluate best attention checkpoint on test set")
    print("=" * 50)
    best_ckpt = safe_torch_load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    eval_dir = Path(output_dir) / "evaluation_resnet152_region_attention"
    eval_dir.mkdir(parents=True, exist_ok=True)
    evaluate_and_show(model, test_loader, os.path.join(data_path, "test.csv"), device, str(eval_dir))
    print(f"--> Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
