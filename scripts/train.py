import os
import sys
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wandb
import torch
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger_wandb import init_wandb

from src.data.dataloader import build_dataloader, build_landmark_dataloader, build_unet_mask_dataloader
from src.models import get_model # in __init__ gfile
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.optimizer import build_scheduler
from src.utils.checkpoint import load_checkpoints
from src.evaluation.evaluator import evaluate_and_show
from src.utils.logger_wandb import save_model_to_wandb
from src.utils.data_stats import get_class_distribution # testing: class weight

from datetime import datetime, timedelta
#-------------------------------------------------------------


def save_training_artifacts(history, train_losses, val_losses, output_dir, run_name):
    curves_dir = Path(output_dir) / "training_curves" / run_name
    curves_dir.mkdir(parents=True, exist_ok=True)

    if not history:
        history = [
            {
                "epoch": idx + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            for idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses))
        ]

    csv_path = curves_dir / "training_history.csv"
    json_path = curves_dir / "training_history.json"
    plot_path = curves_dir / "training_curves.png"

    fieldnames = sorted({key for row in history for key in row.keys()})
    preferred = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "best_val_loss",
        "best_val_accuracy",
        "improved",
        "patience_counter",
        "lr_head",
        "lr_visual_extractor",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    epochs = [row["epoch"] for row in history]
    has_accuracy = all(
        "train_accuracy" in row and "val_accuracy" in row
        for row in history
    )
    if has_accuracy:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        loss_ax, acc_ax = axes
    else:
        fig, loss_ax = plt.subplots(1, 1, figsize=(8, 5))
        acc_ax = None

    loss_ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], marker="x", label="Val loss")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    if acc_ax is not None:
        acc_ax.plot(
            epochs,
            [row["train_accuracy"] for row in history],
            marker="o",
            label="Train accuracy",
        )
        acc_ax.plot(
            epochs,
            [row["val_accuracy"] for row in history],
            marker="x",
            label="Val accuracy",
        )
        acc_ax.set_title("Accuracy")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.grid(True, alpha=0.3)
        acc_ax.legend()

    fig.suptitle(run_name)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"--> Training history CSV: {csv_path}")
    print(f"--> Training history JSON: {json_path}")
    print(f"--> Training curve plot: {plot_path}")

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
            "On Kaggle, switch Accelerator to GPU T4 x2 or set --nproc_per_node to the number of visible GPUs."
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


def _safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint does not contain a valid model state dict.")


def _strip_known_prefixes(state_dict):
    prefixes = ("module.", "_orig_mod.")
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        cleaned[name] = value
    return cleaned


def _resolve_checkpoint_path(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    basename = os.path.basename(checkpoint_path)
    search_roots = [os.getcwd()]
    if os.path.exists("/kaggle/input"):
        search_roots.insert(0, "/kaggle/input")

    for root in search_roots:
        for current_dir, _, files in os.walk(root):
            if basename in files:
                found = os.path.join(current_dir, basename)
                print(f"--> Using discovered init checkpoint: {found}")
                return found

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_model_init_checkpoint(model, checkpoint_path, strict=True):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    print(f"--> Loading model init checkpoint: {checkpoint_path}")
    checkpoint = _safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = _strip_known_prefixes(_extract_state_dict(checkpoint))
    target_model = model.module if hasattr(model, "module") else model
    incompatible = target_model.load_state_dict(state_dict, strict=strict)
    if incompatible.missing_keys:
        print(f"--> Init checkpoint missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"--> Init checkpoint unexpected keys: {len(incompatible.unexpected_keys)}")
    print("--> Model init checkpoint loaded.")


def resolve_data_path(data_path):
    required_files = {"train.csv", "val.csv", "test.csv"}
    if os.path.isdir(data_path):
        if required_files.issubset(set(os.listdir(data_path))):
            return data_path

        for current_dir, _, files in os.walk(data_path):
            if required_files.issubset(set(files)):
                print(f"--> Data path not exact; using discovered split folder: {current_dir}")
                return current_dir

    raise FileNotFoundError(
        f"Could not find train.csv, val.csv, test.csv under data_path: {data_path}"
    )

@record
def main():
    distributed, rank, world_size, local_rank = setup_distributed()
    try:
        if is_main_process():
            print("\t\t--> In main <--\t\t")

        # device
        device = torch.device(f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu"))
        if is_main_process():
            print("--- Use device:", device)
            if distributed:
                print(f"--- DDP enabled: world_size={world_size}")

        # get args 
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
        args = parser.parse_args()
        
        # load config
        config = load_config(args.config, args.env)
        set_seed(config['seed'].get('random_seed', 21) + rank)
        ddp_cfg = config.get('ddp', {})

        # load_config() merges env.yaml into top-level keys: data_path, output_dir, root_path.
        path_cfg = config.get('paths', {})
        data_path = path_cfg.get('data_path', config.get('data_path', "dataset/fer13-split"))
        output_dir = path_cfg.get('output_dir', config.get('output_dir', "outputs"))
        root_path = config.get('root_path', ".")
        data_path = resolve_data_path(data_path)
           

        timestamp = datetime.now().strftime("%d%m%Y_%H%M")
        run_name = f"{config['model'].get('name', 'cnn')}_{timestamp}"

        # load data, loss, optim, model
        model_name = config['model'].get('name')
        if model_name == 'resnet152_landmark_attention':
            dataloader_builder = build_landmark_dataloader
        elif model_name in {'resnet152_unet_mask_attention', 'convnext_tiny_mask_guided_region_attention'}:
            dataloader_builder = build_unet_mask_dataloader
        else:
            dataloader_builder = build_dataloader

        if is_main_process() and dataloader_builder is build_landmark_dataloader:
            print("--> Using landmark dataloader for landmark-guided attention.")
        if is_main_process() and dataloader_builder is build_unet_mask_dataloader:
            print("--> Using precomputed mask dataloader for mask-guided attention.")

        train_loader, val_loader, test_loader = dataloader_builder(
            config=config,
            data_path=data_path,
            distributed=distributed,
            world_size=world_size,
        )
        
        model = get_model(
            name=config['model']['name'],
            config=config)

        init_checkpoint_path = config.get('training', {}).get('init_checkpoint_path')
        if init_checkpoint_path:
            init_strict = bool(config.get('training', {}).get('init_checkpoint_strict', True))
            load_model_init_checkpoint(model, init_checkpoint_path, strict=init_strict)
        

        # ── Transfer Learning: load pretrained backbone weights ──
        pretrained_vgg = config['model'].get('pretrained_vgg_path', None)
        pretrained_resnet = config['model'].get('pretrained_resnet_path', None)
        
        if hasattr(model, 'load_pretrained_backbones'):
            if pretrained_vgg and pretrained_resnet:
                if is_main_process():
                    print("\n" + "="*50 + "\n[Transfer Learning] Loading dual pretrained backbones...\n" + "="*50)
                model.load_pretrained_backbones(pretrained_vgg, pretrained_resnet, device=device)
                model.freeze_backbones()
                if is_main_process():
                    print("="*50 + "\n")
            elif pretrained_resnet:
                if is_main_process():
                    print("\n" + "="*50 + "\n[Transfer Learning] Loading ResNet pretrained backbone...\n" + "="*50)
                model.load_pretrained_backbones(resnet_ckpt_path=pretrained_resnet, device=device)
                model.freeze_backbones()
                if is_main_process():
                    print("="*50 + "\n")
            elif pretrained_vgg:
                if is_main_process():
                    print("\n" + "="*50 + "\n[Transfer Learning] Loading VGG pretrained backbone...\n" + "="*50)
                model.load_pretrained_backbones(vgg_ckpt_path=pretrained_vgg, device=device)
                model.freeze_backbones()
                if is_main_process():
                    print("="*50 + "\n")


        # get class_distribution for class_weights (optional)
        use_class_weights = config['training'].get('use_class_weights', False)
        class_weights = None
        
        if use_class_weights:
            if is_main_process():
                print("--> Using Class Weights to handle imbalance...")
            trainset_path = os.path.join(data_path, "train.csv")
            train_class_distribution = get_class_distribution(trainset_path)
            train_class_distribution_np = train_class_distribution.values
            class_weights = 1.0 / torch.tensor(train_class_distribution_np, dtype=torch.float)
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(device)

        model = model.to(device)
        if distributed:
            if ddp_cfg.get('sync_batchnorm', False):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=ddp_cfg.get('find_unused_parameters', False),
                broadcast_buffers=ddp_cfg.get('broadcast_buffers', True),
            )

        loss = build_loss(config=config, class_weights=class_weights)
        optimizer = build_optimizer(model=model, config=config)
        scheduler = build_scheduler(optimizer=optimizer, config=config)
        
        # set path to save ckpt
        path_save_ckpt = os.path.join(output_dir, f"checkpoints/{config['model'].get('name', 'cnn')}/{run_name}_best.pth")
        if is_main_process():
            os.makedirs(os.path.dirname(path_save_ckpt), exist_ok=True)
        if distributed:
            dist.barrier()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            run_name=run_name,
            save_dir=path_save_ckpt
        )
        train_losses, val_losses = trainer.fit()

        if is_main_process():
            save_training_artifacts(
                history=getattr(trainer, "history", []),
                train_losses=train_losses,
                val_losses=val_losses,
                output_dir=output_dir,
                run_name=run_name,
            )

        # evaluate
        if distributed:
            dist.barrier()

        if not is_main_process():
            return

        print("\n" + "="*51)
        print("Evaluate in test set")
        print("="*51)
        
        eval_model = model.module if hasattr(model, "module") else model

        # Get path of file best  
        load_checkpoints(
            eval_model,
            optimizer=None,
            checkpoint_path=path_save_ckpt,
            device=device,
            load_optimizer=False,
        )
        
        eval_dir_path = os.path.join(output_dir, "figures")
        os.makedirs(eval_dir_path, exist_ok=True)
        print(f"Evaluatoin save path: {eval_dir_path}")


        # test data path
        testset_path = os.path.join(data_path, "test.csv")
        evaluate_and_show(eval_model, test_loader, testset_path, device, eval_dir_path)
        
        # upload best ckpt to wandb
        if config['logging'].get('use_wandb', True):
            print("\n\t--> Uploading best ckpt to WandB, please wait...")
            save_model_to_wandb(path_save_ckpt)
            
            # Đóng cửa sổ WandB, tránh bị kẹt quá trình upload trên hệ thống ngầm của Kaggle
            wandb.finish()

        print("\n\t\tDONE!\n")
    finally:
        cleanup_distributed()

    

if __name__ == "__main__":
    main()
