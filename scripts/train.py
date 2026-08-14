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


def print_parameter_summary(model, label="Model"):
    target_model = model.module if hasattr(model, "module") else model
    total_params = sum(param.numel() for param in target_model.parameters())
    trainable_params = sum(
        param.numel() for param in target_model.parameters() if param.requires_grad
    )
    frozen_params = total_params - trainable_params
    trainable_percent = 100.0 * trainable_params / total_params if total_params else 0.0
    print(
        f"--> [{label}] Parameters: "
        f"total={total_params:,}, "
        f"trainable={trainable_params:,} ({trainable_percent:.2f}%), "
        f"frozen={frozen_params:,}"
    )


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
    if not checkpoint_path:
        raise ValueError("No checkpoint path provided.")

    # 1. Exact file check
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    # 2. Directory check
    if os.path.isdir(checkpoint_path):
        files_in_dir = []
        for root, _, files in os.walk(checkpoint_path):
            for file in sorted(files):
                full_p = os.path.join(root, file)
                if file.endswith('.pth') or file.endswith('.pt'):
                    print(f"--> Discovered checkpoint file in directory: {full_p}")
                    return full_p
                files_in_dir.append(full_p)
        if files_in_dir:
            print(f"--> Using file found in directory: {files_in_dir[0]}")
            return files_in_dir[0]

    # 3. Search in search_roots (/kaggle/input/models, /kaggle/input, cwd, etc.)
    search_roots = []
    if os.path.exists("/kaggle/input/models"):
        search_roots.append("/kaggle/input/models")
    if os.path.exists("/kaggle/input"):
        search_roots.append("/kaggle/input")
    search_roots.append(os.getcwd())

    all_files = []
    pth_files = []
    for root_dir in search_roots:
        for current_dir, _, files in os.walk(root_dir):
            for f in sorted(files):
                full_path = os.path.join(current_dir, f)
                if full_path not in all_files:
                    all_files.append(full_path)
                if f.endswith('.pth') or f.endswith('.pt'):
                    if full_path not in pth_files:
                        pth_files.append(full_path)

    # Match by partial keywords from the provided path
    clean_parts = [
        p.lower()
        for p in checkpoint_path.replace("\\", "/").split("/")
        if p and p.lower() not in ("kaggle", "input", "models", "pytorch", "default", "1", "datasets")
    ]

    # First try matching .pth files by keyword
    for ckpt in pth_files:
        ckpt_lower = ckpt.lower()
        if any(part in ckpt_lower for part in clean_parts):
            print(f"--> Using matched .pth checkpoint: {ckpt}")
            return ckpt

    # Then try matching ANY file under search roots
    for f in all_files:
        f_lower = f.lower()
        if any(part in f_lower for part in clean_parts):
            print(f"--> Using discovered model file: {f}")
            return f

    # If any file exists in /kaggle/input/models, pick the first one
    models_files = [f for f in all_files if "/models/" in f.replace("\\", "/")]
    if models_files:
        print(f"--> Discovered model file in /kaggle/input/models: {models_files[0]}")
        return models_files[0]

    print(f"--> Available files found on system:\n" + "\n".join(f"    - {c}" for c in all_files[:20]))
    raise FileNotFoundError(f"Checkpoint not found for input: {checkpoint_path}")


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
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to checkpoint file (.pth) to resume training state (epoch, model, optimizer, scheduler, baseline score)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Override total epochs in config['training']['epochs']",
        )
        parser.add_argument(
            "--extra_epochs",
            type=int,
            default=None,
            help="Run for N extra epochs beyond the resumed epoch (e.g. 50 extra epochs)",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=None,
            help="Override early stopping patience",
        )
        args = parser.parse_args()
        
        # load config
        config = load_config(args.config, args.env)
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
        if args.patience is not None:
            config['training']['patience'] = args.patience
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

        resume_checkpoint_path = (
            args.resume
            or config.get('training', {}).get('resume_checkpoint_path')
            or config.get('training', {}).get('resume_path')
        )
        init_checkpoint_path = config.get('training', {}).get('init_checkpoint_path')

        start_epoch = 0
        best_score = None
        best_val_loss = None
        best_val_acc = None
        patience_counter = 0
        ckpt_loaded = None

        if resume_checkpoint_path:
            resume_checkpoint_path = _resolve_checkpoint_path(resume_checkpoint_path)
            if is_main_process():
                print("\n" + "="*60 + f"\n[RESUME] Loading training checkpoint from: {resume_checkpoint_path}\n" + "="*60)
            ckpt_loaded = _safe_torch_load(resume_checkpoint_path, map_location="cpu")
            state_dict = _strip_known_prefixes(_extract_state_dict(ckpt_loaded))
            incompatible = model.load_state_dict(
                state_dict,
                strict=bool(config.get('training', {}).get('init_checkpoint_strict', True)),
            )
            if is_main_process():
                if incompatible.missing_keys:
                    print(f"[RESUME] Missing keys: {len(incompatible.missing_keys)}")
                if incompatible.unexpected_keys:
                    print(f"[RESUME] Unexpected keys: {len(incompatible.unexpected_keys)}")

            saved_epoch = ckpt_loaded.get('epoch', 0)
            start_epoch = saved_epoch + 1
            best_score = ckpt_loaded.get('best_score')
            best_val_loss = ckpt_loaded.get('val_loss')
            best_val_acc = ckpt_loaded.get('val_accuracy')
            if best_score is None:
                monitor = config.get('training', {}).get('monitor', 'val_accuracy')
                best_score = best_val_loss if monitor == 'val_loss' else best_val_acc

            if args.extra_epochs is not None:
                config['training']['epochs'] = start_epoch + args.extra_epochs
                if is_main_process():
                    print(f"[RESUME] Target epochs extended by +{args.extra_epochs} -> Total {config['training']['epochs']} epochs.")

            freeze_epochs = int(config.get('model', {}).get('freeze_backbone_epochs', 0) or 0)
            unfreeze_backbone = bool(config.get('model', {}).get('unfreeze_backbone', True))
            if unfreeze_backbone and start_epoch >= freeze_epochs:
                if hasattr(model, 'unfreeze_backbones'):
                    model.unfreeze_backbones()
                elif hasattr(model, 'unfreeze_backbone'):
                    model.unfreeze_backbone()

        elif init_checkpoint_path:
            init_strict = bool(config.get('training', {}).get('init_checkpoint_strict', True))
            load_model_init_checkpoint(model, init_checkpoint_path, strict=init_strict)
        

        # ── Transfer Learning: load pretrained backbone weights ──
        if not resume_checkpoint_path:
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

        if is_main_process():
            print_parameter_summary(model)

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

        # Build optimizer with appropriate learning rates for resume or fresh training
        freeze_epochs = int(config.get('model', {}).get('freeze_backbone_epochs', 0) or 0)
        unfreeze_backbone = bool(config.get('model', {}).get('unfreeze_backbone', True))
        if resume_checkpoint_path and unfreeze_backbone and start_epoch >= freeze_epochs:
            finetune_lr = config['training'].get('finetune_lr')
            visual_extractor_lr = config['training'].get('visual_extractor_lr')
            if visual_extractor_lr is not None:
                old_lr = config['training']['lr']
                config['training']['lr'] = finetune_lr if finetune_lr is not None else old_lr
                optimizer = build_optimizer(model=model, config=config)
                config['training']['lr'] = old_lr
            else:
                optimizer = build_optimizer(model=model, config=config)
        else:
            optimizer = build_optimizer(model=model, config=config)

        scheduler = build_scheduler(optimizer=optimizer, config=config)

        if ckpt_loaded is not None and 'optimizer_state_dict' in ckpt_loaded:
            try:
                optimizer.load_state_dict(ckpt_loaded['optimizer_state_dict'])
                if is_main_process():
                    print("[RESUME] Loaded optimizer state dictionary successfully.")
            except Exception as e:
                if is_main_process():
                    print(f"[RESUME WARNING] Could not load optimizer_state_dict ({e}). Training will proceed with freshly configured optimizer.")

        if resume_checkpoint_path and scheduler is not None and hasattr(scheduler, "last_epoch"):
            scheduler.last_epoch = start_epoch - 1

        if resume_checkpoint_path and is_main_process():
            print(f"[RESUME] Resuming from Epoch {start_epoch + 1}/{config['training'].get('epochs', 70)}")
            print(f"[RESUME] Restored baseline -> best_score: {best_score}, val_acc: {best_val_acc}, val_loss: {best_val_loss}")
            print("="*60 + "\n")
        
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
            save_dir=path_save_ckpt,
            start_epoch=start_epoch,
            best_score=best_score,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            patience_counter=patience_counter,
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
        evaluate_and_show(
            eval_model,
            test_loader,
            testset_path,
            device,
            eval_dir_path,
            use_wandb=config["logging"].get("use_wandb", True),
        )
        
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
