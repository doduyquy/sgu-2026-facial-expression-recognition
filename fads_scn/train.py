import os
import sys
from pathlib import Path
import argparse
import random
import numpy as np
import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import build_dataloaders
from fads_scn.losses.scn_loss import SCNLoss
from fads_scn.training.trainer import AttentiveSCNTrainer
from fads_scn.runtime import load_config, apply_overrides, resolve_data_path, build_model


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] random_seed={seed}")


def compute_class_weights(counts: np.ndarray, train_cfg: dict) -> torch.Tensor:
    mode = train_cfg.get("class_weight_mode", "sqrt_inverse")
    if mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts + 1e-6)
    elif mode == "inverse":
        weights = 1.0 / (counts + 1e-6)
    elif mode == "manual":
        weights = np.asarray(train_cfg.get("manual_class_weights", []), dtype=np.float32)
        if weights.shape[0] != counts.shape[0]:
            raise ValueError("manual_class_weights length must match num_classes")
    else:
        raise ValueError(f"Unsupported class_weight_mode: {mode}")

    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pure Image-Based Attentive-SCN on FER2013")
    parser.add_argument(
        "--config",
        type=str,
        default="fads_scn/configs/scn_convnext.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="local",
        choices=["local", "kaggle"],
        help="Execution environment",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_dir", default=None, help="Base directory; each run gets a unique subdirectory")
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load config
    cfg, config_path = load_config(args.config)
    apply_overrides(cfg, args.set)

    # Overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
        cfg["training"]["backbone_lr"] = args.lr
        cfg["training"]["head_lr"] = args.lr
    if args.backbone_lr is not None:
        cfg["training"]["backbone_lr"] = args.backbone_lr
    if args.head_lr is not None:
        cfg["training"]["head_lr"] = args.head_lr
    if args.seed is not None:
        cfg.setdefault("seed", {})["random_seed"] = args.seed

    # Environment-specific path resolution
    required = ("train", "val", "test") if cfg["training"].get("evaluate_test_at_end", False) else ("train", "val")
    cfg["data"]["data_path"] = resolve_data_path(cfg, args.env, args.data_path, required)
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    elif args.env == "kaggle":
        cfg["training"]["output_dir"] = f"/kaggle/working/outputs/{cfg['model']['backbone']}"

    # Set device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Running Attentive-SCN on {device.upper()} ===")
    print(f"Config: {config_path}")
    seed = int(cfg.get("seed", {}).get("random_seed", 42))
    cfg.setdefault("seed", {})["random_seed"] = seed
    set_seed(seed)

    # 2. Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset) if test_loader else 'not loaded'}")

    # 3. Compute class weights
    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        counts = train_loader.dataset.get_class_counts()
        print(f"Class distribution: {counts}")
        class_weights = compute_class_weights(counts, cfg["training"])
        print(f"Class weight mode: {cfg['training'].get('class_weight_mode', 'sqrt_inverse')}")
        print(f"Computed class weights: {[round(float(w), 3) for w in class_weights]}")

    # 4. Initialize Model
    m_cfg = cfg["model"]
    model = build_model(cfg)

    # 5. Initialize Loss
    scn_cfg = cfg.get("scn", {})
    criterion = SCNLoss(
        num_classes=m_cfg.get("num_classes", 7),
        label_smoothing=cfg["training"].get("label_smoothing", 0.05),
        margin=scn_cfg.get("margin", 0.15),
        clean_ratio=scn_cfg.get("clean_ratio", 0.70),
        rank_loss_weight=scn_cfg.get("rank_loss_weight", 0.10),
        div_loss_weight=scn_cfg.get("div_loss_weight", 0.05),
        sparsity_loss_weight=scn_cfg.get("sparsity_loss_weight", 0.0),
        class_weights=class_weights,
        use_scn=scn_cfg.get("use_scn", True),
        rank_mode=scn_cfg.get("rank_mode", "global"),
    )

    # 6. Initialize Trainer & Run
    trainer = AttentiveSCNTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
