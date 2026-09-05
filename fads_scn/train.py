import os
import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import build_dataloaders
from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.losses.scn_loss import SCNLoss
from fads_scn.training.trainer import AttentiveSCNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pure Image-Based Attentive-SCN on FER2013")
    parser.add_argument(
        "--config",
        type=str,
        default="fads_scn/configs/scn_pure_image.yaml",
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
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try finding relative to repo root
        config_path = repo_root / args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    # Environment-specific path resolution
    if args.env == "kaggle":
        kaggle_candidate_paths = [
            "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
            "/kaggle/input/datasets/doduyquynii/fer13-split",
            "/kaggle/input/fer13-split/fer13-split",
            "/kaggle/input/fer13-split",
            "/kaggle/input/sgu-2026-facial-expression-recognition/dataset/fer13-split",
            "/kaggle/input/sgu-2026-facial-expression-recognition/fer13-split",
            "/kaggle/input/fer2013/dataset/fer13-split",
            "/kaggle/input/fer2013",
        ]
        for p in kaggle_candidate_paths:
            if os.path.exists(p):
                cfg["data"]["data_path"] = p
                print(f"[Kaggle Env] Found data at: {p}")
                break
        cfg["training"]["output_dir"] = "/kaggle/working/outputs/fads_scn"

    # Set device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Running Attentive-SCN on {device.upper()} ===")
    print(f"Config: {config_path}")

    # 2. Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 3. Compute class weights
    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        counts = train_loader.dataset.get_class_counts()
        print(f"Class distribution: {counts}")
        weights = 1.0 / np.sqrt(counts + 1e-6)
        weights = weights / weights.sum() * len(counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Computed class weights: {[round(float(w), 3) for w in class_weights]}")

    # 4. Initialize Model
    m_cfg = cfg["model"]
    model = AttentiveSCNFER(
        backbone_name=m_cfg.get("backbone", "resnet50"),
        num_classes=m_cfg.get("num_classes", 7),
        in_channels=m_cfg.get("in_channels", 1),
        embed_dim=m_cfg.get("embed_dim", 256),
        num_attn_heads=m_cfg.get("num_attn_heads", 4),
        dropout=m_cfg.get("dropout", 0.25),
        use_pretrained=m_cfg.get("use_pretrained", True),
        pretrained_weights_path=m_cfg.get("pretrained_weights_path", ""),
    )

    # 5. Initialize Loss
    scn_cfg = cfg.get("scn", {})
    criterion = SCNLoss(
        num_classes=m_cfg.get("num_classes", 7),
        label_smoothing=cfg["training"].get("label_smoothing", 0.05),
        margin=scn_cfg.get("margin", 0.15),
        clean_ratio=scn_cfg.get("clean_ratio", 0.70),
        rank_loss_weight=scn_cfg.get("rank_loss_weight", 0.10),
        class_weights=class_weights,
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
