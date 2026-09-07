import os
import sys
from pathlib import Path
import argparse
import random
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
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
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
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.seed is not None:
        cfg.setdefault("seed", {})["random_seed"] = args.seed

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
    seed = int(cfg.get("seed", {}).get("random_seed", 42))
    set_seed(seed)

    # 2. Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

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
    model = AttentiveSCNFER(
        backbone_name=m_cfg.get("backbone", "resnet50"),
        num_classes=m_cfg.get("num_classes", 7),
        in_channels=m_cfg.get("in_channels", 1),
        embed_dim=m_cfg.get("embed_dim", 256),
        num_attn_heads=m_cfg.get("num_attn_heads", 8),
        use_latent_graph=m_cfg.get("use_latent_graph", True),
        dropout=m_cfg.get("dropout", 0.25),
        classifier_type=m_cfg.get("classifier_type", "cosface"),
        cosface_scale=m_cfg.get("cosface_scale", 30.0),
        cosface_margin=m_cfg.get("cosface_margin", 0.20),
        use_pretrained=m_cfg.get("use_pretrained", True),
        pretrained_weights_path=m_cfg.get("pretrained_weights_path", ""),
        stem_init=m_cfg.get("stem_init", "mean"),
    )

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
