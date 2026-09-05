import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import numpy as np

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import PureImageFER2013, build_transforms, EMOTION_NAMES
from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.evaluation.evaluator import evaluate_model
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Attentive-SCN on FER2013")
    parser.add_argument(
        "--config",
        type=str,
        default="fads_scn/configs/scn_pure_image.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = repo_root / args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    data_path = cfg["data"].get("data_path", "dataset/fer13-split")
    tf = build_transforms(args.split)
    ds = PureImageFER2013(data_path=data_path, split=args.split, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model
    m_cfg = cfg["model"]
    model = AttentiveSCNFER(
        backbone_name=m_cfg.get("backbone", "resnet50"),
        num_classes=m_cfg.get("num_classes", 7),
        in_channels=m_cfg.get("in_channels", 1),
        embed_dim=m_cfg.get("embed_dim", 256),
        num_attn_heads=m_cfg.get("num_attn_heads", 4),
        dropout=0.0,
        use_pretrained=False,
    )

    # Load weights
    ckpt_path = Path(args.weights)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"\n=======================================================")
    print(f"[EVAL] EVALUATING ATTENTIVE-SCN ON FER2013 ({args.split.upper()} SET)")
    print(f"Weights: {ckpt_path}")
    print(f"Total Samples: {len(ds)} | Device: {device}")
    print(f"=======================================================\n")

    # 1. Standard Forward Evaluation
    metrics_standard = evaluate_model(model, loader, device, use_tta=False)
    print(f"--- Standard Evaluation (Single Image) ---")
    print(f"Accuracy:    {metrics_standard['accuracy'] * 100:.2f}%")
    print(f"Macro F1:    {metrics_standard['macro_f1'] * 100:.2f}%")
    print(f"Mean Alpha:  {metrics_standard['mean_alpha']:.3f}\n")

    # 2. Horizontal Flip TTA Evaluation
    metrics_tta = evaluate_model(model, loader, device, use_tta=True)
    diff = (metrics_tta['accuracy'] - metrics_standard['accuracy']) * 100
    print(f"--- Horizontal Flip TTA Evaluation (Standard SOTA) ---")
    print(f"Accuracy:    {metrics_tta['accuracy'] * 100:.2f}%  (diff: {diff:+.2f}%)")
    print(f"Macro F1:    {metrics_tta['macro_f1'] * 100:.2f}%")
    print(f"Hybrid Score:{metrics_tta['hybrid_score']:.4f}\n")

    print("--- Per-Class Accuracies (with TTA) ---")
    for cls_name, cls_acc in metrics_tta["per_class_acc"].items():
        print(f"  {cls_name.ljust(10)}: {cls_acc:.2f}%")

    print("\n--- Confusion Matrix ---")
    print(metrics_tta["confusion_matrix"])


if __name__ == "__main__":
    main()
