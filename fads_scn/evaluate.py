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
    parser.add_argument("--data_path", type=str, default=None, help="Explicit path to fer13-split dataset folder")
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
    data_path = args.data_path or cfg["data"].get("data_path", "dataset/fer13-split")
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
    if args.data_path is None:
        for p in kaggle_candidate_paths:
            if os.path.exists(p):
                data_path = p
                break
    tf = build_transforms(
        args.split,
        input_size=cfg["data"].get("input_size", 48),
        in_channels=cfg["model"].get("in_channels", 1),
        normalization=cfg["data"].get("normalization", "symmetric"),
    )
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
        use_latent_graph=m_cfg.get("use_latent_graph", True),
        use_spatial_attention=m_cfg.get("use_spatial_attention", True),
        use_multiscale_fusion=m_cfg.get("use_multiscale_fusion", False),
        multiscale_se_reduction=m_cfg.get("multiscale_se_reduction", 16),
        dropout=0.0,
        classifier_type=m_cfg.get("classifier_type", "linear"),
        cosface_scale=m_cfg.get("cosface_scale", 30.0),
        cosface_margin=m_cfg.get("cosface_margin", 0.20),
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

    # 2. Horizontal Flip TTA Evaluation (2-crop)
    metrics_tta = evaluate_model(model, loader, device, use_tta=True)
    diff_flip = (metrics_tta['accuracy'] - metrics_standard['accuracy']) * 100
    print(f"--- Horizontal Flip TTA Evaluation (2-Crop) ---")
    print(f"Accuracy:    {metrics_tta['accuracy'] * 100:.2f}%  (diff: {diff_flip:+.2f}%)")
    print(f"Macro F1:    {metrics_tta['macro_f1'] * 100:.2f}%")
    print(f"Hybrid Score:{metrics_tta['hybrid_score']:.4f}\n")

    # 3. Multi-Scale Zoom TTA Evaluation (4-crop)
    metrics_ms = evaluate_model(model, loader, device, use_tta="multiscale")
    diff_ms = (metrics_ms['accuracy'] - metrics_standard['accuracy']) * 100
    print(f"--- Multi-Scale Zoom TTA Evaluation (4-Crop: Orig, Flip, Zoom 1.05x, Zoom Flip) ---")
    print(f"Accuracy:    {metrics_ms['accuracy'] * 100:.2f}%  (diff vs standard: {diff_ms:+.2f}%)")
    print(f"Macro F1:    {metrics_ms['macro_f1'] * 100:.2f}%")
    print(f"Hybrid Score:{metrics_ms['hybrid_score']:.4f}\n")

    best_eval = metrics_ms if metrics_ms['accuracy'] >= metrics_tta['accuracy'] else metrics_tta
    best_mode = "Multi-Scale TTA" if metrics_ms['accuracy'] >= metrics_tta['accuracy'] else "Flip TTA"

    print(f"--- Per-Class Accuracies ({best_mode}) ---")
    for cls_name, cls_acc in best_eval["per_class_acc"].items():
        print(f"  {cls_name.ljust(10)}: {cls_acc:.2f}%")

    print(f"\n--- Confusion Matrix ({best_mode}) ---")
    print(best_eval["confusion_matrix"])

    # Optional: Save high-res confusion matrix if output dir is specified
    output_dir = Path(cfg.get("training", {}).get("output_dir", "outputs/fads_scn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = output_dir / f"confusion_matrix_{args.split}_{best_mode.lower().replace(' ', '_')}.png"
    from fads_scn.evaluation.evaluator import plot_confusion_matrix
    plot_confusion_matrix(
        best_eval["confusion_matrix"],
        class_names=EMOTION_NAMES,
        save_path=cm_path,
        title=f"FER2013 {args.split.upper()} Confusion Matrix ({best_mode} Acc: {best_eval['accuracy']*100:.2f}%)",
    )
    print(f"\n[SAVE] Confusion matrix saved -> {cm_path}\n")


if __name__ == "__main__":
    main()
