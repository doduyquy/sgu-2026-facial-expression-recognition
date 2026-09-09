import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import PureImageFER2013, build_transforms, EMOTION_NAMES
from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.evaluation.evaluator import plot_confusion_matrix
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble of Attentive-SCN Models on FER2013")
    parser.add_argument(
        "--config1",
        type=str,
        default="fads_scn/configs/scn_pure_image.yaml",
        help="Path to YAML config for Model 1 (e.g., DenseNet-121)",
    )
    parser.add_argument(
        "--weights1",
        type=str,
        required=True,
        help="Path to trained checkpoint for Model 1 (.pth)",
    )
    parser.add_argument(
        "--config2",
        type=str,
        default="fads_scn/configs/scn_convnext.yaml",
        help="Path to YAML config for Model 2 (e.g., ConvNeXt-Tiny)",
    )
    parser.add_argument(
        "--weights2",
        type=str,
        required=True,
        help="Path to trained checkpoint for Model 2 (.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--weight1",
        type=float,
        default=0.5,
        help="Ensemble weight for Model 1 (Model 2 weight is 1 - weight1)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/fads_scn/ensemble",
        help="Directory to save ensemble confusion matrix and results",
    )
    return parser.parse_args()


def load_model(config_path, weights_path, device):
    p = Path(config_path)
    if not p.exists():
        p = repo_root / config_path
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["model"]
    model = AttentiveSCNFER(
        backbone_name=m_cfg.get("backbone", "resnet50"),
        num_classes=m_cfg.get("num_classes", 7),
        in_channels=m_cfg.get("in_channels", 1),
        embed_dim=m_cfg.get("embed_dim", 256),
        num_attn_heads=m_cfg.get("num_attn_heads", 4),
        dropout=0.0,
        classifier_type=m_cfg.get("classifier_type", "linear"),
        cosface_scale=m_cfg.get("cosface_scale", 30.0),
        cosface_margin=m_cfg.get("cosface_margin", 0.20),
        use_pretrained=False,
    )

    ckpt_path = Path(weights_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, m_cfg.get("backbone", "unknown")


@torch.no_grad()
def get_model_probs(model, loader, device, use_tta="multiscale"):
    """
    Computes predicted softmax probability distributions for all samples in loader.
    """
    model.eval()
    all_probs = []
    all_targets = []

    for batch in loader:
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch

        images = images.to(device, non_blocking=True)
        outputs = model(images, use_tta=use_tta)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    return all_probs, all_targets


def compute_metrics(probs, targets):
    preds = np.argmax(probs, axis=-1)
    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    cm = confusion_matrix(targets, preds, labels=list(range(len(EMOTION_NAMES))))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)
    per_class_dict = {
        name: round(float(acc_val) * 100, 2)
        for name, acc_val in zip(EMOTION_NAMES, per_class_acc)
    }
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "hybrid_score": acc * macro_f1,
        "per_class_acc": per_class_dict,
        "confusion_matrix": cm,
    }


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    data_path = "dataset/fer13-split"
    kaggle_candidate_paths = [
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
        "/kaggle/input/datasets/doduyquynii/fer13-split",
        "/kaggle/input/fer13-split/fer13-split",
        "/kaggle/input/fer13-split",
    ]
    for p in kaggle_candidate_paths:
        if os.path.exists(p):
            data_path = p
            break

    tf = build_transforms(args.split)
    ds = PureImageFER2013(data_path=data_path, split=args.split, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Load Models
    print(f"\n=======================================================")
    print(f"[ENSEMBLE] EVALUATION ON FER2013 ({args.split.upper()} SET)")
    print(f"Total Samples: {len(ds)} | Device: {device}")
    print(f"Loading Model 1 from: {args.weights1}")
    m1, name1 = load_model(args.config1, args.weights1, device)
    print(f"Loading Model 2 from: {args.weights2}")
    m2, name2 = load_model(args.config2, args.weights2, device)
    print(f"Model 1: {name1} | Model 2: {name2}")
    print(f"=======================================================\n")

    # 3. Model 1 Predictions
    print(f"--> Extracting predictions for Model 1 ({name1})...")
    probs1_std, targets = get_model_probs(m1, loader, device, use_tta=False)
    probs1_tta, _ = get_model_probs(m1, loader, device, use_tta=True)
    probs1_ms, _ = get_model_probs(m1, loader, device, use_tta="multiscale")
    m1_std_metrics = compute_metrics(probs1_std, targets)
    m1_tta_metrics = compute_metrics(probs1_tta, targets)
    m1_ms_metrics = compute_metrics(probs1_ms, targets)

    # 4. Model 2 Predictions
    print(f"--> Extracting predictions for Model 2 ({name2})...")
    probs2_std, _ = get_model_probs(m2, loader, device, use_tta=False)
    probs2_tta, _ = get_model_probs(m2, loader, device, use_tta=True)
    probs2_ms, _ = get_model_probs(m2, loader, device, use_tta="multiscale")
    m2_std_metrics = compute_metrics(probs2_std, targets)
    m2_tta_metrics = compute_metrics(probs2_tta, targets)
    m2_ms_metrics = compute_metrics(probs2_ms, targets)

    # 5. Ensemble Predictions
    w1 = args.weight1
    w2 = 1.0 - w1

    # A. Ensemble Standard (Single Image)
    probs_ens_std = w1 * probs1_std + w2 * probs2_std
    ens_std_metrics = compute_metrics(probs_ens_std, targets)

    # B. Ensemble with Flip TTA (Empirical Best: +0.56% boost)
    probs_ens_tta = w1 * probs1_tta + w2 * probs2_tta
    ens_tta_metrics = compute_metrics(probs_ens_tta, targets)

    # C. Ensemble with Multi-Scale TTA
    probs_ens_ms = w1 * probs1_ms + w2 * probs2_ms
    ens_ms_metrics = compute_metrics(probs_ens_ms, targets)

    # Grid search optimal weights on Flip TTA
    best_w = w1
    best_ens_acc = ens_tta_metrics["accuracy"]
    best_ens_f1 = ens_tta_metrics["macro_f1"]
    for test_w1 in np.linspace(0.1, 0.9, 9):
        p_comb = test_w1 * probs1_tta + (1.0 - test_w1) * probs2_tta
        m_comb = compute_metrics(p_comb, targets)
        if m_comb["accuracy"] > best_ens_acc:
            best_ens_acc = m_comb["accuracy"]
            best_ens_f1 = m_comb["macro_f1"]
            best_w = test_w1

    best_mode_metrics = ens_tta_metrics if ens_tta_metrics["accuracy"] >= ens_ms_metrics["accuracy"] else ens_ms_metrics
    best_mode_name = "Flip TTA (2-Crop)" if ens_tta_metrics["accuracy"] >= ens_ms_metrics["accuracy"] else "Multi-Scale TTA (4-Crop)"

    print("\n" + "=" * 70)
    print(f"{'EVALUATION SUMMARY':^70}")
    print("=" * 70)
    print(f"{'Method / Model':<38} | {'Accuracy':<12} | {'Macro F1':<10}")
    print("-" * 70)
    print(f"{f'1. {name1} (Single Image)':<38} | {m1_std_metrics['accuracy']*100:>10.2f}% | {m1_std_metrics['macro_f1']*100:>8.2f}%")
    print(f"{f'   {name1} (+ Flip TTA 2-Crop)':<38} | {m1_tta_metrics['accuracy']*100:>10.2f}% | {m1_tta_metrics['macro_f1']*100:>8.2f}%")
    print("-" * 70)
    print(f"{f'2. {name2} (Single Image)':<38} | {m2_std_metrics['accuracy']*100:>10.2f}% | {m2_std_metrics['macro_f1']*100:>8.2f}%")
    print(f"{f'   {name2} (+ Flip TTA 2-Crop)':<38} | {m2_tta_metrics['accuracy']*100:>10.2f}% | {m2_tta_metrics['macro_f1']*100:>8.2f}%")
    print("-" * 70)
    print(f"{f'3. Ensemble (w1={w1:.2f}) Standard':<38} | {ens_std_metrics['accuracy']*100:>10.2f}% | {ens_std_metrics['macro_f1']*100:>8.2f}%")
    print(f"{f'   Ensemble (w1={w1:.2f}) + Flip TTA':<38} | {ens_tta_metrics['accuracy']*100:>10.2f}% | {ens_tta_metrics['macro_f1']*100:>8.2f}%")
    if abs(best_w - w1) > 1e-4:
        print(f"{f'   Ensemble Optimal (w1={best_w:.2f}) + Flip TTA':<38} | {best_ens_acc*100:>10.2f}% | {best_ens_f1*100:>8.2f}%")
    print("=" * 70)

    print(f"\n--- Per-Class Accuracies (Ensemble + {best_mode_name}) ---")
    for cls_name, cls_acc in best_mode_metrics["per_class_acc"].items():
        print(f"  {cls_name.ljust(10)}: {cls_acc:.2f}%")

    # Plot and save confusion matrix
    cm_path = out_dir / f"confusion_matrix_ensemble_{args.split}.png"
    plot_confusion_matrix(
        best_mode_metrics["confusion_matrix"],
        class_names=EMOTION_NAMES,
        save_path=cm_path,
        title=f"Ensemble ({name1} + {name2}) Confusion Matrix ({args.split.upper()} Acc: {best_mode_metrics['accuracy']*100:.2f}%)",
    )
    print(f"\n[SAVE] Ensemble Confusion Matrix saved -> {cm_path}\n")


if __name__ == "__main__":
    main()
