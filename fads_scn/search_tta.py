import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import PureImageFER2013, build_transforms, EMOTION_NAMES
from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.evaluation.evaluator import plot_confusion_matrix
from fads_scn.runtime import load_config, load_checkpoint_model, resolve_data_path
from fads_scn.data.dataset import transforms_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Systematic TTA Strategy Search for Attentive-SCN on FER2013")
    parser.add_argument(
        "--config",
        type=str,
        default="fads_scn/configs/scn_convnext.yaml",
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
        default="val",
        choices=["val"],
        help="Select TTA on validation only; evaluate the fixed mode separately on test",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--env", choices=["local", "kaggle"], default="local")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/fads_scn/tta_search",
        help="Output directory for results",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Transformation Helpers (Pure Tensor Operations - 100% Deterministic)
# ----------------------------------------------------------------------
def tta_flip(x):
    return torch.flip(x, dims=[-1])

def tta_shift_1px(x):
    return _shift_views(x, 1)

def tta_shift_2px(x):
    return _shift_views(x, 2)


def _shift_views(x, n):
    h, w = x.shape[-2:]
    p = F.pad(x, (n, n, n, n), mode="reflect")
    return (p[:, :, n:n+h, :w], p[:, :, n:n+h, 2*n:2*n+w],
            p[:, :, :h, n:n+w], p[:, :, 2*n:2*n+h, n:n+w])

def tta_brightness(x, delta=0.06):
    return torch.clamp(x + delta, -2.0, 2.0), torch.clamp(x - delta, -2.0, 2.0)

def tta_contrast(x, factor=1.08):
    mean = x.mean(dim=[-1, -2], keepdim=True)
    c_plus = (x - mean) * factor + mean
    c_minus = (x - mean) * (2.0 - factor) + mean
    return c_plus, c_minus


# ----------------------------------------------------------------------
# Extract Raw Outputs for All Atomic Augmentations in 1 Forward Pass
# ----------------------------------------------------------------------
@torch.no_grad()
def extract_all_views_outputs(model, loader, device):
    """
    Extracts logits and alpha confidence weights for all atomic TTA views
    for the entire dataset, saving them in memory so testing 20+ TTA combinations
    takes just milliseconds!
    """
    model.eval()
    print("--> Extracting base feature passes across all atomic TTA transforms...")

    store = {
        "orig": [],
        "flip": [],
        "shift1_left": [],
        "shift1_right": [],
        "shift1_up": [],
        "shift1_down": [],
        "flip_shift1_left": [],
        "flip_shift1_right": [],
        "shift2_left": [],
        "shift2_right": [],
        "bright_plus": [],
        "bright_minus": [],
        "contrast_plus": [],
        "contrast_minus": [],
        "targets": [],
    }

    total_batches = len(loader)
    for idx, batch in enumerate(loader):
        if (idx + 1) % 15 == 0 or (idx + 1) == total_batches:
            print(f"    Batch [{idx+1}/{total_batches}] processed...")

        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch

        images = images.to(device, non_blocking=True)
        store["targets"].append(targets.cpu())

        # 1. Original & Flip
        x_orig = images
        x_flip = tta_flip(x_orig)

        # 2. Shift 1px
        s1_l, s1_r, s1_u, s1_d = tta_shift_1px(x_orig)
        fs1_l, fs1_r, _, _ = tta_shift_1px(x_flip)

        # 3. Shift 2px
        s2_l, s2_r, _, _ = tta_shift_2px(x_orig)

        # 4. Brightness & Contrast
        b_p, b_m = tta_brightness(x_orig)
        c_p, c_m = tta_contrast(x_orig)

        # Batch forward all views efficiently
        views = {
            "orig": x_orig,
            "flip": x_flip,
            "shift1_left": s1_l,
            "shift1_right": s1_r,
            "shift1_up": s1_u,
            "shift1_down": s1_d,
            "flip_shift1_left": fs1_l,
            "flip_shift1_right": fs1_r,
            "shift2_left": s2_l,
            "shift2_right": s2_r,
            "bright_plus": b_p,
            "bright_minus": b_m,
            "contrast_plus": c_p,
            "contrast_minus": c_m,
        }

        for key, view_tensor in views.items():
            out = model._forward_single(view_tensor)
            logits = out["logits"].cpu()
            alpha = out["alpha"].view(-1, 1).cpu() if "alpha" in out else torch.ones((logits.size(0), 1))
            store[key].append((logits, alpha))

    # Concatenate all batches
    all_targets = torch.cat(store["targets"], dim=0).numpy()
    view_data = {}
    for key in store:
        if key == "targets":
            continue
        all_logits = torch.cat([b[0] for b in store[key]], dim=0)
        all_alphas = torch.cat([b[1] for b in store[key]], dim=0)
        view_data[key] = {"logits": all_logits, "alpha": all_alphas}

    return view_data, all_targets


def evaluate_predictions(probs_or_logits, targets):
    if isinstance(probs_or_logits, torch.Tensor):
        probs_or_logits = probs_or_logits.numpy()
    preds = np.argmax(probs_or_logits, axis=-1)
    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    cm = confusion_matrix(targets, preds, labels=list(range(len(EMOTION_NAMES))))
    return acc, macro_f1, acc * macro_f1, cm


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fallback = load_config(args.config)[0]
    model, cfg, _ = load_checkpoint_model(args.weights, device, fallback)
    m_cfg = cfg["model"]
    data_path = resolve_data_path(cfg, args.env, args.data_path, ("val",))
    tf = transforms_from_config(cfg, "val")
    ds = PureImageFER2013(data_path=data_path, split="val", transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.get("data", {}).get("num_workers", 2),
                        generator=torch.Generator().manual_seed(43))
    ckpt_path = Path(args.weights)

    print(f"\n=======================================================")
    print(f"[TTA SEARCH] SYSTEMATIC SEARCH FOR BEST TTA ON FER2013")
    print(f"Model Backbone: {m_cfg.get('backbone')} | Split: {args.split.upper()}")
    print(f"Total Samples:  {len(ds)} | Device: {device}")
    print(f"Weights:        {ckpt_path}")
    print(f"=======================================================\n")

    # 3. Extract all views once
    views, targets = extract_all_views_outputs(model, loader, device)

    # Helper lambdas
    def get_logits(key):
        return views[key]["logits"]

    def get_probs(key):
        return F.softmax(views[key]["logits"], dim=-1)

    def get_alpha(key):
        return views[key]["alpha"]

    # ------------------------------------------------------------------
    # 4. Evaluate Grid of TTA Strategies
    # ------------------------------------------------------------------
    results = []

    def record(name, num_crops, probs_tensor):
        acc, f1, score, cm = evaluate_predictions(probs_tensor, targets)
        results.append({
            "name": name,
            "crops": num_crops,
            "acc": acc,
            "f1": f1,
            "score": score,
            "cm": cm,
            "probs": probs_tensor,
        })

    # Strategy 0: Single Image Baseline
    record("Baseline: Single Image (1-Crop)", 1, get_probs("orig"))

    # Strategy 1: Horizontal Flip (Logits Average)
    p_flip_logits = F.softmax(0.5 * (get_logits("orig") + get_logits("flip")), dim=-1)
    record("Flip TTA (2-Crop, Logits Average)", 2, p_flip_logits)

    # Strategy 2: Horizontal Flip (Probabilities Average)
    p_flip_prob = 0.5 * (get_probs("orig") + get_probs("flip"))
    record("Flip TTA (2-Crop, Prob Average)", 2, p_flip_prob)

    # Strategy 3: Horizontal Flip (SCN Alpha-Weighted Average)
    a_orig = get_alpha("orig")
    a_flip = get_alpha("flip")
    p_flip_alpha = (a_orig * get_probs("orig") + a_flip * get_probs("flip")) / (a_orig + a_flip + 1e-6)
    record("Flip TTA (2-Crop, SCN Alpha-Weighted)", 2, p_flip_alpha)

    # Strategy 4: 1-Pixel Shift Horizontal + Flip (4-Crop)
    p_shift1_4c = (
        get_probs("orig") + get_probs("flip") +
        get_probs("shift1_left") + get_probs("shift1_right")
    ) / 4.0
    record("1px Horizontal Shift + Flip (4-Crop)", 4, p_shift1_4c)

    # Strategy 5: Symmetric 1-Pixel Shift + Flips (6-Crop)
    p_shift1_6c = (
        get_probs("orig") + get_probs("flip") +
        get_probs("shift1_left") + get_probs("shift1_right") +
        get_probs("flip_shift1_left") + get_probs("flip_shift1_right")
    ) / 6.0
    record("1px Shift Symm + Flips (6-Crop)", 6, p_shift1_6c)

    # Strategy 6: Full 1-Pixel Shift 4-Way + Flip (6-Crop)
    p_shift1_full = (
        get_probs("orig") + get_probs("flip") +
        get_probs("shift1_left") + get_probs("shift1_right") +
        get_probs("shift1_up") + get_probs("shift1_down")
    ) / 6.0
    record("1px Shift 4-Way Cross + Flip (6-Crop)", 6, p_shift1_full)

    # Strategy 7: 2-Pixel Shift Horizontal + Flip (4-Crop)
    p_shift2_4c = (
        get_probs("orig") + get_probs("flip") +
        get_probs("shift2_left") + get_probs("shift2_right")
    ) / 4.0
    record("2px Shift Horizontal + Flip (4-Crop)", 4, p_shift2_4c)

    # Strategy 8: Photometric Brightness + Flip (4-Crop)
    p_bright_4c = (
        get_probs("orig") + get_probs("flip") +
        get_probs("bright_plus") + get_probs("bright_minus")
    ) / 4.0
    record("Brightness (+/- 0.06) + Flip (4-Crop)", 4, p_bright_4c)

    # Strategy 9: Photometric Contrast + Flip (4-Crop)
    p_contrast_4c = (
        get_probs("orig") + get_probs("flip") +
        get_probs("contrast_plus") + get_probs("contrast_minus")
    ) / 4.0
    record("Contrast (+/- 8%) + Flip (4-Crop)", 4, p_contrast_4c)

    # Strategy 10: Contrast + 1px Shift + Flip (6-Crop Hybrid)
    p_hybrid_6c = (
        2.0 * get_probs("orig") + 2.0 * get_probs("flip") +
        get_probs("shift1_left") + get_probs("shift1_right") +
        get_probs("contrast_plus") + get_probs("contrast_minus")
    ) / 8.0
    record("Hybrid: 1px Shift + Contrast + Flip (6-Crop)", 6, p_hybrid_6c)

    # Scalar temperature after logits averaging cannot change argmax/accuracy.

    # ------------------------------------------------------------------
    # 5. Print Leaderboard Table
    # ------------------------------------------------------------------
    # Sort results by Accuracy (descending), then F1 (descending)
    results.sort(key=lambda x: (x["acc"], x["f1"]), reverse=True)

    baseline_acc = next(r["acc"] for r in results if "Baseline" in r["name"])
    print("\n" + "=" * 78)
    print(f"{'TTA STRATEGY LEADERBOARD (SORTED BY ACCURACY)':^78}")
    print("=" * 78)
    print(f"{'Rank':<4} | {'Strategy Name':<42} | {'Crops':<5} | {'Acc (%)':<9} | {'Diff':<7} | {'Macro F1':<8}")
    print("-" * 78)

    for rank, r in enumerate(results, 1):
        diff = (r["acc"] - baseline_acc) * 100.0
        marker = "🏆 [BEST]" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
        print(
            f"{rank:<4} | {r['name']:<42} | {r['crops']:<5} | "
            f"{r['acc']*100:>7.2f}% | {diff:>+6.2f}% | {r['f1']*100:>6.2f}% {marker}"
        )
    print("=" * 78)

    best = results[0]
    print(f"\n🥇 Best TTA Strategy: {best['name']}")
    print(f"   Accuracy:    {best['acc']*100:.2f}%  (Improvement: +{(best['acc'] - baseline_acc)*100:.2f}%)")
    print(f"   Macro F1:    {best['f1']*100:.2f}%")
    print(f"   Score:       {best['score']:.4f}\n")

    # Per-class accuracies of best strategy
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(best["cm"]) / best["cm"].sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)

    print("--- Per-Class Accuracies of Champion TTA ---")
    for name, acc_val in zip(EMOTION_NAMES, per_class_acc):
        print(f"  {name.ljust(10)}: {acc_val*100:.2f}%")

    # Save Confusion Matrix of Champion TTA
    cm_path = out_dir / f"confusion_matrix_champion_tta_{args.split}.png"
    plot_confusion_matrix(
        best["cm"],
        class_names=EMOTION_NAMES,
        save_path=cm_path,
        title=f"Champion TTA ({best['name']}) - {args.split.upper()} Acc: {best['acc']*100:.2f}%",
    )
    print(f"\n[SAVE] Champion TTA Confusion Matrix saved -> {cm_path}\n")


if __name__ == "__main__":
    main()
