"""Audit motif-filtered dataset alignment and class/motif distributions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _format_hist(hist: Counter, total: int, num_classes: int) -> str:
    parts = []
    for class_id in range(num_classes):
        count = int(hist.get(class_id, 0))
        pct = 100.0 * count / max(total, 1)
        parts.append(f"{class_id}:{count}({pct:.1f}%)")
    return "  ".join(parts)


def _audit_split(data_dir: Path, split: str, num_classes: int) -> None:
    path = data_dir / f"{split}_motif_filtered.pt"
    if not path.exists():
        print(f"\n[{split}] missing: {path}")
        return

    samples = _torch_load(path)
    label_hist = Counter()
    matched_global = Counter()
    matched_by_label = {c: Counter() for c in range(num_classes)}
    motif_argmax_hist = Counter()
    motif_argmax_by_label = {c: Counter() for c in range(num_classes)}
    motif_align_count = 0
    top1_match_align_count = 0
    n_samples = 0
    n_valid_subgraphs = 0
    score_true_sum = torch.zeros(num_classes)
    score_true_count = torch.zeros(num_classes)
    score_margin_sum = torch.zeros(num_classes)

    for sample in samples:
        label = int(sample["label"])
        label_hist[label] += 1
        n_samples += 1

        motif_vec = torch.as_tensor(sample["motif_score_vector"]).float()
        pred_motif = int(motif_vec.argmax().item())
        motif_argmax_hist[pred_motif] += 1
        motif_argmax_by_label[label][pred_motif] += 1
        motif_align_count += int(pred_motif == label)

        score_true = float(motif_vec[label].item())
        other = motif_vec.clone()
        other[label] = -1e9
        margin = score_true - float(other.max().item())
        score_true_sum[label] += score_true
        score_margin_sum[label] += margin
        score_true_count[label] += 1

        mask = torch.as_tensor(sample["mask"]).bool()
        matched_class = torch.as_tensor(sample["matched_class"]).long()
        match_scores = torch.as_tensor(sample["match_scores"]).float()
        valid_idx = torch.where(mask)[0]
        n_valid_subgraphs += int(valid_idx.numel())

        if valid_idx.numel() > 0:
            valid_classes = matched_class[valid_idx]
            valid_scores = match_scores[valid_idx]
            best_local = int(valid_scores.argmax().item())
            top1_match_align_count += int(valid_classes[best_local].item() == label)
            for cls in valid_classes.tolist():
                cls = int(cls)
                matched_global[cls] += 1
                matched_by_label[label][cls] += 1

    print("\n" + "=" * 88)
    print(f"[{split}] Motif Dataset Audit")
    print("=" * 88)
    print(f"path                  : {path}")
    print(f"num samples           : {n_samples}")
    print(f"label histogram       : {_format_hist(label_hist, n_samples, num_classes)}")
    print(f"matched_class global  : {_format_hist(matched_global, n_valid_subgraphs, num_classes)}")
    print(f"motif_argmax global   : {_format_hist(motif_argmax_hist, n_samples, num_classes)}")
    print(f"motif_alignment_acc   : {motif_align_count / max(n_samples, 1):.4f}")
    print(f"top1_match_align_acc  : {top1_match_align_count / max(n_samples, 1):.4f}")

    print("\nmatched_class distribution by true label:")
    for label in range(num_classes):
        total = sum(matched_by_label[label].values())
        label_name = EMOTION_NAMES[label] if label < len(EMOTION_NAMES) else str(label)
        print(f"  true {label} {label_name:<8}: {_format_hist(matched_by_label[label], total, num_classes)}")

    print("\nmotif_score_vector argmax by true label:")
    for label in range(num_classes):
        total = sum(motif_argmax_by_label[label].values())
        label_name = EMOTION_NAMES[label] if label < len(EMOTION_NAMES) else str(label)
        print(f"  true {label} {label_name:<8}: {_format_hist(motif_argmax_by_label[label], total, num_classes)}")

    print("\nmean true motif score and margin(score_true - best_other):")
    for label in range(num_classes):
        count = max(float(score_true_count[label].item()), 1.0)
        mean_true = float((score_true_sum[label] / count).item())
        mean_margin = float((score_margin_sum[label] / count).item())
        label_name = EMOTION_NAMES[label] if label < len(EMOTION_NAMES) else str(label)
        print(f"  {label} {label_name:<8}: true_score={mean_true:.4f}  margin={mean_margin:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="artifacts/motif_filtered_dataset_v1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=7)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.pt"
    print("=" * 88)
    print("Motif-Filtered Dataset Audit")
    print("=" * 88)
    if meta_path.exists():
        meta = _torch_load(meta_path)
        print(f"meta class_names        : {meta.get('class_names')}")
        print(f"meta input_dir          : {meta.get('input_dir')}")
        print(f"meta motif_bank_path    : {meta.get('motif_bank_path')}")
        print(f"meta graph_config_ver   : {meta.get('graph_config_version')}")
    else:
        print(f"meta missing            : {meta_path}")

    for split in args.splits:
        _audit_split(data_dir, split, args.num_classes)


if __name__ == "__main__":
    main()
