"""Audit pixel-preserving motif dataset alignment and coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _hist(counter: Counter, total: int, n: int) -> str:
    return "  ".join(f"{i}:{counter.get(i,0)}({100*counter.get(i,0)/max(total,1):.1f}%)" for i in range(n))


def audit_split(data_dir: Path, split: str, num_classes: int) -> None:
    path = data_dir / f"{split}_pixel_motif.pt"
    if not path.exists():
        print(f"[{split}] missing: {path}")
        return
    samples = _torch_load(path)
    label_hist = Counter()
    matched_global = Counter()
    matched_by_label = {c: Counter() for c in range(num_classes)}
    motif_argmax_hist = Counter()
    coverage_global = Counter()
    coverage_by_label = {c: Counter() for c in range(num_classes)}
    motif_align = 0
    top1_align = 0
    n_valid = 0

    for sample in samples:
        label = int(sample["label"])
        label_hist[label] += 1
        motif_vec = torch.as_tensor(sample["motif_score_vector"]).float()
        motif_pred = int(motif_vec.argmax().item())
        motif_argmax_hist[motif_pred] += 1
        motif_align += int(motif_pred == label)
        valid = torch.as_tensor(sample["mask"]).bool()
        n_valid += int(valid.sum().item())
        matched = torch.as_tensor(sample["matched_class"]).long()
        scores = torch.as_tensor(sample["match_scores"]).float()
        cells = torch.as_tensor(sample["coverage_cell"]).long()
        if valid.any():
            valid_idx = torch.where(valid)[0]
            best_idx = int(valid_idx[scores[valid].argmax()].item())
            top1_align += int(matched[best_idx].item() == label)
        for cls in matched[valid].tolist():
            matched_global[int(cls)] += 1
            matched_by_label[label][int(cls)] += 1
        for cell in cells[valid].tolist():
            coverage_global[int(cell)] += 1
            coverage_by_label[label][int(cell)] += 1

    print("\n" + "=" * 90)
    print(f"[{split}] Pixel Motif Audit")
    print("=" * 90)
    print(f"num samples          : {len(samples)}")
    print(f"label_hist           : {_hist(label_hist, len(samples), num_classes)}")
    print(f"matched_global       : {_hist(matched_global, n_valid, num_classes)}")
    print(f"motif_argmax_global  : {_hist(motif_argmax_hist, len(samples), num_classes)}")
    print(f"motif_alignment_acc  : {motif_align / max(len(samples),1):.4f}")
    print(f"top1_match_align_acc : {top1_align / max(len(samples),1):.4f}")
    print(f"coverage_global      : {dict(sorted(coverage_global.items()))}")

    print("\nmatched_class by true label:")
    for label in range(num_classes):
        total = sum(matched_by_label[label].values())
        print(f"  true {label} {EMOTION_NAMES[label]:<8}: {_hist(matched_by_label[label], total, num_classes)}")

    print("\ncoverage cells by true label:")
    for label in range(num_classes):
        print(f"  true {label} {EMOTION_NAMES[label]:<8}: {dict(sorted(coverage_by_label[label].items()))}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="artifacts/pixel_motif_dataset_v2")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--num_classes", type=int, default=7)
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.pt"
    print("=" * 90)
    print("Audit Pixel-preserving Motif Dataset V2")
    print("=" * 90)
    if meta_path.exists():
        meta = _torch_load(meta_path)
        print(f"class_names          : {meta.get('class_names')}")
        print(f"candidate_dir        : {meta.get('candidate_dir')}")
        print(f"motif_bank_path      : {meta.get('motif_bank_path')}")
        print(f"coverage_grid        : {meta.get('coverage_grid')}")
    for split in args.splits:
        audit_split(data_dir, split, args.num_classes)


if __name__ == "__main__":
    main()
