"""Inspect motif-filtered dataset files."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.motif.motif_scoring import check_finite_tensor


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _inspect_split(data_dir: Path, split: str) -> None:
    path = data_dir / f"{split}_motif_filtered.pt"
    if not path.exists():
        print(f"\n[{split}] missing: {path}")
        return
    samples = _torch_load(path)
    print("\n" + "=" * 72)
    print(f"[{split}] {path}")
    print(f"num samples: {len(samples)}")
    if not samples:
        return

    s0 = samples[0]
    print(f"sample[0] keys          : {list(s0.keys())}")
    print(f"x shape                 : {tuple(s0['x'].shape)}")
    print(f"mask shape              : {tuple(s0['mask'].shape)}")
    print(f"edge_index shape        : {tuple(s0['edge_index'].shape)}")
    print(f"edge_attr shape         : {tuple(s0['edge_attr'].shape)}")
    print(f"match_scores shape      : {tuple(s0['match_scores'].shape)}")
    print(f"matched_class shape     : {tuple(s0['matched_class'].shape)}")
    print(f"motif_score_vector shape: {tuple(s0['motif_score_vector'].shape)}")
    print(f"label                   : {s0['label']}")

    label_hist = Counter()
    matched_hist = Counter()
    match_scores = []

    for idx, sample in enumerate(samples):
        for key in ["x", "centers", "edge_attr", "match_scores", "matched_disc_score", "motif_score_vector"]:
            check_finite_tensor(f"{split}[{idx}].{key}", sample[key])
        edge_index = sample["edge_index"]
        if edge_index.numel() > 0:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= sample["x"].shape[0]:
                raise RuntimeError(f"{split}[{idx}] edge_index out of range")

        label_hist[int(sample["label"])] += 1
        valid = torch.as_tensor(sample["mask"]).bool()
        for cls in torch.as_tensor(sample["matched_class"])[valid].tolist():
            matched_hist[int(cls)] += 1
        valid_scores = torch.as_tensor(sample["match_scores"]).float()[valid]
        if valid_scores.numel() > 0:
            match_scores.append(valid_scores)

    if match_scores:
        ms = torch.cat(match_scores)
        mean = ms.mean().item()
        std = ms.std(unbiased=False).item()
    else:
        mean = std = 0.0

    print(f"matched_class histogram : {dict(sorted(matched_hist.items()))}")
    print(f"label histogram         : {dict(sorted(label_hist.items()))}")
    print(f"match_scores mean/std   : {mean:.4f} / {std:.4f}")
    print("No NaN/Inf detected.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="artifacts/motif_filtered_dataset_v1")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.pt"
    print("=" * 72)
    print("Inspect Motif-Filtered Dataset")
    print("=" * 72)
    if meta_path.exists():
        meta = _torch_load(meta_path)
        print(f"meta: {meta}")
    else:
        print(f"meta missing: {meta_path}")

    for split in ["train", "val", "test"]:
        _inspect_split(data_dir, split)


if __name__ == "__main__":
    main()
