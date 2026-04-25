"""Inspect pixel-preserving motif dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf")


def inspect_split(data_dir: Path, split: str) -> None:
    path = data_dir / f"{split}_pixel_motif.pt"
    if not path.exists():
        print(f"[{split}] missing: {path}")
        return
    samples = _torch_load(path)
    print("\n" + "=" * 80)
    print(f"[{split}] {path}")
    print(f"num samples: {len(samples)}")
    if not samples:
        return
    s0 = samples[0]
    for key in [
        "x", "mask", "centers", "bbox", "selected_indices", "node_indices",
        "node_mask", "edge_index", "edge_attr", "match_scores", "matched_class",
        "matched_motif_id", "matched_disc_score", "motif_score_vector", "coverage_cell",
    ]:
        value = s0[key]
        print(f"{key:<24}: {tuple(value.shape) if torch.is_tensor(value) else type(value)}")

    label_hist = Counter()
    matched_hist = Counter()
    coverage_hist = Counter()
    score_values = []
    for idx, s in enumerate(samples):
        for key in ["x", "centers", "bbox", "edge_attr", "match_scores", "matched_disc_score", "motif_score_vector"]:
            _check_finite(f"{split}[{idx}].{key}", torch.as_tensor(s[key]))
        if s["edge_index"].numel() > 0:
            if int(s["edge_index"].min()) < 0 or int(s["edge_index"].max()) >= s["x"].shape[0]:
                raise RuntimeError(f"{split}[{idx}] edge_index out of range")
        label_hist[int(s["label"])] += 1
        valid = torch.as_tensor(s["mask"]).bool()
        for cls in torch.as_tensor(s["matched_class"])[valid].tolist():
            matched_hist[int(cls)] += 1
        for cell in torch.as_tensor(s["coverage_cell"])[valid].tolist():
            coverage_hist[int(cell)] += 1
        score_values.append(torch.as_tensor(s["match_scores"]).float()[valid])
    scores = torch.cat(score_values) if score_values else torch.tensor([])
    print(f"label_hist       : {dict(sorted(label_hist.items()))}")
    print(f"matched_hist     : {dict(sorted(matched_hist.items()))}")
    print(f"coverage_hist    : {dict(sorted(coverage_hist.items()))}")
    print(f"match mean/std   : {scores.mean().item():.4f} / {scores.std(unbiased=False).item():.4f}")
    print("No NaN/Inf detected.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="artifacts/pixel_motif_dataset_v2")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.pt"
    print("=" * 80)
    print("Inspect Pixel Motif Dataset V2")
    print("=" * 80)
    if meta_path.exists():
        print(_torch_load(meta_path))
    else:
        print(f"meta missing: {meta_path}")
    for split in ["train", "val", "test"]:
        inspect_split(data_dir, split)


if __name__ == "__main__":
    main()
