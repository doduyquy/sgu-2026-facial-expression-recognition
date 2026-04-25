"""Inspect pixel-preserving candidate subgraph artifact."""

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="artifacts/pixel_candidate_subgraphs_v2")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    meta = _torch_load(data_dir / "meta.pt")
    print("=" * 80)
    print("Inspect Pixel Candidate Subgraphs V2")
    print("=" * 80)
    for key in [
        "descriptor_dim", "num_candidates", "max_nodes_per_candidate",
        "height", "width", "coverage_grid", "radii", "seed_stride",
        "graph_config_version", "split_counts",
    ]:
        print(f"{key:<26}: {meta.get(key)}")
    topologies = meta["candidate_topologies"]
    node_counts = Counter(int(t["num_nodes"]) for t in topologies)
    radius_counts = Counter(int(t["radius"]) for t in topologies)
    coverage_counts = Counter(int(t["coverage_cell"]) for t in topologies)
    print(f"topology node_counts      : {dict(sorted(node_counts.items()))}")
    print(f"topology radius_counts    : {dict(sorted(radius_counts.items()))}")
    print(f"topology coverage_counts  : {dict(sorted(coverage_counts.items()))}")
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}_pixel_candidates.pt"
        if not path.exists():
            print(f"\n[{split}] missing: {path}")
            continue
        samples = _torch_load(path)
        print("\n" + "-" * 80)
        print(f"[{split}] samples={len(samples)} path={path}")
        if samples:
            s0 = samples[0]
            for key in ["x", "mask", "centers", "bbox", "coverage_cell"]:
                print(f"  {key:<16}: {tuple(s0[key].shape)}")
            if not torch.isfinite(s0["x"]).all():
                raise ValueError(f"{split}[0].x contains NaN/Inf")
            labels = Counter(int(s["label"]) for s in samples)
            print(f"  label_hist      : {dict(sorted(labels.items()))}")
    print("DONE")


if __name__ == "__main__":
    main()
