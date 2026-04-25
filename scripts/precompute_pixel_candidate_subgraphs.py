"""Precompute pixel-preserving candidate subgraphs with descriptor + trace."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.graph_types import PixelGraphSample
from src.graph.subgraph_descriptor import infer_descriptor_dim
from src.motif_v2.topology import (
    build_candidate_topologies,
    coverage_cell,
    descriptor_from_topology,
    node_center,
)


EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _process_split(
    repo_root: str,
    split: str,
    out_dir: Path,
    topologies: List[dict],
    descriptor_dim: int,
    log_every: int,
    max_samples: int | None = None,
) -> List[dict]:
    reader = GraphRepositoryReader(repo_root)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    n_total = reader.num_samples(split)
    if max_samples is not None:
        n_total = min(int(max_samples), int(n_total or max_samples))

    C = len(topologies)
    samples = []
    t0 = time.time()
    for idx, raw in enumerate(reader.iter_split(split)):
        if max_samples is not None and idx >= max_samples:
            break
        graph = resolver.resolve(raw)
        x = torch.zeros((C, descriptor_dim), dtype=torch.float32)
        mask = torch.ones(C, dtype=torch.bool)
        centers = torch.zeros((C, 2), dtype=torch.float32)
        bboxes = torch.zeros((C, 4), dtype=torch.float32)
        cells = torch.zeros(C, dtype=torch.long)

        for cidx, topo in enumerate(topologies):
            x[cidx] = descriptor_from_topology(graph.node_features, graph.edge_attr, topo)
            centers[cidx] = node_center(topo["node_indices"], graph.node_features, shared.height, shared.width)
            bboxes[cidx] = topo["bbox"]
            cells[cidx] = int(topo["coverage_cell"])

        samples.append(
            {
                "graph_id": int(graph.graph_id),
                "label": int(graph.label),
                "x": x,
                "mask": mask,
                "centers": centers,
                "bbox": bboxes,
                "coverage_cell": cells,
            }
        )

        if (idx + 1) % log_every == 0 or (n_total is not None and idx + 1 == n_total):
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            print(f"  [{split}] {idx+1:6d}/{n_total or '?'} | {rate:6.1f} samp/s", flush=True)

    out_path = out_dir / f"{split}_pixel_candidates.pt"
    torch.save(samples, out_path)
    print(f"[{split}] saved {len(samples)} samples -> {out_path} ({out_path.stat().st_size / 1024**2:.2f} MB)")
    return samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default="artifacts/graph_repo")
    p.add_argument("--out_dir", default="artifacts/pixel_candidate_subgraphs_v2")
    p.add_argument("--max_candidates", type=int, default=128)
    p.add_argument("--seed_stride", type=int, default=4)
    p.add_argument("--radii", nargs="+", type=int, default=[1, 2])
    p.add_argument("--max_nodes_per_subgraph", type=int, default=None)
    p.add_argument("--coverage_grid", nargs=2, type=int, default=[4, 4])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--max_samples_per_split", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Precompute Pixel-preserving Candidate Subgraphs V2")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k:<26}: {v}")

    reader = GraphRepositoryReader(args.repo_root)
    shared = reader.load_shared()
    resolver = GraphResolver(shared)
    first_raw: PixelGraphSample = next(reader.iter_split(args.splits[0]))
    first_graph = resolver.resolve(first_raw)
    descriptor_dim = infer_descriptor_dim(first_graph.num_node_features, first_graph.num_edge_features)

    topologies = build_candidate_topologies(
        edge_index=shared.edge_index,
        num_nodes=shared.num_nodes,
        height=shared.height,
        width=shared.width,
        seed_stride=args.seed_stride,
        radii=args.radii,
        max_candidates=args.max_candidates,
        max_nodes_per_subgraph=args.max_nodes_per_subgraph,
        coverage_grid=tuple(args.coverage_grid),
    )
    max_nodes = max(int(t["node_indices"].numel()) for t in topologies)
    print(f"candidate_topologies : {len(topologies)} | max_nodes={max_nodes} | descriptor_dim={descriptor_dim}")

    split_counts = {}
    for split in args.splits:
        samples = _process_split(
            repo_root=args.repo_root,
            split=split,
            out_dir=out_dir,
            topologies=topologies,
            descriptor_dim=descriptor_dim,
            log_every=args.log_every,
            max_samples=args.max_samples_per_split,
        )
        split_counts[split] = len(samples)

    shared_cfg = shared.config_dict if isinstance(shared.config_dict, dict) else {}
    meta = {
        **vars(args),
        "descriptor_dim": descriptor_dim,
        "num_candidates": len(topologies),
        "max_nodes_per_candidate": max_nodes,
        "height": shared.height,
        "width": shared.width,
        "num_classes": 7,
        "class_names": EMOTION_NAMES,
        "node_feature_names": list(first_graph.node_feature_names),
        "edge_feature_names": list(first_graph.edge_feature_names),
        "graph_config_version": shared_cfg.get("version", "unknown"),
        "graph_config": shared_cfg,
        "candidate_topologies": topologies,
        "split_counts": split_counts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(meta, out_dir / "meta.pt")
    print(f"meta saved -> {out_dir / 'meta.pt'}")
    print("DONE")


if __name__ == "__main__":
    main()
