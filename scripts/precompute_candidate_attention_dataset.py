"""Precompute candidate-level attention dataset from pixel candidate subgraphs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _pad_1d(values: torch.Tensor, max_items: int, fill: float | int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.full((max_items,), fill, dtype=dtype)
    n = min(max_items, int(values.numel()))
    if n > 0:
        out[:n] = values[:n].to(dtype=dtype)
    return out


def _pad_2d(values: torch.Tensor, max_items: int, dim: int, fill: float | int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.full((max_items, dim), fill, dtype=dtype)
    n = min(max_items, int(values.shape[0]))
    if n > 0:
        out[:n, : min(dim, values.shape[1])] = values[:n, :dim].to(dtype=dtype)
    return out


def _topology_field(topologies: list[dict[str, Any]], key: str, max_items: int, fill: int = 0) -> torch.Tensor:
    values = []
    for topo in topologies[:max_items]:
        values.append(int(topo.get(key, fill)))
    if len(values) < max_items:
        values.extend([fill] * (max_items - len(values)))
    return torch.tensor(values, dtype=torch.long)


def _candidate_node_tensors(topologies: list[dict[str, Any]], max_items: int, max_nodes: int):
    node_indices = torch.full((max_items, max_nodes), -1, dtype=torch.long)
    node_mask = torch.zeros((max_items, max_nodes), dtype=torch.bool)
    for idx, topo in enumerate(topologies[:max_items]):
        nodes = torch.as_tensor(topo.get("node_indices", []), dtype=torch.long)
        n = min(max_nodes, int(nodes.numel()))
        if n > 0:
            node_indices[idx, :n] = nodes[:n]
            node_mask[idx, :n] = True
    return node_indices, node_mask


def _knn_edges_from_distance(dist: torch.Tensor, valid: torch.Tensor, k: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    valid_idx = torch.where(valid)[0]
    if int(valid_idx.numel()) <= 1 or k <= 0:
        return edges
    large = torch.full_like(dist, 1e9)
    dist = torch.where(valid.view(1, -1), dist, large)
    dist[:, torch.arange(dist.shape[0])] = 1e9
    for src in valid_idx.tolist():
        kk = min(int(k), int(valid_idx.numel()) - 1)
        nbrs = torch.topk(dist[src], k=kk, largest=False).indices.tolist()
        for dst in nbrs:
            if bool(valid[dst]):
                edges.append((int(src), int(dst)))
    return edges


def build_candidate_edges(
    x: torch.Tensor,
    centers: torch.Tensor,
    valid: torch.Tensor,
    *,
    k_spatial: int,
    k_feature: int,
):
    edge_types: dict[tuple[int, int], int] = {}
    spatial_dist = torch.cdist(centers, centers)
    for edge in _knn_edges_from_distance(spatial_dist, valid, k_spatial):
        edge_types[edge] = edge_types.get(edge, 0) | 1

    if k_feature > 0:
        x_norm = F.normalize(x.float(), dim=1)
        cos_dist = 1.0 - (x_norm @ x_norm.T)
        for edge in _knn_edges_from_distance(cos_dist, valid, k_feature):
            edge_types[edge] = edge_types.get(edge, 0) | 2

    if not edge_types:
        return (
            torch.zeros((2, 1), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.bool),
        )

    items = sorted(edge_types.items())
    edge_index = torch.tensor([[src, dst] for (src, dst), _ in items], dtype=torch.long).T.contiguous()
    attrs = []
    for (src, dst), edge_type in items:
        dx = float(centers[dst, 0] - centers[src, 0])
        dy = float(centers[dst, 1] - centers[src, 1])
        dist = float(torch.sqrt(torch.tensor(dx * dx + dy * dy)).item())
        attrs.append([dx, dy, dist, float(edge_type)])
    return (
        edge_index,
        torch.tensor(attrs, dtype=torch.float32),
        torch.ones((len(items),), dtype=torch.bool),
    )


def process_split(
    split: str,
    candidate_dir: Path,
    out_dir: Path,
    topologies: list[dict[str, Any]],
    max_candidates: int,
    max_nodes: int,
    k_spatial: int,
    k_feature: int,
) -> int:
    path = candidate_dir / f"{split}_pixel_candidates.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    samples = _torch_load(path)
    topo_slice = topologies[:max_candidates]
    radius = _topology_field(topo_slice, "radius", max_candidates, fill=0)
    topo_coverage = _topology_field(topo_slice, "coverage_cell", max_candidates, fill=-1)
    node_indices, node_mask = _candidate_node_tensors(topo_slice, max_candidates, max_nodes)

    out_samples = []
    for idx, sample in enumerate(samples):
        x = torch.as_tensor(sample["x"]).float()
        mask = torch.as_tensor(sample.get("mask", torch.ones(x.shape[0]))).bool()
        centers = torch.as_tensor(sample["centers"]).float()
        bbox = torch.as_tensor(sample["bbox"]).float()
        coverage = torch.as_tensor(sample.get("coverage_cell", topo_coverage[: x.shape[0]])).long()
        n = min(max_candidates, int(x.shape[0]))

        candidate_x = _pad_2d(x, max_candidates, int(x.shape[1]), 0.0, torch.float32)
        candidate_mask = _pad_1d(mask[:n], max_candidates, 0, torch.bool)
        candidate_centers = _pad_2d(centers, max_candidates, 2, 0.0, torch.float32)
        candidate_bbox = _pad_2d(bbox, max_candidates, 4, 0.0, torch.float32)
        candidate_radius = radius.clone()
        candidate_coverage = _pad_1d(coverage[:n], max_candidates, -1, torch.long)

        edge_index, edge_attr, edge_valid = build_candidate_edges(
            candidate_x,
            candidate_centers,
            candidate_mask,
            k_spatial=k_spatial,
            k_feature=k_feature,
        )

        out_samples.append(
            {
                "graph_id": int(sample["graph_id"]),
                "label": int(sample["label"]),
                "candidate_x": candidate_x,
                "candidate_mask": candidate_mask,
                "candidate_centers": candidate_centers,
                "candidate_bbox": candidate_bbox,
                "candidate_radius": candidate_radius,
                "candidate_coverage_cell": candidate_coverage,
                "candidate_node_indices": node_indices.clone(),
                "candidate_node_mask": node_mask.clone(),
                "candidate_edge_index": edge_index,
                "candidate_edge_attr": edge_attr,
                "candidate_edge_valid": edge_valid,
            }
        )
        if (idx + 1) % 2000 == 0 or idx + 1 == len(samples):
            print(f"  [{split}] {idx + 1:6d}/{len(samples)}", flush=True)

    out_path = out_dir / f"{split}_candidate_attention.pt"
    torch.save(out_samples, out_path)
    print(f"[{split}] saved -> {out_path} ({out_path.stat().st_size / 1024**2:.2f} MB)")
    return len(out_samples)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_dir", default="artifacts/pixel_candidate_subgraphs_v2")
    p.add_argument("--out_dir", default="artifacts/candidate_attention_dataset_v1")
    p.add_argument("--max_candidates", type=int, default=128)
    p.add_argument("--k_spatial", type=int, default=8)
    p.add_argument("--k_feature", type=int, default=4)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _torch_load(candidate_dir / "meta.pt")
    topologies = list(meta["candidate_topologies"])
    max_nodes = int(meta.get("max_nodes_per_candidate", 0))
    max_candidates = min(int(args.max_candidates), len(topologies))

    print("=" * 80)
    print("Precompute Candidate Attention Dataset V1")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k:<24}: {v}")
    print(f"resolved_max_candidates : {max_candidates}")
    print(f"max_nodes_per_candidate : {max_nodes}")

    split_counts = {}
    for split in args.splits:
        split_counts[split] = process_split(
            split,
            candidate_dir,
            out_dir,
            topologies,
            max_candidates,
            max_nodes,
            int(args.k_spatial),
            int(args.k_feature),
        )

    out_meta = {
        "dataset_type": "candidate_attention_v1",
        "source_candidate_dir": str(candidate_dir),
        "max_candidates": max_candidates,
        "descriptor_dim": int(meta.get("descriptor_dim", 41)),
        "max_nodes_per_candidate": max_nodes,
        "k_spatial": int(args.k_spatial),
        "k_feature": int(args.k_feature),
        "edge_attr_names": ["dx", "dy", "dist", "edge_type"],
        "edge_attr_dim": 4,
        "height": meta.get("height", 48),
        "width": meta.get("width", 48),
        "coverage_grid": meta.get("coverage_grid"),
        "radii": meta.get("radii"),
        "seed_stride": meta.get("seed_stride"),
        "split_counts": split_counts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(out_meta, out_dir / "meta.pt")
    print(f"meta saved -> {out_dir / 'meta.pt'}")
    print("DONE")


if __name__ == "__main__":
    main()

