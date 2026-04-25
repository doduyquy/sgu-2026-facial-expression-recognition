"""Pixel-preserving candidate topology and descriptor helpers."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[int(src)].append(int(dst))
    return adj


def bfs_nodes(seed: int, radius: int, adj: List[List[int]], max_nodes: Optional[int] = None) -> List[int]:
    visited = {int(seed)}
    queue = deque([(int(seed), 0)])
    out = []
    while queue:
        node, dist = queue.popleft()
        out.append(node)
        if max_nodes is not None and len(out) >= max_nodes:
            break
        if dist >= radius:
            continue
        for nb in adj[node]:
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, dist + 1))
    return sorted(out)


def evenly_limit(items: List[int], max_items: Optional[int]) -> List[int]:
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return items
    step = len(items) / float(max_items)
    return [items[int(i * step)] for i in range(max_items)]


def grid_seeds(height: int, width: int, stride: int, max_seeds: Optional[int] = None) -> List[int]:
    seeds = []
    stride = max(1, int(stride))
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            seeds.append(y * width + x)
    return evenly_limit(seeds, max_seeds)


def node_bbox(node_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
    ys = torch.div(node_ids.long(), width, rounding_mode="floor").float()
    xs = (node_ids.long() % width).float()
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    return torch.tensor(
        [
            float(xs.min().item() / denom_x),
            float(ys.min().item() / denom_y),
            float(xs.max().item() / denom_x),
            float(ys.max().item() / denom_y),
        ],
        dtype=torch.float32,
    )


def node_center(node_ids: torch.Tensor, node_features: torch.Tensor, height: int, width: int) -> torch.Tensor:
    nf = node_features[node_ids].float()
    if nf.ndim == 2 and nf.shape[1] >= 3:
        return torch.tensor([float(nf[:, 1].mean()), float(nf[:, 2].mean())], dtype=torch.float32)
    ys = torch.div(node_ids.long(), width, rounding_mode="floor").float()
    xs = (node_ids.long() % width).float()
    return torch.tensor(
        [float(xs.mean() / max(width - 1, 1)), float(ys.mean() / max(height - 1, 1))],
        dtype=torch.float32,
    )


def coverage_cell(center: torch.Tensor, grid_rows: int, grid_cols: int) -> int:
    x = float(center[0].clamp(0, 1).item())
    y = float(center[1].clamp(0, 1).item())
    col = min(grid_cols - 1, int(x * grid_cols))
    row = min(grid_rows - 1, int(y * grid_rows))
    return int(row * grid_cols + col)


def build_candidate_topologies(
    edge_index: torch.Tensor,
    num_nodes: int,
    height: int,
    width: int,
    seed_stride: int = 4,
    radii: Iterable[int] = (1, 2),
    max_candidates: int = 128,
    max_nodes_per_subgraph: Optional[int] = None,
    coverage_grid: Tuple[int, int] = (4, 4),
) -> List[Dict]:
    """Build image-grid candidate topologies once for all FER samples."""
    adj = build_adjacency(edge_index, num_nodes)
    seeds = grid_seeds(height, width, seed_stride, max_seeds=None)
    rows, cols = coverage_grid
    src_all = edge_index[0]
    dst_all = edge_index[1]

    candidates: List[Dict] = []
    seen = set()
    for radius in radii:
        for seed in seeds:
            nodes = bfs_nodes(seed, int(radius), adj, max_nodes=max_nodes_per_subgraph)
            sig = (int(radius), tuple(nodes))
            if sig in seen:
                continue
            seen.add(sig)
            node_ids = torch.tensor(nodes, dtype=torch.long)
            local = {int(v): i for i, v in enumerate(nodes)}
            node_set = set(nodes)
            keep = torch.tensor(
                [int(s) in node_set and int(d) in node_set for s, d in zip(src_all.tolist(), dst_all.tolist())],
                dtype=torch.bool,
            )
            edge_attr_indices = keep.nonzero(as_tuple=False).view(-1)
            ei = edge_index[:, keep]
            if ei.numel() > 0:
                ei = torch.tensor(
                    [[local[int(v)] for v in ei[0].tolist()], [local[int(v)] for v in ei[1].tolist()]],
                    dtype=torch.long,
                )
            else:
                ei = torch.empty((2, 0), dtype=torch.long)

            bbox = node_bbox(node_ids, height, width)
            seed_center = torch.tensor(
                [(seed % width) / max(width - 1, 1), (seed // width) / max(height - 1, 1)],
                dtype=torch.float32,
            )
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "seed_node": int(seed),
                    "radius": int(radius),
                    "node_indices": node_ids,
                    "edge_index_sub": ei,
                    "edge_mask": keep,
                    "edge_attr_indices": edge_attr_indices,
                    "bbox": bbox,
                    "seed_center": seed_center,
                    "coverage_cell": coverage_cell(seed_center, rows, cols),
                    "num_nodes": int(node_ids.numel()),
                    "num_edges": int(ei.shape[1]),
                }
            )

    if len(candidates) > max_candidates:
        keep_indices = evenly_limit(list(range(len(candidates))), max_candidates)
        candidates = [candidates[i] for i in keep_indices]
        for new_id, cand in enumerate(candidates):
            cand["candidate_id"] = int(new_id)

    return candidates


def descriptor_from_topology(
    node_features: torch.Tensor,
    edge_attr: torch.Tensor,
    topology: Dict,
) -> torch.Tensor:
    nf = node_features[topology["node_indices"]].float()
    edge_selector = topology.get("edge_attr_indices", topology["edge_mask"])
    ea = edge_attr[edge_selector].float()

    node_mean = nf.mean(dim=0)
    node_std = nf.std(dim=0, unbiased=False)
    node_min = nf.min(dim=0).values
    node_max = nf.max(dim=0).values

    n_nodes = float(topology["num_nodes"])
    n_edges = float(topology["num_edges"])
    density = n_edges / max(n_nodes * (n_nodes - 1.0), 1.0)
    struct = torch.tensor([n_nodes, n_edges, density], dtype=torch.float32)

    if ea.ndim == 2 and ea.shape[0] > 0 and ea.shape[1] > 0:
        edge_mean = ea.mean(dim=0)
        edge_std = ea.std(dim=0, unbiased=False)
    else:
        edge_dim = int(ea.shape[1]) if ea.ndim == 2 else 0
        edge_mean = torch.zeros(edge_dim, dtype=torch.float32)
        edge_std = torch.zeros(edge_dim, dtype=torch.float32)

    desc = torch.cat([node_mean, node_std, node_min, node_max, struct, edge_mean, edge_std], dim=0)
    return torch.nan_to_num(desc.float(), nan=0.0, posinf=0.0, neginf=0.0)


def build_directed_knn_edges(centers: torch.Tensor, knn_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    K = int(centers.shape[0])
    if K <= 1:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float32)

    k = max(1, min(int(knn_k), K - 1))
    diff = centers.unsqueeze(1) - centers.unsqueeze(0)
    dist = diff.pow(2).sum(dim=-1).sqrt()
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)

    src, dst, attrs = [], [], []
    for i in range(K):
        for j in nn_idx[i].tolist():
            dx = float(centers[j, 0] - centers[i, 0])
            dy = float(centers[j, 1] - centers[i, 1])
            d = float((dx * dx + dy * dy) ** 0.5)
            src.append(i)
            dst.append(int(j))
            attrs.append([dx, dy, d])

    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(attrs, dtype=torch.float32)


RICH_MOTIF_EDGE_ATTR_NAMES = [
    "dx",
    "dy",
    "dist",
    "bbox_iou",
    "descriptor_cosine",
    "same_matched_class",
    "same_motif_id",
    "abs_match_score_diff",
    "match_score_src",
    "match_score_dst",
    "disc_score_src",
    "disc_score_dst",
    "same_coverage_cell",
]


def bbox_iou_matrix(bbox: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized boxes [K, 4] in xyxy format."""
    bbox = torch.as_tensor(bbox).float()
    K = int(bbox.shape[0])
    if K == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    x1 = torch.maximum(bbox[:, None, 0], bbox[None, :, 0])
    y1 = torch.maximum(bbox[:, None, 1], bbox[None, :, 1])
    x2 = torch.minimum(bbox[:, None, 2], bbox[None, :, 2])
    y2 = torch.minimum(bbox[:, None, 3], bbox[None, :, 3])
    inter = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)

    area = (bbox[:, 2] - bbox[:, 0]).clamp_min(0.0) * (bbox[:, 3] - bbox[:, 1]).clamp_min(0.0)
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp_min(1e-8)


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity for row vectors [K, D]."""
    x = torch.as_tensor(x).float()
    denom = x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    xn = x / denom
    return torch.mm(xn, xn.t()).clamp(-1.0, 1.0)


def build_directed_knn_rich_edges(
    centers: torch.Tensor,
    bbox: torch.Tensor,
    descriptors: torch.Tensor,
    match_scores: torch.Tensor,
    matched_class: torch.Tensor,
    matched_motif_id: torch.Tensor,
    matched_disc_score: torch.Tensor,
    coverage_cell: torch.Tensor,
    knn_k: int,
    edge_attr_names: Sequence[str] = RICH_MOTIF_EDGE_ATTR_NAMES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build KNN edges between selected motif nodes with graph-only relational features.

    The first three dimensions stay compatible with the old spatial edge_attr:
    [dx, dy, dist]. Extra dimensions describe bbox overlap, descriptor similarity,
    motif agreement, score confidence, and region agreement.
    """
    centers = torch.as_tensor(centers).float()
    bbox = torch.as_tensor(bbox).float()
    descriptors = torch.as_tensor(descriptors).float()
    match_scores = torch.as_tensor(match_scores).float()
    matched_class = torch.as_tensor(matched_class).long()
    matched_motif_id = torch.as_tensor(matched_motif_id).long()
    matched_disc_score = torch.as_tensor(matched_disc_score).float()
    coverage_cell = torch.as_tensor(coverage_cell).long()

    K = int(centers.shape[0])
    if K <= 1:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, len(edge_attr_names)), dtype=torch.float32),
        )

    k = max(1, min(int(knn_k), K - 1))
    diff = centers.unsqueeze(1) - centers.unsqueeze(0)
    dist = diff.pow(2).sum(dim=-1).sqrt()
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)

    iou = bbox_iou_matrix(bbox)
    desc_cos = cosine_similarity_matrix(descriptors)

    src, dst, attrs = [], [], []
    for i in range(K):
        for j in nn_idx[i].tolist():
            j = int(j)
            dx = float(centers[j, 0] - centers[i, 0])
            dy = float(centers[j, 1] - centers[i, 1])
            d = float((dx * dx + dy * dy) ** 0.5)
            feature_map = {
                "dx": dx,
                "dy": dy,
                "dist": d,
                "bbox_iou": float(iou[i, j]),
                "descriptor_cosine": float(desc_cos[i, j]),
                "same_matched_class": float(matched_class[i].item() == matched_class[j].item()),
                "same_motif_id": float(matched_motif_id[i].item() == matched_motif_id[j].item()),
                "abs_match_score_diff": float((match_scores[i] - match_scores[j]).abs()),
                "match_score_src": float(match_scores[i]),
                "match_score_dst": float(match_scores[j]),
                "disc_score_src": float(matched_disc_score[i]),
                "disc_score_dst": float(matched_disc_score[j]),
                "same_coverage_cell": float(coverage_cell[i].item() == coverage_cell[j].item()),
            }
            src.append(i)
            dst.append(j)
            attrs.append([feature_map[name] for name in edge_attr_names])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
    return edge_index, edge_attr
