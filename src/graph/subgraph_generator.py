"""
Utilities for generating local candidate subgraphs from a resolved pixel graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from data.graph_types import ResolvedPixelGraph


@dataclass
class CandidateSubgraph:
    """
    Lightweight subgraph container used by the subgraph-first baseline.
    """

    graph_id: int
    label: int
    original_node_indices: torch.Tensor
    node_features_sub: torch.Tensor
    edge_index_sub: torch.Tensor
    edge_attr_sub: torch.Tensor
    seed_node: int
    radius: int
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features_sub.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index_sub.shape[1])


def build_adjacency_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """
    Build adjacency list from COO edge_index [2, E].
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
        )
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency

    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()
    for src, dst in zip(src_nodes, dst_nodes):
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            adjacency[src].append(dst)
    return adjacency


def sample_seed_nodes(
    num_nodes: int,
    stride: Optional[int] = None,
    max_seeds: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> List[int]:
    """
    Sample seed nodes on the image grid if height/width are known.
    Fallback to evenly-spaced node indices otherwise.
    """
    if num_nodes <= 0:
        return []

    effective_stride = max(1, int(stride or 1))
    seeds: List[int] = []

    if height is not None and width is not None and height * width == num_nodes:
        for y in range(0, height, effective_stride):
            for x in range(0, width, effective_stride):
                seeds.append(y * width + x)
    else:
        seeds = list(range(0, num_nodes, effective_stride))

    if max_seeds is not None and len(seeds) > max_seeds:
        seeds = seeds[:max_seeds]
    return seeds


def extract_radius_subgraph(
    graph: ResolvedPixelGraph,
    seed_node: int,
    radius: int,
    adjacency_list: Sequence[Sequence[int]],
    max_nodes: Optional[int] = None,
) -> CandidateSubgraph:
    """
    Extract a local subgraph around one seed node with BFS radius expansion.
    """
    if not (0 <= seed_node < graph.num_nodes):
        raise IndexError(f"seed_node={seed_node} is out of range for {graph.num_nodes} nodes")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    visited = {seed_node}
    queue = deque([(seed_node, 0)])
    ordered_nodes: List[int] = []

    while queue:
        current, dist = queue.popleft()
        ordered_nodes.append(current)

        if max_nodes is not None and len(ordered_nodes) >= max_nodes:
            break
        if dist >= radius:
            continue

        for neighbor in adjacency_list[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, dist + 1))

    original_node_indices = torch.tensor(sorted(ordered_nodes), dtype=torch.long)
    local_index = {
        int(node_idx): local_idx
        for local_idx, node_idx in enumerate(original_node_indices.tolist())
    }

    src_all = graph.edge_index[0]
    dst_all = graph.edge_index[1]
    keep_mask = torch.tensor(
        [
            int(src.item()) in local_index and int(dst.item()) in local_index
            for src, dst in zip(src_all, dst_all)
        ],
        dtype=torch.bool,
    )

    edge_index_sub = graph.edge_index[:, keep_mask]
    if edge_index_sub.numel() > 0:
        remapped_src = [local_index[int(v)] for v in edge_index_sub[0].tolist()]
        remapped_dst = [local_index[int(v)] for v in edge_index_sub[1].tolist()]
        edge_index_sub = torch.tensor(
            [remapped_src, remapped_dst],
            dtype=torch.long,
        )
        edge_attr_sub = graph.edge_attr[keep_mask]
    else:
        edge_index_sub = torch.empty((2, 0), dtype=torch.long)
        edge_attr_sub = torch.empty(
            (0, graph.num_edge_features),
            dtype=graph.edge_attr.dtype,
        )

    node_features_sub = graph.node_features[original_node_indices]
    return CandidateSubgraph(
        graph_id=graph.graph_id,
        label=graph.label,
        original_node_indices=original_node_indices,
        node_features_sub=node_features_sub,
        edge_index_sub=edge_index_sub,
        edge_attr_sub=edge_attr_sub,
        seed_node=seed_node,
        radius=radius,
        metadata={
            "split": graph.split,
            "num_nodes": int(node_features_sub.shape[0]),
            "num_edges": int(edge_index_sub.shape[1]),
        },
    )


def _subgraph_signature(node_indices: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in node_indices)


def generate_candidate_subgraphs(
    graph: ResolvedPixelGraph,
    radius: int = 1,
    seed_stride: int = 4,
    max_candidates: int = 64,
    max_nodes_per_subgraph: Optional[int] = None,
) -> List[CandidateSubgraph]:
    """
    Generate unique local candidate subgraphs from one resolved graph.
    """
    adjacency_list = build_adjacency_list(graph.edge_index, graph.num_nodes)
    seeds = sample_seed_nodes(
        num_nodes=graph.num_nodes,
        stride=seed_stride,
        max_seeds=max_candidates,
        height=graph.metadata.get("height"),
        width=graph.metadata.get("width"),
    )

    candidates: List[CandidateSubgraph] = []
    seen_signatures: set[tuple[int, ...]] = set()

    for seed in seeds:
        subgraph = extract_radius_subgraph(
            graph=graph,
            seed_node=seed,
            radius=radius,
            adjacency_list=adjacency_list,
            max_nodes=max_nodes_per_subgraph,
        )
        signature = _subgraph_signature(subgraph.original_node_indices.tolist())
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidates.append(subgraph)
        if len(candidates) >= max_candidates:
            break

    return candidates
