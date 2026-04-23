"""
Descriptor extraction for candidate subgraphs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.graph.subgraph_generator import CandidateSubgraph


def compute_density(num_nodes: int, num_edges: int, directed: bool = True) -> float:
    """
    Compute graph density safely for very small graphs.
    """
    if num_nodes <= 1:
        return 0.0

    if directed:
        denom = num_nodes * (num_nodes - 1)
    else:
        denom = num_nodes * (num_nodes - 1) / 2.0

    if denom <= 0:
        return 0.0
    return float(num_edges / denom)


def _safe_feature_stats(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor for feature stats, got {tuple(x.shape)}")

    if x.shape[0] == 0:
        feat_dim = int(x.shape[1]) if x.ndim == 2 else 0
        zeros = torch.zeros(feat_dim, dtype=torch.float32)
        return zeros, zeros.clone(), zeros.clone(), zeros.clone()

    x = x.float()
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    min_v = x.min(dim=0).values
    max_v = x.max(dim=0).values
    return mean, std, min_v, max_v


def subgraph_to_descriptor(
    subgraph: CandidateSubgraph,
    directed: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert one candidate subgraph into a fixed-dimensional descriptor tensor [D].
    """
    node_mean, node_std, node_min, node_max = _safe_feature_stats(subgraph.node_features_sub)

    structure_stats = torch.tensor(
        [
            float(subgraph.num_nodes),
            float(subgraph.num_edges),
            compute_density(subgraph.num_nodes, subgraph.num_edges, directed=directed),
        ],
        dtype=torch.float32,
    )

    if subgraph.edge_attr_sub.ndim == 2 and subgraph.edge_attr_sub.shape[1] > 0:
        edge_mean, edge_std, _, _ = _safe_feature_stats(subgraph.edge_attr_sub)
    else:
        edge_feat_dim = (
            int(subgraph.edge_attr_sub.shape[1])
            if subgraph.edge_attr_sub.ndim == 2
            else 0
        )
        edge_mean = torch.zeros(edge_feat_dim, dtype=torch.float32)
        edge_std = torch.zeros(edge_feat_dim, dtype=torch.float32)

    descriptor = torch.cat(
        [
            node_mean,
            node_std,
            node_min,
            node_max,
            structure_stats,
            edge_mean,
            edge_std,
        ],
        dim=0,
    ).to(dtype=dtype)

    descriptor = torch.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)
    return descriptor


def descriptor_to_numpy(descriptor: torch.Tensor) -> np.ndarray:
    """
    Convenience helper for downstream clustering / logging.
    """
    return descriptor.detach().cpu().numpy().astype(np.float32, copy=False)


def infer_descriptor_dim(
    node_feature_dim: int,
    edge_feature_dim: Optional[int],
) -> int:
    """
    Infer fixed descriptor dimension from feature inventory.
    """
    return (4 * int(node_feature_dim)) + 3 + (2 * int(edge_feature_dim or 0))
