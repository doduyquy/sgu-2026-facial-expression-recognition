"""
data/graph_types.py — Canonical graph data types for the FER-2013 pipeline.

Three levels of graph representation:

  1. SharedGraphStructure  — topology + static edge attrs, shared by ALL images.
  2. PixelGraphSample      — per-image data (node features + dynamic edge attrs).
  3. ResolvedPixelGraph    — merged view used by GNN / subgraph / motif code.

Rule: downstream modules must import these types, never define their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. SharedGraphStructure
# ---------------------------------------------------------------------------

@dataclass
class SharedGraphStructure:
    """
    Topology and static features shared by ALL images of the same grid size.

    Because every FER-2013 image is 48×48 with 8-neighbor connectivity,
    the edge_index and static edge attributes (dx, dy, dist) are IDENTICAL
    for every sample. We build this ONCE and reuse it everywhere.

    Attributes
    ----------
    height, width           : image grid dimensions
    connectivity            : 4 or 8
    edge_index              : int64 tensor [2, M] — COO source/destination
    edge_attr_static        : float32 tensor [M, S] — per topology-edge features
    static_feature_names    : ordered names corresponding to columns of edge_attr_static
    config_dict             : serialized GraphConfig used to build this structure
    """

    height: int
    width: int
    connectivity: int

    # [2, M]  — int64 (safe for torch.gather / PyG)
    edge_index: torch.Tensor

    # [M, S]  — float32 static edge attributes (dx, dy, dist …)
    edge_attr_static: torch.Tensor

    # Metadata
    static_feature_names: List[str] = field(default_factory=list)
    config_dict: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.height * self.width

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_static_features(self) -> int:
        return int(self.edge_attr_static.shape[1]) if self.edge_attr_static.ndim == 2 else 0

    def __repr__(self) -> str:
        return (
            f"SharedGraphStructure("
            f"grid={self.height}x{self.width}, "
            f"connectivity={self.connectivity}, "
            f"nodes={self.num_nodes}, "
            f"edges={self.num_edges}, "
            f"static_feats={self.static_feature_names})"
        )


# ---------------------------------------------------------------------------
# 2. PixelGraphSample
# ---------------------------------------------------------------------------

@dataclass
class PixelGraphSample:
    """
    Per-image graph data.  Does NOT contain edge_index or static edge attrs —
    those are in SharedGraphStructure and must be resolved at usage time.

    Attributes
    ----------
    graph_id            : unique integer id (= row index in the CSV split)
    label               : emotion class [0..6]
    split               : "train" | "val" | "test"
    usage               : original CSV Usage string
    height, width       : redundant but useful for validation
    node_features       : float32 tensor [N, d]
    edge_attr_dynamic   : float32 tensor [M, D] — per-image edge features
    node_feature_names  : names for columns of node_features
    dynamic_feature_names : names for columns of edge_attr_dynamic
    metadata            : free-form dict for future extensions
    """

    graph_id: int
    label: int
    split: str
    usage: str
    height: int
    width: int

    # [N, d]  float32
    node_features: torch.Tensor

    # [M, D]  float32
    edge_attr_dynamic: torch.Tensor

    # Feature name lists (must match tensor column order)
    node_feature_names: List[str] = field(default_factory=list)
    dynamic_feature_names: List[str] = field(default_factory=list)

    # Extension slot
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_node_features(self) -> int:
        return int(self.node_features.shape[1]) if self.node_features.ndim == 2 else 0

    @property
    def num_dynamic_features(self) -> int:
        return int(self.edge_attr_dynamic.shape[1]) if self.edge_attr_dynamic.ndim == 2 else 0

    def __repr__(self) -> str:
        return (
            f"PixelGraphSample("
            f"id={self.graph_id}, label={self.label}, split={self.split!r}, "
            f"node_features={tuple(self.node_features.shape)}, "
            f"edge_attr_dynamic={tuple(self.edge_attr_dynamic.shape)})"
        )


# ---------------------------------------------------------------------------
# 3. ResolvedPixelGraph
# ---------------------------------------------------------------------------

@dataclass
class ResolvedPixelGraph:
    """
    Full graph view ready for GNN / subgraph / motif code.

    Created by GraphResolver.resolve(shared, sample).
    Contains everything needed to pass to a PyG model or any custom GNN.

    Attributes
    ----------
    graph_id            : from PixelGraphSample
    label               : from PixelGraphSample
    split               : from PixelGraphSample
    node_features       : [N, d] float32
    edge_index          : [2, M] int64
    edge_attr           : [M, S+D] float32 — concat(static, dynamic)
    node_feature_names  : ordered names
    edge_feature_names  : ordered names (static first, then dynamic)
    metadata            : merged metadata
    """

    graph_id: int
    label: int
    split: str

    # Core tensors
    node_features: torch.Tensor      # [N, d]
    edge_index: torch.Tensor         # [2, M]
    edge_attr: torch.Tensor          # [M, S+D]

    # Feature inventories
    node_feature_names: List[str] = field(default_factory=list)
    edge_feature_names: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_node_features(self) -> int:
        return int(self.node_features.shape[1]) if self.node_features.ndim == 2 else 0

    @property
    def num_edge_features(self) -> int:
        return int(self.edge_attr.shape[1]) if self.edge_attr.ndim == 2 else 0

    def has_nan(self) -> bool:
        return (
            torch.isnan(self.node_features).any().item()
            or torch.isnan(self.edge_attr).any().item()
        )

    def __repr__(self) -> str:
        return (
            f"ResolvedPixelGraph("
            f"id={self.graph_id}, label={self.label}, split={self.split!r}, "
            f"nodes={self.num_nodes}, edges={self.num_edges}, "
            f"node_feat={self.num_node_features}, edge_feat={self.num_edge_features})"
        )
