"""
data/graph_resolver.py — Merge SharedGraphStructure + PixelGraphSample → ResolvedPixelGraph.

This is the ONLY place where the two halves of the canonical representation
are joined into a full graph view.

All downstream code (GNN, subgraph generation, motif matching, visualization)
must go through the resolver — never reconstruct the full graph elsewhere.

Resolution contract
-------------------
  edge_attr = concat(edge_attr_static, edge_attr_dynamic)  along dim=1
  edge_feature_names = static_names + dynamic_names
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from data.graph_types import (
    PixelGraphSample,
    ResolvedPixelGraph,
    SharedGraphStructure,
)

log = logging.getLogger(__name__)


class GraphResolver:
    """
    Resolves a PixelGraphSample against a SharedGraphStructure to produce
    a ResolvedPixelGraph that downstream modules can consume directly.

    Parameters
    ----------
    shared : SharedGraphStructure — loaded once, reused for all samples

    Usage
    -----
    >>> resolver = GraphResolver(shared)
    >>> resolved: ResolvedPixelGraph = resolver.resolve(sample)
    """

    def __init__(self, shared: SharedGraphStructure) -> None:
        self.shared = shared

    # ------------------------------------------------------------------
    # Main resolve method
    # ------------------------------------------------------------------

    def resolve(self, sample: PixelGraphSample) -> ResolvedPixelGraph:
        """
        Merge shared topology + per-sample features into a ResolvedPixelGraph.

        Parameters
        ----------
        sample : PixelGraphSample from the repository

        Returns
        -------
        ResolvedPixelGraph with:
            node_features : [N, d]
            edge_index    : [2, M]  (shared)
            edge_attr     : [M, S+D]  concat(static, dynamic)
        """
        self._validate_compatibility(sample)

        # edge_attr = cat([static, dynamic], dim=1)
        edge_attr = self._merge_edge_attrs(sample)

        edge_feature_names = (
            list(self.shared.static_feature_names)
            + list(sample.dynamic_feature_names)
        )

        return ResolvedPixelGraph(
            graph_id=sample.graph_id,
            label=sample.label,
            split=sample.split,
            node_features=sample.node_features,
            edge_index=self.shared.edge_index,
            edge_attr=edge_attr,
            node_feature_names=list(sample.node_feature_names),
            edge_feature_names=edge_feature_names,
            metadata={
                **sample.metadata,
                "resolved": True,
            },
        )

    def resolve_batch(
        self, samples: list[PixelGraphSample]
    ) -> list[ResolvedPixelGraph]:
        """Resolve a list of samples. Useful for batch inspection."""
        return [self.resolve(s) for s in samples]

    # ------------------------------------------------------------------
    # Partial-resolution helpers (for specialized downstream use)
    # ------------------------------------------------------------------

    def get_edge_index(self) -> torch.Tensor:
        """Return the shared edge_index [2, M]."""
        return self.shared.edge_index

    def get_full_edge_attr(self, sample: PixelGraphSample) -> torch.Tensor:
        """Return merged edge_attr [M, S+D] without building a full ResolvedPixelGraph."""
        return self._merge_edge_attrs(sample)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_edge_attrs(self, sample: PixelGraphSample) -> torch.Tensor:
        """
        Concatenate static and dynamic edge attributes along feature dim.

        Handles edge cases:
          * zero static features → return dynamic only
          * zero dynamic features → return static only
          * both present → cat([S, D], dim=1)
        """
        static = self.shared.edge_attr_static    # [M, S]
        dynamic = sample.edge_attr_dynamic        # [M, D]

        if static.shape[1] == 0:
            return dynamic
        if dynamic.shape[1] == 0:
            return static
        return torch.cat([static, dynamic], dim=1)   # [M, S+D]

    def _validate_compatibility(self, sample: PixelGraphSample) -> None:
        """Sanity-check that sample is compatible with the shared graph."""
        if sample.height != self.shared.height or sample.width != self.shared.width:
            raise ValueError(
                f"Sample grid {sample.height}x{sample.width} ≠ "
                f"shared grid {self.shared.height}x{self.shared.width}"
            )
        M_shared = self.shared.num_edges
        M_sample = int(sample.edge_attr_dynamic.shape[0])
        if M_shared != M_sample:
            raise ValueError(
                f"Edge count mismatch: shared has {M_shared} edges, "
                f"sample.edge_attr_dynamic has {M_sample} rows"
            )
