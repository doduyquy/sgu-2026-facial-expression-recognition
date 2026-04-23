"""
data/shared_graph_builder.py — Build the SharedGraphStructure once for all images.

This module is responsible for:
  * Enumerating all edges in the H×W pixel grid with the given connectivity
  * Computing static edge attributes that depend only on grid topology:
      dx, dy, dist
  * Packaging everything into a SharedGraphStructure

Key design decisions
--------------------
* edge_index is stored as int64 (safe for PyTorch/PyG indexing).
  For a 48×48 grid with 8-connectivity, M ≈ 16,704 edges — fits trivially in memory.
* Static edge attrs use float32.
* This builder is called ONCE per pipeline run.  The result is saved to
  shared/shared_graph.pt and reused by every downstream step.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
import torch

from configs.graph_config import GraphConfig
from data.graph_types import SharedGraphStructure

log = logging.getLogger(__name__)


class SharedGraphBuilder:
    """
    Builds SharedGraphStructure from a GraphConfig.

    Usage
    -----
    >>> cfg = GraphConfig()
    >>> shared = SharedGraphBuilder(cfg).build()
    """

    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.height = config.height
        self.width = config.width
        self.connectivity = config.connectivity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> SharedGraphStructure:
        """Build and return the SharedGraphStructure."""
        log.info(
            "Building shared graph: %dx%d grid, %d-connectivity …",
            self.height, self.width, self.connectivity,
        )

        edge_index_np = self._build_edge_index()   # [2, M]  int64
        edge_attr_np = self._build_static_edge_attr(edge_index_np)  # [M, S]

        edge_index = torch.from_numpy(edge_index_np)        # int64
        edge_attr_static = torch.from_numpy(edge_attr_np)   # float32

        log.info(
            "  edge_index: %s  |  edge_attr_static: %s  (features: %s)",
            tuple(edge_index.shape),
            tuple(edge_attr_static.shape),
            self.config.edge_static_feature_names,
        )

        return SharedGraphStructure(
            height=self.height,
            width=self.width,
            connectivity=self.connectivity,
            edge_index=edge_index,
            edge_attr_static=edge_attr_static,
            static_feature_names=list(self.config.edge_static_feature_names),
            config_dict=self.config.to_dict(),
        )

    # ------------------------------------------------------------------
    # Edge index
    # ------------------------------------------------------------------

    def _neighbor_offsets(self) -> List[Tuple[int, int]]:
        if self.connectivity == 4:
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            return [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity}")

    def _node_id(self, y: int, x: int) -> int:
        return y * self.width + x

    def _build_edge_index(self) -> np.ndarray:
        """
        Enumerate all directed edges (u→v) for the pixel grid.
        Bidirectional: if (u,v) is a neighbor pair, both (u,v) and (v,u) appear.

        Returns
        -------
        np.ndarray shape [2, M], dtype int64
        """
        offsets = self._neighbor_offsets()
        rows, cols = [], []
        for y in range(self.height):
            for x in range(self.width):
                u = self._node_id(y, x)
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        v = self._node_id(ny, nx)
                        rows.append(u)
                        cols.append(v)

        edge_index = np.stack([
            np.array(rows, dtype=np.int64),
            np.array(cols, dtype=np.int64),
        ], axis=0)  # [2, M]
        return edge_index

    # ------------------------------------------------------------------
    # Static edge attributes
    # ------------------------------------------------------------------

    def _build_static_edge_attr(self, edge_index: np.ndarray) -> np.ndarray:
        """
        Compute static edge attributes (dx, dy, dist) for each edge.
        These depend only on grid positions, not on pixel values.

        Parameters
        ----------
        edge_index : [2, M] int64

        Returns
        -------
        np.ndarray [M, S] float32, columns ordered by config.edge_static_feature_names
        """
        M = edge_index.shape[1]
        src = edge_index[0]   # [M]
        dst = edge_index[1]   # [M]

        # Decode (y, x) positions for each node
        src_y = src // self.width
        src_x = src % self.width
        dst_y = dst // self.width
        dst_x = dst % self.width

        dx = (dst_x - src_x).astype(np.float32)
        dy = (dst_y - src_y).astype(np.float32)
        dist = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)

        feature_map = {"dx": dx, "dy": dy, "dist": dist}

        cols = []
        for name in self.config.edge_static_feature_names:
            if name not in feature_map:
                raise ValueError(
                    f"Unknown static edge feature: {name!r}. "
                    f"Supported: {list(feature_map)}"
                )
            cols.append(feature_map[name])

        if not cols:
            return np.zeros((M, 0), dtype=np.float32)

        return np.stack(cols, axis=1)   # [M, S]
