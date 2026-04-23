"""
data/canonical_graph_builder.py — Build PixelGraphSample from RawSample + SharedGraphStructure.

Responsibilities
----------------
* Normalize image (raw [0,255] → [0,1])
* Build node features  (baseline: intensity, x_norm, y_norm)
  Extensible: gx, gy, grad_mag, contrast
* Build dynamic edge attributes (delta_intensity, intensity_similarity)
  These depend on per-image pixel values and use the SAME edge ordering
  as SharedGraphStructure.edge_index.

This builder does NOT touch edge_index or static edge attrs — those live in
SharedGraphStructure and are resolved later by GraphResolver.

Design principle
----------------
One method per feature family → easy to add new features without breaking existing ones.
All heavy numpy work is vectorized (no Python loops over pixels).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from configs.graph_config import GraphConfig
from data.raw_types import RawSample
from data.graph_types import PixelGraphSample, SharedGraphStructure

log = logging.getLogger(__name__)


class CanonicalGraphBuilder:
    """
    Build PixelGraphSample from a RawSample + SharedGraphStructure + GraphConfig.

    Usage
    -----
    >>> cfg = GraphConfig()
    >>> shared = SharedGraphBuilder(cfg).build()
    >>> builder = CanonicalGraphBuilder(cfg, shared)
    >>> sample: PixelGraphSample = builder.build(raw_sample)
    """

    def __init__(
        self,
        config: GraphConfig,
        shared: SharedGraphStructure,
    ) -> None:
        self.config = config
        self.shared = shared
        self.height = config.height
        self.width = config.width

        # Pre-compute static coordinate grids (reused for every image)
        self._x_norm, self._y_norm = self._build_coord_grids()

        # Decode edge endpoints once (for dynamic edge attr computation)
        ei = shared.edge_index.numpy()   # [2, M]
        self._src_ids = ei[0]            # [M]
        self._dst_ids = ei[1]            # [M]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, raw: RawSample) -> PixelGraphSample:
        """
        Convert one RawSample → PixelGraphSample.

        Parameters
        ----------
        raw : RawSample with image in [0,255] float32

        Returns
        -------
        PixelGraphSample — ready to be chunked into the repository
        """
        image = self._normalize(raw.image)     # float32, [0,1], (H,W)

        node_features = self._build_node_features(image)     # [N, d]
        edge_attr_dyn = self._build_dynamic_edge_attr(image) # [M, D]

        return PixelGraphSample(
            graph_id=raw.sample_id,
            label=raw.label,
            split=raw.split,
            usage=raw.usage,
            height=self.height,
            width=self.width,
            node_features=torch.from_numpy(node_features),
            edge_attr_dynamic=torch.from_numpy(edge_attr_dyn),
            node_feature_names=list(self.config.node_feature_names),
            dynamic_feature_names=list(self.config.edge_dynamic_feature_names),
            metadata={},
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Scale raw pixel values [0,255] → [0,1] if configured."""
        img = image.astype(np.float32)
        if self.config.normalize_pixels:
            img = img / 255.0
        return img

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------

    def _build_coord_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute normalized (x, y) coordinate grids, flattened to [N].
        x_norm[i] = x / (W-1),  y_norm[i] = y / (H-1)
        """
        H, W = self.height, self.width
        xs = np.arange(W, dtype=np.float32) / (W - 1)   # [W]
        ys = np.arange(H, dtype=np.float32) / (H - 1)   # [H]
        x_grid = np.tile(xs, (H, 1)).ravel()    # [N]
        y_grid = np.repeat(ys, W)               # [N]
        return x_grid, y_grid

    def _build_node_features(self, image: np.ndarray) -> np.ndarray:
        """
        Build node feature matrix [N, d].

        Supported feature names (ordered by config.node_feature_names):
            "intensity"  — normalized pixel value
            "x_norm"     — column / (W-1)
            "y_norm"     — row    / (H-1)
            "gx"         — horizontal Sobel gradient
            "gy"         — vertical Sobel gradient
            "grad_mag"   — gradient magnitude
            "contrast"   — local contrast (pixel - 3x3 mean)

        Adding a new feature: implement a helper _compute_<name> and add
        its key to the dispatch dict below.  No other changes needed.
        """
        names = self.config.node_feature_names
        flat = image.ravel()    # [N]

        # Lazy computation — only compute what is requested
        computed: Dict[str, np.ndarray] = {}

        def _need(name: str) -> bool:
            return name in names

        if _need("intensity"):
            computed["intensity"] = flat

        if _need("x_norm"):
            computed["x_norm"] = self._x_norm

        if _need("y_norm"):
            computed["y_norm"] = self._y_norm

        if _need("gx") or _need("gy") or _need("grad_mag"):
            gx, gy = self._compute_gradients(image)
            if _need("gx"):
                computed["gx"] = gx.ravel()
            if _need("gy"):
                computed["gy"] = gy.ravel()
            if _need("grad_mag"):
                computed["grad_mag"] = np.sqrt(gx ** 2 + gy ** 2).ravel().astype(np.float32)

        if _need("contrast"):
            computed["contrast"] = self._compute_local_contrast(image).ravel()

        # Assemble columns in the declared order
        cols = []
        for name in names:
            if name not in computed:
                raise ValueError(
                    f"Unknown node feature: {name!r}. "
                    f"Supported: {list(computed) + ['gx','gy','grad_mag','contrast']}"
                )
            cols.append(computed[name])

        return np.stack(cols, axis=1).astype(np.float32)   # [N, d]

    # ------------------------------------------------------------------
    # Dynamic edge attributes
    # ------------------------------------------------------------------

    def _build_dynamic_edge_attr(self, image: np.ndarray) -> np.ndarray:
        """
        Build dynamic edge feature matrix [M, D].

        Supported feature names (ordered by config.edge_dynamic_feature_names):
            "delta_intensity"       — |I_u - I_v|
            "intensity_similarity"  — exp(-alpha * |I_u - I_v|)

        Parameters
        ----------
        image : float32 (H, W), already normalized

        Returns
        -------
        np.ndarray [M, D] float32
        """
        flat = image.ravel()          # [N]
        I_src = flat[self._src_ids]   # [M]
        I_dst = flat[self._dst_ids]   # [M]

        delta = np.abs(I_src - I_dst).astype(np.float32)   # [M]
        alpha = self.config.intensity_similarity_alpha

        feature_map: Dict[str, np.ndarray] = {
            "delta_intensity": delta,
            "intensity_similarity": np.exp(-alpha * delta).astype(np.float32),
        }

        names = self.config.edge_dynamic_feature_names
        cols = []
        for name in names:
            if name not in feature_map:
                raise ValueError(
                    f"Unknown dynamic edge feature: {name!r}. "
                    f"Supported: {list(feature_map)}"
                )
            cols.append(feature_map[name])

        if not cols:
            M = len(self._src_ids)
            return np.zeros((M, 0), dtype=np.float32)

        return np.stack(cols, axis=1).astype(np.float32)   # [M, D]

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def _compute_gradients(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Central-difference gradient, vectorized.

        Returns
        -------
        gx, gy : float32 arrays of shape (H, W)
        """
        gx = np.zeros_like(image, dtype=np.float32)
        gy = np.zeros_like(image, dtype=np.float32)

        # gx: central difference on x-axis
        gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
        gx[:, 0]    = image[:, 1]  - image[:, 0]
        gx[:, -1]   = image[:, -1] - image[:, -2]

        # gy: central difference on y-axis
        gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
        gy[0, :]    = image[1, :]  - image[0, :]
        gy[-1, :]   = image[-1, :] - image[-2, :]

        return gx, gy

    def _compute_local_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Local contrast = pixel - mean(3×3 patch).
        Uses numpy convolution-style approach with reflection padding.

        Returns
        -------
        float32 (H, W)
        """
        from scipy.ndimage import uniform_filter
        mean_patch = uniform_filter(image.astype(np.float64), size=3).astype(np.float32)
        return (image - mean_patch).astype(np.float32)
