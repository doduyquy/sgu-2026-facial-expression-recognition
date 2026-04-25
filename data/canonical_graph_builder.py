"""
data/canonical_graph_builder.py — Build PixelGraphSample from RawSample + SharedGraphStructure.

Responsibilities
----------------
* Normalize image (raw [0,255] → [0,1])
* Build node features:
    intensity, x_norm, y_norm, gx, gy, grad_mag, local_contrast
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
from typing import Dict, Tuple

import numpy as np
import torch

from configs.graph_config import GraphConfig
from data.raw_types import RawSample
from data.graph_types import PixelGraphSample, SharedGraphStructure

log = logging.getLogger(__name__)


def compute_gradients(image_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stable finite-difference gradients from a normalized image.

    Parameters
    ----------
    image_norm : np.ndarray [H, W], float32, values in [0, 1]

    Returns
    -------
    gx, gy, grad_mag : float32 arrays, all shape [H, W]
    """
    if image_norm.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image_norm.shape}")

    img = np.asarray(image_norm, dtype=np.float32)
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    if img.shape[1] > 1:
        gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
        gx[:, 0] = img[:, 1] - img[:, 0]
        gx[:, -1] = img[:, -1] - img[:, -2]

    if img.shape[0] > 1:
        gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
        gy[0, :] = img[1, :] - img[0, :]
        gy[-1, :] = img[-1, :] - img[-2, :]

    gx = np.clip(gx, -1.0, 1.0).astype(np.float32, copy=False)
    gy = np.clip(gy, -1.0, 1.0).astype(np.float32, copy=False)
    grad_mag = np.sqrt(np.clip(gx * gx + gy * gy, a_min=0.0, a_max=None)).astype(np.float32, copy=False)
    grad_mag = np.clip(grad_mag, 0.0, 1.0).astype(np.float32, copy=False)

    if not np.isfinite(gx).all() or not np.isfinite(gy).all() or not np.isfinite(grad_mag).all():
        raise ValueError("Non-finite values detected in gradient features")

    return gx, gy, grad_mag


def compute_local_contrast(image_norm: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Compute local contrast as abs(pixel - local_mean) with edge padding.

    Parameters
    ----------
    image_norm   : np.ndarray [H, W], float32, values in [0, 1]
    window_size  : odd kernel size, default 3

    Returns
    -------
    np.ndarray [H, W], float32, values in [0, 1]
    """
    if image_norm.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image_norm.shape}")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")

    img = np.asarray(image_norm, dtype=np.float32)
    pad = window_size // 2
    padded = np.pad(img, pad_width=pad, mode="edge")

    local_sum = np.zeros_like(img, dtype=np.float32)
    for dy in range(window_size):
        for dx in range(window_size):
            local_sum += padded[dy:dy + img.shape[0], dx:dx + img.shape[1]]

    local_mean = local_sum / float(window_size * window_size)
    contrast = np.abs(img - local_mean).astype(np.float32, copy=False)
    contrast = np.clip(contrast, 0.0, 1.0).astype(np.float32, copy=False)

    if not np.isfinite(contrast).all():
        raise ValueError("Non-finite values detected in local_contrast")

    return contrast


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
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
        if not np.isfinite(img).all():
            raise ValueError("Non-finite values detected after image normalization")
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
            "local_contrast" — abs(pixel - local_mean_3x3)

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
            gx, gy, grad_mag = compute_gradients(image)
            if _need("gx"):
                computed["gx"] = gx.ravel()
            if _need("gy"):
                computed["gy"] = gy.ravel()
            if _need("grad_mag"):
                computed["grad_mag"] = grad_mag.ravel()

        if _need("local_contrast"):
            computed["local_contrast"] = compute_local_contrast(image, window_size=3).ravel()

        # Assemble columns in the declared order
        cols = []
        for name in names:
            if name not in computed:
                raise ValueError(
                    f"Unknown node feature: {name!r}. "
                    f"Supported: {list(computed) + ['gx', 'gy', 'grad_mag', 'local_contrast']}"
                )
            cols.append(computed[name])

        node_features = np.stack(cols, axis=1).astype(np.float32)   # [N, d]
        if not np.isfinite(node_features).all():
            raise ValueError("Non-finite values detected in node_features")
        return node_features

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

        edge_attr = np.stack(cols, axis=1).astype(np.float32)   # [M, D]
        if not np.isfinite(edge_attr).all():
            raise ValueError("Non-finite values detected in dynamic edge attributes")
        return edge_attr
