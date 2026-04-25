"""
configs/graph_config.py — Canonical configuration for the FER-2013 pixel-graph pipeline.

This is the SINGLE source of truth for all graph construction parameters.
Every downstream module (builder, repository, resolver, dataset) must reference
this config — never hard-code graph parameters elsewhere.

Version field allows tracing which config was used to build a given repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Version tag — bump whenever graph structure or feature semantics change.
# This is embedded in the shared_graph.pt so you can detect stale repos.
# ---------------------------------------------------------------------------
GRAPH_CONFIG_VERSION = "2.0.0"


@dataclass
class GraphConfig:
    """
    Full configuration for the canonical pixel-graph pipeline.

    Sections
    --------
    image           : image dimensions & grayscale assumption
    connectivity    : 4- or 8-neighbor topology
    normalization   : how raw pixel values are scaled
    node_features   : ordered list of node feature names to compute
    edge_static     : edge features that depend only on grid topology (dx, dy, dist)
    edge_dynamic    : edge features that depend on per-image pixel values
    repository      : chunk size & output paths
    version         : string tag for traceability
    """

    # ---- Image ---------------------------------------------------------------
    height: int = 48
    width: int = 48

    # ---- Topology ------------------------------------------------------------
    connectivity: int = 8          # 4 or 8

    # ---- Normalization -------------------------------------------------------
    normalize_pixels: bool = True  # divide raw [0,255] → [0,1]

    # ---- Node features -------------------------------------------------------
    node_feature_names: List[str] = field(default_factory=lambda: [
        "intensity",
        "x_norm",
        "y_norm",
        "gx",
        "gy",
        "grad_mag",
        "local_contrast",
    ])

    # ---- Static edge features (topology-only, shared across all images) ------
    edge_static_feature_names: List[str] = field(default_factory=lambda: [
        "dx",
        "dy",
        "dist",
    ])

    # ---- Dynamic edge features (per-image, depend on pixel intensities) ------
    edge_dynamic_feature_names: List[str] = field(default_factory=lambda: [
        "delta_intensity",
        "intensity_similarity",
    ])

    # ---- Intensity-similarity kernel parameter --------------------------------
    intensity_similarity_alpha: float = 1.0

    # ---- Repository ----------------------------------------------------------
    chunk_size: int = 500          # samples per chunk file
    repo_root: str = "artifacts/graph_repo_v2"

    # ---- Traceability --------------------------------------------------------
    version: str = GRAPH_CONFIG_VERSION

    # --------------------------------------------------------------------------
    # Derived / convenience properties
    # --------------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.height * self.width

    @property
    def num_node_features(self) -> int:
        return len(self.node_feature_names)

    @property
    def num_edge_static_features(self) -> int:
        return len(self.edge_static_feature_names)

    @property
    def num_edge_dynamic_features(self) -> int:
        return len(self.edge_dynamic_feature_names)

    @property
    def num_edge_features_total(self) -> int:
        return self.num_edge_static_features + self.num_edge_dynamic_features

    # --------------------------------------------------------------------------
    # Factory helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def default() -> "GraphConfig":
        """Return the canonical baseline config."""
        return GraphConfig()

    @staticmethod
    def from_yaml_dict(cfg: dict) -> "GraphConfig":
        """
        Construct from a nested YAML dict (e.g. loaded via PyYAML).

        Expected top-level keys: 'graph', 'data', 'repo'  (all optional).
        Falls back to dataclass defaults for missing keys.
        """
        g = cfg.get("graph", {})
        d = cfg.get("data", {})
        r = cfg.get("repo", {})

        return GraphConfig(
            height=d.get("image_size", 48),
            width=d.get("image_size", 48),
            connectivity=g.get("connectivity", 8),
            normalize_pixels=g.get("normalize_pixels", True),
            node_feature_names=g.get("node_feature_names",
                                     g.get("node_features",
                                           [
                                               "intensity",
                                               "x_norm",
                                               "y_norm",
                                               "gx",
                                               "gy",
                                               "grad_mag",
                                               "local_contrast",
                                           ])),
            edge_static_feature_names=g.get("edge_static_feature_names",
                                            ["dx", "dy", "dist"]),
            edge_dynamic_feature_names=g.get("edge_dynamic_feature_names",
                                             ["delta_intensity",
                                              "intensity_similarity"]),
            intensity_similarity_alpha=g.get("intensity_similarity_alpha", 1.0),
            chunk_size=r.get("chunk_size", 500),
            repo_root=r.get("repo_root", "artifacts/graph_repo_v2"),
            version=g.get("version", GRAPH_CONFIG_VERSION),
        )

    def to_dict(self) -> dict:
        """Serialize to plain dict (for embedding in .pt metadata)."""
        return {
            "height": self.height,
            "width": self.width,
            "connectivity": self.connectivity,
            "normalize_pixels": self.normalize_pixels,
            "node_feature_names": list(self.node_feature_names),
            "edge_static_feature_names": list(self.edge_static_feature_names),
            "edge_dynamic_feature_names": list(self.edge_dynamic_feature_names),
            "intensity_similarity_alpha": self.intensity_similarity_alpha,
            "chunk_size": self.chunk_size,
            "repo_root": self.repo_root,
            "version": self.version,
        }
