"""
config.py — SemanticRoiGraphConfig dataclass and DEFAULT_SEMANTIC_REGIONS.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SEMANTIC_REGIONS = (
    "forehead",
    "left_eyebrow",
    "right_eyebrow",
    "glabella",
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth_corner",
    "right_mouth_corner",
)


@dataclass
class SemanticRoiGraphConfig:
    name: str = "semantic_roi_graph_fer"
    num_classes: int = 7
    num_regions: int = 9
    roi_grid: int = 4
    feature_dim: int = 256
    motif_per_class: int = 4
    micro_motifs_per_region: int = 8
    macro_motifs_per_class: int = 4
    # Bug 5 fix: defaults now match semantic_roi_graph.yaml
    cross_region_compositions: int = 8   # was 4
    semantic_state_dim: int = 128         # was 9 — must be divisible by semantic_attn_heads
    semantic_latent_dim: int = 256        # was 128
    semantic_attn_heads: int = 4          # was 3 — must divide semantic_state_dim
    hyperedge_count: int = 4
    router_hidden_dim: int = 256          # was 64
    use_pretrained: bool = True
    backbone_out_size: int = 12
    bbox_input_size: int = 48
    micro_layers: int = 2
    macro_layers: int = 2
    attn_heads: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.0
    relation_temperature: float = 0.07
    region_confidence_threshold: float = 0.3
    fusion_scale: float = 0.25
