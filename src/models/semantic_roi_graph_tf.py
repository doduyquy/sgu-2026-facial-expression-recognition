"""
semantic_roi_graph_tf.py — TensorFlow/Keras port của SemanticROIGraphFER.

Các thay đổi chính so với bản PyTorch:
- Tensor format: NHWC thay vì NCHW
- roi_align -> tf.image.crop_and_resize (boxes format: [y1,x1,y2,x2] normalized [0,1])
- nn.Module -> tf.keras.Model / tf.keras.layers.Layer
- nn.Parameter -> tf.Variable (non-trainable weights) hoặc self.add_weight
- nn.Linear -> tf.keras.layers.Dense
- nn.MultiheadAttention -> tf.keras.layers.MultiHeadAttention
- ResNet50 backbone: tf.keras.applications.ResNet50V2 (lấy intermediate layer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
    cross_region_compositions: int = 8
    semantic_state_dim: int = 128
    semantic_latent_dim: int = 256
    semantic_attn_heads: int = 4
    hyperedge_count: int = 4
    router_hidden_dim: int = 256
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
    region_dropout_prob: float = 0.0
    program_dim: int = 128
    programs_per_class: int = 4


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_softmax(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Numerically stable softmax that handles fully masked (all -inf) rows."""
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    all_invalid = tf.reduce_all(tf.math.is_inf(x_shifted) | tf.math.is_nan(x_shifted), axis=axis, keepdims=True)
    x_shifted = tf.where(all_invalid, tf.zeros_like(x_shifted), x_shifted)
    return tf.nn.softmax(x_shifted, axis=axis)


def _canonical_region_boxes_tf(bbox_input_size: int) -> tf.Tensor:
    """Canonical 9-region bounding boxes for 48x48 images. Returns [9, 4] (x1,y1,x2,y2)."""
    boxes = tf.constant([
        [8,  0, 40, 10],   # forehead
        [5,  8, 18, 18],   # left_eyebrow
        [30, 8, 43, 18],   # right_eyebrow
        [18, 12, 30, 22],  # glabella
        [6,  16, 20, 30],  # left_eye
        [28, 16, 42, 30],  # right_eye
        [14, 20, 34, 38],  # nose
        [8,  30, 22, 43],  # left_mouth_corner
        [26, 30, 40, 43],  # right_mouth_corner
    ], dtype=tf.float32)
    scale = float(bbox_input_size) / 48.0
    return boxes * scale


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class SemanticBackbone(tf.keras.layers.Layer):
    """ResNet50V2 backbone (NHWC), outputs (B, H/4, W/4, feature_dim)."""

    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)
        weights = "imagenet" if use_pretrained else None
        # Build full ResNet50V2 in NHWC format
        base = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=weights,
            input_shape=(None, None, 3),
        )
        # Extract up to conv3_block4_out (≈ layer3 of PyTorch ResNet50)
        # This gives stride-8 features (48/8=6 → but we want 12, so we take conv3 which is stride-4)
        # conv2_block3_out → stride 4 → for 48x48 input gives 12x12 ✓
        self.feature_extractor = tf.keras.Model(
            inputs=base.input,
            outputs=base.get_layer("conv3_block4_out").output,
        )
        # conv3_block4_out has 512 channels in ResNet50V2
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Conv2D(feature_dim, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
        ])
        self.out_channels = feature_dim

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: (B, H, W, C) NHWC — ensure 3 channels
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        x = self.feature_extractor(x, training=training)
        return self.proj(x, training=training)


# ---------------------------------------------------------------------------
# ROI Align (using tf.image.crop_and_resize)
# ---------------------------------------------------------------------------

class SemanticRoiAlign(tf.keras.layers.Layer):
    """ROI Align equivalent using tf.image.crop_and_resize.

    Input bboxes format: (B, R, 4) in pixel coords [x1, y1, x2, y2]
    tf.image.crop_and_resize expects boxes [y1, x1, y2, x2] normalized [0, 1].
    """

    def __init__(self, roi_grid: int = 4, bbox_input_size: int = 48,
                 feature_out_size: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.roi_grid = int(roi_grid)
        self.bbox_input_size = int(bbox_input_size)
        self.feature_out_size = int(feature_out_size)

    def _canonical_boxes(self, batch_size: int) -> tf.Tensor:
        boxes = _canonical_region_boxes_tf(self.bbox_input_size)  # [9, 4]
        return tf.tile(tf.expand_dims(boxes, 0), [batch_size, 1, 1])

    def validate_bboxes(self, bboxes: tf.Tensor) -> tf.Tensor:
        """Clamp and repair invalid bboxes. Returns (B, R, 4)."""
        bboxes = tf.cast(bboxes, tf.float32)
        bsize = float(self.bbox_input_size - 1)

        x1 = tf.clip_by_value(bboxes[..., 0], 0.0, bsize)
        y1 = tf.clip_by_value(bboxes[..., 1], 0.0, bsize)
        x2 = tf.clip_by_value(bboxes[..., 2], 0.0, bsize)
        y2 = tf.clip_by_value(bboxes[..., 3], 0.0, bsize)

        x1_n = tf.minimum(x1, x2)
        y1_n = tf.minimum(y1, y2)
        x2_n = tf.maximum(x1, x2)
        y2_n = tf.maximum(y1, y2)

        x2_n = tf.maximum(x2_n, x1_n + 2.0)
        y2_n = tf.maximum(y2_n, y1_n + 2.0)
        x2_n = tf.minimum(x2_n, bsize)
        y2_n = tf.minimum(y2_n, bsize)

        return tf.stack([x1_n, y1_n, x2_n, y2_n], axis=-1)

    def call(self, feature_map: tf.Tensor, bboxes: tf.Tensor) -> tf.Tensor:
        """
        feature_map: (B, H, W, C) NHWC
        bboxes: (B, R, 4) pixel coords [x1, y1, x2, y2]
        Returns: (B, R, roi_grid*roi_grid, C)
        """
        b = tf.shape(feature_map)[0]
        h = tf.cast(tf.shape(feature_map)[1], tf.float32)
        w = tf.cast(tf.shape(feature_map)[2], tf.float32)
        r = tf.shape(bboxes)[1]
        c = feature_map.shape[-1]

        bboxes = self.validate_bboxes(bboxes)

        # Normalize to [0,1] and convert [x1,y1,x2,y2] -> [y1,x1,y2,x2]
        norm_scale = tf.cast(self.bbox_input_size, tf.float32)
        x1 = bboxes[..., 0] / norm_scale
        y1 = bboxes[..., 1] / norm_scale
        x2 = bboxes[..., 2] / norm_scale
        y2 = bboxes[..., 3] / norm_scale
        # boxes for crop_and_resize: [y1, x1, y2, x2]
        boxes_norm = tf.stack([y1, x1, y2, x2], axis=-1)  # (B, R, 4)

        # Flatten for crop_and_resize: boxes=(B*R, 4), box_ind=(B*R,)
        boxes_flat = tf.reshape(boxes_norm, [-1, 4])
        box_ind = tf.repeat(tf.range(b), r)

        # Crop and resize
        crops = tf.image.crop_and_resize(
            feature_map,
            boxes_flat,
            box_ind,
            crop_size=(self.roi_grid, self.roi_grid),
            method="bilinear",
        )  # (B*R, roi_grid, roi_grid, C)

        # Reshape to (B, R, roi_grid*roi_grid, C)
        crops = tf.reshape(crops, [b, r, self.roi_grid * self.roi_grid, -1])
        return crops


# ---------------------------------------------------------------------------
# Graph Attention Block
# ---------------------------------------------------------------------------

class GATBlock(tf.keras.layers.Layer):
    """Multi-head graph attention with learnable adjacency bias."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1,
                 num_nodes: Optional[int] = None, use_locality: bool = False, **kwargs):
        super().__init__(**kwargs)
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_rate = dropout
        self.q_proj = tf.keras.layers.Dense(dim)
        self.k_proj = tf.keras.layers.Dense(dim)
        self.v_proj = tf.keras.layers.Dense(dim)
        self.out_proj = tf.keras.layers.Dense(dim)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.num_nodes = num_nodes
        self.use_locality = use_locality

    def build(self, input_shape):
        if self.num_nodes is not None:
            self.adj_bias = self.add_weight(
                name="adj_bias",
                shape=(1, 1, self.num_nodes, self.num_nodes),
                initializer=tf.initializers.RandomNormal(stddev=0.01),
                trainable=True,
            )
        else:
            self.adj_bias = None

        if self.use_locality and self.num_nodes is not None:
            side = int(self.num_nodes ** 0.5)
            if side * side == self.num_nodes:
                coords_1d = np.arange(side, dtype=np.float32)
                gy, gx = np.meshgrid(coords_1d, coords_1d, indexing="ij")
                coords = np.stack([gy.flatten(), gx.flatten()], axis=-1)
            else:
                coords = np.arange(self.num_nodes, dtype=np.float32)[:, None]
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1))
            dist = dist / (dist.max() + 1e-4)
            self.locality_bias = tf.constant(-dist[None, None], dtype=tf.float32)
        else:
            self.locality_bias = None
        super().build(input_shape)

    def call(self, x: tf.Tensor, edge_prior=None, attn_mask=None, training: bool = False) -> tf.Tensor:
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]

        def split_heads(t):
            t = tf.reshape(t, [b, n, self.heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])  # (B, H, N, D)

        orig_dtype = x.dtype
        q = tf.cast(split_heads(self.q_proj(x)), tf.float32)
        k = tf.cast(split_heads(self.k_proj(x)), tf.float32)
        v = tf.cast(split_heads(self.v_proj(x)), tf.float32)

        # Attention scores (B, H, N, N)
        attn = tf.einsum("bhid,bhjd->bhij", q, k) / (float(self.head_dim) ** 0.5)

        if self.adj_bias is not None:
            attn = attn + tf.cast(self.adj_bias, tf.float32)
        if self.locality_bias is not None:
            attn = attn + tf.cast(self.locality_bias, tf.float32)
        if edge_prior is not None:
            if len(edge_prior.shape) == 2:
                edge_prior = tf.expand_dims(edge_prior, 0)
            attn = attn + tf.math.log(tf.maximum(tf.cast(edge_prior, tf.float32), 1e-6))[:, None]
        if attn_mask is not None:
            # attn_mask: True where should be masked
            attn = tf.where(attn_mask[:, None, None, :], tf.fill(tf.shape(attn), tf.cast(-1e4, tf.float32)), attn)

        attn = safe_softmax(attn, axis=-1)
        attn = self.dropout_layer(attn, training=training)

        out = tf.einsum("bhij,bhjd->bhid", attn, v)
        out = tf.cast(out, orig_dtype)
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, N, H, D)
        out = tf.reshape(out, [b, n, self.dim])
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Gated Pooling
# ---------------------------------------------------------------------------

class GatedPooling(tf.keras.layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gate = tf.keras.layers.Dense(1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        weights = tf.sigmoid(tf.cast(self.gate(x), tf.float32))
        weighted = tf.cast(x, tf.float32) * weights
        pooled = tf.reduce_sum(weighted, axis=1) / (tf.reduce_sum(weights, axis=1) + 1e-4)
        return tf.cast(pooled, x.dtype)


# ---------------------------------------------------------------------------
# Micro Graph Reasoner
# ---------------------------------------------------------------------------

class MicroGraphReasoner(tf.keras.layers.Layer):
    def __init__(self, dim: int, num_nodes: int, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gat_layers = [
            GATBlock(dim, heads=heads, dropout=dropout, num_nodes=num_nodes)
            for _ in range(layers)
        ]
        self.norms = [tf.keras.layers.LayerNormalization() for _ in range(layers)]
        self.pool = GatedPooling(dim)

    def call(self, x: tf.Tensor, training: bool = False):
        # x: (B, R, N, D)
        b = tf.shape(x)[0]
        r = tf.shape(x)[1]
        n = tf.shape(x)[2]
        d = x.shape[-1]

        x_flat = tf.reshape(x, [b * r, n, d])
        for layer, norm in zip(self.gat_layers, self.norms):
            x_flat = x_flat + layer(norm(x_flat), training=training)

        pooled = self.pool(x_flat)  # (B*R, D)
        pooled = tf.reshape(pooled, [b, r, d])
        x_out = tf.reshape(x_flat, [b, r, n, d])
        return x_out, pooled


# ---------------------------------------------------------------------------
# Semantic State Encoder
# ---------------------------------------------------------------------------

class SemanticStateEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, state_dim: int,
                 hidden_dim: Optional[int] = None, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = hidden_dim or max(input_dim // 2, state_dim * 2)
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(state_dim),
        ])
        self.gate_net = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.Activation("sigmoid"),
        ])
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        raw_state = self.proj(x, training=training)
        gate = self.gate_net(x)
        return self.norm(raw_state * gate)


# ---------------------------------------------------------------------------
# Micro Semantic Motif Bank & Matcher
# ---------------------------------------------------------------------------

class MicroSemanticMotifBank(tf.keras.layers.Layer):
    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.state_dim = state_dim

    def build(self, input_shape):
        self.motifs = self.add_weight(
            name="motifs",
            shape=(self.num_regions, self.motifs_per_region, self.state_dim),
            initializer=tf.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs=None) -> tf.Tensor:
        return self.motifs


class MicroSemanticMotifMatcher(tf.keras.layers.Layer):
    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int,
                 temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(temperature)
        self.token_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
        ])

    def call(self, semantic_states: tf.Tensor, motif_bank: tf.Tensor, training: bool = False):
        orig_dtype = semantic_states.dtype
        states_f32 = tf.cast(semantic_states, tf.float32)
        bank_f32 = tf.cast(motif_bank, tf.float32)
        
        state_norm = tf.math.l2_normalize(states_f32, axis=-1, epsilon=1e-4)
        bank_norm = tf.math.l2_normalize(bank_f32, axis=-1, epsilon=1e-4)
        sim = tf.einsum("brs,rks->brk", state_norm, bank_norm) / self.temperature
        attn = safe_softmax(sim, axis=-1)
        tokens = tf.einsum("brk,rks->brs", attn, bank_f32)
        
        tokens = tf.cast(tokens, orig_dtype)
        tokens = self.token_proj(tokens, training=training)
        return tf.cast(attn, orig_dtype), semantic_states + tokens


# ---------------------------------------------------------------------------
# Semantic Interaction Block
# ---------------------------------------------------------------------------

class SemanticInteractionBlock(tf.keras.layers.Layer):
    def __init__(self, state_dim: int, hidden_dim: Optional[int] = None,
                 dropout: float = 0.1, dropedge_rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.dropedge_rate = dropedge_rate
        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        pair_input_dim = state_dim * 4
        self.edge_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation("sigmoid"),
        ])
        self.edge_message = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(state_dim),
        ])
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, semantic_states: tf.Tensor, region_mask=None, training: bool = False):
        b = tf.shape(semantic_states)[0]
        r = tf.shape(semantic_states)[1]
        s = semantic_states.shape[-1]

        # Expand for pairwise interactions
        states_f32 = tf.cast(semantic_states, tf.float32)
        left = tf.broadcast_to(
            tf.expand_dims(states_f32, 2),
            [b, r, r, s]
        )
        right = tf.broadcast_to(
            tf.expand_dims(states_f32, 1),
            [b, r, r, s]
        )
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)
        pair_input = tf.cast(pair_input, semantic_states.dtype)

        gates = self.edge_gate(pair_input, training=training)[..., 0] + 0.1

        if training and self.dropedge_rate > 0:
            gates = tf.nn.dropout(gates, rate=self.dropedge_rate)

        if region_mask is not None:
            pair_mask = (
                tf.expand_dims(region_mask, -1) *
                tf.expand_dims(region_mask, -2)
            )
            gates = gates * pair_mask

        messages = self.edge_message(pair_input, training=training)
        
        gates_f32 = tf.cast(gates, tf.float32)
        messages_f32 = tf.cast(messages, tf.float32)
        
        interaction_tensor_f32 = tf.expand_dims(gates_f32, -1) * messages_f32
        denom_f32 = tf.reduce_sum(gates_f32, axis=2, keepdims=True) + 1e-4
        interaction_summary_f32 = tf.reduce_sum(interaction_tensor_f32, axis=2) / denom_f32
        
        interaction_summary = tf.cast(interaction_summary_f32, semantic_states.dtype)
        interaction_tensor = tf.cast(interaction_tensor_f32, semantic_states.dtype)
        
        updated_states = self.norm(semantic_states + interaction_summary)
        return updated_states, interaction_tensor, gates


# ---------------------------------------------------------------------------
# Cross Region Composition Graph
# ---------------------------------------------------------------------------

class CrossRegionCompositionGraph(tf.keras.layers.Layer):
    def __init__(self, state_dim: int, num_compositions: int, attn_heads: int = 4,
                 hidden_dim: Optional[int] = None, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        self.num_compositions = num_compositions
        self.pair_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(state_dim),
        ])
        self.pair_router = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ])
        self.composition_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=state_dim // attn_heads,
            dropout=dropout,
        )
        self.composition_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        state_dim = input_shape[-1]
        self.composition_queries = self.add_weight(
            name="composition_queries",
            shape=(self.num_compositions, state_dim),
            initializer=tf.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, semantic_states: tf.Tensor, region_mask=None,
             region_confidence=None, training: bool = False):
        b = tf.shape(semantic_states)[0]
        r = tf.shape(semantic_states)[1]
        d = semantic_states.shape[-1]

        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * tf.expand_dims(region_confidence, -1)

        left = tf.broadcast_to(tf.expand_dims(tokens, 2), [b, r, r, d])
        right = tf.broadcast_to(tf.expand_dims(tokens, 1), [b, r, r, d])
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)
        pair_tokens = self.pair_encoder(pair_input, training=training)
        pair_scores = self.pair_router(pair_tokens, training=training)[..., 0]

        if region_mask is not None:
            pair_mask = (
                tf.expand_dims(region_mask, -1) *
                tf.expand_dims(region_mask, -2)
            )
            pair_scores = tf.where(pair_mask <= 0, tf.fill(tf.shape(pair_scores), tf.cast(-1e4, pair_scores.dtype)), pair_scores)

        pair_attention = tf.reshape(
            safe_softmax(tf.reshape(pair_scores, [b, -1]), axis=-1),
            [b, r, r]
        )
        pair_sequence = tf.reshape(pair_tokens, [b, r * r, d])

        composition_queries = tf.tile(
            tf.expand_dims(self.composition_queries, 0), [b, 1, 1]
        )

        cross_region_tokens = self.composition_attn(
            query=composition_queries,
            key=pair_sequence,
            value=pair_sequence,
            training=training,
        )
        cross_region_tokens = self.composition_norm(cross_region_tokens)

        return {
            "cross_region_tokens": cross_region_tokens,
            "pair_tokens": pair_tokens,
            "pair_scores": pair_scores,
            "pair_attention": pair_attention,
        }


# ---------------------------------------------------------------------------
# Semantic Hypergraph Reasoner
# ---------------------------------------------------------------------------

class SemanticHypergraphReasoner(tf.keras.layers.Layer):
    def __init__(self, state_dim: int, latent_dim: int, hyperedge_count: int,
                 attn_heads: int = 4, router_hidden_dim: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hyperedge_count = hyperedge_count
        self.hyperedge_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=state_dim // attn_heads,
            dropout=dropout,
        )
        self.region_back_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=state_dim // attn_heads,
            dropout=dropout,
        )
        self.router = tf.keras.Sequential([
            tf.keras.layers.Dense(router_hidden_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ])
        self.latent_projector = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(latent_dim),
        ])
        self.latent_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        state_dim = input_shape[-1]
        self.hyperedge_queries = self.add_weight(
            name="hyperedge_queries",
            shape=(self.hyperedge_count, state_dim),
            initializer=tf.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, semantic_states: tf.Tensor, region_mask=None,
             region_confidence=None, training: bool = False):
        b = tf.shape(semantic_states)[0]
        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * tf.expand_dims(region_confidence, -1)

        hyper_queries = tf.tile(tf.expand_dims(self.hyperedge_queries, 0), [b, 1, 1])
        hyperedge_tokens = self.hyperedge_attn(
            query=hyper_queries,
            key=tokens,
            value=tokens,
            training=training,
        )
        region_context = self.region_back_attn(
            query=tokens,
            key=hyperedge_tokens,
            value=hyperedge_tokens,
            training=training,
        )

        composed_states = tokens + region_context
        routing_logits = self.router(composed_states, training=training)[..., 0]
        orig_dtype = routing_logits.dtype
        routing_logits_f32 = tf.cast(routing_logits, tf.float32)

        if region_mask is not None:
            region_mask_f32 = tf.cast(region_mask, tf.float32)
            routing_logits_f32 = tf.where(
                region_mask_f32 == 0.0,
                tf.fill(tf.shape(routing_logits_f32), tf.cast(-1e4, tf.float32)),
                routing_logits_f32
            )
        
        routing_weights_f32 = safe_softmax(routing_logits_f32, axis=1)
        
        if region_mask is not None:
            routing_weights_f32 = routing_weights_f32 * region_mask_f32
            routing_weights_f32 = routing_weights_f32 / tf.clip_by_value(
                tf.reduce_sum(routing_weights_f32, axis=1, keepdims=True), 
                clip_value_min=1e-6, clip_value_max=1e9
            )
            
        routing_weights = tf.cast(routing_weights_f32, orig_dtype)
        routing_logits = tf.cast(routing_logits_f32, orig_dtype)

        pooled_state = tf.reduce_sum(
            tf.expand_dims(routing_weights, -1) * composed_states, axis=1
        )
        hyper_summary = tf.reduce_mean(hyperedge_tokens, axis=1)
        emotion_latent = self.latent_projector(
            tf.concat([pooled_state, hyper_summary], axis=-1), training=training
        )
        emotion_latent = self.latent_norm(emotion_latent)

        return {
            "composed_states": composed_states,
            "hyperedge_tokens": hyperedge_tokens,
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "emotion_latent": emotion_latent,
        }


# ---------------------------------------------------------------------------
# Semantic Compositional Program Bank
# ---------------------------------------------------------------------------

class SemanticCompositionalProgramBank(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int,
                 state_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim

    def build(self, input_shape):
        self.programs = self.add_weight(
            name="programs",
            shape=(self.num_classes, self.programs_per_class, self.num_regions, self.state_dim),
            initializer=tf.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.topology_logits = self.add_weight(
            name="topology_logits",
            shape=(self.num_classes, self.programs_per_class, self.num_regions, self.num_regions),
            initializer=tf.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs=None):
        return self.programs, tf.sigmoid(self.topology_logits)


# ---------------------------------------------------------------------------
# Semantic Program Executor
# ---------------------------------------------------------------------------

class SemanticProgramExecutor(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int,
                 state_dim: int, temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.temperature = float(temperature)
        self.program_summary_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
        ])

    def build(self, input_shape):
        init_val = np.ones((1, self.num_classes, 1, 3), dtype=np.float32)
        init_val[..., 0] = 1.0
        init_val[..., 1] = 0.5
        init_val[..., 2] = 0.25
        self.sim_weights = self.add_weight(
            name="sim_weights",
            shape=(1, self.num_classes, 1, 3),
            initializer=tf.constant_initializer(init_val),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, semantic_states, cross_region_tokens, program_bank,
             program_topology, region_mask=None, interaction_gates=None,
             routing_weights=None, training: bool = False):
        
        orig_dtype = semantic_states.dtype
        states_f32 = tf.cast(semantic_states, tf.float32)
        bank_f32 = tf.cast(program_bank, tf.float32)

        state_norm = tf.math.l2_normalize(states_f32, axis=-1, epsilon=1e-4)
        program_norm = tf.math.l2_normalize(bank_f32, axis=-1, epsilon=1e-4)

        # Region similarity: (B, C, M)
        region_sims = tf.einsum("brd,cmrd->bcmr", state_norm, program_norm)
        if routing_weights is not None:
            rw_f32 = tf.cast(routing_weights, tf.float32)
            region_sim = tf.reduce_sum(
                region_sims * rw_f32[:, None, None, :], axis=-1
            )
        elif region_mask is not None:
            valid_mask = tf.cast(region_mask[:, None, None, :], tf.float32)
            region_sims = region_sims * valid_mask
            region_sim = tf.reduce_sum(region_sims, axis=-1) / tf.clip_by_value(
                tf.reduce_sum(valid_mask, axis=-1), clip_value_min=1.0, clip_value_max=1e9
            )
        else:
            region_sim = tf.reduce_mean(region_sims, axis=-1)

        # Topology similarity
        if interaction_gates is not None:
            observed_topo = tf.cast(interaction_gates, tf.float32)[:, None, None, :, :]
            topo_bank = tf.cast(program_topology, tf.float32)[None, :, :, :, :]
            topo_mse = (observed_topo - topo_bank) ** 2
            if region_mask is not None:
                pair_mask = tf.cast(
                    region_mask[:, None, None, :, None] *
                    region_mask[:, None, None, None, :], tf.float32
                )
                topo_mse = topo_mse * pair_mask
                topology_sim = 1.0 - (
                    tf.reduce_sum(topo_mse, axis=[-1, -2]) / tf.clip_by_value(
                        tf.reduce_sum(pair_mask, axis=[-1, -2]), clip_value_min=1.0, clip_value_max=1e9
                    )
                )
            else:
                topology_sim = 1.0 - tf.reduce_mean(topo_mse, axis=[-1, -2])
        else:
            topology_sim = tf.ones_like(region_sim)

        # Composition similarity
        cr_tokens_f32 = tf.cast(cross_region_tokens, tf.float32)
        composition_summary = tf.reduce_mean(cr_tokens_f32, axis=1)  # (B, D)
        composition_summary = tf.cast(composition_summary, orig_dtype)
        composition_summary = self.program_summary_proj(composition_summary, training=training)
        composition_summary_f32 = tf.cast(composition_summary, tf.float32)

        # program_bank mean: (C, M, R, D) -> (C, M, D)
        prog_mean = tf.reduce_mean(bank_f32, axis=2)  # (C, M, D)
        prog_mean = tf.cast(prog_mean, orig_dtype)
        c_dim = tf.shape(prog_mean)[0]
        m_dim = tf.shape(prog_mean)[1]
        d_dim = prog_mean.shape[-1]
        # Reshape to (C*M, D) -> Dense -> reshape back to (C, M, D)
        prog_mean_flat = tf.reshape(prog_mean, [c_dim * m_dim, d_dim])
        prog_mean_flat = self.program_summary_proj(prog_mean_flat, training=training)
        program_summary = tf.reshape(prog_mean_flat, [c_dim, m_dim, -1])  # (C, M, D)
        program_summary_f32 = tf.cast(program_summary, tf.float32)

        composition_sim = tf.einsum(
            "bd,cmd->bcm",
            tf.math.l2_normalize(composition_summary_f32, axis=-1, epsilon=1e-4),
            tf.math.l2_normalize(program_summary_f32, axis=-1, epsilon=1e-4),
        )

        w = tf.cast(tf.nn.softplus(self.sim_weights), tf.float32)
        total_sim = w[..., 0] * region_sim + w[..., 1] * topology_sim + w[..., 2] * composition_sim
        
        region_score = tf.cast(region_sim / self.temperature, orig_dtype)
        topology_score = tf.cast(topology_sim / self.temperature, orig_dtype)
        composition_score = tf.cast(composition_sim / self.temperature, orig_dtype)
        
        compatibility = tf.clip_by_value(total_sim / self.temperature, -50.0, 50.0)

        program_attention = safe_softmax(compatibility, axis=-1)
        class_scores = tf.reduce_logsumexp(compatibility, axis=-1)
        
        # Cast attention back to original dtype before computing tokens
        program_attention = tf.cast(program_attention, orig_dtype)
        class_scores = tf.cast(class_scores, orig_dtype)
        compatibility = tf.cast(compatibility, orig_dtype)
        
        program_tokens = tf.einsum("bcm,cmd->bcd", program_attention, program_summary)

        if routing_weights is not None:
            rw_f32 = tf.cast(routing_weights, tf.float32)
            routing_entropy = -tf.reduce_sum(
                tf.maximum(rw_f32, 1e-6) * tf.math.log(tf.maximum(rw_f32, 1e-6)),
                axis=-1,
            )
            routing_entropy = tf.cast(routing_entropy, orig_dtype)
        else:
            routing_entropy = tf.zeros([tf.shape(semantic_states)[0]], dtype=orig_dtype)

        return {
            "program_scores": class_scores,
            "program_attention": program_attention,
            "program_tokens": program_tokens,
            "compatibility": compatibility,
            "region_score": region_score,
            "topology_score": topology_score,
            "composition_score": composition_score,
            "routing_entropy": routing_entropy,
        }


# ---------------------------------------------------------------------------
# Semantic Emotion Classifier
# ---------------------------------------------------------------------------

class SemanticEmotionClassifier(tf.keras.layers.Layer):
    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes),
        ])

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.net(x, training=training)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class SemanticROIGraphFER(tf.keras.Model):
    """End-to-end semantic compositional facial reasoning model (TensorFlow port)."""

    def __init__(self, config: SemanticRoiGraphConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.backbone = SemanticBackbone(
            feature_dim=config.feature_dim,
            use_pretrained=config.use_pretrained,
        )
        self.roi_align_layer = SemanticRoiAlign(
            roi_grid=config.roi_grid,
            bbox_input_size=config.bbox_input_size,
            feature_out_size=config.backbone_out_size,
        )
        self.micro_reasoner = MicroGraphReasoner(
            dim=config.feature_dim,
            num_nodes=config.roi_grid * config.roi_grid,
            layers=config.micro_layers,
            heads=config.attn_heads,
            dropout=config.dropout,
        )
        self.semantic_state_encoder = SemanticStateEncoder(
            input_dim=config.feature_dim,
            state_dim=config.semantic_state_dim,
            hidden_dim=max(config.feature_dim // 2, config.semantic_state_dim * 2),
            dropout=config.dropout,
        )
        self.semantic_interaction_block = SemanticInteractionBlock(
            state_dim=config.semantic_state_dim,
            hidden_dim=max(config.semantic_state_dim * 2, 32),
            dropout=config.dropout,
            dropedge_rate=0.5,
        )
        self.micro_motif_bank = MicroSemanticMotifBank(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            state_dim=config.semantic_state_dim,
        )
        self.micro_motif_matcher = MicroSemanticMotifMatcher(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            state_dim=config.semantic_state_dim,
            temperature=config.relation_temperature,
        )
        self.semantic_compositional_reasoner = SemanticHypergraphReasoner(
            state_dim=config.semantic_state_dim,
            latent_dim=config.semantic_latent_dim,
            hyperedge_count=config.hyperedge_count,
            attn_heads=config.semantic_attn_heads,
            router_hidden_dim=config.router_hidden_dim,
            dropout=config.dropout,
        )
        self.cross_region_composition_graph = CrossRegionCompositionGraph(
            state_dim=config.semantic_state_dim,
            num_compositions=config.cross_region_compositions,
            attn_heads=config.semantic_attn_heads,
            hidden_dim=max(config.semantic_state_dim * 2, 32),
            dropout=config.dropout,
        )
        self.semantic_program_bank = SemanticCompositionalProgramBank(
            num_classes=config.num_classes,
            programs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            state_dim=config.semantic_state_dim,
        )
        self.semantic_program_executor = SemanticProgramExecutor(
            num_classes=config.num_classes,
            programs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            state_dim=config.semantic_state_dim,
            temperature=config.relation_temperature,
        )
        self.semantic_classifier = SemanticEmotionClassifier(
            latent_dim=config.semantic_latent_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
        # Global context branch (GlobalAvgPool + Dense)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.global_context_net = tf.keras.Sequential([
            tf.keras.layers.Dense(config.semantic_latent_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(config.dropout),
        ])
        self.global_fusion = tf.keras.Sequential([
            tf.keras.layers.Dense(config.semantic_latent_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
        ])
        self.region_reliability_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(config.feature_dim // 2),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation("sigmoid"),
        ])
        self.region_dropout_prob = float(getattr(config, "region_dropout_prob", 0.05))

    def build(self, input_shape):
        self.semantic_structure_gate = self.add_weight(
            name="semantic_structure_gate",
            shape=(self.config.num_classes,),
            initializer=tf.initializers.Constant(-0.5),
            trainable=True,
        )
        self.missing_region_token = self.add_weight(
            name="missing_region_token",
            shape=(self.config.feature_dim,),
            initializer=tf.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def _canonical_bboxes(self, batch_size: int) -> tf.Tensor:
        boxes = _canonical_region_boxes_tf(self.config.bbox_input_size)
        return tf.tile(tf.expand_dims(boxes, 0), [batch_size, 1, 1])

    def _prepare_regions(self, bboxes, batch_size: int, dtype=tf.float32):
        """Validate and repair bboxes. Returns (bboxes, region_mask, region_confidence)."""
        if bboxes is None:
            repaired = self._canonical_bboxes(batch_size)
            region_mask = tf.ones([batch_size, self.config.num_regions], dtype=dtype)
            region_confidence = tf.fill([batch_size, self.config.num_regions], tf.cast(0.95, dtype))
            return tf.cast(repaired, dtype), region_mask, region_confidence

        bboxes = tf.cast(bboxes, dtype)
        x1 = bboxes[..., 0]; y1 = bboxes[..., 1]
        x2 = bboxes[..., 2]; y2 = bboxes[..., 3]

        finite_mask = tf.reduce_all(tf.math.is_finite(bboxes), axis=-1)
        size_mask = ((x2 - x1) >= 2.0) & ((y2 - y1) >= 2.0)
        order_mask = (x2 > x1) & (y2 > y1)
        region_mask = tf.cast(finite_mask & size_mask & order_mask, dtype)

        repaired = self.roi_align_layer.validate_bboxes(bboxes)
        canonical = tf.cast(self._canonical_bboxes(batch_size), dtype)
        valid_bool = tf.cast(region_mask, tf.bool)
        repaired = tf.where(
            tf.expand_dims(valid_bool, -1),
            repaired,
            canonical
        )

        width = tf.maximum(repaired[..., 2] - repaired[..., 0], 1.0)
        height = tf.maximum(repaired[..., 3] - repaired[..., 1], 1.0)
        area = (width * height) / float(self.config.bbox_input_size ** 2)
        area_conf = tf.cast(tf.clip_by_value(area, 0.0, 1.0), dtype)
        region_confidence = tf.where(
            valid_bool,
            tf.cast(0.5, dtype) + tf.cast(0.5, dtype) * area_conf,
            tf.fill(tf.shape(area_conf), tf.cast(0.05, dtype)),
        )
        return repaired, tf.cast(region_mask, dtype), tf.cast(region_confidence, dtype)

    def call(self, inputs, training: bool = False):
        """
        inputs: tuple of (image, bboxes) or just image tensor
        image: (B, H, W, C) NHWC
        bboxes: (B, R, 4) pixel coords [x1, y1, x2, y2], optional
        """
        if isinstance(inputs, (list, tuple)):
            image = inputs[0]
            bboxes = inputs[1] if len(inputs) > 1 else None
            region_mask = inputs[2] if len(inputs) > 2 else None
            region_confidence = inputs[3] if len(inputs) > 3 else None
        else:
            image = inputs
            bboxes = None
            region_mask = None
            region_confidence = None

        # Ensure 3-channel
        if image.shape[-1] == 1:
            image = tf.repeat(image, 3, axis=-1)

        batch_size = tf.shape(image)[0]

        # --- Backbone ---
        feature_map = self.backbone(image, training=training)  # (B, H/8, W/8, feature_dim)

        # --- Region preparation ---
        bboxes_prep, computed_mask, computed_confidence = self._prepare_regions(
            bboxes, batch_size, dtype=image.dtype
        )
        if region_mask is None:
            region_mask = computed_mask
        
        if region_confidence is None:
            region_confidence = computed_confidence

        region_mask = tf.cast(region_mask, tf.float32)
        region_confidence = tf.cast(region_confidence, tf.float32)

        # Region dropout during training
        if training and self.region_dropout_prob > 0:
            drop_mask = tf.cast(
                tf.random.uniform([batch_size, self.config.num_regions]) > self.region_dropout_prob,
                tf.float32
            )
            region_mask = region_mask * drop_mask
            region_confidence = region_confidence * drop_mask

        # --- ROI Align ---
        roi_nodes = self.roi_align_layer(feature_map, bboxes_prep)  # (B, R, G*G, C)

        # --- Micro Graph Reasoning ---
        micro_node_features, region_embeddings = self.micro_reasoner(roi_nodes, training=training)

        # Fill missing regions
        missing_token = tf.reshape(self.missing_region_token, [1, 1, -1])
        region_valid = tf.expand_dims(region_mask, -1) > 0
        region_embeddings = tf.where(
            region_valid,
            region_embeddings,
            tf.broadcast_to(missing_token, tf.shape(region_embeddings)),
        )

        # Predicted confidence
        predicted_conf = self.region_reliability_predictor(
            region_embeddings, training=training
        )[..., 0]
        region_confidence = tf.clip_by_value(
            tf.cast(0.5, predicted_conf.dtype) * tf.cast(region_confidence, predicted_conf.dtype) + 
            tf.cast(0.5, predicted_conf.dtype) * predicted_conf, 
            0.0, 1.0
        ) * tf.cast(region_mask, predicted_conf.dtype)

        # --- Semantic State Encoding ---
        semantic_state_tokens = self.semantic_state_encoder(region_embeddings, training=training)

        # --- Micro Motif Matching ---
        micro_motif_bank = self.micro_motif_bank(None)
        micro_motif_attention, semantic_motif_tokens = self.micro_motif_matcher(
            semantic_state_tokens, micro_motif_bank, training=training
        )

        # --- Semantic Interaction Block ---
        interaction_states, semantic_interaction_tensor, semantic_interaction_gates = (
            self.semantic_interaction_block(
                semantic_motif_tokens, region_mask=region_mask, training=training
            )
        )

        # --- Cross-Region Composition ---
        cross_region_outputs = self.cross_region_composition_graph(
            interaction_states,
            region_mask=region_mask,
            region_confidence=region_confidence,
            training=training,
        )
        cross_region_tokens = cross_region_outputs["cross_region_tokens"]
        cross_region_attention = cross_region_outputs["pair_attention"]

        # --- Hypergraph Reasoner ---
        composition_summary = tf.reduce_mean(cross_region_tokens, axis=1, keepdims=True)
        hypergraph_input = interaction_states + tf.broadcast_to(
            composition_summary, tf.shape(interaction_states)
        )
        compositional_outputs = self.semantic_compositional_reasoner(
            hypergraph_input, region_mask=region_mask,
            region_confidence=region_confidence, training=training
        )
        composed_states = compositional_outputs["composed_states"]
        hyperedge_tokens = compositional_outputs["hyperedge_tokens"]
        routing_weights = compositional_outputs["routing_weights"]
        semantic_latent_embedding = compositional_outputs["emotion_latent"]

        # --- Program Bank & Executor ---
        semantic_program_bank, semantic_program_topology = self.semantic_program_bank(None)
        semantic_program_outputs = self.semantic_program_executor(
            composed_states,
            cross_region_tokens,
            semantic_program_bank,
            semantic_program_topology,
            region_mask=region_mask,
            interaction_gates=semantic_interaction_gates,
            routing_weights=routing_weights,
            training=training,
        )
        semantic_program_scores = semantic_program_outputs["program_scores"]
        semantic_program_attention = semantic_program_outputs["program_attention"]
        semantic_program_tokens = semantic_program_outputs["program_tokens"]
        routing_entropy = semantic_program_outputs["routing_entropy"]

        # --- Global Context Fusion ---
        global_ctx = self.global_pool(feature_map)
        global_ctx = self.global_context_net(global_ctx, training=training)
        fused_latent = self.global_fusion(
            tf.concat([semantic_latent_embedding, global_ctx], axis=-1),
            training=training,
        )
        logits_fused = self.semantic_classifier(fused_latent, training=training)

        # Per-class gate
        structure_gate = tf.sigmoid(self.semantic_structure_gate)[None]  # (1, C)
        logits_motif = semantic_program_scores
        logits = (1.0 - structure_gate) * logits_fused + structure_gate * logits_motif
        logits = tf.cast(logits, tf.float32)

        return {
            "logits": logits,
            "logits_motif": logits_motif,
            "logits_fused": logits_fused,
            "structure_gate": structure_gate,
            "region_embeddings": region_embeddings,
            "semantic_state_tokens": semantic_state_tokens,
            "semantic_motif_tokens": semantic_motif_tokens,
            "micro_motif_attention": micro_motif_attention,
            "cross_region_tokens": cross_region_tokens,
            "cross_region_attention": cross_region_attention,
            "semantic_interaction_tensor": semantic_interaction_tensor,
            "semantic_interaction_gates": semantic_interaction_gates,
            "semantic_routing_weights": routing_weights,
            "hyperedge_tokens": hyperedge_tokens,
            "semantic_program_scores": semantic_program_scores,
            "semantic_program_attention": semantic_program_attention,
            "semantic_program_tokens": semantic_program_tokens,
            "semantic_program_bank": semantic_program_bank,
            "semantic_program_topology": semantic_program_topology,
            "semantic_latent_embedding": semantic_latent_embedding,
            "fused_latent_embedding": fused_latent,
            "region_mask": region_mask,
            "region_confidence": region_confidence,
            "macro_embeddings": composed_states,
            "macro_motif_attention": semantic_program_attention,
            "aux_losses": {
                "semantic_consistency": tf.zeros(()),
            },
        }

    def call_with_tta(self, image, bboxes=None, region_mask=None, region_confidence=None):
        """Horizontal flip TTA during inference."""
        outputs_orig = self.call(
            (image, bboxes, region_mask, region_confidence), training=False
        )

        if bboxes is None:
            return outputs_orig

        w = float(self.config.bbox_input_size)
        flipped_image = image[:, :, ::-1, :]  # flip W axis (NHWC)

        flipped_bboxes = bboxes
        # x1_new = (w-1) - x2, x2_new = (w-1) - x1
        x1_new = (w - 1.0) - bboxes[..., 2]
        y1_new = bboxes[..., 1]
        x2_new = (w - 1.0) - bboxes[..., 0]
        y2_new = bboxes[..., 3]
        flipped_bboxes = tf.stack([x1_new, y1_new, x2_new, y2_new], axis=-1)

        # Swap symmetric pairs: (1,2), (4,5), (7,8)
        swap_pairs = [(1, 2), (4, 5), (7, 8)]
        bboxes_list = tf.unstack(flipped_bboxes, axis=1)
        mask_list = tf.unstack(region_mask, axis=1) if region_mask is not None else None
        conf_list = tf.unstack(region_confidence, axis=1) if region_confidence is not None else None

        for i, j in swap_pairs:
            bboxes_list[i], bboxes_list[j] = bboxes_list[j], bboxes_list[i]
            if mask_list:
                mask_list[i], mask_list[j] = mask_list[j], mask_list[i]
            if conf_list:
                conf_list[i], conf_list[j] = conf_list[j], conf_list[i]

        flipped_bboxes = tf.stack(bboxes_list, axis=1)
        flipped_mask = tf.stack(mask_list, axis=1) if mask_list else None
        flipped_conf = tf.stack(conf_list, axis=1) if conf_list else None

        outputs_flipped = self.call(
            (flipped_image, flipped_bboxes, flipped_mask, flipped_conf), training=False
        )

        avg_keys = ("logits", "logits_motif", "logits_fused", "semantic_program_scores")
        avg_outputs = {}
        for k, val in outputs_orig.items():
            if k in avg_keys and tf.is_tensor(val) and k in outputs_flipped:
                avg_outputs[k] = 0.5 * (val + outputs_flipped[k])
            else:
                avg_outputs[k] = val
        return avg_outputs
