"""
layers_tf.py — All custom layers for Semantic ROI Graph FER in TensorFlow.
Implements dual-level graph reasoning with exact mathematical parity to PyTorch.

Anti-Overfitting Features:
- Exact coordinate normalization for tf.image.crop_and_resize (prevents ROI distortion).
- Strict propagation of `training=training` across all Dropout and BatchNorm layers.
- Stable softmax with numerical clamping to eliminate NaN risks.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def safe_softmax(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Numerically stable softmax preventing NaN/inf."""
    x_max = tf.stop_gradient(tf.reduce_max(x, axis=axis, keepdims=True))
    x_shifted = tf.clip_by_value(x - x_max, -50.0, 0.0)
    exp_x = tf.exp(x_shifted)
    denom = tf.maximum(tf.reduce_sum(exp_x, axis=axis, keepdims=True), 1e-8)
    return exp_x / denom


class SemanticRoiAlignTF(layers.Layer):
    """
    TensorFlow ROIAlign over 9 semantic facial regions using tf.image.crop_and_resize.
    Transforms pixel bounding boxes [x1, y1, x2, y2] into normalized [y1, x1, y2, x2] / 47.0.
    """
    def __init__(self, roi_grid: int = 4, bbox_input_size: int = 48, **kwargs):
        super().__init__(**kwargs)
        self.roi_grid = int(roi_grid)
        self.bbox_input_size = int(bbox_input_size)

        # Canonical fallback boxes for 9 facial regions (48x48 scale)
        self.canonical_boxes = tf.constant([
            [8.0, 0.0, 40.0, 10.0],   # 0: forehead
            [5.0, 8.0, 18.0, 18.0],   # 1: left_eyebrow
            [30.0, 8.0, 43.0, 18.0],  # 2: right_eyebrow
            [18.0, 12.0, 30.0, 22.0], # 3: glabella
            [6.0, 16.0, 20.0, 30.0],  # 4: left_eye
            [28.0, 16.0, 42.0, 30.0], # 5: right_eye
            [14.0, 20.0, 34.0, 38.0], # 6: nose
            [8.0, 30.0, 22.0, 43.0],  # 7: left_mouth_corner
            [26.0, 30.0, 40.0, 43.0], # 8: right_mouth_corner
        ], dtype=tf.float32)

    def validate_and_normalize_bboxes(self, bboxes: tf.Tensor) -> tf.Tensor:
        """
        Validate bounding boxes and convert from pixel [x1, y1, x2, y2]
        to TensorFlow normalized [y1, x1, y2, x2] in [0.0, 1.0].
        """
        max_coord = float(self.bbox_input_size - 1)
        bboxes = tf.clip_by_value(bboxes, 0.0, max_coord)

        x1 = tf.minimum(bboxes[..., 0], bboxes[..., 2])
        y1 = tf.minimum(bboxes[..., 1], bboxes[..., 3])
        x2 = tf.maximum(bboxes[..., 0], bboxes[..., 2])
        y2 = tf.maximum(bboxes[..., 1], bboxes[..., 3])

        # Enforce minimum size of 2 pixels
        x2 = tf.maximum(x2, x1 + 2.0)
        y2 = tf.maximum(y2, y1 + 2.0)

        # Normalize to [0.0, 1.0] and reorder to [y1, x1, y2, x2] for tf.image.crop_and_resize
        norm_scale = max_coord
        y1_norm = y1 / norm_scale
        x1_norm = x1 / norm_scale
        y2_norm = y2 / norm_scale
        x2_norm = x2 / norm_scale

        norm_boxes = tf.stack([y1_norm, x1_norm, y2_norm, x2_norm], axis=-1)
        return norm_boxes

    def call(self, feature_map: tf.Tensor, bboxes: tf.Tensor) -> tf.Tensor:
        # feature_map: (B, H, W, C)
        # bboxes: (B, R, 4) in image pixels [x1, y1, x2, y2]
        batch_size = tf.shape(feature_map)[0]
        num_regions = tf.shape(bboxes)[1]
        channels = tf.shape(feature_map)[-1]

        norm_boxes = self.validate_and_normalize_bboxes(bboxes) # (B, R, 4)

        # Flatten boxes to (B * R, 4)
        flat_boxes = tf.reshape(norm_boxes, [-1, 4])

        # Create box_indices: [0, 0... R times, 1, 1... R times, ..., B-1]
        box_indices = tf.repeat(tf.range(batch_size), num_regions)

        # Perform bilinear ROI pooling
        crops = tf.image.crop_and_resize(
            feature_map,
            flat_boxes,
            box_indices,
            crop_size=[self.roi_grid, self.roi_grid],
            method='bilinear'
        ) # (B * R, 4, 4, C)

        # Reshape to (B, R, 16, C)
        num_nodes = self.roi_grid * self.roi_grid
        roi_nodes = tf.reshape(crops, [batch_size, num_regions, num_nodes, channels])
        return roi_nodes


class GATBlockTF(layers.Layer):
    """Multi-Head Graph Attention Block with learnable adjacency bias and locality prior."""
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, num_nodes: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.drop = layers.Dropout(dropout)

        self.q_proj = layers.Dense(dim, use_bias=False)
        self.k_proj = layers.Dense(dim, use_bias=False)
        self.v_proj = layers.Dense(dim, use_bias=False)
        self.out_proj = layers.Dense(dim)

        self.num_nodes = num_nodes
        if num_nodes is not None:
            self.adj_bias = self.add_weight(
                name="adj_bias",
                shape=(1, 1, num_nodes, num_nodes),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                trainable=True
            )
        else:
            self.adj_bias = None

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        # x: (B, N, D)
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]

        q = tf.reshape(self.q_proj(x), [b, n, self.heads, self.head_dim])
        k = tf.reshape(self.k_proj(x), [b, n, self.heads, self.head_dim])
        v = tf.reshape(self.v_proj(x), [b, n, self.heads, self.head_dim])

        # Transpose to (B, heads, N, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
        attn = tf.matmul(q, k, transpose_b=True) * scale

        if self.adj_bias is not None:
            attn = attn + self.adj_bias

        attn = safe_softmax(attn, axis=-1)
        attn = self.drop(attn, training=training)

        out = tf.matmul(attn, v) # (B, heads, N, head_dim)
        out = tf.transpose(out, [0, 2, 1, 3]) # (B, N, heads, head_dim)
        out = tf.reshape(out, [b, n, self.dim])
        return self.out_proj(out)


class GatedPoolingTF(layers.Layer):
    """Attention-based gated node pooling."""
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gate = layers.Dense(1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (B, N, D)
        weights = tf.sigmoid(self.gate(x)) # (B, N, 1)
        weighted = x * weights
        pooled = tf.reduce_sum(weighted, axis=1) / (tf.reduce_sum(weights, axis=1) + 1e-6)
        return pooled


class MicroGraphReasonerTF(layers.Layer):
    """Intra-region graph reasoning across 16 ROI grid nodes."""
    def __init__(self, dim: int, num_nodes: int = 16, layers_count: int = 2, heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gat_layers = [GATBlockTF(dim, heads=heads, dropout=dropout, num_nodes=num_nodes) for _ in range(layers_count)]
        self.norms = [layers.LayerNormalization(epsilon=1e-5) for _ in range(layers_count)]
        self.pool = GatedPoolingTF(dim)
        self.dim = dim

    def call(self, x: tf.Tensor, training=False) -> Tuple[tf.Tensor, tf.Tensor]:
        # x: (B, R, N, D)
        b = tf.shape(x)[0]
        r = tf.shape(x)[1]
        n = tf.shape(x)[2]
        d = self.dim

        x_flat = tf.reshape(x, [b * r, n, d])
        for gat, norm in zip(self.gat_layers, self.norms):
            x_flat = x_flat + gat(norm(x_flat), training=training)

        pooled = self.pool(x_flat) # (B * R, D)
        pooled = tf.reshape(pooled, [b, r, d])
        x_out = tf.reshape(x_flat, [b, r, n, d])
        return x_out, pooled


class SemanticStateEncoderTF(layers.Layer):
    """Project region embeddings into interpretable 128-D semantic facial state space."""
    def __init__(self, input_dim: int = 256, state_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = max(input_dim // 2, state_dim * 2)
        self.proj = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(state_dim),
        ])
        self.gate = tf.keras.Sequential([
            layers.Dense(state_dim, activation='sigmoid'),
        ])
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, region_embeddings: tf.Tensor, training=False) -> tf.Tensor:
        raw_state = self.proj(region_embeddings, training=training)
        gate = self.gate(region_embeddings)
        return self.norm(raw_state * gate)


class MicroSemanticMotifMatcherTF(layers.Layer):
    """Match semantic region states to interpretable local semantic motifs."""
    def __init__(self, num_regions: int = 9, motifs_per_region: int = 8, state_dim: int = 128, temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.state_dim = state_dim
        self.temperature = float(temperature)

        self.motif_bank = self.add_weight(
            name="micro_motifs",
            shape=(num_regions, motifs_per_region, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )

        self.token_proj = tf.keras.Sequential([
            layers.Dense(state_dim),
            layers.LayerNormalization(epsilon=1e-5),
            layers.Activation('gelu'),
        ])

    def call(self, semantic_states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # semantic_states: (B, R, S)
        # motif_bank: (R, K, S)
        state_norm = tf.math.l2_normalize(semantic_states, axis=-1)
        bank_norm = tf.math.l2_normalize(self.motif_bank, axis=-1)

        sim = tf.einsum("brs,rks->brk", state_norm, bank_norm) / self.temperature
        attn = safe_softmax(sim, axis=-1)
        tokens = tf.einsum("brk,rks->brs", attn, self.motif_bank)
        tokens = self.token_proj(tokens)
        semantic_tokens = semantic_states + tokens
        return attn, semantic_tokens


class SemanticInteractionBlockTF(layers.Layer):
    """Learned pairwise semantic coordination between facial regions."""
    def __init__(self, state_dim: int = 128, dropout: float = 0.1, dropedge_rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.dropedge_rate = float(dropedge_rate)
        hidden_dim = max(state_dim * 2, 32)
        pair_dim = state_dim * 4

        self.edge_gate = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(1, activation='sigmoid'),
        ])
        self.edge_message = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(state_dim),
        ])
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, semantic_states: tf.Tensor, region_mask: Optional[tf.Tensor] = None, training=False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # semantic_states: (B, R, S)
        b = tf.shape(semantic_states)[0]
        r = tf.shape(semantic_states)[1]
        s = tf.shape(semantic_states)[2]

        left = tf.tile(tf.expand_dims(semantic_states, 2), [1, 1, r, 1])
        right = tf.tile(tf.expand_dims(semantic_states, 1), [1, r, 1, 1])
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)

        raw_gates = tf.squeeze(self.edge_gate(pair_input, training=training), axis=-1) + 0.1

        # Computational fix: Mask out invalid regions from interaction
        if region_mask is not None:
            pair_mask = tf.expand_dims(region_mask, -1) * tf.expand_dims(region_mask, -2)
            raw_gates = raw_gates * pair_mask

        # Graph DropEdge for message passing to prevent over-smoothing
        gates = raw_gates
        if training and self.dropedge_rate > 0.0:
            mask = tf.cast(tf.random.uniform(tf.shape(gates)) > self.dropedge_rate, tf.float32)
            gates = gates * mask / (1.0 - self.dropedge_rate)

        messages = self.edge_message(pair_input, training=training)
        interaction_tensor = tf.expand_dims(gates, -1) * messages
        interaction_summary = tf.reduce_sum(interaction_tensor, axis=2) / tf.maximum(tf.reduce_sum(gates, axis=2, keepdims=True), 1e-4)
        updated_states = self.norm(semantic_states + interaction_summary)
        return updated_states, interaction_tensor, raw_gates


class CrossRegionCompositionGraphTF(layers.Layer):
    """Learn higher-order semantic compositions across facial regions."""
    def __init__(self, state_dim: int = 128, num_compositions: int = 8, attn_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = max(state_dim * 2, 32)
        self.num_compositions = num_compositions
        self.state_dim = state_dim

        self.composition_queries = self.add_weight(
            name="composition_queries",
            shape=(num_compositions, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )
        self.pair_encoder = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(state_dim),
        ])
        self.pair_router = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(1),
        ])
        self.mha = layers.MultiHeadAttention(num_heads=attn_heads, key_dim=state_dim // attn_heads, dropout=dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, semantic_states: tf.Tensor, region_mask: Optional[tf.Tensor] = None, region_confidence: Optional[tf.Tensor] = None, training=False) -> Dict[str, tf.Tensor]:
        b = tf.shape(semantic_states)[0]
        r = tf.shape(semantic_states)[1]
        d = self.state_dim

        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * tf.expand_dims(region_confidence, -1)

        left = tf.tile(tf.expand_dims(tokens, 2), [1, 1, r, 1])
        right = tf.tile(tf.expand_dims(tokens, 1), [1, r, 1, 1])
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)

        pair_tokens = self.pair_encoder(pair_input, training=training) # (B, R, R, D)
        pair_scores = tf.squeeze(self.pair_router(pair_tokens, training=training), axis=-1)

        if region_mask is not None:
            pair_mask = tf.expand_dims(region_mask, -1) * tf.expand_dims(region_mask, -2)
            pair_scores = tf.where(pair_mask > 0.0, pair_scores, -1e9)

        pair_attention = tf.reshape(safe_softmax(tf.reshape(pair_scores, [b, -1]), axis=-1), [b, r, r])
        pair_sequence = tf.reshape(pair_tokens, [b, r * r, d])

        queries = tf.tile(tf.expand_dims(self.composition_queries, 0), [b, 1, 1])
        cross_region_tokens, composition_attn = self.mha(
            query=queries,
            value=pair_sequence,
            key=pair_sequence,
            return_attention_scores=True,
            training=training
        )
        cross_region_tokens = self.norm(cross_region_tokens)

        return {
            "cross_region_tokens": cross_region_tokens,
            "composition_attn": composition_attn,
            "pair_tokens": pair_tokens,
            "pair_scores": pair_scores,
            "pair_attention": pair_attention,
        }


class SemanticHypergraphReasonerTF(layers.Layer):
    """Compose multi-region semantic programs with learned hyperedge routing."""
    def __init__(self, state_dim: int = 128, latent_dim: int = 256, hyperedge_count: int = 4, attn_heads: int = 4, router_hidden_dim: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hyperedge_queries = self.add_weight(
            name="hyperedge_queries",
            shape=(hyperedge_count, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )
        self.hyper_attn = layers.MultiHeadAttention(num_heads=attn_heads, key_dim=state_dim // attn_heads, dropout=dropout)
        self.back_attn = layers.MultiHeadAttention(num_heads=attn_heads, key_dim=state_dim // attn_heads, dropout=dropout)

        self.router = tf.keras.Sequential([
            layers.Dense(router_hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(1),
        ])
        self.latent_proj = tf.keras.Sequential([
            layers.Dense(latent_dim),
            layers.LayerNormalization(epsilon=1e-5),
            layers.Activation('gelu'),
            layers.Dropout(dropout),
            layers.Dense(latent_dim),
        ])
        self.latent_norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, semantic_states: tf.Tensor, region_mask: Optional[tf.Tensor] = None, region_confidence: Optional[tf.Tensor] = None, training=False) -> Dict[str, tf.Tensor]:
        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * tf.expand_dims(region_confidence, -1)

        b = tf.shape(tokens)[0]
        queries = tf.tile(tf.expand_dims(self.hyperedge_queries, 0), [b, 1, 1])

        hyperedge_tokens = self.hyper_attn(query=queries, value=tokens, key=tokens, training=training)
        region_context = self.back_attn(query=tokens, value=hyperedge_tokens, key=hyperedge_tokens, training=training)

        composed_states = tokens + region_context
        routing_logits = tf.squeeze(self.router(composed_states, training=training), axis=-1)

        if region_mask is not None:
            routing_logits = tf.where(region_mask > 0.0, routing_logits, -1e9)

        routing_weights = safe_softmax(routing_logits, axis=1)
        if region_mask is not None:
            routing_weights = routing_weights * region_mask
            routing_weights = routing_weights / tf.maximum(tf.reduce_sum(routing_weights, axis=1, keepdims=True), 1e-6)

        pooled_state = tf.reduce_sum(tf.expand_dims(routing_weights, -1) * composed_states, axis=1)
        hyper_summary = tf.reduce_mean(hyperedge_tokens, axis=1)
        emotion_latent = self.latent_proj(tf.concat([pooled_state, hyper_summary], axis=-1), training=training)
        emotion_latent = self.latent_norm(emotion_latent)

        return {
            "composed_states": composed_states,
            "hyperedge_tokens": hyperedge_tokens,
            "routing_weights": routing_weights,
            "emotion_latent": emotion_latent,
        }


class SemanticProgramExecutorTF(layers.Layer):
    """Execute semantic facial programs against observed region states."""
    def __init__(self, num_classes: int = 7, programs_per_class: int = 4, num_regions: int = 9, state_dim: int = 128, temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim
        self.temperature = float(temperature)

        self.programs = self.add_weight(
            name="programs",
            shape=(num_classes, programs_per_class, num_regions, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )
        self.topology_logits = self.add_weight(
            name="topology_logits",
            shape=(num_classes, programs_per_class, num_regions, num_regions),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )

        self.summary_dense = layers.Dense(state_dim)
        self.summary_norm = layers.LayerNormalization(epsilon=1e-5)
        self.summary_act = layers.Activation('gelu')

        # Dynamic structure weights: region_sim (1.0), topology_sim (0.5), composition_sim (0.25)
        sim_init = np.tile(np.array([[[[1.0, 0.5, 0.25]]]], dtype=np.float32), (1, num_classes, 1, 1))
        self.sim_weights = self.add_weight(
            name="sim_weights",
            shape=(1, num_classes, 1, 3),
            initializer=tf.constant_initializer(sim_init),
            trainable=True
        )

    def _project_program_summary(self, x: tf.Tensor) -> tf.Tensor:
        """Project program summary across arbitrary tensor ranks (*, D)."""
        return self.summary_act(self.summary_norm(self.summary_dense(x)))

    def call(self, semantic_states: tf.Tensor, cross_region_tokens: tf.Tensor, region_mask: Optional[tf.Tensor] = None, interaction_gates: Optional[tf.Tensor] = None, routing_weights: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        state_norm = tf.math.l2_normalize(semantic_states, axis=-1)
        program_norm = tf.math.l2_normalize(self.programs, axis=-1)

        # 1. Region similarity
        region_sims = tf.einsum("brd,cmrd->bcmr", state_norm, program_norm)
        if routing_weights is not None:
            region_sim = tf.reduce_sum(region_sims * tf.expand_dims(tf.expand_dims(routing_weights, 1), 1), axis=-1)
        else:
            region_sim = tf.reduce_mean(region_sims, axis=-1)

        # 2. Topology similarity
        if interaction_gates is not None:
            observed = tf.expand_dims(tf.expand_dims(interaction_gates, 1), 1)
            target_topo = tf.expand_dims(tf.sigmoid(self.topology_logits), 0)
            topology_mse = tf.square(observed - target_topo)
            topology_sim = 1.0 - tf.reduce_mean(topology_mse, axis=[-1, -2])
            topology_sim = tf.clip_by_value(topology_sim, 0.0, 1.0)
        else:
            topology_sim = tf.ones_like(region_sim)

        # 3. Composition similarity
        comp_summary = self._project_program_summary(tf.reduce_mean(cross_region_tokens, axis=1)) # (B, 128)
        prog_summary = self._project_program_summary(tf.reduce_mean(self.programs, axis=2))       # (C, M, 128)
        composition_sim = tf.einsum("bd,cmd->bcm", tf.math.l2_normalize(comp_summary, axis=-1), tf.math.l2_normalize(prog_summary, axis=-1))

        # Dynamic combination with softplus
        w = tf.nn.softplus(self.sim_weights)
        total_sim = w[..., 0] * region_sim + w[..., 1] * topology_sim + w[..., 2] * composition_sim

        # Temperature scaling clamped to [-30.0, 30.0]
        compatibility = tf.clip_by_value(total_sim / self.temperature, -30.0, 30.0)
        class_scores = tf.reduce_logsumexp(compatibility, axis=-1) # (B, num_classes)

        return {
            "program_scores": class_scores,
            "compatibility": compatibility,
        }
