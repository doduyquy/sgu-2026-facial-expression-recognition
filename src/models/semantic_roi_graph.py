import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

def safe_softmax(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """A numerically stable softmax that prevents NaN when vectors are fully masked."""
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    all_invalid = tf.logical_or(
        tf.reduce_all(tf.math.is_inf(x_shifted), axis=axis, keepdims=True),
        tf.reduce_all(tf.math.is_nan(x_shifted), axis=axis, keepdims=True)
    )
    x_shifted = tf.where(all_invalid, tf.zeros_like(x_shifted), x_shifted)
    return tf.nn.softmax(x_shifted, axis=axis)

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
    backbone_type: str = "resnet50"
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

class ResNet50Backbone(tf.keras.Model):
    """ResNet50 backbone with projection for high spatial resolution Graph input."""
    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        weights = 'imagenet' if use_pretrained else None
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=weights)
        config = base_model.get_config()
        
        # Modify the config for stride=1 and identity maxpool
        for layer in config['layers']:
            if layer['name'] == 'conv1_conv':
                layer['config']['strides'] = (1, 1)
            elif layer['name'] == 'pool1_pool':
                layer['config']['pool_size'] = (1, 1)
                layer['config']['strides'] = (1, 1)
                layer['config']['padding'] = 'same'
                
        modified_model = tf.keras.Model.from_config(config)
        if weights:
            modified_model.set_weights(base_model.get_weights())
            
        self.feature_extractor = tf.keras.Model(
            inputs=modified_model.input,
            outputs=modified_model.get_layer('conv3_block4_out').output
        )
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Conv2D(feature_dim, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('gelu')
        ])

    def call(self, x, training=False):
        if tf.shape(x)[-1] == 1:
            x = tf.tile(x, [1, 1, 1, 3])
        x = self.feature_extractor(x, training=training)
        return self.proj(x, training=training)

class HRNetBackbone(tf.keras.Model):
    """HRNet-W18 backbone (Removed in TF conversion)."""
    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
    def call(self, x, training=False):
        raise NotImplementedError("HRNet is not supported in TensorFlow implementation.")

class SemanticRoiAlign(tf.keras.layers.Layer):
    """ROIAlign over semantic regions using tf.image.crop_and_resize."""
    def __init__(self, roi_grid: int = 4, bbox_input_size: int = 48, feature_out_size: int = 12):
        super().__init__()
        self.roi_grid = int(roi_grid)
        self.bbox_input_size = int(bbox_input_size)
        self.feature_out_size = int(feature_out_size)

    def _canonical_region_boxes(self):
        boxes = tf.constant([
            [8, 0, 40, 10],   # forehead
            [5, 8, 18, 18],   # left_eyebrow
            [30, 8, 43, 18],  # right_eyebrow
            [18, 12, 30, 22], # glabella
            [6, 16, 20, 30],  # left_eye
            [28, 16, 42, 30], # right_eye
            [14, 20, 34, 38], # nose
            [8, 30, 22, 43],  # left_mouth_corner
            [26, 30, 40, 43], # right_mouth_corner
        ], dtype=tf.float32)
        return boxes * (float(self.bbox_input_size) / 48.0)

    def validate_bboxes(self, bboxes: tf.Tensor) -> tf.Tensor:
        bboxes = tf.cast(bboxes, tf.float32)
        max_val = float(self.bbox_input_size - 1)
        
        x1 = tf.clip_by_value(bboxes[..., 0], 0.0, max_val)
        y1 = tf.clip_by_value(bboxes[..., 1], 0.0, max_val)
        x2 = tf.clip_by_value(bboxes[..., 2], 0.0, max_val)
        y2 = tf.clip_by_value(bboxes[..., 3], 0.0, max_val)
        
        x1_new = tf.minimum(x1, x2)
        y1_new = tf.minimum(y1, y2)
        x2_new = tf.maximum(x1, x2)
        y2_new = tf.maximum(y1, y2)
        
        x2_new = tf.maximum(x2_new, x1_new + 2.0)
        y2_new = tf.maximum(y2_new, y1_new + 2.0)
        
        x2_new = tf.clip_by_value(x2_new, 0.0, max_val)
        y2_new = tf.clip_by_value(y2_new, 0.0, max_val)
        x1_new = tf.clip_by_value(x1_new, 0.0, max_val - 2.0)
        y1_new = tf.clip_by_value(y1_new, 0.0, max_val - 2.0)
        
        repaired = tf.stack([x1_new, y1_new, x2_new, y2_new], axis=-1)
        too_small = tf.logical_or((repaired[..., 2] - repaired[..., 0]) < 2.0,
                                  (repaired[..., 3] - repaired[..., 1]) < 2.0)
        
        canonical = tf.broadcast_to(tf.expand_dims(self._canonical_region_boxes(), axis=0), tf.shape(repaired))
        return tf.where(tf.expand_dims(too_small, axis=-1), canonical, repaired)

    def call(self, feature_map: tf.Tensor, bboxes: tf.Tensor, training=False) -> tf.Tensor:
        b = tf.shape(feature_map)[0]
        bboxes = self.validate_bboxes(bboxes)
        batch_size = tf.shape(bboxes)[0]
        num_regions = tf.shape(bboxes)[1]
        
        scale = 1.0 / float(self.bbox_input_size - 1)
        x1 = bboxes[..., 0] * scale
        y1 = bboxes[..., 1] * scale
        x2 = bboxes[..., 2] * scale
        y2 = bboxes[..., 3] * scale
        normalized_boxes = tf.reshape(tf.stack([y1, x1, y2, x2], axis=-1), [-1, 4])
        
        box_indices = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1]), [1, num_regions]), [-1])
        
        roi_features = tf.image.crop_and_resize(
            feature_map,
            boxes=normalized_boxes,
            box_indices=box_indices,
            crop_size=[self.roi_grid, self.roi_grid],
            method='bilinear'
        )
        
        channels = tf.shape(feature_map)[-1]
        roi_features = tf.reshape(roi_features, [batch_size, num_regions, self.roi_grid * self.roi_grid, channels])
        return roi_features

class GATBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, num_nodes: Optional[int] = None, use_locality: bool = False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.q_proj = tf.keras.layers.Dense(dim)
        self.k_proj = tf.keras.layers.Dense(dim)
        self.v_proj = tf.keras.layers.Dense(dim)
        self.out_proj = tf.keras.layers.Dense(dim)
        self.num_nodes = num_nodes
        self.use_locality = use_locality

    def build(self, input_shape):
        if self.num_nodes is not None:
            self.adj_bias = self.add_weight(
                name='adj_bias', shape=(1, 1, self.num_nodes, self.num_nodes),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), trainable=True
            )
        else:
            self.adj_bias = None
            
        if self.use_locality and self.num_nodes is not None:
            side = int(self.num_nodes ** 0.5)
            if side * side == self.num_nodes:
                coords_1d = tf.cast(tf.range(side), tf.float32)
                grid_y, grid_x = tf.meshgrid(coords_1d, coords_1d, indexing='ij')
                coords = tf.reshape(tf.stack([grid_y, grid_x], axis=-1), [-1, 2])
            else:
                coords = tf.expand_dims(tf.cast(tf.range(self.num_nodes), tf.float32), axis=-1)
            dist = tf.norm(tf.expand_dims(coords, axis=1) - tf.expand_dims(coords, axis=0), axis=-1)
            dist = dist / tf.maximum(tf.reduce_max(dist), 1e-6)
            self.locality_bias = tf.expand_dims(tf.expand_dims(-dist, axis=0), axis=0)
        else:
            self.locality_bias = None
        super().build(input_shape)

    def call(self, x, edge_prior=None, attn_mask=None, training=False):
        b, n = tf.shape(x)[0], tf.shape(x)[1]
        
        q = tf.transpose(tf.reshape(self.q_proj(x), [b, n, self.heads, self.head_dim]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(self.k_proj(x), [b, n, self.heads, self.head_dim]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(self.v_proj(x), [b, n, self.heads, self.head_dim]), [0, 2, 1, 3])
        
        attn = tf.einsum("bhid,bhjd->bhij", q, k) / (float(self.head_dim) ** 0.5)
        
        if self.adj_bias is not None:
            attn += self.adj_bias
        if self.locality_bias is not None:
            attn += self.locality_bias
        if edge_prior is not None:
            if len(edge_prior.shape) == 2:
                edge_prior = tf.expand_dims(edge_prior, axis=0)
            attn += tf.expand_dims(tf.math.log(tf.maximum(edge_prior, 1e-6)), axis=1)
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, axis=1), axis=2)
            elif len(attn_mask.shape) == 3:
                attn_mask = tf.expand_dims(attn_mask, axis=1)
            attn = tf.where(attn_mask == 0, tf.fill(tf.shape(attn), -1e9), attn)
            
        attn = safe_softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)
        
        out = tf.einsum("bhij,bhjd->bhid", attn, v)
        out = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [b, n, self.dim])
        return self.out_proj(out)

class GatedPooling(tf.keras.layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = tf.keras.layers.Dense(1)
    def call(self, x, training=False):
        weights = tf.nn.sigmoid(self.gate(x))
        weighted = x * weights
        return tf.reduce_sum(weighted, axis=1) / (tf.reduce_sum(weights, axis=1) + 1e-6)

class MicroGraphReasoner(tf.keras.layers.Layer):
    def __init__(self, dim: int, num_nodes: int, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat_layers = [GATBlock(dim, heads=heads, dropout=dropout, num_nodes=num_nodes) for _ in range(layers)]
        self.norms = [tf.keras.layers.LayerNormalization(axis=-1) for _ in range(layers)]
        self.pool = GatedPooling(dim)

    def call(self, x, training=False):
        b, r, n, d = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_flat = tf.reshape(x, [b * r, n, d])
        for layer, norm in zip(self.gat_layers, self.norms):
            x_flat = x_flat + layer(norm(x_flat), training=training)
        pooled = tf.reshape(self.pool(x_flat, training=training), [b, r, d])
        return tf.reshape(x_flat, [b, r, n, d]), pooled

class SemanticStateEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, state_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim // 2, state_dim * 2)
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(state_dim)
        ])
        self.gate = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.Activation('sigmoid')
        ])
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, region_embeddings, training=False):
        raw_state = self.proj(region_embeddings, training=training)
        gate = self.gate(region_embeddings, training=training)
        return self.norm(raw_state * gate)

class MicroSemanticMotifBank(tf.keras.layers.Layer):
    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int):
        super().__init__()
        self.motifs = self.add_weight(
            name='motifs', shape=(num_regions, motifs_per_region, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True
        )
    def call(self, training=False):
        return self.motifs

class MicroSemanticMotifMatcher(tf.keras.layers.Layer):
    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.token_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Activation('gelu')
        ])
    def call(self, semantic_states, motif_bank, training=False):
        state_norm = tf.math.l2_normalize(semantic_states, axis=-1)
        bank_norm = tf.math.l2_normalize(motif_bank, axis=-1)
        sim = tf.einsum("brs,rks->brk", state_norm, bank_norm) / self.temperature
        attn = safe_softmax(sim, axis=-1)
        tokens = tf.einsum("brk,rks->brs", attn, motif_bank)
        return attn, semantic_states + self.token_proj(tokens, training=training)

class SemanticInteractionBlock(tf.keras.layers.Layer):
    def __init__(self, state_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, dropedge_rate: float = 0.5):
        super().__init__()
        self.dropedge_rate = dropedge_rate
        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        self.edge_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])
        self.edge_message = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(state_dim)
        ])
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.drop_edge = tf.keras.layers.Dropout(dropedge_rate)

    def call(self, semantic_states, region_mask=None, training=False):
        b, r, s = tf.shape(semantic_states)[0], tf.shape(semantic_states)[1], tf.shape(semantic_states)[2]
        left = tf.broadcast_to(tf.expand_dims(semantic_states, axis=2), [b, r, r, s])
        right = tf.broadcast_to(tf.expand_dims(semantic_states, axis=1), [b, r, r, s])
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)
        
        gates = tf.squeeze(self.edge_gate(pair_input, training=training), axis=-1) + 0.1
        if self.dropedge_rate > 0.0:
            gates = self.drop_edge(gates, training=training)
            
        if region_mask is not None:
            pair_mask = tf.expand_dims(region_mask, axis=-1) * tf.expand_dims(region_mask, axis=-2)
            gates = gates * pair_mask
            
        messages = self.edge_message(pair_input, training=training)
        interaction_tensor = tf.expand_dims(gates, axis=-1) * messages
        interaction_summary = tf.reduce_sum(interaction_tensor, axis=2) / (tf.expand_dims(tf.reduce_sum(gates, axis=2), axis=-1) + 1e-6)
        return self.norm(semantic_states + interaction_summary), interaction_tensor, gates

class CrossRegionCompositionGraph(tf.keras.layers.Layer):
    def __init__(self, state_dim: int, num_compositions: int, attn_heads: int = 3, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        self.composition_queries = self.add_weight(
            name='composition_queries', shape=(num_compositions, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True
        )
        self.pair_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim), tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(state_dim)
        ])
        self.pair_router = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim), tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(1)
        ])
        self.composition_attn = tf.keras.layers.MultiHeadAttention(num_heads=attn_heads, key_dim=state_dim // attn_heads, dropout=dropout)
        self.composition_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, semantic_states, region_mask=None, region_confidence=None, training=False):
        b, r, d = tf.shape(semantic_states)[0], tf.shape(semantic_states)[1], tf.shape(semantic_states)[2]
        tokens = semantic_states if region_confidence is None else semantic_states * tf.expand_dims(region_confidence, axis=-1)
        
        left = tf.broadcast_to(tf.expand_dims(tokens, axis=2), [b, r, r, d])
        right = tf.broadcast_to(tf.expand_dims(tokens, axis=1), [b, r, r, d])
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)
        pair_tokens = self.pair_encoder(pair_input, training=training)
        pair_scores = tf.squeeze(self.pair_router(pair_tokens, training=training), axis=-1)
        
        if region_mask is not None:
            pair_mask = tf.expand_dims(region_mask, axis=-1) * tf.expand_dims(region_mask, axis=-2)
            pair_scores = tf.where(pair_mask <= 0, tf.fill(tf.shape(pair_scores), -1e9), pair_scores)
            
        pair_attention = tf.reshape(safe_softmax(tf.reshape(pair_scores, [b, -1]), axis=-1), [b, r, r])
        pair_sequence = tf.reshape(pair_tokens, [b, r * r, d])
        
        attention_mask = None
        if region_mask is not None:
            # MultiHeadAttention mask expects True for positions to keep.
            attention_mask = tf.expand_dims(tf.reshape(pair_mask > 0, [b, r * r]), axis=1) # (B, 1, R*R)
            
        composition_queries = tf.broadcast_to(tf.expand_dims(self.composition_queries, axis=0), [b, tf.shape(self.composition_queries)[0], d])
        cross_region_tokens, composition_attn = self.composition_attn(
            composition_queries, pair_sequence, pair_sequence,
            attention_mask=attention_mask, return_attention_scores=True, training=training
        )
        return {
            "cross_region_tokens": self.composition_norm(cross_region_tokens),
            "composition_attn": composition_attn,
            "pair_tokens": pair_tokens,
            "pair_scores": pair_scores,
            "pair_attention": pair_attention
        }

class SemanticHypergraphReasoner(tf.keras.layers.Layer):
    def __init__(self, state_dim: int, latent_dim: int, hyperedge_count: int, attn_heads: int, router_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hyperedge_queries = self.add_weight(
            name='hyperedge_queries', shape=(hyperedge_count, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True
        )
        self.hyperedge_attn = tf.keras.layers.MultiHeadAttention(num_heads=attn_heads, key_dim=state_dim // attn_heads, dropout=dropout)
        self.region_back_attn = tf.keras.layers.MultiHeadAttention(num_heads=attn_heads, key_dim=state_dim // attn_heads, dropout=dropout)
        self.router = tf.keras.Sequential([
            tf.keras.layers.Dense(router_hidden_dim), tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(1)
        ])
        self.latent_projector = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim), tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Activation('gelu'), tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(latent_dim)
        ])
        self.latent_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, semantic_states, region_mask=None, region_confidence=None, training=False):
        tokens = semantic_states if region_confidence is None else semantic_states * tf.expand_dims(region_confidence, axis=-1)
        b, r, d = tf.shape(tokens)[0], tf.shape(tokens)[1], tf.shape(tokens)[2]
        
        attention_mask = None
        if region_mask is not None:
            attention_mask = tf.expand_dims(region_mask > 0, axis=1) # (B, 1, R)
            
        hyper_queries = tf.broadcast_to(tf.expand_dims(self.hyperedge_queries, axis=0), [b, tf.shape(self.hyperedge_queries)[0], d])
        hyperedge_tokens, hyperedge_attn = self.hyperedge_attn(
            hyper_queries, tokens, tokens, attention_mask=attention_mask, return_attention_scores=True, training=training
        )
        region_context, region_back_attn = self.region_back_attn(
            tokens, hyperedge_tokens, hyperedge_tokens, return_attention_scores=True, training=training
        )
        composed_states = tokens + region_context
        routing_logits = tf.squeeze(self.router(composed_states, training=training), axis=-1)
        
        if region_mask is not None:
            routing_logits = tf.where(region_mask <= 0, tf.fill(tf.shape(routing_logits), -1e9), routing_logits)
        routing_weights = safe_softmax(routing_logits, axis=1)
        
        if region_mask is not None:
            routing_weights = routing_weights * region_mask
            routing_weights = routing_weights / tf.maximum(tf.reduce_sum(routing_weights, axis=1, keepdims=True), 1e-6)
            
        pooled_state = tf.reduce_sum(tf.expand_dims(routing_weights, axis=-1) * composed_states, axis=1)
        hyper_summary = tf.reduce_mean(hyperedge_tokens, axis=1)
        emotion_latent = self.latent_projector(tf.concat([pooled_state, hyper_summary], axis=-1), training=training)
        
        return {
            "composed_states": composed_states, "hyperedge_tokens": hyperedge_tokens,
            "hyperedge_attn": hyperedge_attn, "region_back_attn": region_back_attn,
            "routing_logits": routing_logits, "routing_weights": routing_weights,
            "emotion_latent": self.latent_norm(emotion_latent)
        }

class SemanticCompositionalProgramBank(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int, state_dim: int):
        super().__init__()
        self.programs = self.add_weight(
            name='programs', shape=(num_classes, programs_per_class, num_regions, state_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True
        )
        self.topology_logits = self.add_weight(
            name='topology_logits', shape=(num_classes, programs_per_class, num_regions, num_regions),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True
        )
    def call(self, training=False):
        return self.programs, tf.nn.sigmoid(self.topology_logits)

class SemanticProgramExecutor(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int, state_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.program_summary_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim), tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Activation('gelu')
        ])
        self.sim_weights = self.add_weight(
            name='sim_weights', shape=(1, num_classes, 1, 3),
            initializer=tf.keras.initializers.Constant([1.0, 0.5, 0.25]), trainable=True
        )

    def call(self, semantic_states, cross_region_tokens, program_bank, program_topology, region_mask=None, interaction_gates=None, routing_weights=None, training=False):
        state_norm = tf.math.l2_normalize(semantic_states, axis=-1)
        program_norm = tf.math.l2_normalize(program_bank, axis=-1)
        
        region_sims = tf.einsum("brd,cmrd->bcmr", state_norm, program_norm)
        if routing_weights is not None:
            region_sim = tf.reduce_sum(region_sims * tf.expand_dims(tf.expand_dims(routing_weights, axis=1), axis=1), axis=-1)
        elif region_mask is not None:
            valid_mask = tf.expand_dims(tf.expand_dims(region_mask, axis=1), axis=1)
            region_sim = tf.reduce_sum(region_sims * valid_mask, axis=-1) / tf.maximum(tf.reduce_sum(valid_mask, axis=-1), 1.0)
        else:
            region_sim = tf.reduce_mean(region_sims, axis=-1)
            
        if interaction_gates is not None:
            observed_topology = tf.expand_dims(tf.expand_dims(interaction_gates, axis=1), axis=1)
            topology_mse = tf.square(observed_topology - tf.expand_dims(program_topology, axis=0))
            if region_mask is not None:
                pair_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(region_mask, axis=-1) * tf.expand_dims(region_mask, axis=-2), axis=1), axis=1)
                topology_sim = 1.0 - (tf.reduce_sum(topology_mse * pair_mask, axis=[-1, -2]) / tf.maximum(tf.reduce_sum(pair_mask, axis=[-1, -2]), 1.0))
            else:
                topology_sim = 1.0 - tf.reduce_mean(topology_mse, axis=[-1, -2])
        else:
            topology_sim = tf.ones_like(region_sim)
            
        composition_summary = self.program_summary_proj(tf.reduce_mean(cross_region_tokens, axis=1), training=training)
        program_summary = self.program_summary_proj(tf.reduce_mean(program_bank, axis=2), training=training)
        composition_sim = tf.einsum("bd,cmd->bcm", tf.math.l2_normalize(composition_summary, axis=-1), tf.math.l2_normalize(program_summary, axis=-1))
        
        w = tf.nn.softplus(self.sim_weights)
        total_sim = w[..., 0] * region_sim + w[..., 1] * topology_sim + w[..., 2] * composition_sim
        
        region_score, topology_score, composition_score = region_sim / self.temperature, topology_sim / self.temperature, composition_sim / self.temperature
        compatibility = tf.clip_by_value(total_sim / self.temperature, -50.0, 50.0)
        program_attention = safe_softmax(compatibility, axis=-1)
        class_scores = tf.reduce_logsumexp(compatibility, axis=-1)
        program_tokens = tf.einsum("bcm,cmd->bcd", program_attention, program_summary)
        
        routing_entropy = -tf.reduce_sum(tf.maximum(routing_weights, 1e-6) * tf.math.log(tf.maximum(routing_weights, 1e-6)), axis=-1) if routing_weights is not None else tf.zeros(tf.shape(semantic_states)[0])
        
        return {
            "program_scores": class_scores, "program_attention": program_attention,
            "program_tokens": program_tokens, "compatibility": compatibility,
            "region_score": region_score, "topology_score": topology_score,
            "composition_score": composition_score, "routing_entropy": routing_entropy
        }

class SemanticEmotionClassifier(tf.keras.layers.Layer):
    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim), tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(num_classes)
        ])
    def call(self, x, training=False):
        return self.net(x, training=training)

class SemanticROIGraphFER(tf.keras.Model):
    def __init__(self, config: SemanticRoiGraphConfig):
        super().__init__()
        self.config = config
        self.backbone = ResNet50Backbone(feature_dim=config.feature_dim, use_pretrained=config.use_pretrained)
        self.roi_align = SemanticRoiAlign(roi_grid=config.roi_grid, bbox_input_size=config.bbox_input_size, feature_out_size=config.backbone_out_size)
        self.micro_reasoner = MicroGraphReasoner(dim=config.feature_dim, num_nodes=config.roi_grid * config.roi_grid, layers=config.micro_layers, heads=config.attn_heads, dropout=config.dropout)
        self.semantic_state_encoder = SemanticStateEncoder(input_dim=config.feature_dim, state_dim=config.semantic_state_dim, hidden_dim=max(config.feature_dim // 2, config.semantic_state_dim * 2), dropout=config.dropout)
        self.semantic_interaction_block = SemanticInteractionBlock(state_dim=config.semantic_state_dim, hidden_dim=max(config.semantic_state_dim * 2, 32), dropout=config.dropout, dropedge_rate=0.5)
        self.micro_motif_bank = MicroSemanticMotifBank(num_regions=config.num_regions, motifs_per_region=config.micro_motifs_per_region, state_dim=config.semantic_state_dim)
        self.micro_motif_matcher = MicroSemanticMotifMatcher(num_regions=config.num_regions, motifs_per_region=config.micro_motifs_per_region, state_dim=config.semantic_state_dim, temperature=config.relation_temperature)
        self.semantic_compositional_reasoner = SemanticHypergraphReasoner(state_dim=config.semantic_state_dim, latent_dim=config.semantic_latent_dim, hyperedge_count=config.hyperedge_count, attn_heads=config.semantic_attn_heads, router_hidden_dim=config.router_hidden_dim, dropout=config.dropout)
        self.cross_region_composition_graph = CrossRegionCompositionGraph(state_dim=config.semantic_state_dim, num_compositions=config.cross_region_compositions, attn_heads=config.semantic_attn_heads, hidden_dim=max(config.semantic_state_dim * 2, 32), dropout=config.dropout)
        self.semantic_program_bank = SemanticCompositionalProgramBank(num_classes=config.num_classes, programs_per_class=config.macro_motifs_per_class, num_regions=config.num_regions, state_dim=config.semantic_state_dim)
        self.semantic_program_executor = SemanticProgramExecutor(num_classes=config.num_classes, programs_per_class=config.macro_motifs_per_class, num_regions=config.num_regions, state_dim=config.semantic_state_dim, temperature=config.relation_temperature)
        self.semantic_classifier = SemanticEmotionClassifier(latent_dim=config.semantic_latent_dim, num_classes=config.num_classes, dropout=config.dropout)
        self.global_context = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(config.semantic_latent_dim),
            tf.keras.layers.Activation('gelu'), tf.keras.layers.Dropout(config.dropout)
        ])
        self.global_fusion = tf.keras.Sequential([
            tf.keras.layers.Dense(config.semantic_latent_dim), tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Activation('gelu')
        ])
        self.semantic_structure_gate = self.add_weight(
            name='semantic_structure_gate', shape=(config.num_classes,),
            initializer=tf.keras.initializers.Constant(-0.5), trainable=True
        )
        self.missing_region_token = self.add_weight(
            name='missing_region_token', shape=(config.feature_dim,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True
        )
        self.region_reliability_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(config.feature_dim // 2), tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1), tf.keras.layers.Activation('sigmoid')
        ])
        self.region_dropout_prob = float(config.region_dropout_prob)

    def _canonical_bboxes(self, batch_size: int):
        boxes = self.roi_align._canonical_region_boxes()
        return tf.broadcast_to(tf.expand_dims(boxes, axis=0), [batch_size, tf.shape(boxes)[0], tf.shape(boxes)[1]])

    def _prepare_regions(self, bboxes, batch_size):
        if bboxes is None:
            repaired = self._canonical_bboxes(batch_size)
            region_mask = tf.ones([batch_size, self.config.num_regions], dtype=tf.float32)
            return repaired, region_mask, tf.fill(tf.shape(region_mask), 0.95), tf.zeros([0, 2], dtype=tf.int32)
            
        repaired = self.roi_align.validate_bboxes(bboxes)
        finite_mask = tf.reduce_all(tf.math.is_finite(bboxes), axis=-1)
        x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        size_mask = tf.logical_and((x2 - x1) >= 2.0, (y2 - y1) >= 2.0)
        order_mask = tf.logical_and(x2 > x1, y2 > y1)
        region_mask = tf.cast(tf.logical_and(tf.logical_and(finite_mask, size_mask), order_mask), tf.float32)
        
        repaired = tf.where(tf.expand_dims(region_mask > 0, axis=-1), repaired, self._canonical_bboxes(batch_size))
        
        width = tf.maximum(repaired[..., 2] - repaired[..., 0], 1.0)
        height = tf.maximum(repaired[..., 3] - repaired[..., 1], 1.0)
        area = (width * height) / float(self.config.bbox_input_size * self.config.bbox_input_size)
        area_conf = tf.clip_by_value(area, 0.0, 1.0)
        region_confidence = tf.where(region_mask > 0, 0.5 + 0.5 * area_conf, tf.fill(tf.shape(area_conf), 0.05))
        
        invalid_indices = tf.where(region_mask == 0)
        return repaired, region_mask, region_confidence, invalid_indices

    def call(self, image, bboxes=None, region_mask=None, region_confidence=None, training=False):
        batch_size = tf.shape(image)[0]
        feature_map = self.backbone(image, training=training)
        comp_bboxes, comp_mask, comp_conf, invalid_indices = self._prepare_regions(bboxes, batch_size)
        
        bboxes = comp_bboxes if bboxes is None else bboxes
        region_mask = comp_mask if region_mask is None else tf.cast(region_mask, tf.float32)
        region_confidence = comp_conf if region_confidence is None else tf.cast(region_confidence, tf.float32)
        
        if training and self.region_dropout_prob > 0:
            drop_mask = tf.cast(tf.random.uniform([batch_size, self.config.num_regions]) > self.region_dropout_prob, tf.float32)
            region_mask *= drop_mask
            region_confidence *= drop_mask
            
        roi_nodes = self.roi_align(feature_map, bboxes, training=training)
        micro_node_features, region_embeddings = self.micro_reasoner(roi_nodes, training=training)
        
        missing_token = tf.broadcast_to(tf.reshape(self.missing_region_token, [1, 1, -1]), tf.shape(region_embeddings))
        region_embeddings = tf.where(tf.expand_dims(region_mask > 0, axis=-1), region_embeddings, missing_token)
        
        predicted_confidence = tf.squeeze(self.region_reliability_predictor(region_embeddings, training=training), axis=-1)
        region_confidence = tf.clip_by_value(0.5 * region_confidence + 0.5 * predicted_confidence, 0.0, 1.0) * region_mask
        
        semantic_state_tokens = self.semantic_state_encoder(region_embeddings, training=training)
        micro_motif_bank = self.micro_motif_bank(training=training)
        micro_motif_attention, semantic_motif_tokens = self.micro_motif_matcher(semantic_state_tokens, micro_motif_bank, training=training)
        
        interaction_states, semantic_interaction_tensor, semantic_interaction_gates = self.semantic_interaction_block(
            semantic_motif_tokens, region_mask=region_mask, training=training
        )
        
        cross_region_outputs = self.cross_region_composition_graph(
            interaction_states, region_mask=region_mask, region_confidence=region_confidence, training=training
        )
        
        hypergraph_input = interaction_states + tf.broadcast_to(tf.reduce_mean(cross_region_outputs["cross_region_tokens"], axis=1, keepdims=True), tf.shape(interaction_states))
        compositional_outputs = self.semantic_compositional_reasoner(
            hypergraph_input, region_mask=region_mask, region_confidence=region_confidence, training=training
        )
        
        semantic_program_bank, semantic_program_topology = self.semantic_program_bank(training=training)
        semantic_program_outputs = self.semantic_program_executor(
            compositional_outputs["composed_states"], cross_region_outputs["cross_region_tokens"],
            semantic_program_bank, semantic_program_topology, region_mask=region_mask,
            interaction_gates=semantic_interaction_gates, routing_weights=compositional_outputs["routing_weights"], training=training
        )
        
        global_semantic_context = self.global_context(feature_map, training=training)
        fused_latent = self.global_fusion(tf.concat([compositional_outputs["emotion_latent"], global_semantic_context], axis=-1), training=training)
        logits_fused = self.semantic_classifier(fused_latent, training=training)
        
        structure_gate = tf.reshape(tf.nn.sigmoid(self.semantic_structure_gate), [1, -1])
        logits = (1 - structure_gate) * logits_fused + structure_gate * semantic_program_outputs["program_scores"]
        
        return {
            "logits": logits, "logits_motif": semantic_program_outputs["program_scores"],
            "logits_fused": logits_fused, "structure_gate": structure_gate
        }
