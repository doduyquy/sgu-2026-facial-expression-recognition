"""
semantic_roi_graph_tf.py — TensorFlow/Keras port of semantic_roi_graph.py (PyTorch).

Translated directly from the PyTorch source, class by class, function by function.
NHWC format is used throughout (TF standard), converted from NCHW where needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_softmax(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Numerically stable softmax — mirrors PyTorch safe_softmax."""
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    # Replace any all-inf / all-nan slices with zeros
    all_invalid = tf.cast(tf.reduce_all(tf.math.is_inf(x_shifted), axis=axis, keepdims=True), tf.bool)
    x_shifted = tf.where(all_invalid, tf.zeros_like(x_shifted), x_shifted)
    return tf.nn.softmax(x_shifted, axis=axis)


DEFAULT_SEMANTIC_REGIONS = (
    "forehead", "left_eyebrow", "right_eyebrow", "glabella",
    "left_eye", "right_eye", "nose", "left_mouth_corner", "right_mouth_corner",
)

# Canonical bounding boxes (pixel coords in 48x48 space) — same as PyTorch
_CANONICAL_BOXES_48 = np.array([
    [ 8,  0, 40, 10],   # forehead
    [ 5,  8, 18, 18],   # left_eyebrow
    [30,  8, 43, 18],   # right_eyebrow
    [18, 12, 30, 22],   # glabella
    [ 6, 16, 20, 30],   # left_eye
    [28, 16, 42, 30],   # right_eye
    [14, 20, 34, 38],   # nose
    [ 8, 30, 22, 43],   # left_mouth_corner
    [26, 30, 40, 43],   # right_mouth_corner
], dtype=np.float32)


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
    fusion_scale: float = 0.25
    region_dropout_prob: float = 0.0
    program_dim: int = 128
    programs_per_class: int = 4
    backbone_type: str = 'hrnet_w18'


class ResidualBasicBlock(tf.keras.layers.Layer):
    """BasicBlock with residual connection for HRNet."""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, x, training=False):
        residual = x
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        return tf.nn.relu(out + residual)


class BottleneckBlock(tf.keras.layers.Layer):
    """Bottleneck for HRNet Stage1."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        mid = out_channels // 4
        self.conv1 = tf.keras.layers.Conv2D(mid, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(mid, 3, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(out_channels, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        # Shortcut
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_channels, 1, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
    
    def call(self, x, training=False):
        residual = x if self.shortcut is None else self.shortcut(x, training=training)
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)
        return tf.nn.relu(out + residual)


class HRNetBackboneTF(tf.keras.layers.Layer):
    """HRNet-W18 backbone implemented in TF/Keras.
    Mirrors PyTorch HRNetBackbone using timm hrnet_w18.
    
    Architecture:
    - Stem: 2x Conv3x3 (stride=1) -> 64ch
    - Stage1: 4x Bottleneck -> 256ch
    - Stage2: 2 branches [W=18, W=36], 1 BasicBlock module each
    - Stage3: 3 branches [W=18, W=36, W=72], 4 BasicBlock modules each
    - Fusion: Upsample all to H/4 resolution, concat, project to feature_dim
    Output: (B, H//4, W//4, feature_dim) -- for 48x48 input -> (B, 12, 12, 256)
    """
    def __init__(self, feature_dim=256, use_pretrained=False, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        # Branch widths
        W = [18, 36, 72]  # W18 config
        
        # Stem
        self.stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ], name='stem')
        
        # Stage1: 4 bottlenecks (64->256)
        self.layer1 = self._make_layer(64, 256, n_blocks=4, name='layer1')
        
        # Transition 1: 256 -> [W[0], W[1]]
        self.trans1_0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(W[0], 3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()
        ])
        self.trans1_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(W[1], 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()
        ])
        
        # Stage2: 2 branches
        self.stage2_b0 = self._make_basic_module(W[0], n_blocks=2, name='s2_b0')
        self.stage2_b1 = self._make_basic_module(W[1], n_blocks=2, name='s2_b1')
        # Stage2 fuse: b1->b0 (upsample), b0->b1 (downsample)
        self.fuse2_1to0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(W[0], 1, use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.fuse2_0to1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(W[1], 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        
        # Transition 2: add 3rd branch
        self.trans2_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(W[2], 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()
        ])
        
        # Stage3: 3 branches, 4 blocks each
        self.stage3_b0 = self._make_basic_module(W[0], n_blocks=4, name='s3_b0')
        self.stage3_b1 = self._make_basic_module(W[1], n_blocks=4, name='s3_b1')
        self.stage3_b2 = self._make_basic_module(W[2], n_blocks=4, name='s3_b2')
        # Stage3 fuse layers (3x3 matrix of fusions)
        total_ch = W[0] + W[1] + W[2]  # 18+36+72=126
        
        # Final projection
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Conv2D(feature_dim, 1, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ], name='proj')
        
        self.W = W

    def _basic_block(self, channels, name=None):
        """Single BasicBlock: Conv-BN-ReLU-Conv-BN + residual"""
        return ResidualBasicBlock(channels, name=name)

    def _make_basic_module(self, channels, n_blocks, name=None):
        """Stack of BasicBlocks"""
        layers = [ResidualBasicBlock(channels) for _ in range(n_blocks)]
        return tf.keras.Sequential(layers, name=name)

    def _make_layer(self, in_ch, out_ch, n_blocks, name=None):
        """Bottleneck layer"""
        layers = [BottleneckBlock(in_ch, out_ch)]
        for _ in range(n_blocks - 1):
            layers.append(BottleneckBlock(out_ch, out_ch))
        return tf.keras.Sequential(layers, name=name)

    def call(self, x, training=False):
        # x: (B, H, W, 1)
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)  # grayscale to 3ch
        x = tf.cast(x, tf.float32)
        
        # Stem
        x = self.stem(x, training=training)  # (B, H, W, 64)
        
        # Stage1
        x = self.layer1(x, training=training)  # (B, H, W, 256)
        
        # Transition 1
        b0 = self.trans1_0(x, training=training)   # (B, H, W, 18)
        b1 = self.trans1_1(x, training=training)   # (B, H/2, W/2, 36)
        
        # Stage2
        b0 = self.stage2_b0(b0, training=training)
        b1 = self.stage2_b1(b1, training=training)
        
        # Stage2 fusion
        h0, w0 = tf.shape(b0)[1], tf.shape(b0)[2]
        b1_resized = tf.cast(tf.image.resize(b1, [h0, w0]), b0.dtype)
        b0_new = tf.nn.relu(b0 + self.fuse2_1to0(b1_resized, training=training))
        b1_new = tf.nn.relu(b1 + self.fuse2_0to1(b0, training=training))
        b0, b1 = b0_new, b1_new
        
        # Transition 2: create branch 2
        b2 = self.trans2_2(b1, training=training)  # (B, H/4, W/4, 72)
        
        # Stage3
        b0 = self.stage3_b0(b0, training=training)
        b1 = self.stage3_b1(b1, training=training)
        b2 = self.stage3_b2(b2, training=training)
        
        # Final fusion: upsample b1, b2 to b0's resolution, concat
        h0, w0 = tf.shape(b0)[1], tf.shape(b0)[2]
        b1_up = tf.cast(tf.image.resize(b1, [h0, w0]), b0.dtype)
        b2_up = tf.cast(tf.image.resize(b2, [h0, w0]), b0.dtype)
        fused = tf.concat([b0, b1_up, b2_up], axis=-1)  # (B, H, W, 18+36+72=126)
        
        # Project to feature_dim
        out = self.proj(fused, training=training)  # (B, H, W, feature_dim)
        return out


# ---------------------------------------------------------------------------
# SemanticBackbone — mirrors PyTorch exactly
# (stem + layer1 + layer2 + layer3 of ResNet50, conv1.stride=(1,1), maxpool=Identity)
# ---------------------------------------------------------------------------

class SemanticBackbone(tf.keras.layers.Layer):
    """ResNet50 backbone with high-resolution features.

    PyTorch equivalent:
        resnet.conv1.stride = (1,1)
        resnet.maxpool = nn.Identity()
        uses stem + layer1 + layer2 + layer3
        projects 1024ch -> feature_dim
    """

    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)
        weights = "imagenet" if use_pretrained else None

        # Build ResNet50V2 (closest Keras equivalent to torchvision ResNet50)
        base = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=weights,
            input_shape=(48, 48, 3),
        )

        # Slice to conv4_block6_out (=layer3 end in PyTorch, 1024 channels)
        base_sliced = tf.keras.Model(
            inputs=base.input,
            outputs=base.get_layer("conv4_block6_out").output,
        )

        # Remove early downsampling: conv1 stride -> (1,1), maxpool -> identity
        def clone_fn(layer):
            if layer.name == "conv1_conv":
                cfg = layer.get_config()
                cfg["strides"] = (1, 1)
                return tf.keras.layers.Conv2D.from_config(cfg)
            if layer.name == "pool1_pad":
                cfg = layer.get_config()
                cfg["padding"] = ((0, 0), (0, 0))
                return tf.keras.layers.ZeroPadding2D.from_config(cfg)
            if layer.name == "pool1_pool":
                cfg = layer.get_config()
                cfg["strides"] = (1, 1)
                cfg["pool_size"] = (1, 1)
                cfg["padding"] = "same"
                return tf.keras.layers.MaxPooling2D.from_config(cfg)
            return layer

        self.feature_extractor = tf.keras.models.clone_model(
            base_sliced, clone_function=clone_fn
        )
        if weights is not None:
            self.feature_extractor.set_weights(base_sliced.get_weights())

        # Ensure all layers trainable (clone_model can freeze BN layers)
        for layer in self.feature_extractor.layers:
            layer.trainable = True

        # Projection: 1024 -> feature_dim  (mirrors PyTorch proj)
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Conv2D(feature_dim, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.SpatialDropout2D(0.1),
        ])
        self.out_channels = feature_dim

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: (B, H, W, 1) or (B, H, W, 3) NHWC
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        x = tf.cast(x, tf.float32)
        x = self.feature_extractor(x, training=training)  # (B, H', W', 1024)
        return self.proj(x, training=training)              # (B, H', W', feature_dim)


# ---------------------------------------------------------------------------
# SemanticRoiAlign — mirrors PyTorch roi_align behavior
# ---------------------------------------------------------------------------

class SemanticRoiAlign(tf.keras.layers.Layer):
    """ROIAlign via tf.image.crop_and_resize (mirrors PyTorch roi_align)."""

    def __init__(
        self,
        roi_grid: int = 4,
        bbox_input_size: int = 48,
        feature_out_size: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.roi_grid = int(roi_grid)
        self.bbox_input_size = int(bbox_input_size)
        self.feature_out_size = int(feature_out_size)

    def _canonical_boxes(self, batch_size: int) -> tf.Tensor:
        """Return canonical 9-region boxes for a batch, in [0,1] normalized coords."""
        boxes = _CANONICAL_BOXES_48.copy() * (float(self.bbox_input_size) / 48.0)
        # normalize to [0, 1]
        s = float(self.bbox_input_size - 1)
        boxes_norm = boxes / s  # (9, 4) [x1,y1,x2,y2] -> [0,1]
        # crop_and_resize expects [y1,x1,y2,x2]
        boxes_yx = np.stack([boxes_norm[:, 1], boxes_norm[:, 0],
                              boxes_norm[:, 3], boxes_norm[:, 2]], axis=-1)
        boxes_tf = tf.constant(boxes_yx[np.newaxis], dtype=tf.float32)
        return tf.tile(boxes_tf, [batch_size, 1, 1])  # (B, 9, 4)

    def _validate_bboxes(self, bboxes: tf.Tensor) -> tf.Tensor:
        """Clamp and repair bboxes — mirrors PyTorch validate_bboxes."""
        bboxes = tf.cast(bboxes, tf.float32)
        size = float(self.bbox_input_size - 1)

        x1 = tf.clip_by_value(bboxes[..., 0], 0.0, size)
        y1 = tf.clip_by_value(bboxes[..., 1], 0.0, size)
        x2 = tf.clip_by_value(bboxes[..., 2], 0.0, size)
        y2 = tf.clip_by_value(bboxes[..., 3], 0.0, size)

        # Ensure x2 > x1, y2 > y1 by at least 2 px
        x1_f = tf.minimum(x1, x2)
        x2_f = tf.maximum(x1, x2)
        y1_f = tf.minimum(y1, y2)
        y2_f = tf.maximum(y1, y2)
        x2_f = tf.maximum(x2_f, x1_f + 2.0)
        y2_f = tf.maximum(y2_f, y1_f + 2.0)
        x2_f = tf.minimum(x2_f, size)
        y2_f = tf.minimum(y2_f, size)

        return tf.stack([x1_f, y1_f, x2_f, y2_f], axis=-1)

    def call(self, feature_map: tf.Tensor, bboxes: tf.Tensor) -> tf.Tensor:
        """
        feature_map: (B, H, W, C) NHWC
        bboxes: (B, R, 4) in pixel coords [x1,y1,x2,y2]
        returns: (B, R, G*G, C)
        """
        b = tf.shape(feature_map)[0]
        b_static = feature_map.shape[0]
        num_regions = bboxes.shape[1] or tf.shape(bboxes)[1]

        bboxes = self._validate_bboxes(bboxes)

        # Normalize to [0,1] for crop_and_resize
        s = float(self.bbox_input_size - 1)
        x1 = bboxes[..., 0] / s
        y1 = bboxes[..., 1] / s
        x2 = bboxes[..., 2] / s
        y2 = bboxes[..., 3] / s
        # crop_and_resize wants [y1, x1, y2, x2]
        boxes_yx = tf.stack([y1, x1, y2, x2], axis=-1)  # (B, R, 4)

        # Flatten to (B*R, 4) + box_ind (B*R,)
        boxes_flat = tf.reshape(boxes_yx, [-1, 4])  # (B*R, 4)
        batch_range = tf.range(b)
        box_ind = tf.repeat(batch_range, num_regions)  # (B*R,)

        # Crop and resize: (B*R, G, G, C)
        crops = tf.image.crop_and_resize(
            feature_map,
            boxes_flat,
            box_ind,
            crop_size=[self.roi_grid, self.roi_grid],
            method="bilinear",
        )
        C = feature_map.shape[-1]
        # Reshape to (B, R, G*G, C)
        crops = tf.reshape(crops, [b, num_regions, self.roi_grid * self.roi_grid, C])
        return crops


# ---------------------------------------------------------------------------
# GATBlock — mirrors PyTorch GATBlock
# ---------------------------------------------------------------------------

class GATBlock(tf.keras.layers.Layer):
    """Multi-head graph attention — direct translation of PyTorch GATBlock."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        num_nodes: Optional[int] = None,
        use_locality: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_nodes = num_nodes

        self.q_proj = tf.keras.layers.Dense(dim)
        self.k_proj = tf.keras.layers.Dense(dim)
        self.v_proj = tf.keras.layers.Dense(dim)
        self.out_proj = tf.keras.layers.Dense(dim)
        self.attn_drop = tf.keras.layers.Dropout(dropout)

        # Learnable adjacency bias
        self.adj_bias = None
        if num_nodes is not None:
            self.adj_bias = self.add_weight(
                name="adj_bias",
                shape=(1, 1, num_nodes, num_nodes),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                trainable=True,
            )

        # Locality bias (fixed)
        self.locality_bias_val = None
        if use_locality and num_nodes is not None:
            side = int(num_nodes ** 0.5)
            if side * side == num_nodes:
                coords_1d = np.arange(side, dtype=np.float32)
                gy, gx = np.meshgrid(coords_1d, coords_1d, indexing="ij")
                coords = np.stack([gy.ravel(), gx.ravel()], axis=-1)
            else:
                coords = np.arange(num_nodes, dtype=np.float32)[:, None]
            dist = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
            dist = dist / max(dist.max(), 1e-6)
            self.locality_bias_val = tf.constant(
                -dist[np.newaxis, np.newaxis], dtype=tf.float32
            )  # (1,1,N,N)

    def call(self, x: tf.Tensor, training: bool = False,
             edge_prior: Optional[tf.Tensor] = None,
             attn_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        # x: (B, N, D)
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]

        def _proj_heads(proj, inp):
            out = proj(inp)  # (B,N,D)
            out = tf.reshape(out, [b, n, self.heads, self.head_dim])
            return tf.transpose(out, [0, 2, 1, 3])  # (B,H,N,head_dim)

        q = _proj_heads(self.q_proj, x)
        k = _proj_heads(self.k_proj, x)
        v = _proj_heads(self.v_proj, x)

        # Attention scores: (B,H,N,N)
        scale = float(self.head_dim) ** -0.5
        attn = tf.einsum("bhid,bhjd->bhij", q, k) * scale

        if self.adj_bias is not None:
            attn = attn + self.adj_bias
        if self.locality_bias_val is not None:
            attn = attn + tf.cast(self.locality_bias_val, attn.dtype)
        if edge_prior is not None:
            edge_prior = tf.cast(tf.maximum(edge_prior, 1e-6), attn.dtype)
            log_ep = tf.math.log(edge_prior)
            if len(log_ep.shape) == 2:
                log_ep = log_ep[tf.newaxis, tf.newaxis]
            elif len(log_ep.shape) == 3:
                log_ep = log_ep[:, tf.newaxis]
            attn = attn + log_ep
        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask, attn.dtype)
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask[:, tf.newaxis, tf.newaxis, :]
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask[:, tf.newaxis, :, :]
            attn = attn + (1.0 - attn_mask) * (-1e9)

        attn = safe_softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        # (B,H,N,N) x (B,H,N,head_dim) -> (B,H,N,head_dim)
        out = tf.einsum("bhij,bhjd->bhid", attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])        # (B,N,H,head_dim)
        out = tf.reshape(out, [b, n, self.dim])       # (B,N,D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# GatedPooling — mirrors PyTorch GatedPooling
# ---------------------------------------------------------------------------

class GatedPooling(tf.keras.layers.Layer):
    """Attention-based gated pooling."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gate = tf.keras.layers.Dense(1)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: (B, N, D)
        weights = tf.nn.sigmoid(self.gate(x))        # (B,N,1)
        weighted = x * weights
        pooled = tf.reduce_sum(weighted, axis=1) / (tf.reduce_sum(weights, axis=1) + 1e-6)
        return pooled


# ---------------------------------------------------------------------------
# MicroGraphReasoner — mirrors PyTorch MicroGraphReasoner
# ---------------------------------------------------------------------------

class MicroGraphReasoner(tf.keras.layers.Layer):
    """Intra-region reasoning with GAT — direct translation."""

    def __init__(self, dim: int, num_nodes: int, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gat_layers = [
            GATBlock(dim, heads=heads, dropout=dropout, num_nodes=num_nodes)
            for _ in range(layers)
        ]
        self.norms = [tf.keras.layers.LayerNormalization() for _ in range(layers)]
        self.pool = GatedPooling(dim)

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        # x: (B, R, N, D)
        b = tf.shape(x)[0]
        r = x.shape[1]
        n = x.shape[2]
        d = x.shape[3]
        x_flat = tf.reshape(x, [b * r, n, d])  # (B*R, N, D)

        for layer, norm in zip(self.gat_layers, self.norms):
            x_flat = x_flat + layer(norm(x_flat), training=training)

        pooled = self.pool(x_flat, training=training)  # (B*R, D)
        pooled = tf.reshape(pooled, [b, r, d])          # (B, R, D)
        x_out = tf.reshape(x_flat, [b, r, n, d])        # (B, R, N, D)
        return x_out, pooled


# ---------------------------------------------------------------------------
# SemanticStateEncoder — mirrors PyTorch SemanticStateEncoder
# ---------------------------------------------------------------------------

class SemanticStateEncoder(tf.keras.layers.Layer):
    """Project region embeddings -> semantic state space."""

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
        self.gate = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.Activation("sigmoid"),
        ])
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, region_embeddings: tf.Tensor, training: bool = False) -> tf.Tensor:
        raw_state = self.proj(region_embeddings, training=training)
        gate = self.gate(region_embeddings, training=training)
        return self.norm(raw_state * gate)


# ---------------------------------------------------------------------------
# MicroSemanticMotifBank — mirrors PyTorch MicroSemanticMotifBank
# ---------------------------------------------------------------------------

class MicroSemanticMotifBank(tf.keras.layers.Layer):
    """Learnable local semantic motifs."""

    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.state_dim = state_dim

    def build(self, input_shape):
        self.motifs = self.add_weight(
            name="motifs",
            shape=(self.num_regions, self.motifs_per_region, self.state_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs=None, training: bool = False) -> tf.Tensor:
        return self.motifs


# ---------------------------------------------------------------------------
# MicroSemanticMotifMatcher — mirrors PyTorch MicroSemanticMotifMatcher
# ---------------------------------------------------------------------------

class MicroSemanticMotifMatcher(tf.keras.layers.Layer):
    """Match semantic region states to local motifs."""

    def __init__(self, num_regions: int, motifs_per_region: int,
                 state_dim: int, temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(temperature)
        self.token_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
        ])

    def call(self, semantic_states: tf.Tensor, motif_bank: tf.Tensor,
             training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        # semantic_states: (B, R, D), motif_bank: (R, K, D)
        state_norm = tf.nn.l2_normalize(semantic_states, axis=-1)
        bank_norm = tf.nn.l2_normalize(motif_bank, axis=-1)

        # sim: (B, R, K)
        sim = tf.einsum("brd,rks->brk", state_norm, bank_norm) / self.temperature
        attn = safe_softmax(sim, axis=-1)

        # tokens: (B, R, D)
        tokens = tf.einsum("brk,rks->brs", attn, motif_bank)
        tokens = self.token_proj(tokens, training=training)
        semantic_tokens = semantic_states + tokens
        return attn, semantic_tokens


# ---------------------------------------------------------------------------
# SemanticInteractionBlock — mirrors PyTorch SemanticInteractionBlock
# ---------------------------------------------------------------------------

class SemanticInteractionBlock(tf.keras.layers.Layer):
    """Pairwise region interaction — direct translation."""

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

    def call(self, semantic_states: tf.Tensor, training: bool = False,
             region_mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # semantic_states: (B, R, S)
        b = tf.shape(semantic_states)[0]
        r = semantic_states.shape[1]
        s = semantic_states.shape[2]

        left = tf.tile(semantic_states[:, :, tf.newaxis, :], [1, 1, r, 1])   # (B,R,R,S)
        right = tf.tile(semantic_states[:, tf.newaxis, :, :], [1, r, 1, 1])  # (B,R,R,S)
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)  # (B,R,R,4S)

        gates = self.edge_gate(pair_input, training=training)[..., 0] + 0.1  # (B,R,R)

        # DropEdge during training
        if self.dropedge_rate > 0.0 and training:
            keep_prob = 1.0 - self.dropedge_rate
            noise = tf.random.uniform(tf.shape(gates))
            gates = tf.where(tf.cast(noise < keep_prob, tf.bool), gates / keep_prob, tf.zeros_like(gates))

        if region_mask is not None:
            pair_mask = tf.cast(region_mask[:, :, tf.newaxis], gates.dtype) * \
                        tf.cast(region_mask[:, tf.newaxis, :], gates.dtype)
            gates = gates * pair_mask

        messages = self.edge_message(pair_input, training=training)  # (B,R,R,S)
        interaction_tensor = gates[..., tf.newaxis] * messages        # (B,R,R,S)

        gate_sum = tf.reduce_sum(gates, axis=2, keepdims=True) + 1e-6
        interaction_summary = tf.reduce_sum(interaction_tensor, axis=2) / gate_sum  # (B,R,S)
        updated_states = self.norm(semantic_states + interaction_summary)
        return updated_states, interaction_tensor, gates


# ---------------------------------------------------------------------------
# CrossRegionCompositionGraph — mirrors PyTorch CrossRegionCompositionGraph
# ---------------------------------------------------------------------------

class CrossRegionCompositionGraph(tf.keras.layers.Layer):
    """Higher-order cross-region compositions."""

    def __init__(self, state_dim: int, num_compositions: int, attn_heads: int = 4,
                 hidden_dim: Optional[int] = None, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert state_dim % attn_heads == 0
        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        self.num_compositions = num_compositions
        self.state_dim = state_dim
        self.attn_heads = attn_heads
        self.head_dim = state_dim // attn_heads

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
        # MHA: composition_queries attend to pair_sequence
        self.q_proj = tf.keras.layers.Dense(state_dim)
        self.k_proj = tf.keras.layers.Dense(state_dim)
        self.v_proj = tf.keras.layers.Dense(state_dim)
        self.out_proj = tf.keras.layers.Dense(state_dim)
        self.attn_drop = tf.keras.layers.Dropout(dropout)
        self.composition_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        self.composition_queries = self.add_weight(
            name="composition_queries",
            shape=(self.num_compositions, self.state_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def _mha(self, q, kv, training, key_mask=None):
        """Multi-head cross-attention: q attends to kv."""
        b = tf.shape(q)[0]
        qn = tf.shape(q)[1]
        kvn = tf.shape(kv)[1]

        def heads(proj, x, seq_len):
            out = proj(x)
            out = tf.reshape(out, [b, seq_len, self.attn_heads, self.head_dim])
            return tf.transpose(out, [0, 2, 1, 3])

        Q = heads(self.q_proj, q, qn)
        K = heads(self.k_proj, kv, kvn)
        V = heads(self.v_proj, kv, kvn)

        scale = float(self.head_dim) ** -0.5
        attn_w = tf.einsum("bhid,bhjd->bhij", Q, K) * scale  # (B,H,qn,kvn)
        if key_mask is not None:
            # key_mask: (B, kvn) True = ignore
            key_mask = tf.cast(key_mask, attn_w.dtype)
            attn_w = attn_w + key_mask[:, tf.newaxis, tf.newaxis, :] * (-1e9)
        attn_w = tf.nn.softmax(attn_w, axis=-1)
        attn_w = self.attn_drop(attn_w, training=training)

        out = tf.einsum("bhij,bhjd->bhid", attn_w, V)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [b, qn, self.state_dim])
        out = self.out_proj(out)
        return out, attn_w

    def call(self, semantic_states: tf.Tensor, training: bool = False,
             region_mask: Optional[tf.Tensor] = None,
             region_confidence: Optional[tf.Tensor] = None) -> Dict:
        b = tf.shape(semantic_states)[0]
        r = semantic_states.shape[1]
        d = semantic_states.shape[2]

        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * region_confidence[..., tf.newaxis]

        left = tf.tile(tokens[:, :, tf.newaxis, :], [1, 1, r, 1])
        right = tf.tile(tokens[:, tf.newaxis, :, :], [1, r, 1, 1])
        pair_input = tf.concat([left, right, left - right, left * right], axis=-1)

        pair_tokens = self.pair_encoder(pair_input, training=training)  # (B,R,R,D)
        pair_scores = self.pair_router(pair_tokens, training=training)[..., 0]  # (B,R,R)

        if region_mask is not None:
            pair_mask = tf.cast(region_mask[:, :, tf.newaxis], pair_scores.dtype) * \
                        tf.cast(region_mask[:, tf.newaxis, :], pair_scores.dtype)
            pair_scores = pair_scores + (1.0 - pair_mask) * (-1e9)

        pair_attention = safe_softmax(tf.reshape(pair_scores, [b, -1]), axis=-1)
        pair_attention = tf.reshape(pair_attention, [b, r, r])
        pair_sequence = tf.reshape(pair_tokens, [b, r * r, d])

        # key_mask for MHA
        key_padding_mask = None
        if region_mask is not None:
            pair_mask_flat = tf.reshape(
                tf.cast(region_mask[:, :, tf.newaxis], pair_scores.dtype) *
                tf.cast(region_mask[:, tf.newaxis, :], pair_scores.dtype),
                [b, r * r]
            )
            key_padding_mask = 1.0 - pair_mask_flat  # 1.0 = ignore

        comp_queries = tf.tile(self.composition_queries[tf.newaxis], [b, 1, 1])
        cross_region_tokens, composition_attn = self._mha(
            comp_queries, pair_sequence, training, key_mask=key_padding_mask
        )
        cross_region_tokens = self.composition_norm(cross_region_tokens)

        return {
            "cross_region_tokens": cross_region_tokens,
            "composition_attn": composition_attn,
            "pair_tokens": pair_tokens,
            "pair_scores": pair_scores,
            "pair_attention": pair_attention,
        }


# ---------------------------------------------------------------------------
# SemanticHypergraphReasoner — mirrors PyTorch SemanticHypergraphReasoner
# ---------------------------------------------------------------------------

class SemanticHypergraphReasoner(tf.keras.layers.Layer):
    """Hypergraph multi-region reasoning."""

    def __init__(self, state_dim: int, latent_dim: int, hyperedge_count: int,
                 attn_heads: int = 4, router_hidden_dim: int = 256,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert state_dim % attn_heads == 0
        self.hyperedge_count = hyperedge_count
        self.state_dim = state_dim
        self.attn_heads = attn_heads
        self.head_dim = state_dim // attn_heads

        # hyperedge attention: hyper_queries -> tokens
        self.he_q = tf.keras.layers.Dense(state_dim)
        self.he_k = tf.keras.layers.Dense(state_dim)
        self.he_v = tf.keras.layers.Dense(state_dim)
        self.he_out = tf.keras.layers.Dense(state_dim)
        self.he_drop = tf.keras.layers.Dropout(dropout)

        # region back-attention: tokens -> hyperedge_tokens
        self.rb_q = tf.keras.layers.Dense(state_dim)
        self.rb_k = tf.keras.layers.Dense(state_dim)
        self.rb_v = tf.keras.layers.Dense(state_dim)
        self.rb_out = tf.keras.layers.Dense(state_dim)
        self.rb_drop = tf.keras.layers.Dropout(dropout)

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
        self.hyperedge_queries = self.add_weight(
            name="hyperedge_queries",
            shape=(self.hyperedge_count, self.state_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def _mha(self, q_proj, k_proj, v_proj, out_proj, drop,
             query, kv, training, key_mask=None):
        b = tf.shape(query)[0]
        qn = tf.shape(query)[1]
        kvn = tf.shape(kv)[1]

        def heads(proj, x, n):
            out = proj(x)
            out = tf.reshape(out, [b, n, self.attn_heads, self.head_dim])
            return tf.transpose(out, [0, 2, 1, 3])

        Q = heads(q_proj, query, qn)
        K = heads(k_proj, kv, kvn)
        V = heads(v_proj, kv, kvn)

        scale = float(self.head_dim) ** -0.5
        attn_w = tf.einsum("bhid,bhjd->bhij", Q, K) * scale
        if key_mask is not None:
            attn_w = attn_w + tf.cast(key_mask, attn_w.dtype)[:, tf.newaxis, tf.newaxis, :] * (-1e9)
        attn_w = tf.nn.softmax(attn_w, axis=-1)
        attn_w = drop(attn_w, training=training)
        out = tf.einsum("bhij,bhjd->bhid", attn_w, V)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [b, qn, self.state_dim])
        return out_proj(out), attn_w

    def call(self, semantic_states: tf.Tensor, training: bool = False,
             region_mask: Optional[tf.Tensor] = None,
             region_confidence: Optional[tf.Tensor] = None) -> Dict:
        b = tf.shape(semantic_states)[0]
        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * region_confidence[..., tf.newaxis]

        key_mask = None
        if region_mask is not None:
            key_mask = 1.0 - tf.cast(region_mask, tokens.dtype)

        hyper_queries = tf.tile(self.hyperedge_queries[tf.newaxis], [b, 1, 1])
        hyperedge_tokens, hyperedge_attn = self._mha(
            self.he_q, self.he_k, self.he_v, self.he_out, self.he_drop,
            hyper_queries, tokens, training, key_mask=key_mask
        )
        region_context, region_back_attn = self._mha(
            self.rb_q, self.rb_k, self.rb_v, self.rb_out, self.rb_drop,
            tokens, hyperedge_tokens, training, key_mask=None
        )

        composed_states = tokens + region_context
        routing_logits = self.router(composed_states, training=training)[..., 0]
        if region_mask is not None:
            routing_logits = routing_logits + (1.0 - tf.cast(region_mask, routing_logits.dtype)) * (-1e9)
        routing_weights = safe_softmax(routing_logits, axis=1)
        if region_mask is not None:
            routing_weights = routing_weights * tf.cast(region_mask, routing_weights.dtype)
            routing_weights = routing_weights / (tf.reduce_sum(routing_weights, axis=1, keepdims=True) + 1e-6)

        pooled_state = tf.reduce_sum(routing_weights[..., tf.newaxis] * composed_states, axis=1)
        hyper_summary = tf.reduce_mean(hyperedge_tokens, axis=1)
        emotion_latent = self.latent_projector(
            tf.concat([pooled_state, hyper_summary], axis=-1), training=training
        )
        emotion_latent = self.latent_norm(emotion_latent)

        return {
            "composed_states": composed_states,
            "hyperedge_tokens": hyperedge_tokens,
            "hyperedge_attn": hyperedge_attn,
            "region_back_attn": region_back_attn,
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "emotion_latent": emotion_latent,
        }


# ---------------------------------------------------------------------------
# SemanticCompositionalProgramBank — mirrors PyTorch
# ---------------------------------------------------------------------------

class SemanticCompositionalProgramBank(tf.keras.layers.Layer):
    """Learnable semantic facial programs + topology."""

    def __init__(self, num_classes: int, programs_per_class: int,
                 num_regions: int, state_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim

    def build(self, input_shape):
        self.programs = self.add_weight(
            name="programs",
            shape=(self.num_classes, self.programs_per_class, self.num_regions, self.state_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.topology_logits = self.add_weight(
            name="topology_logits",
            shape=(self.num_classes, self.programs_per_class, self.num_regions, self.num_regions),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs=None, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.programs, tf.nn.sigmoid(self.topology_logits)


# ---------------------------------------------------------------------------
# SemanticProgramExecutor — mirrors PyTorch SemanticProgramExecutor
# ---------------------------------------------------------------------------

class SemanticProgramExecutor(tf.keras.layers.Layer):
    """Execute semantic facial programs against observed states."""

    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int,
                 state_dim: int, temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim
        self.temperature = float(temperature)

        self.program_summary_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(state_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
        ])

    def build(self, input_shape):
        # sim_weights: (1, num_classes, 1, 3) — adaptive structure weights
        init = np.array([[[[1.0, 0.5, 0.25]]]] * self.num_classes
                        ).reshape(1, self.num_classes, 1, 3).astype(np.float32)
        self.sim_weights = self.add_weight(
            name="sim_weights",
            shape=(1, self.num_classes, 1, 3),
            initializer=tf.keras.initializers.Constant(init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, semantic_states: tf.Tensor, cross_region_tokens: tf.Tensor,
             program_bank: tf.Tensor, program_topology: tf.Tensor,
             training: bool = False,
             region_mask: Optional[tf.Tensor] = None,
             interaction_gates: Optional[tf.Tensor] = None,
             routing_weights: Optional[tf.Tensor] = None) -> Dict:
        state_norm = tf.nn.l2_normalize(semantic_states, axis=-1)
        program_norm = tf.nn.l2_normalize(program_bank, axis=-1)

        # region_sims: (B, C, M, R)
        region_sims = tf.einsum("brd,cmrd->bcmr", state_norm, program_norm)
        if routing_weights is not None:
            region_sim = tf.reduce_sum(
                region_sims * routing_weights[:, tf.newaxis, tf.newaxis, :], axis=-1
            )
        elif region_mask is not None:
            valid_mask = region_mask[:, tf.newaxis, tf.newaxis, :]
            region_sims = region_sims * tf.cast(valid_mask, region_sims.dtype)
            region_sim = tf.reduce_sum(region_sims, axis=-1) / \
                         tf.maximum(tf.reduce_sum(tf.cast(valid_mask, tf.float32), axis=-1), 1.0)
        else:
            region_sim = tf.reduce_mean(region_sims, axis=-1)

        # topology_sim: (B, C, M)
        if interaction_gates is not None:
            obs_topo = interaction_gates[:, tf.newaxis, tf.newaxis, :, :]
            topo_mse = (obs_topo - program_topology[tf.newaxis]) ** 2
            if region_mask is not None:
                pair_mask = tf.cast(region_mask[:, :, tf.newaxis], topo_mse.dtype) * \
                            tf.cast(region_mask[:, tf.newaxis, :], topo_mse.dtype)
                pair_mask = pair_mask[:, tf.newaxis, tf.newaxis, :, :]
                topo_mse = topo_mse * pair_mask
                topology_sim = 1.0 - (tf.reduce_sum(topo_mse, axis=[-2, -1]) /
                                      tf.maximum(tf.reduce_sum(pair_mask, axis=[-2, -1]), 1.0))
            else:
                topology_sim = 1.0 - tf.reduce_mean(topo_mse, axis=[-2, -1])
        else:
            topology_sim = tf.ones_like(region_sim)

        # composition_sim: (B, C, M)
        comp_summary = tf.reduce_mean(cross_region_tokens, axis=1)  # (B, D)
        comp_summary = self.program_summary_proj(comp_summary, training=training)
        prog_mean = tf.reduce_mean(program_bank, axis=2)  # (C, M, D)
        prog_mean_flat = tf.reshape(prog_mean, [-1, self.state_dim])
        prog_summary_flat = self.program_summary_proj(prog_mean_flat, training=training)
        prog_summary = tf.reshape(prog_summary_flat, [self.num_classes, self.programs_per_class, self.state_dim])
        composition_sim = tf.einsum(
            "bd,cmd->bcm",
            tf.nn.l2_normalize(comp_summary, axis=-1),
            tf.nn.l2_normalize(prog_summary, axis=-1),
        )

        # Adaptive weighting — softplus ensures positive
        w = tf.nn.softplus(self.sim_weights)  # (1, C, 1, 3)
        total_sim = (w[..., 0] * region_sim +
                     w[..., 1] * topology_sim +
                     w[..., 2] * composition_sim)

        region_score = region_sim / self.temperature
        topology_score = topology_sim / self.temperature
        composition_score = composition_sim / self.temperature

        compatibility = tf.clip_by_value(total_sim / self.temperature, -50.0, 50.0)
        program_attention = safe_softmax(compatibility, axis=-1)
        class_scores = tf.reduce_logsumexp(compatibility, axis=-1)

        # program_tokens: (B, C, D)
        prog_summ_2d = tf.reshape(prog_summary, [self.num_classes, self.programs_per_class, -1])
        program_tokens = tf.einsum("bcm,cmd->bcd", program_attention, prog_summ_2d)

        if routing_weights is not None:
            routing_entropy = -tf.reduce_sum(
                tf.maximum(routing_weights, 1e-6) * tf.math.log(tf.maximum(routing_weights, 1e-6)),
                axis=-1
            )
        else:
            routing_entropy = tf.zeros([tf.shape(semantic_states)[0]], dtype=semantic_states.dtype)

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
# SemanticEmotionClassifier — mirrors PyTorch SemanticEmotionClassifier
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
# SemanticROIGraphFER — main model (mirrors PyTorch SemanticROIGraphFER)
# ---------------------------------------------------------------------------

class SemanticROIGraphFER(tf.keras.Model):
    """End-to-end semantic compositional facial reasoning model."""

    def __init__(self, config: SemanticRoiGraphConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.training_cfg: Dict = {}

        backbone_type = getattr(config, 'backbone_type', 'resnet50')
        if backbone_type == 'hrnet_w18':
            self.backbone = HRNetBackboneTF(
                feature_dim=config.feature_dim,
                use_pretrained=config.use_pretrained,
            )
        else:
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

        # Global context branch (mirrors PyTorch global_context)
        self.global_context_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.global_context_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(config.semantic_latent_dim),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.Dropout(config.dropout),
        ])
        self.global_fusion = tf.keras.Sequential([
            tf.keras.layers.Dense(config.semantic_latent_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("gelu"),
        ])

        # Missing region token
        self.region_reliability_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(config.feature_dim // 2),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation("sigmoid"),
        ])

        self.region_dropout_prob = float(getattr(config, "region_dropout_prob", 0.0))

    def build(self, input_shape):
        # semantic_structure_gate: (num_classes,) — per-class gate init -0.5
        self.semantic_structure_gate = self.add_weight(
            name="semantic_structure_gate",
            shape=(self.config.num_classes,),
            initializer=tf.keras.initializers.Constant(-0.5),
            trainable=True,
        )
        # missing_region_token: (feature_dim,)
        self.missing_region_token = self.add_weight(
            name="missing_region_token",
            shape=(self.config.feature_dim,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    # ------------------------------------------------------------------
    # Region preparation (mirrors PyTorch _prepare_regions)
    # ------------------------------------------------------------------

    def _canonical_bboxes(self, batch_size: int) -> tf.Tensor:
        boxes = _CANONICAL_BOXES_48.copy() * (float(self.config.bbox_input_size) / 48.0)
        boxes_tf = tf.constant(boxes[np.newaxis], dtype=tf.float32)
        return tf.tile(boxes_tf, [batch_size, 1, 1])

    def _prepare_regions(self, bboxes, batch_size, training):
        R = self.config.num_regions

        if bboxes is None:
            repaired = self._canonical_bboxes(batch_size)
            region_mask = tf.ones([batch_size, R], dtype=tf.float32)
            region_confidence = tf.fill([batch_size, R], 0.95)
            return repaired, region_mask, region_confidence

        bboxes = tf.cast(bboxes, tf.float32)

        x1 = bboxes[..., 0]; y1 = bboxes[..., 1]
        x2 = bboxes[..., 2]; y2 = bboxes[..., 3]

        finite_mask = tf.reduce_all(tf.math.is_finite(bboxes), axis=-1)
        size_mask = ((x2 - x1) >= 2.0) & ((y2 - y1) >= 2.0)
        order_mask = (x2 > x1) & (y2 > y1)
        region_mask = tf.cast(finite_mask & size_mask & order_mask, tf.float32)

        # Repair bboxes
        repaired = self.roi_align_layer._validate_bboxes(bboxes)
        canonical = self._canonical_bboxes(batch_size)
        mask_4 = tf.cast(region_mask[..., tf.newaxis] > 0, tf.bool)
        repaired = tf.where(mask_4, repaired, canonical)

        width  = tf.maximum(repaired[..., 2] - repaired[..., 0], 1.0)
        height = tf.maximum(repaired[..., 3] - repaired[..., 1], 1.0)
        size = float(self.config.bbox_input_size)
        area = (width * height) / (size * size)
        area_conf = tf.clip_by_value(area, 0.0, 1.0)
        region_confidence = tf.where(
            tf.cast(region_mask > 0, tf.bool),
            0.5 + 0.5 * area_conf,
            tf.fill(tf.shape(area_conf), 0.05),
        )
        return repaired, region_mask, region_confidence

    # ------------------------------------------------------------------
    # Core forward (mirrors PyTorch _forward_single)
    # ------------------------------------------------------------------

    def _forward_single(self, image: tf.Tensor, bboxes=None, region_mask=None, region_confidence=None, training: bool = False) -> Dict:
        image = tf.cast(image, tf.float32)
        if image.shape[-1] == 1:
            image = tf.repeat(image, 3, axis=-1)  # grayscale -> 3ch

        batch_size = tf.shape(image)[0]
        batch_size_s = image.shape[0]

        # Backbone: NHWC -> NHWC
        feature_map = self.backbone(image, training=training)  # (B, H', W', C)

        repaired, computed_mask, computed_confidence = self._prepare_regions(
            bboxes, batch_size, training
        )
        
        if region_mask is None:
            region_mask = computed_mask
        else:
            region_mask = tf.cast(region_mask, tf.float32)
            
        if region_confidence is None:
            region_confidence = computed_confidence
        else:
            region_confidence = tf.cast(region_confidence, tf.float32)

        # Region dropout (mirrors PyTorch training drop_mask)
        if training and self.region_dropout_prob > 0.0:
            drop_mask = tf.cast(
                tf.random.uniform([batch_size, self.config.num_regions]) > self.region_dropout_prob,
                tf.float32,
            )
            region_mask = region_mask * drop_mask
            region_confidence = region_confidence * drop_mask

        # ROI Align: (B, R, G*G, C)
        roi_nodes = self.roi_align_layer(feature_map, repaired)

        # MicroGraphReasoner: (B, R, G*G, C) -> (B, R, G*G, C), (B, R, C)
        micro_node_features, region_embeddings = self.micro_reasoner(roi_nodes, training=training)

        # Missing region token substitution
        missing_token = tf.cast(tf.reshape(self.missing_region_token, [1, 1, -1]), region_embeddings.dtype)
        region_valid_mask = tf.cast(region_mask[..., tf.newaxis] > 0, tf.bool)
        region_embeddings = tf.where(region_valid_mask, region_embeddings,
                                     tf.broadcast_to(missing_token, tf.shape(region_embeddings)))

        # Predicted confidence
        predicted_confidence = self.region_reliability_predictor(
            region_embeddings, training=training
        )[..., 0]
        
        predicted_confidence = tf.cast(predicted_confidence, region_confidence.dtype)
        
        region_confidence = tf.clip_by_value(
            0.5 * region_confidence + 0.5 * predicted_confidence, 0.0, 1.0
        )
        region_confidence = region_confidence * region_mask

        # Semantic state encoding
        semantic_state_tokens = self.semantic_state_encoder(region_embeddings, training=training)

        # Micro motif matching
        micro_motif_bank = self.micro_motif_bank(training=training)
        micro_motif_attention, semantic_motif_tokens = self.micro_motif_matcher(
            semantic_state_tokens, micro_motif_bank, training=training
        )

        # Pairwise interaction
        interaction_states, semantic_interaction_tensor, semantic_interaction_gates = \
            self.semantic_interaction_block(semantic_motif_tokens, training=training, region_mask=region_mask)

        # Cross-region composition
        cross_region_outputs = self.cross_region_composition_graph(
            interaction_states, training=training,
            region_mask=region_mask, region_confidence=region_confidence,
        )
        cross_region_tokens = cross_region_outputs["cross_region_tokens"]
        cross_region_attention = cross_region_outputs["composition_attn"]
        cross_region_pair_tokens = cross_region_outputs["pair_tokens"]
        cross_region_pair_scores = cross_region_outputs["pair_scores"]
        cross_region_pair_attention = cross_region_outputs["pair_attention"]

        # Enrich with composition context
        composition_summary = tf.reduce_mean(cross_region_tokens, axis=1, keepdims=True)
        hypergraph_input = interaction_states + tf.broadcast_to(
            composition_summary, tf.shape(interaction_states)
        )

        # Hypergraph reasoning
        compositional_outputs = self.semantic_compositional_reasoner(
            hypergraph_input, training=training,
            region_mask=region_mask, region_confidence=region_confidence,
        )
        composed_states = compositional_outputs["composed_states"]
        hyperedge_tokens = compositional_outputs["hyperedge_tokens"]
        routing_weights = compositional_outputs["routing_weights"]
        semantic_latent_embedding = compositional_outputs["emotion_latent"]

        # Program bank + executor
        semantic_program_bank, semantic_program_topology = self.semantic_program_bank(training=training)
        semantic_program_outputs = self.semantic_program_executor(
            composed_states, cross_region_tokens,
            semantic_program_bank, semantic_program_topology,
            training=training,
            region_mask=region_mask,
            interaction_gates=semantic_interaction_gates,
            routing_weights=routing_weights,
        )
        semantic_program_scores = semantic_program_outputs["program_scores"]
        semantic_program_attention = semantic_program_outputs["program_attention"]
        semantic_program_tokens = semantic_program_outputs["program_tokens"]

        # Global context branch
        global_ctx = self.global_context_pool(feature_map)
        global_ctx = self.global_context_dense(global_ctx, training=training)
        fused_latent = self.global_fusion(
            tf.concat([semantic_latent_embedding, global_ctx], axis=-1), training=training
        )
        logits_fused = self.semantic_classifier(fused_latent, training=training)

        # Per-class gate
        structure_gate = tf.nn.sigmoid(self.semantic_structure_gate)[tf.newaxis, :]  # (1, C)
        logits_motif = semantic_program_scores
        logits = (1.0 - structure_gate) * logits_fused + structure_gate * logits_motif

        result_dict = {
            "logits": logits,
            "logits_motif": logits_motif,
            "logits_fused": logits_fused,
            "structure_gate": structure_gate,
            "micro_node_features": micro_node_features,
            "micro_motif_attention": micro_motif_attention,
            "region_embeddings": region_embeddings,
            "semantic_state_tokens": semantic_state_tokens,
            "semantic_motif_tokens": semantic_motif_tokens,
            "cross_region_tokens": cross_region_tokens,
            "cross_region_attention": cross_region_attention,
            "cross_region_pair_tokens": cross_region_pair_tokens,
            "cross_region_pair_scores": cross_region_pair_scores,
            "cross_region_pair_attention": cross_region_pair_attention,
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
            "micro_motif_bank": micro_motif_bank,
            "macro_motif_bank": semantic_program_bank,
        }

        # Build aux_losses dict (mirrors PyTorch output structure)
        aux_losses = {
            "micro_motif_bank": micro_motif_bank,
            "macro_motif_bank": semantic_program_bank,
            "micro_motif_attention": micro_motif_attention,
            "macro_motif_attention": semantic_program_attention,
        }
        result_dict["aux_losses"] = aux_losses
        
        return result_dict

    def call(self, inputs, training: bool = False):
        """Accept (image,) or (image, bboxes) or dict."""
        region_mask, region_confidence = None, None
        if isinstance(inputs, dict):
            image = inputs["image"]
            bboxes = inputs.get("bboxes", None)
            region_mask = inputs.get("region_mask", None)
            region_confidence = inputs.get("region_confidence", None)
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                image, bboxes = inputs
            elif len(inputs) == 4:
                image, bboxes, region_mask, region_confidence = inputs
            else:
                image = inputs[0]
                bboxes = None
        else:
            image = inputs
            bboxes = None
        
        # Public forward: dispatch to TTA or single-image path (mimicking PyTorch)
        if len(image.shape) == 5:
            return self._forward_tta(image, bboxes, region_mask, region_confidence, training)
            
        if not training and bboxes is not None:
            # Horizontal Flip TTA
            outputs_orig = self._forward_single(image, bboxes, region_mask, region_confidence, training)
            flipped_image = tf.reverse(image, axis=[-2]) # flip width
            
            # Flip bboxes
            w = float(self.config.bbox_input_size)
            x1 = (w - 1.0) - bboxes[..., 2]
            y1 = bboxes[..., 1]
            x2 = (w - 1.0) - bboxes[..., 0]
            y2 = bboxes[..., 3]
            flipped_bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
            
            swap_pairs = [(1, 2), (4, 5), (7, 8)]
            
            def _swap(tensor):
                if tensor is None: return None
                indices = list(range(self.config.num_regions))
                for l, r in swap_pairs:
                    indices[l], indices[r] = indices[r], indices[l]
                return tf.gather(tensor, indices, axis=1)

            flipped_bboxes = _swap(flipped_bboxes)
            flipped_region_mask = _swap(region_mask)
            flipped_region_confidence = _swap(region_confidence)
            
            outputs_flipped = self._forward_single(
                flipped_image, flipped_bboxes, flipped_region_mask, flipped_region_confidence, training
            )
            
            avg_outputs = {}
            _avg_keys = ("logits", "logits_motif", "logits_fused", "semantic_program_scores")
            for k, val in outputs_orig.items():
                if k in _avg_keys and k in outputs_flipped:
                    avg_outputs[k] = 0.5 * (val + outputs_flipped[k])
                else:
                    avg_outputs[k] = val
            return avg_outputs

        return self._forward_single(image, bboxes, region_mask, region_confidence, training=training)

    def _forward_tta(self, image, bboxes, region_mask, region_confidence, training):
        # image: (B, T, H, W, C)
        b = tf.shape(image)[0]
        t = tf.shape(image)[1]
        h, w, c = image.shape[2:]
        flat_image = tf.reshape(image, [b * t, h, w, c])
        
        flat_bboxes, flat_rmask, flat_rconf = None, None, None
        if bboxes is not None:
            flat_bboxes = tf.reshape(tf.tile(bboxes[:, tf.newaxis, :, :], [1, t, 1, 1]), [b * t, -1, 4])
        if region_mask is not None:
            flat_rmask = tf.reshape(tf.tile(region_mask[:, tf.newaxis, :], [1, t, 1]), [b * t, -1])
        if region_confidence is not None:
            flat_rconf = tf.reshape(tf.tile(region_confidence[:, tf.newaxis, :], [1, t, 1]), [b * t, -1])
            
        outputs = self._forward_single(flat_image, flat_bboxes, flat_rmask, flat_rconf, training)
        
        _avg_keys = ("logits", "logits_motif", "logits_fused", "semantic_program_scores")
        for key in _avg_keys:
            if key in outputs:
                x = outputs[key]
                outputs[key] = tf.reduce_mean(tf.reshape(x, [b, t, -1]), axis=1)
                
        # For non-averaged keys, keep center-crop
        center_idx = 4 if image.shape[1] > 4 else image.shape[1] // 2
        for key, val in list(outputs.items()):
            if key not in _avg_keys:
                if isinstance(val, tf.Tensor) and len(val.shape) >= 1 and tf.shape(val)[0] == b * t:
                    outputs[key] = tf.reshape(val, [b, t] + list(val.shape[1:]))[:, center_idx]
                    
        return outputs

