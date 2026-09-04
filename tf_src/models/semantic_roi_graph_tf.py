"""
semantic_roi_graph_tf.py — Complete Semantic ROI Graph FER Model in TensorFlow.
Replicates the golden 72.92% architecture with exact dual-level graph reasoning,
semantic programs, logit scale alignment, and built-in horizontal flip TTA.
"""

from typing import Dict, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model

from tf_src.models.backbones_tf import HRNetW18TF, ResNet50TF
from tf_src.models.layers_tf import (
    SemanticRoiAlignTF,
    MicroGraphReasonerTF,
    SemanticStateEncoderTF,
    MicroSemanticMotifMatcherTF,
    SemanticInteractionBlockTF,
    CrossRegionCompositionGraphTF,
    SemanticHypergraphReasonerTF,
    SemanticProgramExecutorTF,
)


class SemanticROIGraphFERTF(Model):
    """
    Dual-level semantic compositional facial expression recognition model in TensorFlow.
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        m_cfg = config.get("model", {})

        self.num_classes = int(m_cfg.get("num_classes", 7))
        self.num_regions = int(m_cfg.get("num_regions", 9))
        self.feature_dim = int(m_cfg.get("feature_dim", 256))
        self.state_dim = int(m_cfg.get("semantic_state_dim", 128))
        self.latent_dim = int(m_cfg.get("semantic_latent_dim", 256))
        self.enable_logit_alignment = bool(m_cfg.get("enable_logit_alignment", True))
        self.enable_manifold_mixup = bool(m_cfg.get("enable_manifold_mixup", True))
        self.manifold_mixup_prob = float(m_cfg.get("manifold_mixup_prob", 0.3))
        self.manifold_mixup_alpha = float(m_cfg.get("manifold_mixup_alpha", 0.2))

        # 1. Backbone
        backbone_type = m_cfg.get("backbone_type", "hrnet_w18")
        out_size = int(m_cfg.get("backbone_out_size", 12))
        if backbone_type == "hrnet_w18":
            self.backbone = HRNetW18TF(feature_dim=self.feature_dim, out_size=out_size)
        else:
            self.backbone = ResNet50TF(feature_dim=self.feature_dim)

        # 2. ROI Alignment & Micro Reasoner
        self.roi_align = SemanticRoiAlignTF(
            roi_grid=int(m_cfg.get("roi_grid", 4)),
            bbox_input_size=int(m_cfg.get("bbox_input_size", 48)),
        )
        self.micro_reasoner = MicroGraphReasonerTF(
            dim=self.feature_dim,
            num_nodes=int(m_cfg.get("roi_grid", 4)) ** 2,
            layers_count=int(m_cfg.get("micro_layers", 2)),
            heads=int(m_cfg.get("attn_heads", 4)),
            dropout=float(m_cfg.get("dropout", 0.25)),
        )

        # 3. Semantic State Encoder & Motif Matcher
        self.state_encoder = SemanticStateEncoderTF(
            input_dim=self.feature_dim,
            state_dim=self.state_dim,
            dropout=float(m_cfg.get("dropout", 0.25)),
        )
        self.motif_matcher = MicroSemanticMotifMatcherTF(
            num_regions=self.num_regions,
            motifs_per_region=int(m_cfg.get("micro_motifs_per_region", 8)),
            state_dim=self.state_dim,
            temperature=float(m_cfg.get("relation_temperature", 0.07)),
        )

        # 4. Semantic Interaction & Composition
        self.interaction_block = SemanticInteractionBlockTF(
            state_dim=self.state_dim,
            dropout=float(m_cfg.get("dropout", 0.25)),
            dropedge_rate=0.5,
        )
        self.composition_graph = CrossRegionCompositionGraphTF(
            state_dim=self.state_dim,
            num_compositions=int(m_cfg.get("cross_region_compositions", 8)),
            attn_heads=int(m_cfg.get("semantic_attn_heads", 4)),
            dropout=float(m_cfg.get("dropout", 0.25)),
        )

        # 5. Hypergraph Reasoner & Program Executor
        self.hypergraph_reasoner = SemanticHypergraphReasonerTF(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            hyperedge_count=int(m_cfg.get("hyperedge_count", 4)),
            attn_heads=int(m_cfg.get("semantic_attn_heads", 4)),
            router_hidden_dim=int(m_cfg.get("router_hidden_dim", 256)),
            dropout=float(m_cfg.get("dropout", 0.25)),
        )
        self.program_executor = SemanticProgramExecutorTF(
            num_classes=self.num_classes,
            programs_per_class=int(m_cfg.get("programs_per_class", 4)),
            num_regions=self.num_regions,
            state_dim=self.state_dim,
            temperature=float(m_cfg.get("relation_temperature", 0.07)),
        )

        # 6. Global Context & Emotion Classifier
        self.global_gap = layers.GlobalAveragePooling2D()
        self.global_context = tf.keras.Sequential([
            layers.Dense(self.latent_dim),
            layers.LayerNormalization(epsilon=1e-5),
            layers.Activation('gelu'),
        ])
        self.global_fusion = tf.keras.Sequential([
            layers.Dense(self.latent_dim),
            layers.LayerNormalization(epsilon=1e-5),
            layers.Activation('gelu'),
            layers.Dropout(float(m_cfg.get("dropout", 0.25))),
        ])
        self.classifier = layers.Dense(self.num_classes)

        # 7. Logit Alignment & Ensembling
        self.fused_logit_norm = layers.LayerNormalization(epsilon=1e-5)
        self.motif_logit_norm = layers.LayerNormalization(epsilon=1e-5)
        self.structure_gate = self.add_weight(
            name="structure_gate",
            shape=(1, self.num_classes),
            initializer="zeros",
            trainable=True
        )
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(1,),
            initializer="ones",
            trainable=True
        )

    def _forward_single(
        self,
        images: tf.Tensor,
        bboxes: tf.Tensor,
        region_mask: Optional[tf.Tensor] = None,
        region_confidence: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Core single forward pass."""
        batch_size = tf.shape(images)[0]

        # 1. Feature extraction
        feature_map = self.backbone(images, training=training) # (B, 12, 12, 256)
        global_ctx = self.global_context(self.global_gap(feature_map)) # (B, 256)

        # 2. ROI extraction & micro reasoning
        roi_nodes = self.roi_align(feature_map, bboxes) # (B, 9, 16, 256)
        _, region_embeddings = self.micro_reasoner(roi_nodes, training=training) # (B, 9, 256)

        # 3. State encoding & motif matching
        semantic_states = self.state_encoder(region_embeddings, training=training) # (B, 9, 128)

        # Semantic Manifold Mixup (Feature-Level Mixup, 100% bbox-safe)
        mixup_lam = tf.constant(1.0, dtype=tf.float32)
        mixup_perm = None
        if training and self.enable_manifold_mixup and self.manifold_mixup_prob > 0.0:
            rand_val = tf.random.uniform([], 0.0, 1.0)
            if rand_val < self.manifold_mixup_prob:
                alpha = self.manifold_mixup_alpha
                gamma_a = tf.random.gamma([], alpha, 1.0)
                gamma_b = tf.random.gamma([], alpha, 1.0)
                lam = gamma_a / (gamma_a + gamma_b + 1e-8)
                mixup_lam = tf.where(lam < 0.5, 1.0 - lam, lam)
                mixup_perm = tf.random.shuffle(tf.range(batch_size))

                semantic_states_perm = tf.gather(semantic_states, mixup_perm)
                global_ctx_perm = tf.gather(global_ctx, mixup_perm)
                semantic_states = mixup_lam * semantic_states + (1.0 - mixup_lam) * semantic_states_perm
                global_ctx = mixup_lam * global_ctx + (1.0 - mixup_lam) * global_ctx_perm

        micro_attn, motif_tokens = self.motif_matcher(semantic_states) # (B, 9, 128)

        # 4. Pairwise interaction & higher-order composition
        interaction_states, _, raw_gates = self.interaction_block(motif_tokens, region_mask=region_mask, training=training)
        comp_out = self.composition_graph(interaction_states, region_mask=region_mask, region_confidence=region_confidence, training=training)
        cross_tokens = comp_out["cross_region_tokens"] # (B, 8, 128)

        # 5. Hypergraph reasoning
        comp_summary = tf.reduce_mean(cross_tokens, axis=1, keepdims=True)
        hyper_in = interaction_states + tf.tile(comp_summary, [1, self.num_regions, 1])
        hyper_out = self.hypergraph_reasoner(hyper_in, region_mask=region_mask, region_confidence=region_confidence, training=training)
        composed_states = hyper_out["composed_states"]
        routing_weights = hyper_out["routing_weights"]
        emotion_latent = hyper_out["emotion_latent"] # (B, 256)

        # 6. Program execution -> logits_motif
        prog_out = self.program_executor(
            composed_states,
            cross_tokens,
            region_mask=region_mask,
            interaction_gates=raw_gates,
            routing_weights=routing_weights
        )
        logits_motif = prog_out["program_scores"] # (B, 7)

        # 7. Global fusion -> logits_fused
        fused_latent = self.global_fusion(tf.concat([emotion_latent, global_ctx], axis=-1), training=training)
        logits_fused = self.classifier(fused_latent) # (B, 7)

        # 8. Logit Alignment & Ensembling
        gate = tf.sigmoid(self.structure_gate) # (1, 7)
        if self.enable_logit_alignment:
            fused_norm = self.fused_logit_norm(logits_fused)
            motif_norm = self.motif_logit_norm(logits_motif)
            blended = (1.0 - gate) * fused_norm + gate * motif_norm
            logits = self.logit_scale * blended
        else:
            logits = (1.0 - gate) * logits_fused + gate * logits_motif

        return {
            "logits": logits,
            "logits_motif": logits_motif,
            "logits_fused": logits_fused,
            "structure_gate": gate,
            "semantic_states": semantic_states,
            "cross_region_tokens": cross_tokens,
            "emotion_latent": emotion_latent,
            "mixup_lam": mixup_lam,
            "mixup_perm": mixup_perm,
        }

    def _flip_bboxes(self, bboxes: tf.Tensor) -> tf.Tensor:
        """Horizontally flip bounding boxes and swap symmetric left/right pairs."""
        # bboxes: (B, 9, 4) in [x1, y1, x2, y2]
        w = float(self.config.get("model", {}).get("bbox_input_size", 48) - 1)
        x1 = bboxes[..., 0]
        y1 = bboxes[..., 1]
        x2 = bboxes[..., 2]
        y2 = bboxes[..., 3]

        flipped_x1 = w - x2
        flipped_x2 = w - x1
        flipped_boxes = tf.stack([flipped_x1, y1, flipped_x2, y2], axis=-1)

        # Symmetric swap pairs: 1 <-> 2 (eyebrows), 4 <-> 5 (eyes), 7 <-> 8 (mouth corners)
        swap_indices = [0, 2, 1, 3, 5, 4, 6, 8, 7]
        flipped_boxes = tf.gather(flipped_boxes, swap_indices, axis=1)
        return flipped_boxes

    def _flip_mask_or_conf(self, tensor: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
        if tensor is None:
            return None
        swap_indices = [0, 2, 1, 3, 5, 4, 6, 8, 7]
        return tf.gather(tensor, swap_indices, axis=1)

    def call(
        self,
        images: tf.Tensor,
        bboxes: tf.Tensor,
        region_mask: Optional[tf.Tensor] = None,
        region_confidence: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """
        Public call: In inference mode (training=False), executes the golden 72.92%
        built-in Horizontal Flip TTA with symmetric ROI transforms.
        """
        if training or bboxes is None:
            return self._forward_single(images, bboxes, region_mask, region_confidence, training=training)

        # 1. Normal forward
        out_orig = self._forward_single(images, bboxes, region_mask, region_confidence, training=False)

        # 2. Flipped forward
        flipped_images = tf.image.flip_left_right(images)
        flipped_bboxes = self._flip_bboxes(bboxes)
        flipped_mask = self._flip_mask_or_conf(region_mask)
        flipped_conf = self._flip_mask_or_conf(region_confidence)

        out_flipped = self._forward_single(flipped_images, flipped_bboxes, flipped_mask, flipped_conf, training=False)

        # 3. Average ensemble logits
        avg_out = {}
        for k in out_orig:
            if k in ["logits", "logits_motif", "logits_fused"]:
                avg_out[k] = 0.5 * (out_orig[k] + out_flipped[k])
            else:
                avg_out[k] = out_orig[k]
        return avg_out
