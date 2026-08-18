"""
Loss functions for Semantic ROI Graph FER model (TensorFlow version).

This module provides standalone loss functions for the dual-level
semantic ROI graph architecture:
- micro motif diversity
- macro semantic program diversity
- semantic supervised contrastive loss
- semantic consistency loss
- compositional motif consistency
- semantic disentanglement
- region coordination regularization
"""

from typing import Dict, Optional, Callable, Union

import tensorflow as tf


def _unwrap_model(model):
    return model


def _get_training_cfg(model) -> Dict:
    try:
        base_model = _unwrap_model(model)
        return getattr(base_model, "training_cfg", {})
    except Exception:
        return {}


def micro_motif_diversity_loss(motifs: tf.Tensor) -> tf.Tensor:
    """Encourage diverse motifs within each semantic region bank."""
    shape = tf.shape(motifs)
    r, k, d = shape[0], shape[1], shape[2]
    
    motifs = tf.reshape(motifs, [r, k, d])
    motifs = tf.math.l2_normalize(motifs, axis=-1)
    sim = tf.einsum("rkd,rgd->rkg", motifs, motifs)
    
    identity = tf.expand_dims(tf.eye(k, dtype=sim.dtype), 0)
    off_diag = sim * (1.0 - identity)
    return tf.reduce_mean(tf.square(off_diag))


def macro_motif_diversity_loss(motifs: tf.Tensor) -> tf.Tensor:
    """Encourage diverse macro motifs across class topology prototypes."""
    if isinstance(motifs, tuple) or isinstance(motifs, list):
        motifs = motifs[0]
        
    shape = tf.shape(motifs)
    if len(motifs.shape) == 3:
        c, m, d = shape[0], shape[1], shape[2]
        motifs = tf.reshape(motifs, [c, m, d])
    else:
        c, m, r, d = shape[0], shape[1], shape[2], shape[3]
        motifs = tf.reshape(motifs, [c, m, r * d])
        
    motifs = tf.math.l2_normalize(motifs, axis=-1)
    sim = tf.einsum("cmd,cnd->cmn", motifs, motifs)
    
    m_dim = tf.shape(sim)[-1]
    identity = tf.expand_dims(tf.eye(m_dim, dtype=sim.dtype), 0)
    off_diag = sim * (1.0 - identity)
    
    # Use margin to avoid forcing 100% orthogonality which conflicts with CE
    off_diag = tf.nn.relu(tf.abs(sim) - 0.3) * (1.0 - identity)
    return tf.reduce_mean(tf.square(off_diag))


def motif_diversity_loss(motif_bank_fn: Callable) -> tf.Tensor:
    """Backward-compatible alias for macro motif diversity."""
    motifs = motif_bank_fn()
    if isinstance(motifs, tuple) or isinstance(motifs, list):
        motifs = motifs[0]
    if len(motifs.shape) == 3:
        return micro_motif_diversity_loss(lambda: motifs)
    return macro_motif_diversity_loss(lambda: motifs)


def compositional_program_consistency_loss(program_scores: Optional[tf.Tensor], labels: tf.Tensor) -> tf.Tensor:
    """Encourage the correct semantic facial program to dominate execution output."""
    if program_scores is None:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=program_scores))


def topology_alignment_loss(
    predicted_topology: Optional[tf.Tensor],
    program_topology: Optional[tf.Tensor],
    labels: tf.Tensor,
    program_attention: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Align observed region coordination with the selected semantic program topology."""
    if predicted_topology is None or program_topology is None:
        return tf.constant(0.0, dtype=tf.float32)

    selected_topology = tf.gather(program_topology, labels)
    
    if program_attention is not None:
        batch_size = tf.shape(labels)[0]
        batch_indices = tf.range(batch_size, dtype=labels.dtype)
        indices = tf.stack([batch_indices, labels], axis=-1)
        selected_attention = tf.gather_nd(program_attention, indices)
        
        selected_topology = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(selected_attention, -1), -1) * selected_topology, 
            axis=1
        )
    else:
        selected_topology = tf.reduce_mean(selected_topology, axis=1)

    if len(predicted_topology.shape) == 4:
        predicted_topology = tf.reduce_mean(predicted_topology, axis=1)

    return tf.reduce_mean(tf.square(predicted_topology - selected_topology))


def region_composition_contrastive_loss(
    cross_region_tokens: Optional[tf.Tensor],
    labels: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
    temperature: float = 0.07,
) -> tf.Tensor:
    """Contrast higher-order cross-region semantic compositions across emotions."""
    if cross_region_tokens is None:
        return tf.constant(0.0, dtype=tf.float32)
    return region_supervised_contrastive_loss(cross_region_tokens, labels, temperature=temperature, region_mask=None)


def semantic_program_sparsity_loss(
    program_attention: Optional[tf.Tensor] = None,
    routing_weights: Optional[tf.Tensor] = None,
    cross_region_attention: Optional[tf.Tensor] = None,
    mode: str = "l1",
) -> tf.Tensor:
    """Sparsity / load-balance loss for programs and routing.

    Args:
        mode: 'l1' to use L1 sparsity (encourages sparse activations),
              'entropy' to use entropy-maximization (encourage balanced use).
    """
    losses = []

    def _entropy(attn: tf.Tensor) -> tf.Tensor:
        attn = tf.maximum(attn, 1e-6)
        entropy = -tf.reduce_sum(attn * tf.math.log(attn), axis=-1)
        denom = tf.maximum(tf.math.log(tf.cast(tf.shape(attn)[-1], attn.dtype)), 1e-6)
        return tf.reduce_mean(entropy / denom)

    def _l1(attn: tf.Tensor) -> tf.Tensor:
        # Negative L2 norm to encourage sparsity on softmax distributions
        return -tf.reduce_mean(tf.square(attn))

    if program_attention is not None:
        losses.append(program_attention)
    if routing_weights is not None:
        pass # Not applying sparsity to routing_weights
    if cross_region_attention is not None:
        if len(cross_region_attention.shape) == 4:
            attn = tf.reduce_mean(cross_region_attention, axis=1)
        else:
            attn = cross_region_attention
        losses.append(attn)

    if not losses:
        return tf.constant(0.0, dtype=tf.float32)

    if mode == "entropy":
        vals = [_entropy(x) for x in losses]
        # We want to MAXIMIZE entropy to encourage load balancing.
        # Return negative entropy so that minimizing loss increases entropy.
        return -tf.reduce_sum(vals) / float(len(vals))
    else:
        vals = [_l1(x) for x in losses]
        return tf.reduce_sum(vals) / float(len(vals))


def program_diversity_loss(program_bank: Union[tf.Tensor, Callable]) -> tf.Tensor:
    """Encourage different semantic facial programs to specialize."""
    if callable(program_bank):
        program_bank = program_bank()
    if isinstance(program_bank, tuple) or isinstance(program_bank, list):
        program_bank = program_bank[0]

    if len(program_bank.shape) == 4:
        summaries = tf.reduce_mean(program_bank, axis=2)
    else:
        summaries = program_bank

    summaries = tf.reshape(summaries, [-1, tf.shape(summaries)[-1]])
    
    def compute_div():
        norm_summaries = tf.math.l2_normalize(summaries, axis=-1)
        sim = tf.matmul(norm_summaries, norm_summaries, transpose_b=True)
        identity = tf.eye(tf.shape(sim)[0], dtype=sim.dtype)
        off_diag = tf.nn.relu(tf.abs(sim) - 0.3) * (1.0 - identity)
        return tf.reduce_mean(tf.square(off_diag))
        
    return tf.cond(
        tf.shape(summaries)[0] < 2,
        lambda: tf.constant(0.0, dtype=tf.float32),
        compute_div
    )


def semantic_consistency_loss(
    semantic_states: tf.Tensor,
    labels: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Encourage samples from the same class to share similar semantic facial states."""
    if len(semantic_states.shape) == 3:
        if region_mask is not None:
            weights = tf.expand_dims(tf.cast(region_mask, semantic_states.dtype), -1)
            pooled = tf.reduce_sum(semantic_states * weights, axis=1) / tf.maximum(tf.reduce_sum(weights, axis=1), 1.0)
        else:
            pooled = tf.reduce_mean(semantic_states, axis=1)
    else:
        pooled = semantic_states

    labels = tf.reshape(labels, [-1])
    
    # Vectorized computation of class variances
    # Create mask of shape (B, B) where mask[i, j] = 1 if item i and j have the same label
    label_mask = tf.cast(tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)), pooled.dtype)
    counts = tf.reduce_sum(label_mask, axis=1, keepdims=True)
    
    # Center for each item's class
    centers = tf.matmul(label_mask, pooled) / tf.maximum(counts, 1e-6)
    
    # Variance of each item from its class center
    item_vars = tf.reduce_mean(tf.square(pooled - centers), axis=-1)
    
    # Keep classes with >= 2 items
    valid_mask = counts[:, 0] >= 2.0
    
    def compute_loss():
        valid_item_vars = tf.boolean_mask(item_vars, valid_mask)
        valid_counts = tf.boolean_mask(counts[:, 0], valid_mask)
        num_valid_classes = tf.reduce_sum(1.0 / valid_counts)
        return tf.reduce_sum(valid_item_vars / valid_counts) / tf.maximum(num_valid_classes, 1e-6)
        
    return tf.cond(
        tf.reduce_any(valid_mask),
        compute_loss,
        lambda: tf.constant(0.0, dtype=pooled.dtype)
    )


def compositional_motif_consistency_loss(program_scores: Optional[tf.Tensor], labels: tf.Tensor) -> tf.Tensor:
    """Align semantic latent emotion representations with the correct class program."""
    if program_scores is None:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=program_scores))


def semantic_disentanglement_loss(
    semantic_states: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Reduce redundancy across semantic state channels."""
    if len(semantic_states.shape) == 3:
        if region_mask is not None:
            flat_mask = tf.reshape(region_mask, [-1]) > 0
            flat_states = tf.reshape(semantic_states, [-1, tf.shape(semantic_states)[-1]])
            tokens = tf.boolean_mask(flat_states, flat_mask)
        else:
            tokens = tf.reshape(semantic_states, [-1, tf.shape(semantic_states)[-1]])
    else:
        tokens = tf.reshape(semantic_states, [-1, tf.shape(semantic_states)[-1]])

    def compute_disentangle():
        centered = tokens - tf.reduce_mean(tokens, axis=0, keepdims=True)
        cov = tf.matmul(centered, centered, transpose_a=True) / (tf.cast(tf.shape(tokens)[0], tokens.dtype) - 1.0)
        diag = tf.linalg.diag(tf.linalg.diag_part(cov))
        off_diag = cov - diag
        return tf.reduce_mean(tf.square(off_diag))

    return tf.cond(
        tf.shape(tokens)[0] < 2,
        lambda: tf.constant(0.0, dtype=tf.float32),
        compute_disentangle
    )


def region_coordination_regularization(
    routing_weights: Optional[tf.Tensor],
    interaction_gates: Optional[tf.Tensor] = None,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Regularize how strongly regions coordinate through routing and interactions."""
    loss = None

    if routing_weights is not None:
        weights = tf.maximum(routing_weights, 1e-6)
        entropy = -tf.reduce_sum(weights * tf.math.log(weights), axis=1)
        denom = tf.maximum(tf.math.log(tf.cast(tf.shape(weights)[1], weights.dtype)), 1e-6)
        loss = tf.reduce_mean(entropy / denom)

    if interaction_gates is not None:
        gates = interaction_gates
        if region_mask is not None:
            region_mask_f = tf.cast(region_mask, gates.dtype)
            pair_mask = tf.expand_dims(region_mask_f, -1) * tf.expand_dims(region_mask_f, -2)
            gates = gates * pair_mask
            
        active_mean = tf.reduce_mean(gates, axis=[-1, -2])
        gate_balance = tf.reduce_mean(tf.square(active_mean - 0.6))
        
        # Calculate variance
        gate_mean = tf.reduce_mean(gates, axis=[-1, -2], keepdims=True)
        gate_variance = tf.reduce_mean(tf.square(gates - gate_mean), axis=[-1, -2])
        gate_variance = tf.reduce_mean(gate_variance)
        
        gate_loss = gate_balance + 0.05 * gate_variance
        loss = gate_loss if loss is None else loss + gate_loss

    if loss is None:
        return tf.constant(0.0, dtype=tf.float32)

    return loss


def relation_consistency_loss(
    topology_matrix: tf.Tensor,
    labels: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Backward-compatible alias for semantic coordination regularization."""
    return region_coordination_regularization(topology_matrix, None, region_mask)


def topology_regularization_loss(topology_matrix: tf.Tensor) -> tf.Tensor:
    """Backward-compatible alias for semantic disentanglement loss."""
    return semantic_disentanglement_loss(topology_matrix)


def region_supervised_contrastive_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    temperature: float = 0.07,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Supervised contrastive loss on pooled region embeddings."""
    if len(embeddings.shape) == 3:
        if region_mask is not None:
            weights = tf.expand_dims(tf.cast(region_mask, embeddings.dtype), -1)
            pooled = tf.reduce_sum(embeddings * weights, axis=1) / tf.maximum(tf.reduce_sum(weights, axis=1), 1.0)
        else:
            pooled = tf.reduce_mean(embeddings, axis=1)
    else:
        pooled = embeddings
        
    pooled = tf.math.l2_normalize(pooled, axis=-1)
    sim = tf.matmul(pooled, pooled, transpose_b=True) / temperature
    
    labels = tf.reshape(labels, [-1, 1])
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), sim.dtype)
    
    logits_mask = 1.0 - tf.eye(tf.shape(mask)[0], dtype=mask.dtype)
    mask = mask * logits_mask

    exp_sim = tf.exp(sim) * logits_mask
    log_prob = sim - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8)
    
    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / tf.maximum(mask_sum, 1e-8)
    
    return -tf.reduce_mean(mean_log_prob_pos)


def supervised_contrastive_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    temperature: float = 0.07,
) -> tf.Tensor:
    """Backward-compatible alias for region supervised contrastive loss."""
    return region_supervised_contrastive_loss(embeddings, labels, temperature=temperature)


def region_consistency_loss(region_embeddings: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Backward-compatible alias for semantic consistency loss."""
    return semantic_consistency_loss(region_embeddings, labels)


def compute_ce_loss(logits, labels, label_smoothing=0.0, class_weights=None):
    num_classes = tf.shape(logits)[-1]
    if label_smoothing > 0:
        one_hot = tf.one_hot(labels, num_classes, dtype=logits.dtype)
        labels_smooth = one_hot * (1.0 - label_smoothing) + (label_smoothing / tf.cast(num_classes, logits.dtype))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_smooth, logits=logits)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    
    if class_weights is not None:
        weights = tf.gather(class_weights, labels)
        loss = loss * weights
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(weights), 1e-6)
    return tf.reduce_mean(loss)


def compute_semantic_roi_graph_losses(
    model,
    outputs: Dict[str, tf.Tensor],
    labels: tf.Tensor,
    class_weights: Optional[tf.Tensor] = None,
    temperature: Optional[float] = None,
    region_contrastive_weight: Optional[float] = None,
    micro_diversity_weight: Optional[float] = None,
    macro_diversity_weight: Optional[float] = None,
    relation_consistency_weight: Optional[float] = None,
    topology_reg_weight: Optional[float] = None,
    semantic_consistency_weight: Optional[float] = None,
    compositional_motif_weight: Optional[float] = None,
    semantic_disentanglement_weight: Optional[float] = None,
    region_coordination_weight: Optional[float] = None,
    compositional_program_weight: Optional[float] = None,
    topology_alignment_weight: Optional[float] = None,
    region_composition_contrastive_weight: Optional[float] = None,
    program_sparsity_weight: Optional[float] = None,
    program_diversity_weight: Optional[float] = None,
) -> Dict[str, tf.Tensor]:
    """
    Compute all losses for Semantic ROI Graph FER in TensorFlow.
    """
    logits = outputs["logits"]

    training_cfg = _get_training_cfg(model)

    if temperature is None:
        temperature = float(training_cfg.get("contrastive_temperature", 0.15))
    if region_contrastive_weight is None:
        region_contrastive_weight = float(training_cfg.get("region_contrastive_weight", training_cfg.get("au_contrastive_weight", 0.1)))
    if micro_diversity_weight is None:
        micro_diversity_weight = float(training_cfg.get("micro_motif_diversity_weight", training_cfg.get("motif_diversity_weight", 0.05)))
    if macro_diversity_weight is None:
        macro_diversity_weight = float(training_cfg.get("macro_motif_diversity_weight", training_cfg.get("motif_diversity_weight", 0.05))) * 2.0
    if relation_consistency_weight is None:
        relation_consistency_weight = float(training_cfg.get("region_coordination_weight", training_cfg.get("relation_consistency_weight", 0.1)))
    if topology_reg_weight is None:
        topology_reg_weight = float(training_cfg.get("semantic_disentanglement_weight", training_cfg.get("topology_reg_weight", 0.0)))
    if semantic_consistency_weight is None:
        semantic_consistency_weight = float(training_cfg.get("semantic_consistency_weight", training_cfg.get("region_consistency_weight", 0.1)))
    if compositional_motif_weight is None:
        compositional_motif_weight = float(training_cfg.get("compositional_motif_weight", training_cfg.get("macro_motif_consistency_weight", 0.1)))
    if semantic_disentanglement_weight is None:
        semantic_disentanglement_weight = float(training_cfg.get("semantic_disentanglement_weight", training_cfg.get("topology_reg_weight", 0.01)))
    if region_coordination_weight is None:
        region_coordination_weight = float(training_cfg.get("region_coordination_weight", training_cfg.get("relation_consistency_weight", 0.1)))
    if compositional_program_weight is None:
        compositional_program_weight = float(training_cfg.get("compositional_program_weight", 0.1))
    if topology_alignment_weight is None:
        topology_alignment_weight = float(training_cfg.get("topology_alignment_weight", 0.05))
    if region_composition_contrastive_weight is None:
        region_composition_contrastive_weight = float(training_cfg.get("region_composition_contrastive_weight", training_cfg.get("region_contrastive_weight", 0.1)))
    if program_sparsity_weight is None:
        program_sparsity_weight = float(training_cfg.get("program_sparsity_weight", 0.05))
    if program_diversity_weight is None:
        program_diversity_weight = float(training_cfg.get("program_diversity_weight", 0.05))

    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    ce_loss = compute_ce_loss(logits, labels, label_smoothing, class_weights)

    logits_fused = outputs.get("logits_fused")
    fused_aux_ce_weight = float(training_cfg.get("fused_aux_ce_weight", 0.0))
    if logits_fused is not None:
        fused_ce_loss = compute_ce_loss(logits_fused, labels, label_smoothing, class_weights)
    else:
        fused_ce_loss = tf.constant(0.0, dtype=tf.float32)

    base_model = _unwrap_model(model)

    semantic_states = outputs.get("semantic_state_tokens")
    if semantic_states is None:
        semantic_states = outputs.get("region_embeddings")

    region_mask = outputs.get("region_mask")
    routing_weights = outputs.get("semantic_routing_weights")
    interaction_gates = outputs.get("semantic_interaction_gates")
    program_scores = outputs.get("semantic_program_scores")
    program_attention = outputs.get("semantic_program_attention")
    semantic_latent = outputs.get("semantic_latent_embedding")
    if semantic_latent is None:
        semantic_latent = outputs.get("macro_embeddings")
    cross_region_tokens = outputs.get("cross_region_tokens")
    cross_region_attention = outputs.get("cross_region_attention")
    program_topology = outputs.get("semantic_program_topology")

    micro_diversity_loss = micro_motif_diversity_loss(base_model.micro_motif_bank.motifs)
    macro_diversity_loss = macro_motif_diversity_loss(base_model.semantic_program_bank.programs)
    
    contrastive_source = semantic_states if semantic_states is not None else semantic_latent
    contrastive_region_mask = region_mask if contrastive_source is not None and len(contrastive_source.shape) == 3 else None
    
    if contrastive_source is not None:
        contrastive_loss = region_supervised_contrastive_loss(
            contrastive_source,
            labels,
            temperature=temperature,
            region_mask=contrastive_region_mask,
        )
    else:
        contrastive_loss = tf.constant(0.0, dtype=tf.float32)
        
    semantic_consistency = semantic_consistency_loss(semantic_states, labels, region_mask=region_mask)
    compositional_loss = compositional_program_consistency_loss(program_scores, labels)
    disentanglement_loss = semantic_disentanglement_loss(semantic_states, region_mask=region_mask)
    coordination_loss = region_coordination_regularization(routing_weights, interaction_gates, region_mask=region_mask)
    topology_loss = topology_alignment_loss(interaction_gates, program_topology, labels, program_attention=program_attention)
    composition_contrastive_loss = region_composition_contrastive_loss(cross_region_tokens, labels, region_mask=region_mask, temperature=temperature)
    
    use_entropy = bool(training_cfg.get("use_entropy_sparsity", False))
    sparsity_mode = "entropy" if use_entropy else "l1"
    sparsity_loss = semantic_program_sparsity_loss(
        program_attention=program_attention,
        routing_weights=routing_weights,
        cross_region_attention=cross_region_attention,
        mode=sparsity_mode,
    )
    diversity_loss = program_diversity_loss(lambda: base_model.semantic_program_bank)

    def _flag(name: str, default: bool = True) -> bool:
        return bool(training_cfg.get(name, default))

    total = ce_loss
    if _flag("enable_micro_diversity"):
        total = total + float(micro_diversity_weight) * micro_diversity_loss
    if _flag("enable_macro_diversity"):
        total = total + float(macro_diversity_weight) * macro_diversity_loss
    if _flag("enable_region_contrastive"):
        total = total + float(region_contrastive_weight) * contrastive_loss
    if _flag("enable_semantic_consistency"):
        total = total + float(semantic_consistency_weight) * semantic_consistency
    if _flag("enable_compositional_program"):
        total = total + float(compositional_program_weight) * compositional_loss
    if _flag("enable_semantic_disentanglement"):
        total = total + float(semantic_disentanglement_weight) * disentanglement_loss
    if _flag("enable_region_coordination"):
        total = total + float(region_coordination_weight) * coordination_loss
    if _flag("enable_topology_alignment"):
        total = total + float(topology_alignment_weight) * topology_loss
    if _flag("enable_region_composition_contrastive"):
        total = total + float(region_composition_contrastive_weight) * composition_contrastive_loss
    if _flag("enable_program_sparsity"):
        total = total + float(program_sparsity_weight) * sparsity_loss
    if _flag("enable_program_diversity"):
        total = total + float(program_diversity_weight) * diversity_loss
    if _flag("enable_fused_aux_ce", False):
        total = total + fused_aux_ce_weight * fused_ce_loss
    
    return {
        "loss": total,
        "loss_ce": ce_loss,
        "loss_micro_motif_diversity": micro_diversity_loss,
        "loss_macro_motif_diversity": macro_diversity_loss,
        "loss_motif_diversity": micro_diversity_loss + macro_diversity_loss,
        "loss_contrastive": contrastive_loss,
        "loss_semantic_consistency": semantic_consistency,
        "loss_region_consistency": semantic_consistency,
        "loss_compositional_motif_consistency": compositional_loss,
        "loss_compositional_program_consistency": compositional_loss,
        "loss_semantic_disentanglement": disentanglement_loss,
        "loss_topology_reg": topology_loss,
        "loss_topology_alignment": topology_loss,
        "loss_region_composition_contrastive": composition_contrastive_loss,
        "loss_program_sparsity": sparsity_loss,
        "loss_program_diversity": diversity_loss,
        "loss_region_coordination": coordination_loss,
        "loss_relation_consistency": coordination_loss,
        "loss_fused_aux_ce": fused_ce_loss,
    }