"""
semantic_roi_graph_losses_tf.py — TensorFlow port của semantic_roi_graph_losses.py

Tất cả PyTorch ops được thay thế bằng TensorFlow equivalents:
- torch.Tensor -> tf.Tensor
- F.cross_entropy -> tf.keras.losses.SparseCategoricalCrossentropy
- F.normalize -> tf.math.l2_normalize
- .mm() -> tf.linalg.matmul
- .clamp_min -> tf.maximum
"""

from typing import Dict, Optional

import tensorflow as tf


def _ce_loss_with_smoothing(
    labels_int: tf.Tensor,
    logits: tf.Tensor,
    label_smoothing: float = 0.0,
) -> tf.Tensor:
    """Cross-entropy with optional label smoothing.
    Works across all Keras versions (sparse_categorical_crossentropy does not
    accept label_smoothing in older versions).
    """
    num_classes = tf.shape(logits)[-1]
    if label_smoothing > 0.0:
        one_hot = tf.one_hot(labels_int, num_classes, dtype=logits.dtype)
        smooth_val = label_smoothing / tf.cast(num_classes, logits.dtype)
        one_hot = one_hot * (1.0 - label_smoothing) + smooth_val
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                one_hot, logits, from_logits=True
            )
        )
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels_int, logits, from_logits=True
        )
    )

def _get_training_cfg(model) -> Dict:
    try:
        return getattr(model, "training_cfg", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Component losses
# ---------------------------------------------------------------------------

def micro_motif_diversity_loss(motif_bank_layer) -> tf.Tensor:
    """Encourage diverse motifs within each semantic region bank."""
    motifs = motif_bank_layer(None)  # (R, K, D)
    r, k, d = motifs.shape
    motifs = tf.math.l2_normalize(tf.reshape(motifs, [r, k, d]), axis=-1)
    sim = tf.einsum("rkd,rgd->rkg", motifs, motifs)
    identity = tf.eye(k)[None]  # (1, K, K)
    off_diag = sim * (1.0 - identity)
    return tf.reduce_mean(off_diag ** 2)


def macro_motif_diversity_loss(program_bank_layer) -> tf.Tensor:
    """Encourage diverse macro motifs across class topology prototypes."""
    result = program_bank_layer(None)
    if isinstance(result, (list, tuple)):
        motifs = result[0]
    else:
        motifs = result

    if len(motifs.shape) == 3:
        c, m, d = motifs.shape
        motifs_flat = tf.reshape(motifs, [c, m, d])
    else:
        c, m, r, d = motifs.shape
        motifs_flat = tf.reshape(motifs, [c, m, r * d])

    motifs_flat = tf.math.l2_normalize(motifs_flat, axis=-1)
    sim = tf.einsum("cmd,cnd->cmn", motifs_flat, motifs_flat)
    identity = tf.eye(m)[None]  # (1, M, M)
    off_diag = tf.nn.relu(tf.abs(sim) - 0.3) * (1.0 - identity)
    return tf.reduce_mean(off_diag ** 2)


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
            pooled = (
                tf.reduce_sum(embeddings * weights, axis=1) /
                tf.maximum(tf.reduce_sum(weights, axis=1), 1.0)
            )
        else:
            pooled = tf.reduce_mean(embeddings, axis=1)
    else:
        pooled = embeddings

    pooled = tf.math.l2_normalize(pooled, axis=-1)
    sim = tf.linalg.matmul(pooled, pooled, transpose_b=True) / temperature

    labels_col = tf.expand_dims(tf.cast(labels, tf.int32), 1)
    labels_row = tf.expand_dims(tf.cast(labels, tf.int32), 0)
    mask_pos = tf.cast(tf.equal(labels_col, labels_row), pooled.dtype)

    n = tf.shape(pooled)[0]
    logits_mask = 1.0 - tf.eye(n, dtype=pooled.dtype)
    mask = mask_pos * logits_mask

    exp_sim = tf.exp(sim) * logits_mask
    log_prob = sim - tf.math.log(
        tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8
    )
    mean_log_prob_pos = (
        tf.reduce_sum(mask * log_prob, axis=1) /
        (tf.reduce_sum(mask, axis=1) + 1e-8)
    )
    return -tf.reduce_mean(mean_log_prob_pos)


def semantic_consistency_loss(
    semantic_states: tf.Tensor,
    labels: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Encourage same-class samples to share similar semantic states."""
    if len(semantic_states.shape) == 3:
        if region_mask is not None:
            weights = tf.expand_dims(tf.cast(region_mask, semantic_states.dtype), -1)
            pooled = (
                tf.reduce_sum(semantic_states * weights, axis=1) /
                tf.maximum(tf.reduce_sum(weights, axis=1), 1.0)
            )
        else:
            pooled = tf.reduce_mean(semantic_states, axis=1)
    else:
        pooled = semantic_states

    labels_flat = tf.reshape(labels, [-1])
    unique_classes, _ = tf.unique(labels_flat)
    losses = []
    for cls in unique_classes.numpy():
        mask = tf.equal(labels_flat, cls)
        cls_states = tf.boolean_mask(pooled, mask)
        if tf.shape(cls_states)[0] < 2:
            continue
        center = tf.reduce_mean(cls_states, axis=0, keepdims=True)
        losses.append(tf.reduce_mean((cls_states - center) ** 2))

    if not losses:
        return tf.zeros(())
    return tf.reduce_mean(losses)


def semantic_disentanglement_loss(
    semantic_states: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Reduce redundancy across semantic state channels."""
    if len(semantic_states.shape) == 3:
        if region_mask is not None:
            flat_mask = tf.reshape(region_mask, [-1]) > 0
            tokens = tf.boolean_mask(
                tf.reshape(semantic_states, [-1, semantic_states.shape[-1]]),
                flat_mask
            )
        else:
            tokens = tf.reshape(semantic_states, [-1, semantic_states.shape[-1]])
    else:
        tokens = tf.reshape(semantic_states, [-1, semantic_states.shape[-1]])

    if tf.shape(tokens)[0] < 2:
        return tf.zeros(())

    centered = tokens - tf.reduce_mean(tokens, axis=0, keepdims=True)
    n = tf.cast(tf.shape(tokens)[0] - 1, tokens.dtype)
    cov = tf.linalg.matmul(centered, centered, transpose_a=True) / n
    off_diag = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    return tf.reduce_mean(off_diag ** 2)


def region_coordination_regularization(
    routing_weights: Optional[tf.Tensor],
    interaction_gates: Optional[tf.Tensor] = None,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Regularize routing and interaction gate patterns."""
    loss = None

    if routing_weights is not None:
        weights = tf.maximum(routing_weights, 1e-6)
        entropy = -tf.reduce_sum(weights * tf.math.log(weights), axis=1)
        n = tf.cast(tf.shape(weights)[1], weights.dtype)
        denom = tf.maximum(tf.math.log(n), 1e-6)
        loss = tf.reduce_mean(entropy / denom)

    if interaction_gates is not None:
        gates = interaction_gates
        if region_mask is not None:
            pair_mask = (
                tf.expand_dims(region_mask, -1) *
                tf.expand_dims(region_mask, -2)
            )
            gates = gates * pair_mask
        active_mean = tf.reduce_mean(gates, axis=[-1, -2])
        gate_balance = tf.reduce_mean((active_mean - 0.6) ** 2)
        gate_variance = tf.reduce_mean(tf.math.reduce_variance(gates, axis=[-1, -2]))
        gate_loss = gate_balance + 0.05 * gate_variance
        loss = gate_loss if loss is None else loss + gate_loss

    if loss is None:
        return tf.zeros(())
    return loss


def topology_alignment_loss(
    predicted_topology: Optional[tf.Tensor],
    program_topology: Optional[tf.Tensor],
    labels: tf.Tensor,
    program_attention: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    if predicted_topology is None or program_topology is None:
        return tf.zeros(())

    labels_idx = tf.cast(labels, tf.int32)
    selected_topology = tf.gather(program_topology, labels_idx)  # (B, M, R, R)

    if program_attention is not None:
        batch_idx = tf.range(tf.shape(labels_idx)[0])
        selected_attention = tf.gather_nd(
            program_attention,
            tf.stack([batch_idx, labels_idx], axis=1)
        )  # (B, M)
        selected_topology = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(selected_attention, -1), -1) * selected_topology,
            axis=1
        )
    else:
        selected_topology = tf.reduce_mean(selected_topology, axis=1)

    if len(predicted_topology.shape) == 4:
        predicted_topology = tf.reduce_mean(predicted_topology, axis=1)

    return tf.reduce_mean((predicted_topology - selected_topology) ** 2)


def compositional_program_consistency_loss(
    program_scores: Optional[tf.Tensor],
    labels: tf.Tensor,
) -> tf.Tensor:
    if program_scores is None:
        return tf.zeros(())
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, program_scores, from_logits=True
        )
    )


def region_composition_contrastive_loss(
    cross_region_tokens: Optional[tf.Tensor],
    labels: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
    temperature: float = 0.07,
) -> tf.Tensor:
    if cross_region_tokens is None:
        return tf.zeros(())
    return region_supervised_contrastive_loss(
        cross_region_tokens, labels, temperature=temperature, region_mask=None
    )


def semantic_program_sparsity_loss(
    program_attention: Optional[tf.Tensor] = None,
    routing_weights: Optional[tf.Tensor] = None,
    cross_region_attention: Optional[tf.Tensor] = None,
    mode: str = "l1",
) -> tf.Tensor:
    losses = []

    def _entropy(attn):
        attn = tf.maximum(attn, 1e-6)
        entropy = -tf.reduce_sum(attn * tf.math.log(attn), axis=-1)
        n = tf.cast(tf.shape(attn)[-1], attn.dtype)
        return tf.reduce_mean(entropy / tf.maximum(tf.math.log(n), 1e-6))

    def _l1(attn):
        return -tf.reduce_mean(attn ** 2)

    if program_attention is not None:
        losses.append(program_attention)
    if cross_region_attention is not None:
        attn = cross_region_attention
        if len(attn.shape) == 4:
            attn = tf.reduce_mean(attn, axis=1)
        losses.append(attn)

    if not losses:
        return tf.zeros(())

    if mode == "entropy":
        vals = [_entropy(x) for x in losses]
        return -tf.add_n(vals) / float(len(vals))
    else:
        vals = [_l1(x) for x in losses]
        return tf.add_n(vals) / float(len(vals))


def program_diversity_loss(program_bank_layer) -> tf.Tensor:
    result = program_bank_layer(None)
    if isinstance(result, (list, tuple)):
        program_bank = result[0]
    else:
        program_bank = result

    if len(program_bank.shape) == 4:
        summaries = tf.reduce_mean(program_bank, axis=2)
    else:
        summaries = program_bank

    summaries = tf.reshape(summaries, [-1, summaries.shape[-1]])
    if summaries.shape[0] < 2:
        return tf.zeros(())

    summaries = tf.math.l2_normalize(summaries, axis=-1)
    sim = tf.linalg.matmul(summaries, summaries, transpose_b=True)
    n = tf.shape(summaries)[0]
    identity = tf.eye(n, dtype=summaries.dtype)
    off_diag = tf.nn.relu(tf.abs(sim) - 0.3) * (1.0 - identity)
    return tf.reduce_mean(off_diag ** 2)


# ---------------------------------------------------------------------------
# Master loss function
# ---------------------------------------------------------------------------

def compute_semantic_roi_graph_losses_tf(
    model,
    outputs: Dict,
    labels: tf.Tensor,
    class_weights: Optional[tf.Tensor] = None,
    temperature: Optional[float] = None,
    region_contrastive_weight: Optional[float] = None,
    micro_diversity_weight: Optional[float] = None,
    macro_diversity_weight: Optional[float] = None,
    semantic_consistency_weight: Optional[float] = None,
    compositional_program_weight: Optional[float] = None,
    semantic_disentanglement_weight: Optional[float] = None,
    region_coordination_weight: Optional[float] = None,
    topology_alignment_weight: Optional[float] = None,
    region_composition_contrastive_weight: Optional[float] = None,
    program_sparsity_weight: Optional[float] = None,
    program_diversity_weight: Optional[float] = None,
    fused_aux_ce_weight: Optional[float] = None,
) -> Dict:
    """Compute all losses for TF SemanticROIGraphFER."""
    training_cfg = _get_training_cfg(model)

    # Fill defaults from training_cfg
    if temperature is None:
        temperature = float(training_cfg.get("contrastive_temperature", 0.15))
    if region_contrastive_weight is None:
        region_contrastive_weight = float(training_cfg.get("region_contrastive_weight", 0.1))
    if micro_diversity_weight is None:
        micro_diversity_weight = float(training_cfg.get("micro_motif_diversity_weight", 0.05))
    if macro_diversity_weight is None:
        macro_diversity_weight = float(training_cfg.get("macro_motif_diversity_weight", 0.05)) * 2.0
    if semantic_consistency_weight is None:
        semantic_consistency_weight = float(training_cfg.get("semantic_consistency_weight", 0.1))
    if compositional_program_weight is None:
        compositional_program_weight = float(training_cfg.get("compositional_program_weight", 0.1))
    if semantic_disentanglement_weight is None:
        semantic_disentanglement_weight = float(training_cfg.get("semantic_disentanglement_weight", 0.01))
    if region_coordination_weight is None:
        region_coordination_weight = float(training_cfg.get("region_coordination_weight", 0.1))
    if topology_alignment_weight is None:
        topology_alignment_weight = float(training_cfg.get("topology_alignment_weight", 0.05))
    if region_composition_contrastive_weight is None:
        region_composition_contrastive_weight = float(training_cfg.get("region_composition_contrastive_weight", 0.1))
    if program_sparsity_weight is None:
        program_sparsity_weight = float(training_cfg.get("program_sparsity_weight", 0.05))
    if program_diversity_weight is None:
        program_diversity_weight = float(training_cfg.get("program_diversity_weight", 0.05))
    if fused_aux_ce_weight is None:
        fused_aux_ce_weight = float(training_cfg.get("fused_aux_ce_weight", 0.0))

    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))

    logits = outputs["logits"]
    labels_int = tf.cast(labels, tf.int32)

    # Main CE loss
    ce_loss = _ce_loss_with_smoothing(labels_int, logits, label_smoothing)

    # Fused auxiliary CE
    logits_fused = outputs.get("logits_fused")
    if logits_fused is not None:
        fused_ce_loss = _ce_loss_with_smoothing(labels_int, logits_fused, label_smoothing)
    else:
        fused_ce_loss = tf.zeros(())

    # Extract outputs
    semantic_states = outputs.get("semantic_state_tokens") or outputs.get("region_embeddings")
    region_mask = outputs.get("region_mask")
    routing_weights = outputs.get("semantic_routing_weights")
    interaction_gates = outputs.get("semantic_interaction_gates")
    program_scores = outputs.get("semantic_program_scores")
    program_attention = outputs.get("semantic_program_attention")
    semantic_latent = outputs.get("semantic_latent_embedding") or outputs.get("macro_embeddings")
    cross_region_tokens = outputs.get("cross_region_tokens")
    cross_region_attention = outputs.get("cross_region_attention")
    program_topology = outputs.get("semantic_program_topology")

    # Compute all component losses
    _micro_div = micro_motif_diversity_loss(model.micro_motif_bank)
    _macro_div = macro_motif_diversity_loss(model.semantic_program_bank)

    contrastive_source = semantic_states if semantic_states is not None else semantic_latent
    contrastive_rmask = region_mask if (contrastive_source is not None and len(contrastive_source.shape) == 3) else None
    if contrastive_source is not None:
        _contrastive = region_supervised_contrastive_loss(
            contrastive_source, labels_int, temperature=temperature, region_mask=contrastive_rmask
        )
    else:
        _contrastive = tf.zeros(())

    _sem_consistency = semantic_consistency_loss(semantic_states, labels_int, region_mask=region_mask) if semantic_states is not None else tf.zeros(())
    _compositional = compositional_program_consistency_loss(program_scores, labels_int)
    _disentangle = semantic_disentanglement_loss(semantic_states, region_mask=region_mask) if semantic_states is not None else tf.zeros(())
    _coordination = region_coordination_regularization(routing_weights, interaction_gates, region_mask=region_mask)
    _topology = topology_alignment_loss(interaction_gates, program_topology, labels_int, program_attention=program_attention)
    _comp_contrastive = region_composition_contrastive_loss(cross_region_tokens, labels_int, region_mask=region_mask, temperature=temperature)

    use_entropy = bool(training_cfg.get("use_entropy_sparsity", False))
    _sparsity = semantic_program_sparsity_loss(
        program_attention=program_attention,
        routing_weights=routing_weights,
        cross_region_attention=cross_region_attention,
        mode="entropy" if use_entropy else "l1",
    )
    _prog_diversity = program_diversity_loss(model.semantic_program_bank)

    def _flag(name, default=True):
        return bool(training_cfg.get(name, default))

    total = ce_loss
    if _flag("enable_micro_diversity"):
        total = total + micro_diversity_weight * _micro_div
    if _flag("enable_macro_diversity"):
        total = total + macro_diversity_weight * _macro_div
    if _flag("enable_region_contrastive"):
        total = total + region_contrastive_weight * _contrastive
    if _flag("enable_semantic_consistency"):
        total = total + semantic_consistency_weight * _sem_consistency
    if _flag("enable_compositional_program"):
        total = total + compositional_program_weight * _compositional
    if _flag("enable_semantic_disentanglement"):
        total = total + semantic_disentanglement_weight * _disentangle
    if _flag("enable_region_coordination"):
        total = total + region_coordination_weight * _coordination
    if _flag("enable_topology_alignment"):
        total = total + topology_alignment_weight * _topology
    if _flag("enable_region_composition_contrastive"):
        total = total + region_composition_contrastive_weight * _comp_contrastive
    if _flag("enable_program_sparsity"):
        total = total + program_sparsity_weight * _sparsity
    if _flag("enable_program_diversity"):
        total = total + program_diversity_weight * _prog_diversity
    if _flag("enable_fused_aux_ce", False):
        total = total + fused_aux_ce_weight * fused_ce_loss

    return {
        "loss": total,
        "loss_ce": ce_loss,
        "loss_micro_motif_diversity": _micro_div,
        "loss_macro_motif_diversity": _macro_div,
        "loss_contrastive": _contrastive,
        "loss_semantic_consistency": _sem_consistency,
        "loss_compositional_program_consistency": _compositional,
        "loss_semantic_disentanglement": _disentangle,
        "loss_region_coordination": _coordination,
        "loss_topology_alignment": _topology,
        "loss_region_composition_contrastive": _comp_contrastive,
        "loss_program_sparsity": _sparsity,
        "loss_program_diversity": _prog_diversity,
        "loss_fused_aux_ce": fused_ce_loss,
    }
