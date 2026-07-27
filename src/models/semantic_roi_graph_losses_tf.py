"""
semantic_roi_graph_losses_tf.py — TensorFlow port of semantic_roi_graph_losses.py.

Translated directly from PyTorch source, function by function.
All operations are Graph-Mode and mixed_float16 safe.
"""

from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_training_cfg(model) -> Dict:
    try:
        return getattr(model, "training_cfg", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# micro_motif_diversity_loss — mirrors PyTorch exactly
# ---------------------------------------------------------------------------

def micro_motif_diversity_loss(motif_bank: tf.Tensor) -> tf.Tensor:
    """Encourage diverse motifs within each region bank. (R, K, D)"""
    motifs = tf.nn.l2_normalize(tf.cast(motif_bank, tf.float32), axis=-1)  # (R, K, D)
    sim = tf.einsum("rkd,rgd->rkg", motifs, motifs)    # (R, K, K)
    k = motifs.shape[1] or tf.shape(motifs)[1]
    identity = tf.eye(k, dtype=sim.dtype)[tf.newaxis]   # (1, K, K)
    off_diag = sim * (1.0 - identity)
    return tf.reduce_mean(off_diag ** 2)


# ---------------------------------------------------------------------------
# macro_motif_diversity_loss — mirrors PyTorch exactly
# ---------------------------------------------------------------------------

def macro_motif_diversity_loss(program_bank: tf.Tensor) -> tf.Tensor:
    """Encourage diverse programs. program_bank: (C, M, R, D)"""
    program_bank = tf.cast(program_bank, tf.float32)
    if len(program_bank.shape) == 3:
        c, m, d = program_bank.shape
        motifs = tf.reshape(program_bank, [c, m, d])
    else:
        c, m, r, d = program_bank.shape
        motifs = tf.reshape(program_bank, [c, m, r * d])
    motifs = tf.nn.l2_normalize(motifs, axis=-1)
    sim = tf.einsum("cmd,cnd->cmn", motifs, motifs)  # (C, M, M)
    m_dim = motifs.shape[1] or tf.shape(motifs)[1]
    identity = tf.eye(m_dim, dtype=sim.dtype)[tf.newaxis]
    off_diag = tf.nn.relu(tf.abs(sim) - 0.3) * (1.0 - identity)
    return tf.reduce_mean(off_diag ** 2)


# ---------------------------------------------------------------------------
# program_diversity_loss — mirrors PyTorch
# ---------------------------------------------------------------------------

def program_diversity_loss(program_bank: tf.Tensor) -> tf.Tensor:
    """program_bank: (C, M, R, D)"""
    if isinstance(program_bank, tuple):
        program_bank = program_bank[0]
    program_bank = tf.cast(program_bank, tf.float32)
    if len(program_bank.shape) == 4:
        summaries = tf.reduce_mean(program_bank, axis=2)  # (C, M, D)
    else:
        summaries = program_bank
    # Static shape: C and M are always known at build time (num_classes, programs_per_class)
    # For safety, use tf.shape for the flattened dimension
    cm = tf.shape(summaries)[0] * tf.shape(summaries)[1]
    d = tf.shape(summaries)[2]
    summaries_flat = tf.reshape(summaries, [cm, d])
    summaries_flat = tf.nn.l2_normalize(summaries_flat, axis=-1)
    sim = tf.matmul(summaries_flat, tf.transpose(summaries_flat))  # (C*M, C*M)
    identity = tf.eye(cm, dtype=sim.dtype)
    off_diag = tf.nn.relu(tf.abs(sim) - 0.3) * (1.0 - identity)
    return tf.reduce_mean(off_diag ** 2)


# ---------------------------------------------------------------------------
# compositional_program_consistency_loss — mirrors PyTorch
# ---------------------------------------------------------------------------

def compositional_program_consistency_loss(
    program_scores: Optional[tf.Tensor], labels: tf.Tensor
) -> tf.Tensor:
    """Cross-entropy on program_scores (B, C) vs labels."""
    if program_scores is None:
        return tf.zeros((), dtype=tf.float32)
    labels_int = tf.cast(labels, tf.int32)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_int, logits=tf.cast(program_scores, tf.float32)
        )
    )


# ---------------------------------------------------------------------------
# semantic_consistency_loss — REWRITTEN for Graph Mode safety
# ---------------------------------------------------------------------------

def semantic_consistency_loss(
    semantic_states: Optional[tf.Tensor],
    labels: tf.Tensor,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Intra-class variance loss — vectorized, no tf.boolean_mask or tf.range loops.

    For each class c present in the batch, compute the variance of pooled
    semantic states belonging to class c, then average across classes.

    Vectorized approach:
    1. Pool semantic_states to (B, D)
    2. Build one-hot class membership (B, C)
    3. Compute per-class mean via matmul
    4. Gather each sample's class mean
    5. MSE between sample and its class mean, averaged
    """
    if semantic_states is None:
        return tf.zeros((), dtype=tf.float32)

    states = tf.cast(semantic_states, tf.float32)

    if len(states.shape) == 3:
        if region_mask is not None:
            weights = tf.cast(region_mask[..., tf.newaxis], tf.float32)
            pooled = tf.reduce_sum(states * weights, axis=1) / \
                     tf.maximum(tf.reduce_sum(weights, axis=1), 1.0)
        else:
            pooled = tf.reduce_mean(states, axis=1)
    else:
        pooled = states  # already (B, D)

    labels_flat = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    B = tf.shape(pooled)[0]

    # One-hot: (B, C) where C = max_label + 1
    num_classes = tf.reduce_max(labels_flat) + 1
    one_hot = tf.one_hot(labels_flat, num_classes, dtype=tf.float32)  # (B, C)

    # Count per class: (C,)
    counts = tf.reduce_sum(one_hot, axis=0)  # (C,)

    # Per-class sum of pooled states: (C, D)
    class_sums = tf.matmul(tf.transpose(one_hot), pooled)  # (C, D)

    # Per-class mean: (C, D) — safe divide
    class_means = class_sums / tf.maximum(counts[:, tf.newaxis], 1.0)

    # Gather each sample's class mean: (B, D)
    sample_means = tf.gather(class_means, labels_flat)  # (B, D)

    # Per-sample squared deviation
    deviations = (pooled - sample_means) ** 2  # (B, D)

    # Only count classes with >= 2 samples
    # Mask: (C,) -> gather to (B,) -> mask
    valid_class_mask = tf.cast(counts >= 2.0, tf.float32)  # (C,)
    sample_valid = tf.gather(valid_class_mask, labels_flat)  # (B,)

    # Weighted mean
    weighted_dev = tf.reduce_mean(deviations, axis=-1) * sample_valid  # (B,)
    total_valid = tf.reduce_sum(sample_valid)

    # Return 0 if no valid classes
    return tf.where(
        tf.cast(total_valid > 0.0, tf.bool),
        tf.reduce_sum(weighted_dev) / total_valid,
        tf.zeros((), dtype=tf.float32),
    )


# ---------------------------------------------------------------------------
# semantic_disentanglement_loss — Graph Mode safe
# ---------------------------------------------------------------------------

def semantic_disentanglement_loss(
    semantic_states: Optional[tf.Tensor],
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    if semantic_states is None:
        return tf.zeros((), dtype=tf.float32)

    states = tf.cast(semantic_states, tf.float32)

    if len(states.shape) == 3:
        if region_mask is not None:
            flat_mask = tf.reshape(tf.cast(region_mask, tf.bool), [-1])
            tokens = tf.boolean_mask(tf.reshape(states, [-1, tf.shape(states)[-1]]), flat_mask)
        else:
            tokens = tf.reshape(states, [-1, tf.shape(states)[-1]])
    else:
        tokens = tf.reshape(states, [-1, tf.shape(states)[-1]])

    # Use tf.where instead of Python if on tensor for the < 2 guard
    n_tokens = tf.shape(tokens)[0]

    def _compute():
        centered = tokens - tf.reduce_mean(tokens, axis=0, keepdims=True)
        n_f = tf.cast(tf.maximum(n_tokens - 1, 1), tf.float32)
        cov = tf.matmul(tf.transpose(centered), centered) / n_f
        diag = tf.linalg.diag(tf.linalg.diag_part(cov))
        off_diag = cov - diag
        return tf.reduce_mean(off_diag ** 2)

    return tf.cond(
        tf.cast(n_tokens >= 2, tf.bool),
        _compute,
        lambda: tf.zeros((), dtype=tf.float32),
    )


# ---------------------------------------------------------------------------
# region_coordination_regularization — mirrors PyTorch
# ---------------------------------------------------------------------------

def region_coordination_regularization(
    routing_weights: Optional[tf.Tensor],
    interaction_gates: Optional[tf.Tensor] = None,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    loss = None

    if routing_weights is not None:
        weights = tf.maximum(tf.cast(routing_weights, tf.float32), 1e-6)
        entropy = -tf.reduce_sum(weights * tf.math.log(weights), axis=1)
        k = tf.cast(tf.shape(weights)[1], tf.float32)
        denom = tf.maximum(tf.math.log(k), 1e-6)
        loss = tf.reduce_mean(entropy / denom)

    if interaction_gates is not None:
        gates = tf.cast(interaction_gates, tf.float32)
        if region_mask is not None:
            pair_mask = tf.cast(region_mask[:, :, tf.newaxis], gates.dtype) * \
                        tf.cast(region_mask[:, tf.newaxis, :], gates.dtype)
            gates = gates * pair_mask
        active_mean = tf.reduce_mean(gates, axis=[-2, -1])
        gate_balance = tf.reduce_mean((active_mean - 0.6) ** 2)
        gate_variance = tf.reduce_mean(tf.math.reduce_variance(gates, axis=[-2, -1]))
        gate_loss = gate_balance + 0.05 * gate_variance
        loss = gate_loss if loss is None else loss + gate_loss

    if loss is None:
        return tf.zeros((), dtype=tf.float32)
    return loss


# ---------------------------------------------------------------------------
# region_supervised_contrastive_loss — mirrors PyTorch
# ---------------------------------------------------------------------------

def region_supervised_contrastive_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    temperature: float = 0.07,
    region_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    embeddings = tf.cast(embeddings, tf.float32)
    if len(embeddings.shape) == 3:
        if region_mask is not None:
            w = tf.cast(region_mask[..., tf.newaxis], embeddings.dtype)
            pooled = tf.reduce_sum(embeddings * w, axis=1) / tf.maximum(tf.reduce_sum(w, axis=1), 1.0)
        else:
            pooled = tf.reduce_mean(embeddings, axis=1)
    else:
        pooled = embeddings

    pooled = tf.nn.l2_normalize(pooled, axis=-1)
    sim = tf.matmul(pooled, tf.transpose(pooled)) / float(temperature)

    labels_flat = tf.reshape(tf.cast(labels, tf.int32), [-1, 1])
    mask = tf.cast(tf.equal(labels_flat, tf.transpose(labels_flat)), tf.float32)
    b = tf.shape(pooled)[0]
    eye = tf.eye(b, dtype=tf.float32)
    logits_mask = 1.0 - eye
    mask = mask * logits_mask

    exp_sim = tf.exp(sim) * logits_mask
    log_prob = sim - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8)
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)
    return -tf.reduce_mean(mean_log_prob_pos)


# ---------------------------------------------------------------------------
# semantic_program_sparsity_loss — mirrors PyTorch
# ---------------------------------------------------------------------------

def semantic_program_sparsity_loss(
    program_attention: Optional[tf.Tensor] = None,
    routing_weights: Optional[tf.Tensor] = None,
    cross_region_attention: Optional[tf.Tensor] = None,
    mode: str = "l1",
) -> tf.Tensor:
    losses = []

    def _entropy(attn):
        attn = tf.maximum(tf.cast(attn, tf.float32), 1e-6)
        entropy = -tf.reduce_sum(attn * tf.math.log(attn), axis=-1)
        k = tf.cast(tf.shape(attn)[-1], tf.float32)
        denom = tf.maximum(tf.math.log(k), 1e-6)
        return tf.reduce_mean(entropy / denom)

    def _l1(attn):
        return -tf.reduce_mean(tf.cast(attn, tf.float32) ** 2)

    if program_attention is not None:
        losses.append(tf.cast(program_attention, tf.float32))
    if cross_region_attention is not None:
        attn = tf.cast(cross_region_attention, tf.float32)
        if len(attn.shape) == 4:
            attn = tf.reduce_mean(attn, axis=1)
        losses.append(attn)

    if not losses:
        return tf.zeros((), dtype=tf.float32)

    if mode == "entropy":
        vals = [_entropy(x) for x in losses]
        return -tf.add_n(vals) / float(len(vals))
    else:
        vals = [_l1(x) for x in losses]
        return tf.add_n(vals) / float(len(vals))


# ---------------------------------------------------------------------------
# topology_alignment_loss — mirrors PyTorch
# ---------------------------------------------------------------------------

def topology_alignment_loss(
    predicted_topology: Optional[tf.Tensor],
    program_topology: Optional[tf.Tensor],
    labels: tf.Tensor,
    program_attention: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    if predicted_topology is None or program_topology is None:
        return tf.zeros((), dtype=tf.float32)
    labels_int = tf.cast(labels, tf.int32)
    selected_topology = tf.gather(program_topology, labels_int)  # (B, M, R, R)
    if program_attention is not None:
        # program_attention is (B, C, M). labels_int is (B,).
        # We want to select the attention for the target class for each batch item -> (B, M)
        sel_attn = tf.gather(program_attention, labels_int, axis=1, batch_dims=1)

        # sel_attn is now (B, M)
        # selected_topology is (B, M, R, R)
        selected_topology = tf.reduce_sum(
            sel_attn[:, :, tf.newaxis, tf.newaxis] * selected_topology, axis=1
        )
    else:
        selected_topology = tf.reduce_mean(selected_topology, axis=1)

    pred = tf.cast(predicted_topology, tf.float32)
    if len(pred.shape) == 4:
        pred = tf.reduce_mean(pred, axis=1)

    return tf.reduce_mean((pred - tf.cast(selected_topology, tf.float32)) ** 2)


# ---------------------------------------------------------------------------
# compute_semantic_roi_graph_losses — mirrors PyTorch compute_semantic_roi_graph_losses
# ---------------------------------------------------------------------------

def compute_semantic_roi_graph_losses_tf(
    model,
    outputs: Dict,
    labels: tf.Tensor,
    class_weights: Optional[tf.Tensor] = None,
) -> Dict:
    """
    Full loss computation — direct TF translation of PyTorch
    compute_semantic_roi_graph_losses().
    """
    training_cfg = _get_training_cfg(model)

    # --- Read weights from config (same defaults as PyTorch) ---
    temperature           = float(training_cfg.get("contrastive_temperature", 0.15))
    region_contrastive_w  = float(training_cfg.get("region_contrastive_weight",
                                   training_cfg.get("au_contrastive_weight", 0.1)))
    micro_diversity_w     = float(training_cfg.get("micro_motif_diversity_weight",
                                   training_cfg.get("motif_diversity_weight", 0.05)))
    macro_diversity_w     = float(training_cfg.get("macro_motif_diversity_weight",
                                   training_cfg.get("motif_diversity_weight", 0.05))) * 2.0
    semantic_consistency_w = float(training_cfg.get("semantic_consistency_weight",
                                   training_cfg.get("region_consistency_weight", 0.1)))
    compositional_program_w = float(training_cfg.get("compositional_program_weight", 0.1))
    semantic_disentangle_w  = float(training_cfg.get("semantic_disentanglement_weight",
                                    training_cfg.get("topology_reg_weight", 0.01)))
    region_coord_w          = float(training_cfg.get("region_coordination_weight",
                                    training_cfg.get("relation_consistency_weight", 0.1)))
    topology_align_w        = float(training_cfg.get("topology_alignment_weight", 0.05))
    program_sparsity_w      = float(training_cfg.get("program_sparsity_weight", 0.05))
    program_diversity_w     = float(training_cfg.get("program_diversity_weight", 0.05))
    fused_aux_ce_w          = float(training_cfg.get("fused_aux_ce_weight", 0.0))
    label_smoothing         = float(training_cfg.get("label_smoothing", 0.0))

    def _flag(name: str, default: bool = True) -> bool:
        return bool(training_cfg.get(name, default))

    # --- Main CE loss ---
    logits = tf.cast(outputs["logits"], tf.float32)
    labels_int = tf.cast(labels, tf.int32)
    num_classes = tf.shape(logits)[-1]

    if label_smoothing > 0.0:
        one_hot = tf.one_hot(labels_int, num_classes, dtype=tf.float32)
        smooth_labels = one_hot * (1.0 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
        per_sample_ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=smooth_labels, logits=logits
        )
    else:
        per_sample_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_int, logits=logits
        )

    if class_weights is not None:
        w = tf.cast(tf.gather(class_weights, labels_int), tf.float32)
        ce_loss = tf.reduce_sum(w * per_sample_ce) / (tf.reduce_sum(w) + 1e-8)
    else:
        ce_loss = tf.reduce_mean(per_sample_ce)

    # --- Fused branch aux CE ---
    logits_fused = outputs.get("logits_fused")
    if logits_fused is not None:
        logits_fused = tf.cast(logits_fused, tf.float32)
        if label_smoothing > 0.0:
            nc_fused = tf.shape(logits_fused)[-1]
            one_hot = tf.one_hot(labels_int, nc_fused, dtype=tf.float32)
            smooth_labels = one_hot * (1.0 - label_smoothing) + label_smoothing / tf.cast(nc_fused, tf.float32)
            fused_per = tf.nn.softmax_cross_entropy_with_logits(labels=smooth_labels, logits=logits_fused)
        else:
            fused_per = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_int, logits=logits_fused)
        if class_weights is not None:
            w = tf.cast(tf.gather(class_weights, labels_int), tf.float32)
            fused_ce_loss = tf.reduce_sum(w * fused_per) / (tf.reduce_sum(w) + 1e-8)
        else:
            fused_ce_loss = tf.reduce_mean(fused_per)
    else:
        fused_ce_loss = tf.zeros((), dtype=tf.float32)

    # --- Auxiliary losses ---
    semantic_states    = outputs.get("semantic_state_tokens") or outputs.get("region_embeddings")
    region_mask        = outputs.get("region_mask")
    routing_weights    = outputs.get("semantic_routing_weights")
    interaction_gates  = outputs.get("semantic_interaction_gates")
    program_scores     = outputs.get("semantic_program_scores")
    program_attention  = outputs.get("semantic_program_attention")
    semantic_latent    = outputs.get("semantic_latent_embedding") or outputs.get("macro_embeddings")
    cross_region_tokens = outputs.get("cross_region_tokens")
    cross_region_attn  = outputs.get("cross_region_attention")
    prog_topology      = outputs.get("semantic_program_topology")
    prog_bank          = outputs.get("semantic_program_bank") or model.semantic_program_bank.programs

    micro_div_loss  = micro_motif_diversity_loss(model.micro_motif_bank.motifs)
    macro_div_loss  = macro_motif_diversity_loss(prog_bank)
    prog_div_loss   = program_diversity_loss(prog_bank)

    contrastive_source = semantic_states if semantic_states is not None else semantic_latent
    contrastive_region_mask = region_mask if (contrastive_source is not None and
                                               len(contrastive_source.shape) == 3) else None
    if contrastive_source is not None:
        contrastive_loss = region_supervised_contrastive_loss(
            contrastive_source, labels, temperature=temperature,
            region_mask=contrastive_region_mask,
        )
    else:
        contrastive_loss = tf.zeros((), dtype=tf.float32)

    semantic_cons   = semantic_consistency_loss(semantic_states, labels, region_mask=region_mask)
    comp_loss       = compositional_program_consistency_loss(program_scores, labels)
    disen_loss      = semantic_disentanglement_loss(semantic_states, region_mask=region_mask)
    coord_loss      = region_coordination_regularization(routing_weights, interaction_gates, region_mask=region_mask)
    topo_loss       = topology_alignment_loss(interaction_gates, prog_topology, labels, program_attention=program_attention)

    use_entropy = bool(training_cfg.get("use_entropy_sparsity", False))
    sparsity_mode = "entropy" if use_entropy else "l1"
    sparsity_loss = semantic_program_sparsity_loss(
        program_attention=program_attention,
        routing_weights=routing_weights,
        cross_region_attention=cross_region_attn,
        mode=sparsity_mode,
    )

    # --- Combine losses ---
    total = tf.cast(ce_loss, tf.float32)

    if _flag("enable_micro_diversity"):
        total = total + micro_diversity_w * tf.cast(micro_div_loss, tf.float32)
    if _flag("enable_macro_diversity"):
        total = total + macro_diversity_w * tf.cast(macro_div_loss, tf.float32)
    if _flag("enable_region_contrastive"):
        total = total + region_contrastive_w * tf.cast(contrastive_loss, tf.float32)
    if _flag("enable_semantic_consistency"):
        total = total + semantic_consistency_w * tf.cast(semantic_cons, tf.float32)
    if _flag("enable_compositional_program"):
        total = total + compositional_program_w * tf.cast(comp_loss, tf.float32)
    if _flag("enable_semantic_disentanglement"):
        total = total + semantic_disentangle_w * tf.cast(disen_loss, tf.float32)
    if _flag("enable_region_coordination"):
        total = total + region_coord_w * tf.cast(coord_loss, tf.float32)
    if _flag("enable_topology_alignment"):
        total = total + topology_align_w * tf.cast(topo_loss, tf.float32)
    if _flag("enable_program_sparsity"):
        total = total + program_sparsity_w * tf.cast(sparsity_loss, tf.float32)
    if _flag("enable_program_diversity"):
        total = total + program_diversity_w * tf.cast(prog_div_loss, tf.float32)
    if _flag("enable_fused_aux_ce", False):
        total = total + fused_aux_ce_w * tf.cast(fused_ce_loss, tf.float32)

    return {
        "loss":                          total,
        "loss_ce":                       ce_loss,
        "loss_micro_motif_diversity":    micro_div_loss,
        "loss_macro_motif_diversity":    macro_div_loss,
        "loss_contrastive":              contrastive_loss,
        "loss_semantic_consistency":     semantic_cons,
        "loss_compositional_program":    comp_loss,
        "loss_semantic_disentanglement": disen_loss,
        "loss_region_coordination":      coord_loss,
        "loss_topology_alignment":       topo_loss,
        "loss_program_sparsity":         sparsity_loss,
        "loss_program_diversity":        prog_div_loss,
        "loss_fused_aux_ce":             fused_ce_loss,
    }
