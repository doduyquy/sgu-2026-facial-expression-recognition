"""
losses_tf.py — TensorFlow loss functions for Semantic ROI Graph FER.
Includes:
- Weighted Categorical Cross-Entropy with Label Smoothing (0.1).
- Auxiliary Fused Branch Cross-Entropy.
- Micro Motif Diversity Loss.
- Program Diversity Loss (with margin 0.3).
- Semantic Consistency & Supervised Contrastive Loss.
"""

from typing import Dict, Optional
import tensorflow as tf


def compute_class_weights_sqrt_inverse(class_counts: list) -> tf.Tensor:
    """Compute sqrt_inverse class weights normalized to mean=1.0."""
    counts = tf.constant(class_counts, dtype=tf.float32)
    total = tf.reduce_sum(counts)
    weights = tf.sqrt(total / (counts + 1e-6))
    weights = weights / tf.reduce_mean(weights)
    return weights


def micro_motif_diversity_loss(motif_bank: tf.Tensor) -> tf.Tensor:
    """Encourage diverse motifs within each facial region bank."""
    # motif_bank: (R, K, D)
    motifs_norm = tf.math.l2_normalize(motif_bank, axis=-1)
    # Cosine similarity matrix within each region: (R, K, K)
    sim = tf.matmul(motifs_norm, motifs_norm, transpose_b=True)
    k = tf.shape(sim)[1]
    identity = tf.eye(k, batch_shape=[tf.shape(sim)[0]])
    off_diag = sim * (1.0 - identity)
    return tf.reduce_mean(tf.square(off_diag))


def program_diversity_loss(programs: tf.Tensor) -> tf.Tensor:
    """Encourage diverse semantic programs per emotion class with margin=0.3."""
    # programs: (C, M, R, D) -> reshape to (C, M, R * D)
    c = tf.shape(programs)[0]
    m = tf.shape(programs)[1]
    flat_progs = tf.reshape(programs, [c, m, -1])
    progs_norm = tf.math.l2_normalize(flat_progs, axis=-1)

    sim = tf.matmul(progs_norm, progs_norm, transpose_b=True) # (C, M, M)
    identity = tf.eye(m, batch_shape=[c])
    # Margin 0.3 avoids forcing 100% orthogonality which conflicts with classification
    off_diag = tf.nn.relu(tf.abs(sim) - 0.3) * (1.0 - identity)
    return tf.reduce_mean(tf.square(off_diag))


def semantic_consistency_loss(latent_embeddings: tf.Tensor, labels: tf.Tensor, num_classes: int = 7) -> tf.Tensor:
    """Pull latent representations of the same emotion class closer to their batch centroid."""
    labels = tf.cast(labels, tf.int32)
    total_loss = tf.constant(0.0, dtype=tf.float32)
    valid_classes = tf.constant(0.0, dtype=tf.float32)

    for c in range(num_classes):
        mask = tf.equal(labels, c)
        class_samples = tf.boolean_mask(latent_embeddings, mask)
        n = tf.shape(class_samples)[0]
        if n > 1:
            centroid = tf.reduce_mean(class_samples, axis=0, keepdims=True)
            dist = tf.reduce_mean(tf.reduce_sum(tf.square(class_samples - centroid), axis=-1))
            total_loss = total_loss + dist
            valid_classes = valid_classes + 1.0

    return tf.where(valid_classes > 0.0, total_loss / valid_classes, 0.0)


def compute_semantic_roi_graph_losses_tf(
    model: tf.keras.Model,
    outputs: Dict[str, tf.Tensor],
    labels: tf.Tensor,
    class_weights: Optional[tf.Tensor] = None,
    label_smoothing: float = 0.1,
    train_cfg: Optional[dict] = None,
) -> Dict[str, tf.Tensor]:
    """Compute combined classification and auxiliary graph losses in TensorFlow."""
    train_cfg = train_cfg or {}
    logits = outputs["logits"]
    num_classes = tf.shape(logits)[-1]

    # Convert integer labels to one-hot for smooth cross entropy
    one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)

    # 1. Primary Classification Loss with Label Smoothing & Manifold Mixup support
    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=label_smoothing,
        reduction=tf.keras.losses.Reduction.NONE
    )

    mixup_lam = outputs.get("mixup_lam", None)
    mixup_perm = outputs.get("mixup_perm", None)

    if mixup_perm is not None and mixup_lam is not None:
        labels_b = tf.gather(labels, mixup_perm)
        one_hot_b = tf.one_hot(tf.cast(labels_b, tf.int32), depth=num_classes)

        per_sample_a = ce_loss_fn(one_hot_labels, logits)
        per_sample_b = ce_loss_fn(one_hot_b, logits)

        if class_weights is not None:
            weights_a = tf.gather(class_weights, tf.cast(labels, tf.int32))
            weights_b = tf.gather(class_weights, tf.cast(labels_b, tf.int32))
            loss_a = tf.reduce_mean(per_sample_a * weights_a)
            loss_b = tf.reduce_mean(per_sample_b * weights_b)
        else:
            loss_a = tf.reduce_mean(per_sample_a)
            loss_b = tf.reduce_mean(per_sample_b)

        loss_ce = mixup_lam * loss_a + (1.0 - mixup_lam) * loss_b
        total_loss = loss_ce
        loss_dict = {"loss_ce": loss_ce}

        # 2. Auxiliary Fused Branch Cross-Entropy with Mixup
        if train_cfg.get("enable_fused_aux_ce", True) and "logits_fused" in outputs:
            fused_a = tf.reduce_mean(ce_loss_fn(one_hot_labels, outputs["logits_fused"]))
            fused_b = tf.reduce_mean(ce_loss_fn(one_hot_b, outputs["logits_fused"]))
            loss_fused = mixup_lam * fused_a + (1.0 - mixup_lam) * fused_b
            w_fused = float(train_cfg.get("fused_aux_ce_weight", 0.2))
            total_loss = total_loss + w_fused * loss_fused
            loss_dict["loss_fused"] = loss_fused

        # 3. Compositional Program Branch Cross-Entropy with Mixup
        if train_cfg.get("enable_compositional_program", True) and "logits_motif" in outputs:
            prog_a = tf.reduce_mean(ce_loss_fn(one_hot_labels, outputs["logits_motif"]))
            prog_b = tf.reduce_mean(ce_loss_fn(one_hot_b, outputs["logits_motif"]))
            loss_prog = mixup_lam * prog_a + (1.0 - mixup_lam) * prog_b
            w_prog = float(train_cfg.get("compositional_program_weight", 0.02))
            total_loss = total_loss + w_prog * loss_prog
            loss_dict["loss_prog_ce"] = loss_prog
    else:
        per_sample_loss = ce_loss_fn(one_hot_labels, logits)
        if class_weights is not None:
            sample_weights = tf.gather(class_weights, tf.cast(labels, tf.int32))
            loss_ce = tf.reduce_mean(per_sample_loss * sample_weights)
        else:
            loss_ce = tf.reduce_mean(per_sample_loss)

        total_loss = loss_ce
        loss_dict = {"loss_ce": loss_ce}

        # 2. Auxiliary Fused Branch Cross-Entropy
        if train_cfg.get("enable_fused_aux_ce", True) and "logits_fused" in outputs:
            loss_fused = tf.reduce_mean(ce_loss_fn(one_hot_labels, outputs["logits_fused"]))
            w_fused = float(train_cfg.get("fused_aux_ce_weight", 0.2))
            total_loss = total_loss + w_fused * loss_fused
            loss_dict["loss_fused"] = loss_fused

        # 3. Compositional Program Branch Cross-Entropy
        if train_cfg.get("enable_compositional_program", True) and "logits_motif" in outputs:
            loss_prog = tf.reduce_mean(ce_loss_fn(one_hot_labels, outputs["logits_motif"]))
            w_prog = float(train_cfg.get("compositional_program_weight", 0.02))
            total_loss = total_loss + w_prog * loss_prog
            loss_dict["loss_prog_ce"] = loss_prog

    # 4. Micro Motif Diversity Loss
    if train_cfg.get("enable_micro_diversity", True) and hasattr(model, "motif_matcher"):
        loss_micro = micro_motif_diversity_loss(model.motif_matcher.motif_bank)
        w_micro = float(train_cfg.get("micro_motif_diversity_weight", 0.02))
        total_loss = total_loss + w_micro * loss_micro
        loss_dict["loss_micro_div"] = loss_micro

    # 5. Program Diversity Loss
    if train_cfg.get("enable_program_diversity", True) and hasattr(model, "program_executor"):
        loss_prog_div = program_diversity_loss(model.program_executor.programs)
        w_prog_div = float(train_cfg.get("program_diversity_weight", 0.01))
        total_loss = total_loss + w_prog_div * loss_prog_div
        loss_dict["loss_prog_div"] = loss_prog_div

    # 6. Semantic Consistency Loss (only if enabled)
    if train_cfg.get("enable_semantic_consistency", False) and "emotion_latent" in outputs:
        loss_consist = semantic_consistency_loss(outputs["emotion_latent"], labels, num_classes=num_classes)
        w_consist = float(train_cfg.get("semantic_consistency_weight", 0.03))
        total_loss = total_loss + w_consist * loss_consist
        loss_dict["loss_consist"] = loss_consist

    loss_dict["loss"] = total_loss
    return loss_dict
