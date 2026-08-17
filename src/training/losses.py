import tensorflow as tf
import numpy as np


class SymmetricCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=7, label_smoothing=0.0, weight=None, name='sce'):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.weight = weight  # class weight tensor (num_classes,)

    def call(self, labels, pred):
        """
        Args:
            labels: (B,) int tensor
            pred: (B, num_classes) logit tensor
        """
        # Standard Cross Entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=pred)
        if self.label_smoothing > 0.0:
            # Apply label smoothing manually for SCE compatibility
            one_hot_smooth = tf.one_hot(labels, self.num_classes)
            one_hot_smooth = one_hot_smooth * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            log_probs = tf.nn.log_softmax(pred, axis=-1)
            ce = -tf.reduce_sum(one_hot_smooth * log_probs, axis=-1)
        
        if self.weight is not None:
            sample_weights = tf.gather(self.weight, labels)
            ce = ce * sample_weights
            ce = tf.reduce_sum(ce) / tf.reduce_sum(sample_weights)
        else:
            ce = tf.reduce_mean(ce)

        # Reverse Cross Entropy
        pred_softmax = tf.nn.softmax(pred, axis=-1)
        pred_softmax = tf.clip_by_value(pred_softmax, 1e-7, 1.0)
        
        one_hot = tf.one_hot(labels, self.num_classes)
        if self.label_smoothing > 0.0:
            one_hot = one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
        one_hot = tf.clip_by_value(one_hot, 1e-4, 1.0)
        
        rce_per_sample = -1.0 * tf.reduce_sum(pred_softmax * tf.math.log(one_hot), axis=-1)
        if self.weight is not None:
            sample_weights = tf.gather(self.weight, labels)
            rce = tf.reduce_sum(rce_per_sample * sample_weights) / tf.reduce_sum(sample_weights)
        else:
            rce = tf.reduce_mean(rce_per_sample)

        return self.alpha * ce + self.beta * rce


class MotifConsistencyLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=7, motifs_per_class=8, tau=0.1, margin=0.5, name='motif_consistency'):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.tau = tau
        self.margin = margin

    def call(self, targets, scores, top_k_idx=None):
        """
        Args:
            targets: (B,) labels
            scores: (B, num_candidates, Total_Motifs)
            top_k_idx: (B, top_k) — if None, use all candidates
        """
        B = tf.shape(scores)[0]
        Total_Motifs = tf.shape(scores)[2]
        
        if top_k_idx is not None:
            top_k = tf.shape(top_k_idx)[1]
            batch_idx = tf.tile(
                tf.expand_dims(tf.range(B), 1),
                [1, top_k]
            )
            gather_indices = tf.stack([batch_idx, top_k_idx], axis=-1)
            selected_scores = tf.gather_nd(scores, gather_indices)
        else:
            selected_scores = scores
        
        # Create mask for correct class motifs
        # For each sample, motifs [c*M : (c+1)*M] are positive
        target_start = tf.cast(targets, tf.int32) * self.motifs_per_class
        motif_indices = tf.range(Total_Motifs)
        # mask: (B, Total_Motifs)
        mask = tf.cast(
            (motif_indices[None, :] >= target_start[:, None]) & 
            (motif_indices[None, :] < (target_start[:, None] + self.motifs_per_class)),
            tf.float32
        )
        mask = tf.expand_dims(mask, 1)  # (B, 1, Total_Motifs)
        
        # 1. InfoNCE: log-sum-exp of positive vs all
        pos_scores = tf.where(mask > 0, selected_scores, tf.constant(-1e9))
        log_sum_exp_pos = tf.reduce_logsumexp(pos_scores / self.tau, axis=-1)
        log_sum_exp_all = tf.reduce_logsumexp(selected_scores / self.tau, axis=-1)
        loss_intra = -tf.reduce_mean(log_sum_exp_pos - log_sum_exp_all, axis=1)
        
        # 2. Triplet margin: avg(pos) > avg(neg) + margin
        pos_avg = tf.reduce_sum(selected_scores * mask, axis=-1) / tf.cast(self.motifs_per_class, tf.float32)
        neg_avg = tf.reduce_sum(selected_scores * (1.0 - mask), axis=-1) / tf.cast(
            Total_Motifs - self.motifs_per_class, tf.float32)
        loss_inter = tf.reduce_mean(tf.nn.relu(self.margin + neg_avg - pos_avg), axis=1)
        
        total_loss = loss_intra + loss_inter
        return tf.reduce_mean(total_loss)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name='focal'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha  # per-class weights tensor

    def call(self, targets, inputs):
        """
        Args:
            targets: (B,) int labels
            inputs: (B, C) logits
        """
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=inputs)
        p_t = tf.exp(-ce)
        loss = ((1.0 - p_t) ** self.gamma) * ce
        if self.alpha is not None:
            at = tf.gather(self.alpha, targets)
            loss = at * loss
        return tf.reduce_mean(loss)


class CombinedMotifLoss(tf.keras.losses.Loss):
    def __init__(self, ce_loss, motif_loss, weight, div_weight=0.1, name='combined_motif'):
        super().__init__(name=name)
        self.ce = ce_loss
        self.motif = motif_loss
        self.weight_motif = weight
        self.div_weight = div_weight

    def call(self, targets, logits, scores=None, top_k_idx=None, model=None):
        l_ce = self.ce(targets, logits)
        
        if scores is not None and top_k_idx is not None:
            l_motif = self.motif(targets, scores, top_k_idx)
            loss = l_ce + self.weight_motif * l_motif
        else:
            loss = l_ce
        
        if model is not None and hasattr(model, 'compute_motif_diversity_loss'):
            l_div = model.compute_motif_diversity_loss()
            loss = loss + self.div_weight * l_div
        return loss


def build_loss(config, class_weights=None):
    """Define loss for training, cross_entropy: default.
    
    Args:
        config: all config loaded from yaml
        class_weights: tf.Tensor or None, per-class weights
    """
    loss_name = config['training'].get('loss', 'cross_entropy')

    if loss_name == 'cross_entropy':
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        loss = CrossEntropyWithWeights(
            label_smoothing=label_smoothing,
            class_weights=class_weights
        )

    elif loss_name == 'sce':
        sce_alpha = config['training'].get('sce_alpha', 1.0)
        sce_beta = config['training'].get('sce_beta', 1.0)
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        num_classes = config['model'].get('num_classes', 7)
        loss = SymmetricCrossEntropy(
            alpha=sce_alpha, beta=sce_beta,
            num_classes=num_classes, label_smoothing=label_smoothing,
            weight=class_weights
        )

    elif loss_name == 'focal':
        gamma = config['training'].get('focal_gamma', 2.0)
        alpha = config['training'].get('focal_alpha', None)
        alpha_tensor = None
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        loss = FocalLoss(gamma=gamma, alpha=alpha_tensor)

    elif loss_name == 'motif_combined':
        alpha_weight = config['training'].get('motif_loss_weight', 0.5)
        
        base_loss_name = config['training'].get('base_loss', 'cross_entropy')
        use_sce_base = config['training'].get('use_sce_base', False)
        if base_loss_name == 'sce' or use_sce_base:
            sce_alpha = config['training'].get('sce_alpha', 1.0)
            sce_beta = config['training'].get('sce_beta', 1.0)
            label_smoothing = config['training'].get('label_smoothing', 0.0)
            num_classes = config['model'].get('num_classes', 7)
            ce_loss = SymmetricCrossEntropy(
                alpha=sce_alpha, beta=sce_beta,
                num_classes=num_classes, label_smoothing=label_smoothing,
                weight=class_weights
            )
        else:
            label_smoothing = config['training'].get('label_smoothing', 0.0)
            ce_loss = CrossEntropyWithWeights(
                label_smoothing=label_smoothing,
                class_weights=class_weights
            )
            
        motif_loss = MotifConsistencyLoss(
            num_classes=config['model'].get('num_classes', 7),
            motifs_per_class=config['model'].get('motifs_per_class', 8),
            tau=config['training'].get('motif_tau', 0.1),
            margin=config['training'].get('motif_margin', 0.5)
        )
        
        loss = CombinedMotifLoss(
            ce_loss, motif_loss, alpha_weight,
            div_weight=config['training'].get('motif_div_weight', 0.1)
        )

    elif loss_name == 'semantic_roi_graph':
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        use_sce = config['training'].get('use_sce', False)
        
        if use_sce:
            sce_alpha = config['training'].get('sce_alpha', 1.0)
            sce_beta = config['training'].get('sce_beta', 1.0)
            num_classes = config['model'].get('num_classes', 7)
            loss = SymmetricCrossEntropy(
                alpha=sce_alpha, beta=sce_beta,
                num_classes=num_classes, label_smoothing=label_smoothing,
                weight=class_weights
            )
        else:
            loss = CrossEntropyWithWeights(
                label_smoothing=label_smoothing,
                class_weights=class_weights
            )

    else:
        raise ValueError(f"\n[!!!] Not support {loss_name} loss!\n")

    return loss


class CrossEntropyWithWeights(tf.keras.losses.Loss):
    """Cross-entropy loss with optional class weights and label smoothing."""
    
    def __init__(self, label_smoothing=0.0, class_weights=None, name='ce_weighted'):
        super().__init__(name=name)
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
    
    def call(self, labels, logits):
        """
        Args:
            labels: (B,) int tensor
            logits: (B, C) float tensor
        """
        if self.label_smoothing > 0.0:
            num_classes = tf.shape(logits)[-1]
            one_hot = tf.one_hot(labels, num_classes)
            one_hot = one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / tf.cast(num_classes, tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            per_sample = -tf.reduce_sum(one_hot * log_probs, axis=-1)
        else:
            per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        if self.class_weights is not None:
            sample_weights = tf.gather(self.class_weights, labels)
            return tf.reduce_sum(per_sample * sample_weights) / tf.reduce_sum(sample_weights)
        else:
            return tf.reduce_mean(per_sample)


if __name__ == "__main__":
    # Test block
    config_default = {'training': {}, 'model': {'num_classes': 7}}
    loss_fn = build_loss(config_default)
    print(f"Test 1 (Default): {type(loss_fn)}")

    config_explicit = {'training': {'loss': 'cross_entropy'}, 'model': {'num_classes': 7}}
    loss_fn = build_loss(config_explicit)
    print(f"Test 2 (Explicit): {type(loss_fn)}")

    config_sce = {'training': {'loss': 'sce', 'sce_alpha': 1.0, 'sce_beta': 1.0}, 'model': {'num_classes': 7}}
    loss_fn = build_loss(config_sce)
    print(f"Test 3 (SCE standalone): {type(loss_fn)}")
    
    dummy_logits = tf.random.normal((4, 7))
    dummy_targets = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    loss_val = loss_fn(dummy_targets, dummy_logits)
    print(f"Test 3 loss val: {loss_val.numpy():.4f}")