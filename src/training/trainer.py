"""
trainer_tf.py — TensorFlow/Keras training loop cho SemanticROIGraphFER.

Thay thế PyTorch Trainer bằng Keras GradientTape-based custom training loop.
Hỗ trợ:
- Multi-GPU qua tf.distribute.MirroredStrategy
- Mixed Precision (LossScaleOptimizer)
- Graph Mode (@tf.function) để tăng tốc độ x5-x10
- Multi-loss (via compute_semantic_roi_graph_losses)
- Mixup
- SCN-light (sample weighting by confidence)
- W&B logging
- Early stopping
- EMA (Exponential Moving Average) weights
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from src.models.semantic_roi_graph_losses import compute_semantic_roi_graph_losses


def _spce(labels, logits):
    """Sparse categorical cross-entropy per-sample."""
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def _lr_multiplier_for_name(name: str) -> float:
    if "backbone" in name:
        return 1.0
    return 2.0


class Trainer:
    """TensorFlow GradientTape training loop for SemanticROIGraphFER."""

    def __init__(
        self,
        model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        scheduler=None,
        config: dict = None,
        run_name: str = "run",
        save_path: str = "best_model.weights.h5",
        strategy: tf.distribute.Strategy = None,
        class_weights: Optional[tf.Tensor] = None,
    ):
        self.strategy = strategy or tf.distribute.get_strategy()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.run_name = run_name
        self.save_path = save_path
        self.class_weights = class_weights

        # Distribute datasets
        self.train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        self.val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)

        train_cfg = self.config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 100))
        self.patience = int(train_cfg.get("patience", 20))
        self.model_name = self.config.get("model", {}).get("name", "semantic_roi_graph_fer")
        self.use_wandb = self.config.get("logging", {}).get("use_wandb", False)

        # SCN params
        self.use_scn = bool(train_cfg.get("use_scn", True))
        self.scn_warmup_epochs = int(train_cfg.get("scn_warmup_epochs", 0))
        self.scn_alpha = float(train_cfg.get("scn_alpha", 1.0))
        self.scn_rank_lambda = float(train_cfg.get("scn_rank_lambda", 0.5))
        self.scn_min_weight = float(train_cfg.get("scn_min_weight", 0.2))
        self.scn_margin = float(train_cfg.get("scn_margin", 0.6))
        self.mixup_alpha = float(train_cfg.get("mixup_alpha", 0.2))
        self.ema_decay = 0.999

        # EMA variables — initialized as None, will be populated in fit() BEFORE
        # any @tf.function is traced so the graph captures the real list.
        self._ema_initialized = False
        self.ema_vars = []

        # Mixed Precision flag (for logging only — we do NOT use LossScaleOptimizer)
        # LossScaleOptimizer creates internal tf.cond nodes that crash under
        # Keras 3 + mixed_float16 + MirroredStrategy. Instead we handle
        # gradients manually: cast all grads to float32 before processing.
        self.use_mixed_precision = (
            tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        )
        
        with self.strategy.scope():
            # Run a dummy batch to fully build all dynamic model layers (like Sequential/Dense)
            # BEFORE initializing the optimizer slot variables.
            for dummy_batch in train_dataset.take(1):
                try:
                    self.model(
                        image=dummy_batch["image"],
                        bboxes=dummy_batch.get("bboxes", None),
                        region_mask=dummy_batch.get("region_mask", None),
                        region_confidence=dummy_batch.get("region_confidence", None),
                        training=False
                    )
                except Exception as e:
                    print(f"Warning: Eager dummy pass failed: {e}")
                break
                
            # Eagerly initialize optimizer variables to prevent them from being created
            # lazily inside tf.cond during the first @tf.function apply_gradients call.
            if hasattr(self.optimizer, "build"):
                self.optimizer.build(self.model.trainable_variables)
            elif hasattr(self.optimizer, "_create_all_weights"):
                self.optimizer._create_all_weights(self.model.trainable_variables)

    # ------------------------------------------------------------------
    # SCN Light
    # ------------------------------------------------------------------

    def _scn_loss(self, logits: tf.Tensor, labels: tf.Tensor, epoch_tensor: tf.Tensor):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int32)
        ce = _spce(labels, logits)  # (B,)

        probs = tf.nn.softmax(logits, axis=-1)
        num_classes = tf.shape(logits)[-1]
        one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        conf = tf.reduce_sum(probs * one_hot, axis=-1)  # (B,)
        weights = tf.maximum((1.0 - conf) ** 2, self.scn_min_weight)
        loss = tf.reduce_mean(weights * ce)

        # Ranking loss
        sorted_idx = tf.argsort(conf)
        B = tf.shape(logits)[0]
        k = tf.maximum(2, tf.cast(tf.cast(B, tf.float32) * 0.2, tf.int32))
        hard_idx = sorted_idx[:k]
        easy_idx = sorted_idx[k:]

        hard_loss = tf.reduce_mean(tf.gather(ce, hard_idx))
        easy_loss = tf.reduce_mean(tf.gather(ce, easy_idx))

        raw_ranking_loss = tf.maximum(easy_loss - hard_loss + self.scn_margin, 0.0)
        is_warm = tf.cast(epoch_tensor >= self.scn_warmup_epochs, tf.float32)
        ranking_loss = is_warm * raw_ranking_loss

        return self.scn_alpha * loss + self.scn_rank_lambda * ranking_loss

    # ------------------------------------------------------------------
    # Forward pass helper
    # ------------------------------------------------------------------

    def _forward_batch(self, batch, epoch_tensor: tf.Tensor, use_scn_tensor: tf.Tensor, lam_tensor: tf.Tensor, training: bool = True):
        """Unpack batch and call model. Returns (loss, logits, labels)."""
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
            bboxes = batch.get("bboxes", None)
            region_mask = batch.get("region_mask", None)
            region_confidence = batch.get("region_confidence", None)
            semantic_meta = None
            if region_mask is not None:
                semantic_meta = {"region_mask": region_mask, "region_confidence": region_confidence}
        elif len(batch) == 4:
            images, labels, bboxes, semantic_meta = batch
        elif len(batch) == 3:
            images, labels, bboxes = batch
            semantic_meta = None
        else:
            images, labels = batch[:2]
            bboxes = None
            semantic_meta = None

        labels = tf.cast(labels, tf.int32)
        loss_mode = self.config.get("training", {}).get("loss", "cross_entropy")

        # Mixup (Unconditional mathematical interpolation to avoid tf.cond backprop bugs with float16)
        perm = tf.random.shuffle(tf.range(tf.shape(images)[0]))
        images_b = tf.gather(images, perm)
        labels_b = tf.gather(labels, perm)

        lam_cast = tf.cast(lam_tensor, tf.float32)
        lam_cast_img = tf.reshape(lam_cast, [1, 1, 1, 1])
        
        mixed_img = lam_cast_img * tf.cast(images, tf.float32) + (1.0 - lam_cast_img) * tf.cast(images_b, tf.float32)
        images = tf.cast(mixed_img, images.dtype)

        # Build model inputs
        region_mask, region_confidence = None, None
        if isinstance(semantic_meta, dict):
            region_mask = semantic_meta.get("region_mask")
            region_confidence = semantic_meta.get("region_confidence")

        outputs = self.model(
            image=images, 
            bboxes=bboxes, 
            region_mask=region_mask, 
            region_confidence=region_confidence, 
            training=training
        )
        logits = tf.cast(outputs["logits"], tf.float32)

        # Compute loss
        if loss_mode == "semantic_roi_graph":
            loss_dict = compute_semantic_roi_graph_losses(
                self.model, outputs, labels, class_weights=self.class_weights
            )
            cls_loss = loss_dict["loss"]
        else:
            loss_a = tf.reduce_mean(_spce(labels, logits))
            loss_b = tf.reduce_mean(_spce(labels_b, logits))
            
            scn_loss_val = self._scn_loss(logits, labels, epoch_tensor)
            
            # If use_scn_tensor is True, use SCN loss for A. Otherwise SPCE.
            use_scn_f = tf.cast(use_scn_tensor, tf.float32)
            base_loss_a = use_scn_f * scn_loss_val + (1.0 - use_scn_f) * loss_a
            
            cls_loss = lam_cast * base_loss_a + (1.0 - lam_cast) * loss_b

        return cls_loss, logits, labels, outputs

    # ------------------------------------------------------------------
    # Distributed Training step
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _distributed_train_step(self, batch, epoch_tensor, use_scn_tensor, lam_tensor):
        def step_fn(dist_batch):
            with tf.GradientTape() as tape:
                cls_loss, logits, labels, _ = self._forward_batch(
                    dist_batch, epoch_tensor, use_scn_tensor, lam_tensor, training=True
                )
                # Manual constant loss scaling to prevent float16 underflow,
                # avoiding the dynamic LossScaleOptimizer which creates tf.cond traps.
                loss_scale = 1024.0
                scaled_loss = cls_loss * loss_scale

            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            
            # 1. Unscale and cast ALL gradients to float32 immediately.
            # This ensures no float16 tensors enter tf.where or optimizer logic.
            grads = [
                tf.cast(g, tf.float32) / loss_scale if g is not None else None
                for g in scaled_grads
            ]

            # 2. Sanitize gradients: replace NaN/Inf with zeros.
            # Since grads are now float32, tf.where is completely safe.
            grads = [
                tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                if g is not None else None
                for g in grads
            ]
            
            # 3. Clip by global norm
            valid_grads = [g for g in grads if g is not None]
            clipped_grads, _ = tf.clip_by_global_norm(valid_grads, 5.0)
            
            # 4. Reconstruct grads with zeros instead of None
            final_grads = []
            idx = 0
            for v, g in zip(self.model.trainable_variables, grads):
                if g is not None:
                    final_grads.append(clipped_grads[idx] * _lr_multiplier_for_name(v.name))
                    idx += 1
                else:
                    # Explicit zeros prevent optimizer skipping bugs in Keras/XLA
                    final_grads.append(tf.zeros_like(v, dtype=tf.float32))
                    
            self.optimizer.apply_gradients(zip(final_grads, self.model.trainable_variables))

            # Update EMA weights — self.ema_vars is populated BEFORE first trace
            for v, ema_v in zip(self.model.trainable_variables, self.ema_vars):
                ema_v.assign(self.ema_decay * ema_v + (1.0 - self.ema_decay) * v)

            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc = tf.cast(tf.equal(preds, labels), tf.float32)
            
            return cls_loss, tf.reduce_mean(acc)

        per_replica_losses, per_replica_accs = self.strategy.run(
            step_fn, args=(batch,)
        )
        
        # Aggregate across replicas
        loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
        return loss, acc

    @tf.function(reduce_retracing=True)
    def _distributed_val_step(self, batch):
        def step_fn(dist_batch):
            cls_loss, logits, labels, _ = self._forward_batch(
                dist_batch, 
                epoch_tensor=tf.constant(0, dtype=tf.int32), 
                use_scn_tensor=tf.constant(False), 
                lam_tensor=tf.constant(1.0), 
                training=False
            )
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc = tf.cast(tf.equal(preds, labels), tf.float32)
            return cls_loss, tf.reduce_mean(acc), preds, labels

        per_replica_loss, per_replica_acc, per_replica_preds, per_replica_labels = self.strategy.run(
            step_fn, args=(batch,)
        )

        loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
        acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_acc, axis=None)
        
        # Gather predictions and labels across all GPUs
        if self.strategy.num_replicas_in_sync > 1:
            preds = self.strategy.gather(per_replica_preds, axis=0)
            labels = self.strategy.gather(per_replica_labels, axis=0)
        else:
            preds = per_replica_preds
            labels = per_replica_labels
            
        return loss, acc, preds, labels

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int):
        loss_values = []
        acc_values = []

        use_scn_tensor = tf.constant(getattr(self, "_runtime_use_scn", self.use_scn), dtype=tf.bool)
        use_mixup = getattr(self, "_runtime_use_mixup", False)
        
        for batch in self.train_dataset:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if use_mixup else 1.0
            
            loss, acc = self._distributed_train_step(
                batch, 
                epoch_tensor=tf.constant(epoch, dtype=tf.int32),
                use_scn_tensor=use_scn_tensor,
                lam_tensor=tf.constant(lam, dtype=tf.float32)
            )

            loss_values.append(tf.cast(loss, tf.float32))
            acc_values.append(tf.cast(acc, tf.float32))

        if not loss_values:
            return 0.0, 0.0

        return float(tf.reduce_mean(tf.stack(loss_values)).numpy()), float(tf.reduce_mean(tf.stack(acc_values)).numpy())

    def validate(self, epoch: int):
        # --- SWAP TO EMA WEIGHTS FOR VALIDATION ---
        current_weights = []
        if getattr(self, "_ema_initialized", False) and getattr(self, "ema_vars", None) is not None:
            # Backup current weights and assign EMA weights
            for v, ema_v in zip(self.model.trainable_variables, self.ema_vars):
                current_weights.append(tf.identity(v))
                v.assign(ema_v)
                
        loss_values = []
        acc_values = []
        pred_batches = []
        label_batches = []

        for batch in self.val_dataset:
            loss, acc, preds, labels = self._distributed_val_step(batch)

            loss_values.append(tf.cast(loss, tf.float32))
            acc_values.append(tf.cast(acc, tf.float32))
            pred_batches.append(tf.cast(preds, tf.int32))
            label_batches.append(tf.cast(labels, tf.int32))
            
        # --- RESTORE ORIGINAL WEIGHTS ---
        if current_weights:
            for v, w in zip(self.model.trainable_variables, current_weights):
                v.assign(w)

        if not loss_values:
            return 0.0, 0.0, 0.0, 0.0

        all_preds = tf.concat(pred_batches, axis=0).numpy().tolist()
        all_labels = tf.concat(label_batches, axis=0).numpy().tolist()

        from sklearn.metrics import f1_score, balanced_accuracy_score
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)

        return (
            float(tf.reduce_mean(tf.stack(loss_values)).numpy()),
            float(tf.reduce_mean(tf.stack(acc_values)).numpy()),
            macro_f1,
            bal_acc,
        )

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self):
        print(f"\n--> Start training {self.epochs} epochs\n")

        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="FER2013", name=self.run_name, config=self.config)
            except Exception:
                self.use_wandb = False

        best_score = -float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []

        # Wire scheduler to model (required for Keras callbacks like ReduceLROnPlateau)
        if self.scheduler is not None and hasattr(self.scheduler, "set_model"):
            self.scheduler.set_model(self.model)
            # ReduceLROnPlateau also needs optimizer to be set on model
            if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
                self.model.optimizer = self.optimizer

        print(f"--> [Training] Starts for {self.epochs} epochs. Mixed Precision: {self.use_mixed_precision}")

        # ---------------------------------------------------------------
        # Initialize EMA variables BEFORE any @tf.function is called.
        # This ensures the EMA update loop is captured in the traced graph.
        # ---------------------------------------------------------------
        if not self._ema_initialized:
            with self.strategy.scope():
                self.ema_vars = [
                    tf.Variable(
                        tf.identity(v),
                        trainable=False,
                        name=v.name.split(':')[0].replace('/', '_') + '_ema',
                    )
                    for v in self.model.trainable_variables
                ]
            self._ema_initialized = True
            print(f"--> [EMA] Initialized {len(self.ema_vars)} EMA shadow variables.")

        for ep in range(self.epochs):
            progress = ep / max(self.epochs - 1, 1)

            # Phase scheduling
            if progress <= 0.7:
                self._runtime_use_scn = False
                self._runtime_use_mixup = False
                self._runtime_motif_diversity_weight = float(self.config.get('training', {}).get('micro_motif_diversity_weight', 0.02))
            else:
                self._runtime_use_scn = True
                self._runtime_use_mixup = False
                base_motif_w = float(self.config.get('training', {}).get('micro_motif_diversity_weight', 0.02))
                self._runtime_motif_diversity_weight = base_motif_w * 1.5  # Phase 3 boost

            base_au_w = float(self.config.get('training', {}).get('au_contrastive_weight', 0.03))
            if ep < 5:
                self._runtime_au_contrastive_weight = 0.0
            elif ep < 10:
                self._runtime_au_contrastive_weight = base_au_w * ((ep - 4) / 5.0)
            else:
                self._runtime_au_contrastive_weight = base_au_w

            # Notify model of training progress
            set_prog = getattr(self.model, "set_training_progress", None)
            if callable(set_prog):
                try:
                    set_prog(progress)
                except Exception:
                    pass

            train_loss, train_acc = self.train_one_epoch(ep)
            val_loss, val_acc, macro_f1, bal_acc = self.validate(ep)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch {ep+1}/{self.epochs} - "
                f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
                f"val_f1: {macro_f1:.4f}"
            )

            # Selection score
            selection_score = 0.5 * val_acc + 0.5 * macro_f1

            # LR Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, tf.keras.callbacks.ReduceLROnPlateau):
                    self.scheduler.on_epoch_end(ep, logs={"selection_score": selection_score})
                elif hasattr(self.scheduler, "step"):
                    self.scheduler.step()

            if selection_score > best_score:
                best_score = selection_score
                patience_counter = 0
                # Save EMA weights
                original_vars = [tf.identity(v) for v in self.model.trainable_variables]
                for v, ema_v in zip(self.model.trainable_variables, self.ema_vars):
                    v.assign(ema_v)
                self.model.save_weights(self.save_path)
                # Restore original weights
                for v, orig in zip(self.model.trainable_variables, original_vars):
                    v.assign(orig)
                print(
                    f"\t--- Save best at ep {ep+1}, "
                    f"score: {selection_score:.4f}, val_acc: {val_acc:.4f}, f1: {macro_f1:.4f}"
                )
            else:
                patience_counter += 1
                print(f"\t-!- No improvement: {patience_counter}/{self.patience}")

            # W&B logging
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "Epoch": ep + 1,
                        "Train/Loss": train_loss,
                        "Train/Accuracy": train_acc,
                        "Val/Loss": val_loss,
                        "Val/Accuracy": val_acc,
                        "Val/MacroF1": macro_f1,
                        "Val/BalancedAccuracy": bal_acc,
                        "Learning_Rate": float(self.optimizer.learning_rate),
                    }, step=ep)
                except Exception:
                    pass

            # Early stopping
            if patience_counter >= self.patience:
                print(f"\n--> Early stopping at epoch {ep+1}")
                break

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        return train_losses, val_losses
