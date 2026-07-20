"""
trainer_tf.py — TensorFlow/Keras training loop cho SemanticROIGraphFER.

Thay thế PyTorch Trainer bằng Keras GradientTape-based custom training loop.
Hỗ trợ:
- Multi-GPU qua tf.distribute.MirroredStrategy
- Mixed Precision (LossScaleOptimizer)
- Graph Mode (@tf.function) để tăng tốc độ x5-x10
- Multi-loss (via compute_semantic_roi_graph_losses_tf)
- Mixup
- SCN-light (sample weighting by confidence)
- W&B logging
- Early stopping
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from src.models.semantic_roi_graph_losses_tf import compute_semantic_roi_graph_losses_tf


def _spce(labels, logits):
    """Sparse categorical cross-entropy per-sample."""
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


class TrainerTF:
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
        backbone_freeze_epochs: int = 10,
        backbone_lr_scale: float = 0.1,
    ):
        self.strategy = strategy or tf.distribute.get_strategy()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.run_name = run_name
        self.save_path = save_path
        self.class_weights = class_weights
        self.backbone_freeze_epochs = backbone_freeze_epochs
        self.backbone_lr_scale = backbone_lr_scale
        self._phase = 1  # 1 = head only, 2 = full model

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

        # Handle Mixed Precision Loss Scaling safely (Keras 2 vs Keras 3)
        self.use_mixed_precision = (
            tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        )
        
        with self.strategy.scope():
            if self.use_mixed_precision:
                # If LossScaleOptimizer is available (TF 2.15- / Keras 2)
                if hasattr(tf.keras.mixed_precision, "LossScaleOptimizer"):
                    # Avoid re-wrapping if already wrapped
                    if not isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
            
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
        one_hot = tf.one_hot(labels, depth=logits.shape[-1], dtype=tf.float32)
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

        def calc_ranking_loss():
            return tf.maximum(easy_loss - hard_loss + self.scn_margin, 0.0)

        ranking_loss = tf.cond(
            epoch_tensor >= self.scn_warmup_epochs,
            calc_ranking_loss,
            lambda: tf.zeros(())
        )

        return self.scn_alpha * loss + self.scn_rank_lambda * ranking_loss

    # ------------------------------------------------------------------
    # Forward pass helper
    # ------------------------------------------------------------------

    def _forward_batch(self, batch, epoch_tensor: tf.Tensor, use_scn_tensor: tf.Tensor, lam_tensor: tf.Tensor, training: bool = True):
        """Unpack batch and call model. Returns (loss, logits, labels)."""
        if len(batch) == 4:
            images, labels, bboxes, semantic_meta = batch
        elif len(batch) == 3:
            images, labels, bboxes = batch
            semantic_meta = None
        else:
            images, labels = batch
            bboxes = None
            semantic_meta = None

        labels = tf.cast(labels, tf.int32)
        loss_mode = self.config.get("training", {}).get("loss", "cross_entropy")

        # Mixup
        def apply_mixup():
            perm = tf.random.shuffle(tf.range(tf.shape(images)[0]))
            mixed_img = lam_tensor * tf.cast(images, tf.float32) + (1.0 - lam_tensor) * tf.cast(tf.gather(images, perm), tf.float32)
            return tf.cast(mixed_img, images.dtype), tf.gather(labels, perm)

        images, labels_b = tf.cond(
            lam_tensor < 1.0,
            apply_mixup,
            lambda: (images, labels)
        )

        # Build model inputs
        region_mask, region_confidence = None, None
        if isinstance(semantic_meta, dict):
            region_mask = semantic_meta.get("region_mask")
            region_confidence = semantic_meta.get("region_confidence")

        if bboxes is not None:
            model_inputs = (images, bboxes, region_mask, region_confidence)
        else:
            model_inputs = images

        outputs = self.model(model_inputs, training=training)
        logits = tf.cast(outputs["logits"], tf.float32)

        # Compute loss
        if loss_mode == "semantic_roi_graph":
            loss_dict = compute_semantic_roi_graph_losses_tf(
                self.model, outputs, labels, class_weights=self.class_weights
            )
            cls_loss = loss_dict["loss"]
        else:
            def compute_mixup_loss():
                return lam_tensor * tf.reduce_mean(_spce(labels, logits)) + (1.0 - lam_tensor) * tf.reduce_mean(_spce(labels_b, logits))

            def compute_scn_or_normal():
                return tf.cond(
                    use_scn_tensor,
                    lambda: self._scn_loss(logits, labels, epoch_tensor),
                    lambda: tf.reduce_mean(_spce(labels, logits))
                )

            cls_loss = tf.cond(
                lam_tensor < 1.0,
                compute_mixup_loss,
                compute_scn_or_normal
            )

        # Scale loss by number of replicas for `tf.distribute`
        # Because we will reduce_sum or reduce_mean later.
        # Actually strategy.reduce with MEAN will do the averaging. 
        # But normally we divide by global batch size. 
        # Using tf.nn.compute_average_loss is standard, but tf.reduce_mean inside the replica is ok 
        # if we reduce with MEAN across replicas.
        return cls_loss, logits, labels, outputs

    # ------------------------------------------------------------------
    # Distributed Training step
    # ------------------------------------------------------------------

    @tf.function
    def _distributed_train_step(self, batch, epoch_tensor, use_scn_tensor, lam_tensor):
        def step_fn(dist_batch):
            with tf.GradientTape() as tape:
                cls_loss, logits, labels, _ = self._forward_batch(
                    dist_batch, epoch_tensor, use_scn_tensor, lam_tensor, training=True
                )
                
                # Mixed precision loss scaling
                if hasattr(self.optimizer, "get_scaled_loss"):
                    scaled_loss = self.optimizer.get_scaled_loss(cls_loss)
                elif hasattr(self.optimizer, "scale_loss"):
                    scaled_loss = self.optimizer.scale_loss(cls_loss)
                else:
                    scaled_loss = cls_loss

            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            if hasattr(self.optimizer, "get_unscaled_gradients"):
                grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            elif hasattr(self.optimizer, "unscale_gradients"):
                grads = self.optimizer.unscale_gradients(scaled_grads)
            elif hasattr(self.optimizer, "unscale_grads"):
                grads = self.optimizer.unscale_grads(scaled_grads)
            else:
                if hasattr(self.optimizer, "loss_scale"):
                    scale = self.optimizer.loss_scale
                    grads = [g / tf.cast(scale, g.dtype) if g is not None else g for g in scaled_grads]
                else:
                    grads = scaled_grads

            # Sanitize gradients: replace NaN/Inf with zeros BEFORE clipping.
            # clip_by_norm(Inf) = Inf * (1.0/Inf) = NaN, which would corrupt weights.
            grads = [
                tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                if g is not None else g
                for g in grads
            ]
            grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc = tf.cast(tf.equal(preds, labels), tf.float32)
            
            # Note: cls_loss and acc are averages inside this replica
            return cls_loss, tf.reduce_mean(acc)

        per_replica_losses, per_replica_accs = self.strategy.run(
            step_fn, args=(batch,)
        )
        
        # Aggregate across replicas
        loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
        return loss, acc

    @tf.function
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
        total_loss, total_acc, n = 0.0, 0.0, 0

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
            
            total_loss += loss.numpy()
            total_acc += acc.numpy()
            n += 1

        return total_loss / max(n, 1), total_acc / max(n, 1)

    def validate(self, epoch: int):
        total_loss, total_acc, n = 0.0, 0.0, 0
        all_preds, all_labels = [], []

        for batch in self.val_dataset:
            loss, acc, preds, labels = self._distributed_val_step(batch)
            
            total_loss += loss.numpy()
            total_acc += acc.numpy()
            n += 1
            
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        from sklearn.metrics import f1_score, balanced_accuracy_score
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)

        return total_loss / max(n, 1), total_acc / max(n, 1), macro_f1, bal_acc

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

        for ep in range(self.epochs):
            progress = ep / max(self.epochs - 1, 1)

            # ---------------------------------------------------------------
            # Phase switching: unfreeze backbone after backbone_freeze_epochs
            # After unfreeze, set LR to base_lr (same as backbone in PyTorch)
            # ---------------------------------------------------------------
            if self._phase == 1 and ep >= self.backbone_freeze_epochs:
                self._phase = 2
                base_lr = float(self.config.get("training", {}).get("lr", 3e-4))
                # Unfreeze backbone
                self.model.backbone.feature_extractor.trainable = True
                for layer in self.model.backbone.feature_extractor.layers:
                    layer.trainable = True
                # Reset optimizer LR to base_lr for full model fine-tuning
                self.optimizer.learning_rate.assign(base_lr)
                # Re-build optimizer for newly unfrozen variables
                try:
                    self.optimizer.build(self.model.trainable_variables)
                except Exception:
                    pass
                print(f"\n--> [Phase 2] Backbone UNFROZEN at epoch {ep+1}. LR reset to {base_lr:.6f}")

            # Phase scheduling
            if progress <= 0.7:
                self._runtime_use_scn = False
                self._runtime_use_mixup = False
            else:
                self._runtime_use_scn = True
                self._runtime_use_mixup = False

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

            # LR Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, tf.keras.callbacks.ReduceLROnPlateau):
                    self.scheduler.on_epoch_end(ep, logs={"val_loss": val_loss})
                elif hasattr(self.scheduler, "step"):
                    self.scheduler.step()

            # Selection score
            selection_score = 0.5 * val_acc + 0.5 * macro_f1

            if selection_score > best_score:
                best_score = selection_score
                patience_counter = 0
                self.model.save_weights(self.save_path)
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
