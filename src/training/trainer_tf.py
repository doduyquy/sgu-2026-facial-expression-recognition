"""
trainer_tf.py — TensorFlow/Keras training loop cho SemanticROIGraphFER.

Thay thế PyTorch Trainer bằng Keras GradientTape-based custom training loop.
Hỗ trợ:
- EMA model (built-in Keras optimizer EMA)
- Multi-loss (via compute_semantic_roi_graph_losses_tf)
- Mixup
- SCN-light (sample weighting by confidence)
- W&B logging
- Early stopping
- LR scheduling (ReduceLROnPlateau, CosineAnnealing, Step)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from src.models.semantic_roi_graph_losses_tf import compute_semantic_roi_graph_losses_tf


def _spce(labels, logits):
    """Sparse categorical cross-entropy per-sample — compatible with all Keras versions."""
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
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.run_name = run_name
        self.save_path = save_path

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

        # Aux loss weights
        self.motif_diversity_weight = float(train_cfg.get("motif_diversity_weight", 0.05))
        self.au_contrastive_weight = float(train_cfg.get("au_contrastive_weight", 0.03))

        # EMA
        self._ema_weights = None

    # ------------------------------------------------------------------
    # SCN Light
    # ------------------------------------------------------------------

    def _scn_loss(self, logits: tf.Tensor, labels: tf.Tensor, epoch: int):
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

        hard_loss = tf.reduce_mean(tf.gather(ce, hard_idx)) if tf.size(hard_idx) > 0 else tf.zeros(())
        easy_loss = tf.reduce_mean(tf.gather(ce, easy_idx)) if tf.size(easy_idx) > 0 else tf.zeros(())

        if epoch >= self.scn_warmup_epochs:
            ranking_loss = tf.maximum(easy_loss - hard_loss + self.scn_margin, 0.0)
        else:
            ranking_loss = tf.zeros(())

        return self.scn_alpha * loss + self.scn_rank_lambda * ranking_loss

    # ------------------------------------------------------------------
    # Forward pass helper
    # ------------------------------------------------------------------

    def _forward_batch(self, batch, epoch: int, training: bool = True):
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

        # Mixup (Phase 3 only)
        mixup_active = getattr(self, "_runtime_use_mixup", False) and training
        labels_a, labels_b, lam = labels, labels, 1.0
        if mixup_active and self.mixup_alpha > 0:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            perm = tf.random.shuffle(tf.range(tf.shape(images)[0]))
            images = lam * images + (1.0 - lam) * tf.gather(images, perm)
            labels_b = tf.gather(labels, perm)

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
        logits = outputs["logits"]

        # Compute loss
        runtime_use_scn = getattr(self, "_runtime_use_scn", self.use_scn)

        if mixup_active:
            cls_loss = (
                lam * tf.reduce_mean(_spce(labels_a, logits)) +
                (1.0 - lam) * tf.reduce_mean(_spce(labels_b, logits))
            )
        elif loss_mode == "semantic_roi_graph":
            loss_dict = compute_semantic_roi_graph_losses_tf(self.model, outputs, labels)
            cls_loss = loss_dict["loss"]
        elif runtime_use_scn and epoch >= self.scn_warmup_epochs:
            cls_loss = self._scn_loss(logits, labels, epoch)
        else:
            cls_loss = tf.reduce_mean(_spce(labels, logits))

        return cls_loss, logits, labels, outputs

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self, batch, epoch: int):
        with tf.GradientTape() as tape:
            cls_loss, logits, labels, outputs = self._forward_batch(batch, epoch, training=True)

        grads = tape.gradient(cls_loss, self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return cls_loss, acc

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int):
        total_loss, total_acc, n = 0.0, 0.0, 0

        for batch in self.train_dataset:
            loss, acc = self._train_step(batch, epoch)
            batch_size = tf.shape(batch[0])[0].numpy()
            total_loss += loss.numpy() * batch_size
            total_acc += acc.numpy() * batch_size
            n += batch_size

        return total_loss / max(n, 1), total_acc / max(n, 1)

    def validate(self, epoch: int):
        total_loss, total_acc, n = 0.0, 0.0, 0
        all_preds, all_labels = [], []

        for batch in self.val_dataset:
            cls_loss, logits, labels, _ = self._forward_batch(batch, epoch, training=False)
            batch_size = tf.shape(batch[0])[0].numpy()
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
            total_loss += cls_loss.numpy() * batch_size
            total_acc += acc.numpy() * batch_size
            n += batch_size
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
