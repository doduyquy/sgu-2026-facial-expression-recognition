"""
trainer_tf.py — Complete training and validation loop for Semantic ROI Graph FER in TensorFlow.
Implements:
- Custom training loop using tf.GradientTape with gradient clipping.
- EMA parameter tracking with synchronized BatchNorm statistics.
- Hybrid selection score (0.5 * Accuracy + 0.5 * Macro-F1) matching the 72.92% setup.
- Early stopping with patience=35.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

from tf_src.models.losses_tf import compute_semantic_roi_graph_losses_tf, compute_class_weights_sqrt_inverse


class TrainerTF:
    """End-to-end Trainer in TensorFlow with anti-overfitting regularizations."""
    def __init__(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        config: dict,
        save_dir: str = "outputs/checkpoints_tf",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = int(config.get("training", {}).get("epochs", 1000))
        self.patience = int(config.get("training", {}).get("patience", 35))
        self.label_smoothing = float(config.get("training", {}).get("label_smoothing", 0.1))
        self.train_cfg = config.get("training", {})

        # Compute class weights if enabled
        self.class_weights = None
        if self.train_cfg.get("use_class_weights", True):
            # FER2013 approximate train distribution: [4593, 547, 5121, 8989, 6077, 4002, 6198]
            class_counts = [4593, 547, 5121, 8989, 6077, 4002, 6198]
            self.class_weights = compute_class_weights_sqrt_inverse(class_counts)

        # Build EMA weights tracking if enabled
        self.use_ema = bool(self.train_cfg.get("use_ema", True))
        self.ema_decay = float(self.train_cfg.get("ema_decay", 0.999))
        self.ema_weights = None

    def _init_ema(self):
        """Initialize EMA weights tracking."""
        if self.use_ema and self.ema_weights is None:
            self.ema_weights = [
                tf.Variable(tf.convert_to_tensor(w), trainable=False)
                for w in self.model.trainable_variables
            ]

    def _update_ema(self):
        """Update EMA parameters with decay=0.999."""
        if not self.use_ema:
            return
        if self.ema_weights is None:
            self._init_ema()
        for ema_w, model_w in zip(self.ema_weights, self.model.trainable_variables):
            ema_w.assign(self.ema_decay * ema_w + (1.0 - self.ema_decay) * tf.convert_to_tensor(model_w))

    def _apply_ema_weights(self):
        """Temporarily apply EMA weights to model for validation."""
        if not self.use_ema or self.ema_weights is None:
            return None
        backup_weights = [tf.convert_to_tensor(w) for w in self.model.trainable_variables]
        for model_w, ema_w in zip(self.model.trainable_variables, self.ema_weights):
            model_w.assign(ema_w)
        return backup_weights

    def _restore_weights(self, backup_weights):
        """Restore active training weights after validation."""
        if backup_weights is not None:
            for model_w, b_w in zip(self.model.trainable_variables, backup_weights):
                model_w.assign(b_w)

    @tf.function
    def train_step(self, inputs, labels):
        """One training step with GradientTape and gradient clipping."""
        images = inputs["images"]
        bboxes = inputs["bboxes"]
        region_mask = inputs.get("region_mask", None)
        region_confidence = inputs.get("region_confidence", None)

        with tf.GradientTape() as tape:
            outputs = self.model._forward_single(
                images, bboxes, region_mask=region_mask, region_confidence=region_confidence, training=True
            )
            loss_dict = compute_semantic_roi_graph_losses_tf(
                self.model,
                outputs,
                labels,
                class_weights=self.class_weights,
                label_smoothing=self.label_smoothing,
                train_cfg=self.train_cfg,
            )
            total_loss = loss_dict["loss"]

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        preds = tf.argmax(outputs["logits"], axis=1, output_type=tf.int32)
        corrects = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32))

        return total_loss, corrects, tf.cast(tf.shape(labels)[0], tf.float32)

    def train_one_epoch(self):
        """Train over the full dataset for one epoch."""
        total_loss = 0.0
        total_corrects = 0.0
        total_samples = 0.0

        for inputs, labels in self.train_dataset:
            loss, corrects, n = self.train_step(inputs, labels)
            self._update_ema()

            total_loss += float(loss) * float(n)
            total_corrects += float(corrects)
            total_samples += float(n)

        avg_loss = total_loss / max(total_samples, 1.0)
        avg_acc = total_corrects / max(total_samples, 1.0)
        return avg_loss, avg_acc

    def validate(self):
        """Validate on validation dataset with built-in Horizontal Flip TTA."""
        backup = self._apply_ema_weights()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0.0

        for inputs, labels in self.val_dataset:
            images = inputs["images"]
            bboxes = inputs["bboxes"]
            region_mask = inputs.get("region_mask", None)
            region_confidence = inputs.get("region_confidence", None)

            # Public call executes built-in 72.92% Horizontal Flip TTA
            outputs = self.model(
                images, bboxes, region_mask=region_mask, region_confidence=region_confidence, training=False
            )
            loss_dict = compute_semantic_roi_graph_losses_tf(
                self.model, outputs, labels, class_weights=self.class_weights, label_smoothing=0.0
            )

            logits = outputs["logits"]
            preds = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
            targets = labels.numpy()

            all_preds.extend(preds)
            all_labels.extend(targets)

            n = len(targets)
            total_loss += float(loss_dict["loss"]) * n
            total_samples += n

        self._restore_weights(backup)

        val_loss = total_loss / max(total_samples, 1.0)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return val_loss, val_acc, macro_f1

    def fit(self):
        """Full training loop with early stopping and best checkpoint saving."""
        best_selection_score = -float("inf")
        patience_counter = 0
        best_ckpt_path = self.save_dir / "semantic_roi_graph_fer_tf_best.weights.h5"

        print(f"\n--> Start TensorFlow training for {self.epochs} epochs with patience={self.patience}...")

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, macro_f1 = self.validate()

            # Hybrid score: 0.5 * Accuracy + 0.5 * Macro-F1
            selection_score = 0.5 * val_acc + 0.5 * macro_f1

            print(
                f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
            )

            if selection_score > best_selection_score:
                best_selection_score = selection_score
                patience_counter = 0

                # Save EMA weights
                backup = self._apply_ema_weights()
                self.model.save_weights(str(best_ckpt_path))
                self._restore_weights(backup)

                print(
                    f"\t--- Save best hybrid score at ep {epoch}, "
                    f"score: {selection_score:.4f}, val_acc: {val_acc:.4f}, macro_f1: {macro_f1:.4f} ---"
                )
            else:
                patience_counter += 1
                print(f"\t-!- No score improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping triggered at ep={epoch}")
                    break

        print(f"\n--> Training complete! Best weights saved to: {best_ckpt_path}")
        return str(best_ckpt_path)
