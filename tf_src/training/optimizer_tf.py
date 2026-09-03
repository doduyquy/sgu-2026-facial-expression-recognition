"""
optimizer_tf.py — Optimizer and Learning Rate Scheduler in TensorFlow.
Implements:
- AdamW with decoupled weight decay (avoids zero L2 regularization trap).
- CosineDecayRestarts stepped accurately per batch matching the 30-epoch cycle.
"""

from typing import Tuple, Optional
import tensorflow as tf


def build_lr_schedule_tf(
    lr: float = 0.0003,
    epochs: int = 1000,
    steps_per_epoch: int = 448,
    scheduler_name: str = "cosine_annealing_warm_restart",
    t_0: int = 30,
    t_mult: int = 2,
    eta_min: float = 1e-6,
):
    """Build accurate learning rate schedule stepped per batch."""
    if scheduler_name in ["cosine_warm_restart", "cosine_annealing_warm_restart"]:
        # Multiply epochs by steps_per_epoch so the restart cycle aligns with epoch boundaries
        first_decay_steps = int(t_0 * steps_per_epoch)
        alpha = float(eta_min / lr) if lr > 0 else 0.0
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=first_decay_steps,
            t_mul=float(t_mult),
            m_mul=1.0,
            alpha=alpha,
        )
    return lr


def build_optimizer_tf(config: dict, steps_per_epoch: int = 448) -> tf.keras.optimizers.Optimizer:
    """Build AdamW or Adam optimizer in TensorFlow with decoupled weight decay."""
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 0.0003))
    weight_decay = float(train_cfg.get("weight_decay", 0.001))
    t_0 = int(train_cfg.get("T_0", 30))
    t_mult = int(train_cfg.get("T_mult", 2))
    eta_min = float(train_cfg.get("eta_min", 1e-6))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 5.0))

    lr_schedule = build_lr_schedule_tf(
        lr=lr,
        steps_per_epoch=steps_per_epoch,
        t_0=t_0,
        t_mult=t_mult,
        eta_min=eta_min,
    )

    opt_name = train_cfg.get("optimizer", "adamw").lower()

    if hasattr(tf.keras.optimizers, "AdamW") and opt_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            clipnorm=clip_norm,
        )
    else:
        # Fallback to Adam with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=clip_norm,
        )

    return optimizer
