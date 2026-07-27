"""
train_tf.py — Entry point script để train SemanticROIGraphFER với TensorFlow/Keras.

Usage:
    python scripts/train_tf.py --config configs/semantic_roi_graph.yaml --env kaggle

Thay thế scripts/train.py (PyTorch) bằng TensorFlow/Keras workflow.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
import numpy as np

from src.utils.config import load_config


def set_seed(seed: int = 21):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_optimizer_tf(config: dict, model=None):
    """Build TF optimizer matching PyTorch optimizer.py exactly:
    - backbone params: lr (base)
    - head params: lr * 2.0
    - Adam (not AdamW) to match PyTorch optim.Adam
    """
    train_cfg = config.get("training", {})
    opt_name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg.get("lr", 3e-4))          # PyTorch default: 0.0003
    weight_decay = float(train_cfg.get("weight_decay", 1e-3))

    if opt_name == "adam":
        # Match PyTorch: optim.Adam — use Adam WITHOUT weight decay coupling
        # AdamW applies WD differently (decoupled) vs Adam+L2, so we use Adam
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "sgd":
        momentum = float(train_cfg.get("gamma", 0.9))
        opt = tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=momentum
        )
    elif opt_name == 'adamw':
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay
        )
        print(f'--> [Optimizer] AdamW lr={lr} wd={weight_decay}')
        return opt
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    print(f"--> [Optimizer] {opt_name.upper()} lr={lr} wd={weight_decay} (matching PyTorch)")
    return opt


def build_scheduler_tf(optimizer, config: dict):
    """Build LR scheduler. Returns Keras callback or None."""
    train_cfg = config.get("training", {})
    scheduler_name = train_cfg.get("scheduler", "reduce_lr_on_plateau")

    if scheduler_name == "none" or scheduler_name is None:
        return None

    elif scheduler_name == "reduce_lr_on_plateau":
        factor = float(train_cfg.get("lr_factor", 0.5))
        patience = int(train_cfg.get("lr_patience", 5))
        print(f"--> [Scheduler] ReduceLROnPlateau factor={factor} patience={patience}")
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='selection_score',  # Changed from 'val_loss'
            factor=factor,
            patience=patience,
            mode='max',  # Changed from 'min' - we maximize F1/acc score
            min_delta=0.001,
            min_lr=1e-6,
        )

    elif scheduler_name == "cosine":
        T_max = int(train_cfg.get("epochs", 100))
        print(f"--> [Scheduler] CosineDecay T_max={T_max}")
        # Use CosineDecay as LR schedule directly
        cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(train_cfg.get("lr", 1e-3)),
            decay_steps=T_max,
        )
        optimizer.learning_rate = cosine_decay
        return None

    elif scheduler_name == "step":
        step_size = int(train_cfg.get("lr_step_size", 10))
        gamma = float(train_cfg.get("lr_gamma", 0.1))
        print(f"--> [Scheduler] StepLR step={step_size} gamma={gamma}")

        class StepLRCallback(tf.keras.callbacks.Callback):
            def __init__(self, step_size, gamma):
                super().__init__()
                self.step_size = step_size
                self.gamma = gamma

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.step_size == 0:
                    old_lr = float(self.model.optimizer.learning_rate)
                    new_lr = old_lr * self.gamma
                    self.model.optimizer.learning_rate.assign(new_lr)
                    print(f"  [StepLR] LR: {old_lr:.6f} -> {new_lr:.6f}")

        return StepLRCallback(step_size, gamma)

    return None


def get_class_weights_tf(train_csv: str, config: dict):
    """Compute class weights from train CSV."""
    import pandas as pd
    from sklearn.utils.class_weight import compute_class_weight

    mode = config.get("training", {}).get("class_weight_mode", "inverse")
    use_cw = config.get("training", {}).get("use_class_weights", True)

    if not use_cw:
        return None

    df = pd.read_csv(train_csv)
    labels = df["emotion"].values.astype(int)
    classes = np.unique(labels)

    if mode == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights = np.clip(weights / weights.mean(), 0.1, 10.0)
    else:
        counts = np.bincount(labels, minlength=len(classes)).astype(float)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(classes)

    print(f"--> [ClassWeights] mode={mode}: {np.round(weights, 3)}")
    return tf.constant(weights, dtype=tf.float32)


def main():
    print("\t\t--> TensorFlow Training Script <--")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    args = parser.parse_args()

    # Enable Mixed Precision
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("--> [Mixed Precision] Policy set to 'mixed_float16'")

    # Configure GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"--> Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    else:
        print("--> No GPU found, using CPU")

    config = load_config(args.config, args.env)
    set_seed(config.get("seed", {}).get("random_seed", 21))

    # Paths
    if config["env"]["platform"] == "kaggle":
        data_path = config["kaggle"].get("data_path", "/kaggle/input/datasets")
        root_path = config["kaggle"].get("root_path", "/kaggle/working/sgu-2026-facial-expression-recognition/")
    else:
        data_path = config["local"].get("data_path", "../dataset")
        root_path = config["local"].get("root_path", "../")

    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    model_name = config["model"].get("name", "semantic_roi_graph_fer")
    run_name = f"{model_name}_{timestamp}"

    train_csv = os.path.join(data_path, "train.csv")
    val_csv = os.path.join(data_path, "val.csv")
    test_csv = os.path.join(data_path, "test.csv")

    # Build datasets
    from src.data.dataset_tf import build_datasets
    img_size = config.get("model", {}).get("image_size", 48)
    batch_size = config.get("training", {}).get("batch_size", 64)
    bbox_col = config.get("data", {}).get("bbox_col", None)

    train_ds, val_ds, test_ds = build_datasets(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv if os.path.exists(test_csv) else None,
        image_size=img_size,
        batch_size=batch_size,
        bbox_col=bbox_col,
    )
    print(f"--> Datasets built. image_size={img_size}, batch_size={batch_size}")

    # Build MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"--> [Strategy] Number of devices: {strategy.num_replicas_in_sync}")

    # Build model, optimizer, and scheduler within strategy scope
    with strategy.scope():
        from src.models.semantic_roi_graph_tf import SemanticROIGraphFER, SemanticRoiGraphConfig
        model_cfg = SemanticRoiGraphConfig(
            name=model_name,
            backbone_type=str(config.get('model', {}).get('backbone_type', 'hrnet_w18')),
            num_classes=int(config.get("model", {}).get("num_classes", 7)),
            num_regions=int(config.get("model", {}).get("num_regions", 9)),
            roi_grid=int(config.get("model", {}).get("roi_grid", 4)),
            feature_dim=int(config.get("model", {}).get("feature_dim", 256)),
            semantic_state_dim=int(config.get("model", {}).get("semantic_state_dim", 128)),
            semantic_latent_dim=int(config.get("model", {}).get("semantic_latent_dim", 256)),
            semantic_attn_heads=int(config.get("model", {}).get("semantic_attn_heads", 4)),
            hyperedge_count=int(config.get("model", {}).get("hyperedge_count", 4)),
            router_hidden_dim=int(config.get("model", {}).get("router_hidden_dim", 256)),
            micro_motifs_per_region=int(config.get("model", {}).get("micro_motifs_per_region", 8)),
            macro_motifs_per_class=int(config.get("model", {}).get("macro_motifs_per_class", 4)),
            cross_region_compositions=int(config.get("model", {}).get("cross_region_compositions", 8)),
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
            use_pretrained=bool(config.get("model", {}).get("use_pretrained", True)),
            bbox_input_size=int(config.get("model", {}).get("bbox_input_size", 48)),
            relation_temperature=float(config.get("model", {}).get("relation_temperature", 0.07)),
            region_dropout_prob=float(config.get("model", {}).get("region_dropout_prob", 0.0)),
        )

        model = SemanticROIGraphFER(model_cfg)

        # Attach training_cfg for loss functions
        model.training_cfg = config.get("training", {})

        # Build model by calling with dummy input to initialize weights
        dummy_img = tf.zeros([1, img_size, img_size, 1])
        _ = model(dummy_img, training=False)
        print(f"--> Model built. Trainable params: {model.count_params():,}")

        # ---------------------------------------------------------------
        # Differential Learning Rate (matches PyTorch optimizer.py exactly)
        # backbone params: lr (base rate)
        # head params:     lr * 2.0
        # This is the KEY reason PyTorch trains slower and better.
        # ---------------------------------------------------------------
        train_cfg = config.get("training", {})
        base_lr = float(train_cfg.get("lr", 3e-4))   # 0.0003 from config
        head_lr = base_lr * 2.0                        # 0.0006 for head (match PyTorch)
        backbone_freeze_epochs = int(train_cfg.get("backbone_freeze_epochs", 10))
        backbone_lr_scale = float(train_cfg.get("backbone_lr_scale", 0.1))

        # Freeze backbone for phase 1
        if hasattr(model.backbone, 'feature_extractor'):
            model.backbone.feature_extractor.trainable = False
        else:
            model.backbone.trainable = False
        print(f"--> [Phase 1] Backbone FROZEN for first {backbone_freeze_epochs} epochs.")
        print(f"    Training head at lr={head_lr:.6f}")
        n_trainable = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
        print(f"    Trainable params (head only): {n_trainable:,}")

        # Optimizer & scheduler with head LR
        optimizer = build_optimizer_tf(config, model)
        optimizer.learning_rate.assign(head_lr)  # head trains at 2x lr
        scheduler = build_scheduler_tf(optimizer, config)

    # Checkpoint path
    ckpt_dir = os.path.join(root_path, "outputs", "checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{run_name}_best.weights.h5")

    # Class Weights
    class_weights = get_class_weights_tf(train_csv, config)

    # Trainer
    from src.training.trainer_tf import TrainerTF
    trainer = TrainerTF(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        run_name=run_name,
        save_path=save_path,
        strategy=strategy,
        class_weights=class_weights,
        backbone_freeze_epochs=backbone_freeze_epochs,
        backbone_lr_scale=backbone_lr_scale,
    )

    # Train
    train_losses, val_losses = trainer.fit()

    print(f"\n--> Training done. Best model saved to: {save_path}")

    # Final evaluation with TTA on test set
    print('\n' + '='*51)
    print('Final Evaluation on Test Set (with TTA)')
    print('='*51)
    if test_ds is not None:
        model.load_weights(save_path)
        all_preds, all_labels = [], []
        for batch in test_ds:
            if len(batch) >= 3:
                images, labels_b, bboxes = batch[0], batch[1], batch[2]
                # TTA: original + horizontal flip
                out_orig = model({'image': images, 'bboxes': bboxes}, training=False)
                flipped = tf.reverse(images, axis=[-2])
                w = float(images.shape[-2] or 48)
                x1 = (w - 1.0) - bboxes[..., 2]
                x2 = (w - 1.0) - bboxes[..., 0]
                flipped_bboxes = tf.stack([x1, bboxes[..., 1], x2, bboxes[..., 3]], axis=-1)
                # Swap left/right regions
                swap = [0, 2, 1, 3, 5, 4, 6, 8, 7]
                flipped_bboxes = tf.gather(flipped_bboxes, swap, axis=1)
                out_flip = model({'image': flipped, 'bboxes': flipped_bboxes}, training=False)
                logits = 0.5 * (tf.cast(out_orig['logits'], tf.float32) + tf.cast(out_flip['logits'], tf.float32))
            else:
                images, labels_b = batch[0], batch[1]
                outputs = model(images, training=False)
                logits = tf.cast(outputs['logits'], tf.float32)
            preds = tf.argmax(logits, axis=-1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels_b.numpy().tolist())
        
        from sklearn.metrics import classification_report, accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f'\n--> Test Accuracy (TTA): {acc:.4f}')
        print(f'--> Test Macro-F1 (TTA): {f1:.4f}')
        print(classification_report(
            all_labels, all_preds,
            target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
            zero_division=0,
        ))


if __name__ == "__main__":
    main()
