import tensorflow as tf
import os
import json


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save a TF checkpoint with metadata.
    
    Args:
        model: tf.keras.Model
        optimizer: tf.keras.optimizers.Optimizer
        epoch: int, current epoch number
        metrics: dict, validation metrics to save alongside
        filepath: str, path to save checkpoint (without extension)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=tf.Variable(epoch, dtype=tf.int64),
    )
    ckpt.write(filepath)
    
    # Save metadata (epoch, metrics) as JSON alongside the checkpoint
    meta_path = filepath + "_meta.json"
    meta = {"epoch": epoch}
    if metrics:
        meta["metrics"] = {k: float(v) if hasattr(v, '__float__') else v 
                           for k, v in metrics.items()}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"[Checkpoint] Saved to {filepath} (epoch {epoch})")


def load_partial_weights(model, checkpoint_path):
    """Load weights partially — skip mismatched shapes.
    
    Args:
        model: tf.keras.Model (must be built)
        checkpoint_path: str, path to checkpoint
    
    Returns:
        tuple: (loaded_count, skipped_names)
    """
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(checkpoint_path).expect_partial()
    
    print(f"[PartialLoad] Restored from {checkpoint_path}")
    return status


def load_checkpoints(model, optimizer, checkpoint_path, partial=False):
    """Load model and optimizer from a TF checkpoint.
    
    Args:
        model: tf.keras.Model
        optimizer: tf.keras.optimizers.Optimizer  
        checkpoint_path: str, path to checkpoint
        partial: bool, if True use expect_partial()
    
    Returns:
        int: saved epoch number
    """
    if not os.path.exists(checkpoint_path + ".index"):
        # Try without extension
        if not tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)):
            raise FileNotFoundError(f"Not found checkpoint at {checkpoint_path}")

    print(f"--> Loading ckpt from {checkpoint_path}")

    epoch_var = tf.Variable(0, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch_var,
    )

    if partial:
        ckpt.restore(checkpoint_path).expect_partial()
        print("[PartialLoad] optimizer state may be skipped.")
    else:
        ckpt.restore(checkpoint_path)

    # Load metadata
    meta_path = checkpoint_path + "_meta.json"
    epoch = int(epoch_var.numpy())
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            epoch = meta.get("epoch", epoch)

    print(f"[Checkpoint] Loaded epoch {epoch}")
    return epoch
