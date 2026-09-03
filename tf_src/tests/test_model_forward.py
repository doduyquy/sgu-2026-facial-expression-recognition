"""
test_model_forward.py — Unit test verifying model forward, backward gradients, and TTA in TensorFlow.
Run:
    python tf_src/tests/test_model_forward.py
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import yaml
import numpy as np
import tensorflow as tf

from tf_src.models.semantic_roi_graph_tf import SemanticROIGraphFERTF
from tf_src.models.losses_tf import compute_semantic_roi_graph_losses_tf


def test_pipeline():
    print("--> [1/4] Loading default configuration...")
    config_path = ROOT_DIR / "configs/semantic_roi_graph_tf.yaml"
    if not config_path.exists():
        config_path = ROOT_DIR / "tf_src/configs/semantic_roi_graph_tf.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("--> [2/4] Initializing SemanticROIGraphFERTF model...")
    model = SemanticROIGraphFERTF(config=config)

    batch_size = 4
    dummy_images = tf.random.normal([batch_size, 48, 48, 1], dtype=tf.float32)
    dummy_bboxes = tf.constant([
        [[8.0, 0.0, 40.0, 10.0],
         [5.0, 8.0, 18.0, 18.0],
         [30.0, 8.0, 43.0, 18.0],
         [18.0, 12.0, 30.0, 22.0],
         [6.0, 16.0, 20.0, 30.0],
         [28.0, 16.0, 42.0, 30.0],
         [14.0, 20.0, 34.0, 38.0],
         [8.0, 30.0, 22.0, 43.0],
         [26.0, 30.0, 40.0, 43.0]]
    ], dtype=tf.float32)
    dummy_bboxes = tf.tile(dummy_bboxes, [batch_size, 1, 1])
    dummy_labels = tf.constant([0, 3, 4, 6], dtype=tf.int32)

    print("--> [3/4] Testing forward pass and GradientTape backward loss...")
    with tf.GradientTape() as tape:
        outputs = model._forward_single(dummy_images, dummy_bboxes, training=True)
        loss_dict = compute_semantic_roi_graph_losses_tf(
            model, outputs, dummy_labels, label_smoothing=0.1, train_cfg=config.get("training", {})
        )
        total_loss = loss_dict["loss"]

    assert outputs["logits"].shape == (batch_size, 7), f"Expected shape ({batch_size}, 7), got {outputs['logits'].shape}"
    print(f"    [OK] Forward outputs shape: {outputs['logits'].shape}")
    print(f"    [OK] Computed total loss: {float(total_loss):.4f}")

    trainable_vars = model.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)

    all_finite = True
    for g, v in zip(gradients, trainable_vars):
        if g is None:
            print(f"    [WARNING] Variable {v.name} has None gradient!")
        elif tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
            all_finite = False
            print(f"    [ERROR] Variable {v.name} has NaN/Inf gradient!")

    assert all_finite, "Some gradients contain NaN or Inf!"
    print(f"    [OK] All {len(trainable_vars)} trainable variables have finite gradients!")

    print("--> [4/4] Testing built-in Horizontal Flip TTA (inference mode)...")
    out_tta = model(dummy_images, dummy_bboxes, training=False)
    assert out_tta["logits"].shape == (batch_size, 7)
    assert not tf.reduce_any(tf.math.is_nan(out_tta["logits"]))
    print(f"    [OK] TTA logits shape: {out_tta['logits'].shape}")
    print("\n🎉 ALL TENSORFLOW TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_pipeline()
