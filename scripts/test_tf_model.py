"""
test_tf_model.py — Quick smoke test để verify TF model hoạt động đúng.

Chạy: python scripts/test_tf_model.py
Nếu không có lỗi ImportError / shape error là OK.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}\n")


def test_backbone():
    from src.models.semantic_roi_graph import ResNet50Backbone
    backbone = ResNet50Backbone(feature_dim=256, use_pretrained=False)
    dummy = tf.zeros([2, 48, 48, 3])
    out = backbone(dummy, training=False)
    print(f"[Backbone] output shape: {out.shape}")
    assert out.shape[0] == 2 and out.shape[-1] == 256, f"Unexpected backbone shape: {out.shape}"
    print("[Backbone] PASS ✓\n")


def test_roi_align():
    from src.models.semantic_roi_graph import SemanticRoiAlign
    roi = SemanticRoiAlign(roi_grid=4, bbox_input_size=48, feature_out_size=6)
    feature_map = tf.random.normal([2, 6, 6, 256])
    bboxes = tf.random.uniform([2, 9, 4], minval=0, maxval=40)
    out = roi(feature_map, bboxes)
    print(f"[ROI Align] output shape: {out.shape}")
    assert out.shape == (2, 9, 16, 256), f"Unexpected ROI shape: {out.shape}"
    print("[ROI Align] PASS ✓\n")


def test_full_model():
    from src.models.semantic_roi_graph import SemanticROIGraphFER, SemanticRoiGraphConfig

    config = SemanticRoiGraphConfig(use_pretrained=False)
    model = SemanticROIGraphFER(config)

    # NHWC format (grayscale)
    images = tf.zeros([2, 48, 48, 1])
    bboxes = tf.random.uniform([2, 9, 4], minval=0, maxval=40)

    outputs = model(images, bboxes, training=False)
    logits = outputs["logits"]
    print(f"[Full Model] logits shape: {logits.shape}")
    assert logits.shape == (2, 7), f"Unexpected logits shape: {logits.shape}"
    print(f"[Full Model] all output keys: {list(outputs.keys())}")
    print("[Full Model] PASS ✓\n")


def test_losses():
    from src.models.semantic_roi_graph import SemanticROIGraphFER, SemanticRoiGraphConfig
    from src.models.semantic_roi_graph_losses import compute_semantic_roi_graph_losses

    config = SemanticRoiGraphConfig(use_pretrained=False)
    model = SemanticROIGraphFER(config)

    images = tf.zeros([4, 48, 48, 1])
    bboxes = tf.random.uniform([4, 9, 4], minval=0, maxval=40)
    labels = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    outputs = model(images, bboxes, training=True)
    loss_dict = compute_semantic_roi_graph_losses(model, outputs, labels)
    total_loss = loss_dict["loss"]
    print(f"[Losses] Total loss: {total_loss.numpy():.4f}")
    print(f"[Losses] CE loss: {loss_dict['loss_ce'].numpy():.4f}")
    assert not tf.math.is_nan(total_loss), "NaN loss detected!"
    print("[Losses] PASS ✓\n")


def test_tta():
    from src.models.semantic_roi_graph import SemanticROIGraphFER, SemanticRoiGraphConfig
    from src.models.utils import apply_multi_scale_tta

    config = SemanticRoiGraphConfig(use_pretrained=False)
    model = SemanticROIGraphFER(config)

    images = tf.zeros([2, 48, 48, 1])
    bboxes = tf.random.uniform([2, 9, 4], minval=5, maxval=40)
    region_mask = tf.ones([2, 9])
    region_confidence = tf.ones([2, 9]) * 0.9

    outputs = apply_multi_scale_tta(model, images, bboxes, region_mask, region_confidence)
    logits = outputs["logits"]
    print(f"[TTA] logits shape: {logits.shape}")
    assert logits.shape == (2, 7), f"Unexpected TTA logits shape: {logits.shape}"
    print("[TTA] PASS ✓\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  TF Model Smoke Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Backbone", test_backbone),
        ("ROI Align", test_roi_align),
        ("Full Model", test_full_model),
        ("Losses", test_losses),
        ("TTA", test_tta),
    ]

    passed, failed = 0, []
    for name, fn in tests:
        print(f"--- Testing {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            failed.append((name, str(e)))
            print(f"[{name}] FAIL ✗  — {e}\n")

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print("Failed tests:")
        for n, err in failed:
            print(f"  - {n}: {err}")
    else:
        print("All tests PASSED! ✓")
    print("=" * 60)
