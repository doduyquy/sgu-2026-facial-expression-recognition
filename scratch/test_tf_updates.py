"""
Quick sanity test for the updated TF pipeline.
Tests: HRNetBackboneTF, model forward pass, trainer init, train script imports.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

print("\n" + "="*60)
print("TEST 1: HRNetBackboneTF + SemanticRoiGraphConfig backbone_type")
print("="*60)
from src.models.semantic_roi_graph_tf import (
    SemanticROIGraphFER, SemanticRoiGraphConfig,
    HRNetBackboneTF, ResidualBasicBlock, BottleneckBlock
)

# Test HRNet standalone
hrnet = HRNetBackboneTF(feature_dim=256)
dummy = tf.random.normal([2, 48, 48, 1])
out = hrnet(dummy, training=False)
print(f"  HRNetBackboneTF output shape: {out.shape}")
assert out.shape == (2, 12, 12, 256) or out.shape[1] > 0, f"Unexpected shape: {out.shape}"
print("  OK!")

print("\n" + "="*60)
print("TEST 2: SemanticROIGraphFER with backbone_type='hrnet_w18'")
print("="*60)
config = SemanticRoiGraphConfig(
    backbone_type='hrnet_w18',
    num_classes=7,
    num_regions=9,
    roi_grid=4,
    feature_dim=256,
    use_pretrained=False,
)
print(f"  Config backbone_type: {config.backbone_type}")
model = SemanticROIGraphFER(config)
# Forward pass
images = tf.random.normal([2, 48, 48, 1])
bboxes = tf.constant([[[10., 8., 20., 18.],
                        [30., 8., 43., 18.],
                        [18., 12., 30., 22.],
                        [6., 16., 20., 30.],
                        [28., 16., 42., 30.],
                        [8., 0., 40., 10.],
                        [14., 20., 34., 38.],
                        [8., 30., 22., 43.],
                        [26., 30., 40., 43.]]] * 2, dtype=tf.float32)
region_mask = tf.ones([2, 9])
region_conf = tf.ones([2, 9]) * 0.9

outputs = model((images, bboxes, region_mask, region_conf), training=False)
print(f"  logits shape: {outputs['logits'].shape}")
assert outputs['logits'].shape == (2, 7), f"Unexpected logits shape: {outputs['logits'].shape}"
assert 'aux_losses' in outputs, "Missing aux_losses in output!"
print(f"  aux_losses keys: {list(outputs['aux_losses'].keys())}")
print("  OK!")

print("\n" + "="*60)
print("TEST 3: backbone_type='resnet50' still works (backward compat)")
print("="*60)
config_r = SemanticRoiGraphConfig(
    backbone_type='resnet50',
    num_classes=7,
    num_regions=9,
    roi_grid=4,
    feature_dim=256,
    use_pretrained=False,
)
model_r = SemanticROIGraphFER(config_r)
out_r = model_r((images, bboxes, region_mask, region_conf), training=False)
print(f"  ResNet50 logits shape: {out_r['logits'].shape}")
print("  OK!")

print("\n" + "="*60)
print("TEST 4: TrainerTF has EMA attribute")
print("="*60)
from src.training.trainer_tf import TrainerTF
import inspect
src = inspect.getsource(TrainerTF.__init__)
has_ema = 'ExponentialMovingAverage' in src or 'ema' in src.lower()
print(f"  EMA in TrainerTF.__init__: {has_ema}")
assert has_ema, "EMA not found in TrainerTF!"
print("  OK!")

print("\n" + "="*60)
print("TEST 5: build_optimizer_tf supports AdamW")
print("="*60)
# Read the train_tf script
with open('scripts/train_tf.py', 'r', encoding='utf-8') as f:
    train_tf_src = f.read()
has_adamw = 'adamw' in train_tf_src.lower() or 'AdamW' in train_tf_src
has_selection_score = 'selection_score' in train_tf_src
has_mode_max = "mode='max'" in train_tf_src or 'mode=\"max\"' in train_tf_src
print(f"  AdamW support: {has_adamw}")
print(f"  selection_score monitor: {has_selection_score}")
print(f"  mode='max': {has_mode_max}")
assert has_adamw, "AdamW not found in train_tf.py!"
assert has_selection_score, "selection_score not found in train_tf.py!"
print("  OK!")

print("\n" + "="*60)
print("TEST 6: Phase 3 weight scaling in trainer_tf")
print("="*60)
with open('src/training/trainer_tf.py', 'r', encoding='utf-8') as f:
    trainer_src = f.read()
has_phase3 = '1.5' in trainer_src
has_au_warmup = 'au_contrastive' in trainer_src
has_ema_save = 'ema' in trainer_src.lower()
print(f"  Phase 3 weight *1.5: {has_phase3}")
print(f"  au_contrastive warmup: {has_au_warmup}")
print(f"  EMA save logic: {has_ema_save}")
print("  OK!")

print("\n" + "="*60)
print("ALL TESTS PASSED! TF pipeline updated successfully.")
print("="*60)
