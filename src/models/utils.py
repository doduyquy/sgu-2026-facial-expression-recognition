"""
utils.py — Shared helper functions for the semantic_roi_graph package.
TensorFlow implementation.
"""

from __future__ import annotations

import tensorflow as tf


def safe_softmax(x, axis=-1):
    """A numerically stable softmax that prevents NaN when vectors are fully masked."""
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    # Handle the case where x was all -inf (which results in NaN after subtraction)
    all_invalid = tf.reduce_all(
        tf.math.is_inf(x_shifted) | tf.math.is_nan(x_shifted),
        axis=axis, keepdims=True
    )
    x_shifted = tf.where(all_invalid, tf.zeros_like(x_shifted), x_shifted)
    return tf.nn.softmax(x_shifted, axis=axis)


def apply_multi_scale_tta(model, images, bboxes=None, region_mask=None, region_confidence=None, scale=1.05):
    """Run model with Multi-scale TTA: Original, Flipped, Scaled (1.05x)+Flipped.
    
    Note: TF uses NHWC format (B, H, W, C).
    """
    # 1. Normal forward
    logits_normal = model(images, bboxes, region_mask, region_confidence, training=False)
    
    swap_pairs = [(1, 2), (4, 5), (7, 8)]
    
    def flip_bboxes_fn(boxes):
        if boxes is None:
            return None
        fb = tf.identity(boxes)
        # boxes: (B, R, 4) in [x1, y1, x2, y2] format
        x1 = boxes[:, :, 0]
        x2 = boxes[:, :, 2]
        new_x1 = 47.0 - x2
        new_x2 = 47.0 - x1
        fb = tf.concat([
            tf.expand_dims(new_x1, -1),
            boxes[:, :, 1:2],
            tf.expand_dims(new_x2, -1),
            boxes[:, :, 3:4],
        ], axis=-1)
        # Swap left/right pairs
        indices = list(range(fb.shape[1] if fb.shape[1] is not None else 9))
        for i, j in swap_pairs:
            indices[i], indices[j] = indices[j], indices[i]
        fb = tf.gather(fb, indices, axis=1)
        return fb
        
    def flip_meta_fn(meta):
        if meta is None:
            return None
        indices = list(range(meta.shape[1] if meta.shape[1] is not None else 9))
        for i, j in swap_pairs:
            indices[i], indices[j] = indices[j], indices[i]
        return tf.gather(meta, indices, axis=1)
        
    # 2. Flipped forward (flip along width axis — NHWC: axis=2)
    flipped_images = tf.reverse(images, axis=[2])
    flipped_bboxes = flip_bboxes_fn(bboxes)
    flipped_region_mask = flip_meta_fn(region_mask)
    flipped_region_confidence = flip_meta_fn(region_confidence)
    
    logits_flipped = model(flipped_images, flipped_bboxes, flipped_region_mask, flipped_region_confidence, training=False)
    
    # 3. Scaled (1.05x) and Flipped forward
    img_shape = tf.shape(images)
    h = img_shape[1]
    w = img_shape[2]
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    
    scaled_images = tf.image.resize(images, [new_h, new_w], method='bilinear')
    
    top = (new_h - h) // 2
    left = (new_w - w) // 2
    scaled_images = scaled_images[:, top:top+h, left:left+w, :]
    
    scaled_flipped_images = tf.reverse(scaled_images, axis=[2])
    
    scaled_flipped_bboxes = None
    if bboxes is not None:
        h_f = tf.cast(h, tf.float32)
        w_f = tf.cast(w, tf.float32)
        new_h_f = tf.cast(new_h, tf.float32)
        new_w_f = tf.cast(new_w, tf.float32)
        left_f = tf.cast(left, tf.float32)
        top_f = tf.cast(top, tf.float32)
        
        scaled_bboxes_x1 = bboxes[:, :, 0] * (new_w_f / w_f) - left_f
        scaled_bboxes_x2 = bboxes[:, :, 2] * (new_w_f / w_f) - left_f
        scaled_bboxes_y1 = bboxes[:, :, 1] * (new_h_f / h_f) - top_f
        scaled_bboxes_y2 = bboxes[:, :, 3] * (new_h_f / h_f) - top_f
        
        scaled_bboxes_x1 = tf.clip_by_value(scaled_bboxes_x1, 0.0, w_f - 1.0)
        scaled_bboxes_x2 = tf.clip_by_value(scaled_bboxes_x2, 0.0, w_f - 1.0)
        scaled_bboxes_y1 = tf.clip_by_value(scaled_bboxes_y1, 0.0, h_f - 1.0)
        scaled_bboxes_y2 = tf.clip_by_value(scaled_bboxes_y2, 0.0, h_f - 1.0)
        
        scaled_bboxes = tf.stack([scaled_bboxes_x1, scaled_bboxes_y1, scaled_bboxes_x2, scaled_bboxes_y2], axis=-1)
        scaled_flipped_bboxes = flip_bboxes_fn(scaled_bboxes)
        
    logits_scaled_flipped = model(scaled_flipped_images, scaled_flipped_bboxes, 
                                   flipped_region_mask, flipped_region_confidence, training=False)
    
    # 4. Average 3 predictions
    result = {}
    for k in logits_normal:
        val = logits_normal[k]
        if isinstance(val, tf.Tensor) and val.dtype.is_floating:
            result[k] = (logits_normal[k] + logits_flipped[k] + logits_scaled_flipped[k]) / 3.0
        else:
            result[k] = logits_normal[k]
    return result


def count_parameters(model):
    """Count total and trainable parameters of a Keras model."""
    total = sum(tf.size(v).numpy() for v in model.variables)
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    non_trainable = total - trainable
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
    return total, trainable


def freeze_backbone(model):
    """Freeze backbone parameters."""
    if hasattr(model, 'backbone'):
        model.backbone.trainable = False
        print("[Freeze] Backbone frozen")


def unfreeze_backbone(model):
    """Unfreeze backbone parameters."""
    if hasattr(model, 'backbone'):
        model.backbone.trainable = True
        print("[Unfreeze] Backbone unfrozen")