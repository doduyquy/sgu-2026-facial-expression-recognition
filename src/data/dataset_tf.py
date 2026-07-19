"""
dataset_tf.py — tf.data.Dataset pipeline cho FER dataset.

Thay thế torch.utils.data.Dataset + DataLoader bằng tf.data.Dataset.
Input: CSV file với cột 'pixels' (space-separated) và 'emotion'.
Output: tf.data pipeline với NHWC tensor format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------------------------
# Augmentation layers (TF equivalent của torchvision transforms)
# ---------------------------------------------------------------------------

def build_augmentation_layers(
    image_size: int = 48,
    use_augment: bool = True,
) -> tf.keras.Sequential:
    """Build augmentation pipeline using tf.keras.layers."""
    layers = []
    if use_augment:
        # NOTE: We disabled spatial augmentations (Rotation, Zoom, Translation, Flip) 
        # because the bounding boxes are static and not updated in tf.data pipeline.
        # This matches PyTorch's `use_semantic_masks=True` fallback logic.
        layers += [
            tf.keras.layers.RandomContrast(factor=0.2),
            tf.keras.layers.RandomBrightness(factor=0.2),
        ]
    return tf.keras.Sequential(layers)


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Normalize to [-1, 1] (same as torchvision Normalize mean=0.5, std=0.5)."""
    image = tf.cast(image, tf.float32) / 255.0
    return (image - 0.5) / 0.5


# ---------------------------------------------------------------------------
# CSV-based FER Dataset → tf.data.Dataset
# ---------------------------------------------------------------------------

def augment_spatial_py(image, bbox):
    import torch
    import torchvision.transforms.functional as TF
    
    # image: (48, 48, 1) float32, bbox: (9, 4) float32
    image_t = torch.from_numpy(image).float().permute(2, 0, 1)
    bboxes_np = bbox.copy()
    
    # Synchronized Horizontal Flip
    if np.random.rand() < 0.5:
        image_t = torch.flip(image_t, dims=[-1])
        flipped_bboxes = bboxes_np.copy()
        flipped_bboxes[:, 0] = 47.0 - bboxes_np[:, 2]
        flipped_bboxes[:, 2] = 47.0 - bboxes_np[:, 0]
        
        swap_pairs = [(1, 2), (4, 5), (7, 8)]
        for i, j in swap_pairs:
            tmp = flipped_bboxes[i].copy()
            flipped_bboxes[i] = flipped_bboxes[j]
            flipped_bboxes[j] = tmp
            
        bboxes_np = flipped_bboxes
        
    # Synchronized Random Affine
    if np.random.rand() < 0.5:
        angle_deg = np.random.uniform(-10.0, 10.0)
        tx = np.random.uniform(-4.8, 4.8)
        ty = np.random.uniform(-4.8, 4.8)
        scale = np.random.uniform(0.9, 1.1)
        
        image_t = TF.affine(
            image_t,
            angle=angle_deg,
            translate=[int(tx), int(ty)],
            scale=scale,
            shear=0.0,
            interpolation=TF.InterpolationMode.BILINEAR
        )
        
        cx, cy = 23.5, 23.5
        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        new_bboxes = bboxes_np.copy()
        for r in range(len(bboxes_np)):
            x1, y1, x2, y2 = bboxes_np[r]
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue
                
            corners = np.array([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
            ])
            dx = corners[:, 0] - cx
            dy = corners[:, 1] - cy
            x_new = dx * scale * cos_t + dy * scale * sin_t + cx + tx
            y_new = -dx * scale * sin_t + dy * scale * cos_t + cy + ty
            
            x1_n = np.clip(np.min(x_new), 0.0, 47.0)
            y1_n = np.clip(np.min(y_new), 0.0, 47.0)
            x2_n = np.clip(np.max(x_new), 0.0, 47.0)
            y2_n = np.clip(np.max(y_new), 0.0, 47.0)
            
            if (x2_n - x1_n < 2.0) or (y2_n - y1_n < 2.0):
                new_bboxes[r] = [0.0, 0.0, 47.0, 47.0] # Fallback
            else:
                new_bboxes[r] = [x1_n, y1_n, x2_n, y2_n]
                
        bboxes_np = new_bboxes
        
    image_out = image_t.permute(1, 2, 0).numpy()
    return image_out, bboxes_np.astype(np.float32)

def augment_spatial_tf(image, bbox):
    image = tf.cast(image, tf.float32)
    img_out, bbox_out = tf.py_function(
        func=augment_spatial_py,
        inp=[image, bbox],
        Tout=[tf.float32, tf.float32]
    )
    img_out.set_shape([48, 48, 1])
    bbox_out.set_shape([9, 4])
    return img_out, bbox_out

class FERDatasetTF:
    """
    Reads FER-format CSV (columns: 'emotion', 'pixels', optional 'bboxes_json').
    Returns tf.data.Dataset yielding (image_nhwc, label) or
    (image_nhwc, label, bboxes) tuples.
    """

    def __init__(
        self,
        csv_path: str,
        image_size: int = 48,
        use_augment: bool = True,
        batch_size: int = 64,
        shuffle: bool = True,
        bbox_col: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.image_size = image_size
        self.use_augment = use_augment
        self.batch_size = batch_size
        self.shuffle_data = shuffle
        self.bbox_col = bbox_col

        df = pd.read_csv(csv_path)
        self.pixels = df["pixels"].values
        self.labels = df["emotion"].values.astype(np.int32)
        self.bboxes = None
        if bbox_col and bbox_col in df.columns:
            import json
            self.bboxes = np.array([
                json.loads(b) if isinstance(b, str) else b
                for b in df[bbox_col].values
            ], dtype=np.float32)

        self.augment_layers = build_augmentation_layers(image_size, use_augment)
        self._n = len(self.labels)

    def _parse_pixels(self, pixel_str: bytes) -> np.ndarray:
        """Parse space-separated pixel string to (H, W, 1) uint8."""
        arr = np.frombuffer(pixel_str, dtype=np.uint8)
        arr = np.array(pixel_str.decode("utf-8").split(), dtype=np.uint8)
        arr = arr.reshape(self.image_size, self.image_size, 1)
        return arr

    def _preprocess(self, image: tf.Tensor, label: tf.Tensor, training: bool):
        image = tf.cast(image, tf.float32)
        image = tf.repeat(image, 3, axis=-1)  # grayscale → 3ch
        if training:
            image = self.augment_layers(image, training=True)
        image = normalize_image(image)
        return image, label

    def build_dataset(self, training: bool = True) -> tf.data.Dataset:
        """Build and return tf.data.Dataset."""
        all_images = []
        for px_str in self.pixels:
            arr = np.array(px_str.split(), dtype=np.uint8).reshape(
                self.image_size, self.image_size, 1
            )
            all_images.append(arr)
        images_np = np.stack(all_images, axis=0)  # (N, H, W, 1)

        if self.bboxes is not None:
            ds = tf.data.Dataset.from_tensor_slices(
                (images_np, self.labels, self.bboxes)
            )
        else:
            ds = tf.data.Dataset.from_tensor_slices((images_np, self.labels))

        if self.shuffle_data:
            ds = ds.shuffle(buffer_size=min(self._n, 10000))

        def _process_with_bbox(image, label, bbox):
            if training and self.use_augment:
                image, bbox = augment_spatial_tf(image, bbox)
            image, label = self._preprocess(image, label, training)
            return image, label, bbox

        def _process_no_bbox(image, label):
            return self._preprocess(image, label, training)

        if self.bboxes is not None:
            ds = ds.map(
                lambda img, lbl, bbox: _process_with_bbox(img, lbl, bbox),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            ds = ds.map(
                lambda img, lbl: _process_no_bbox(img, lbl),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.batch(self.batch_size, drop_remainder=training)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


def build_datasets(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    image_size: int = 48,
    batch_size: int = 64,
    bbox_col: Optional[str] = None,
) -> Tuple:
    """Build train/val/test tf.data.Datasets."""
    train_ds = FERDatasetTF(
        csv_path=train_csv,
        image_size=image_size,
        use_augment=True,
        batch_size=batch_size,
        shuffle=True,
        bbox_col=bbox_col,
    ).build_dataset(training=True)

    val_ds = FERDatasetTF(
        csv_path=val_csv,
        image_size=image_size,
        use_augment=False,
        batch_size=batch_size,
        shuffle=False,
        bbox_col=bbox_col,
    ).build_dataset(training=False)

    test_ds = None
    if test_csv:
        test_ds = FERDatasetTF(
            csv_path=test_csv,
            image_size=image_size,
            use_augment=False,
            batch_size=batch_size,
            shuffle=False,
            bbox_col=bbox_col,
        ).build_dataset(training=False)

    return train_ds, val_ds, test_ds
