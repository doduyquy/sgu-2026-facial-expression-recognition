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
    # Native Keras layers for color jitter (spatial is handled in pure TF function now)
    if use_augment:
        layers += [
            tf.keras.layers.RandomContrast(factor=0.3),
            tf.keras.layers.RandomBrightness(factor=0.3),
        ]
    return tf.keras.Sequential(layers)


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Normalize to [-1, 1] (same as torchvision Normalize mean=0.5, std=0.5)."""
    image = tf.cast(image, tf.float32) / 255.0
    return (image - 0.5) / 0.5


# ---------------------------------------------------------------------------
# CSV-based FER Dataset → tf.data.Dataset
# ---------------------------------------------------------------------------

@tf.function
def augment_spatial_tf(image: tf.Tensor, bbox: tf.Tensor):
    """
    Pure TensorFlow spatial augmentation (Random Flip) to avoid py_function 
    random state caching issues in graph mode.
    """
    image = tf.cast(image, tf.float32)
    bbox = tf.cast(bbox, tf.float32)
    
    do_flip = tf.random.uniform([]) < 0.5
    
    def apply_flip():
        flipped_img = tf.image.flip_left_right(image)
        # Bbox flip: x1_new = 47.0 - x2, x2_new = 47.0 - x1
        x1 = 47.0 - bbox[:, 2]
        y1 = bbox[:, 1]
        x2 = 47.0 - bbox[:, 0]
        y2 = bbox[:, 3]
        
        flipped_bbox = tf.stack([x1, y1, x2, y2], axis=-1)
        
        # Swap symmetric pairs: (1, 2), (4, 5), (7, 8)
        indices = tf.constant([0, 2, 1, 3, 5, 4, 6, 8, 7], dtype=tf.int32)
        flipped_bbox = tf.gather(flipped_bbox, indices)
        return flipped_img, flipped_bbox
        
    image, bbox = tf.cond(do_flip, apply_flip, lambda: (image, bbox))
    return image, bbox

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
