"""
dataset_tf.py — High-performance tf.data pipeline for FER2013 with 9-region semantic masks.
Preserves bounding-box synchronicity during data augmentation (flip + affine).
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf


class FER2013TFDataset:
    """Load FER2013 dataset with synchronized 9-region bounding box transforms."""
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        semantic_masks_dir: Optional[str] = None,
        num_regions: int = 9,
    ):
        self.split = split
        self.data_csv = Path(data_path) / f"{split}.csv"
        self.semantic_masks_dir = Path(semantic_masks_dir) if semantic_masks_dir else None
        self.num_regions = num_regions

        df = pd.read_csv(self.data_csv, usecols=[0, 1])
        self.labels = df.iloc[:, 0].values.astype(np.int32)
        # Parse space-delimited pixel strings into (N, 48, 48, 1) float32 arrays in [0, 1]
        print(f"--> Loading {split} pixels into memory ({len(df)} samples)...")
        pixel_arrays = np.array([np.fromstring(p, sep=" ", dtype=np.uint8) for p in df.iloc[:, 1].values])
        self.images = (pixel_arrays.reshape(-1, 48, 48, 1).astype(np.float32) / 255.0)

        # Standard FER2013 normalization: (img - 0.5) / 0.5
        self.images = (self.images - 0.5) / 0.5

        # Pre-load or generate fallback bboxes
        self.bboxes = np.zeros((len(df), num_regions, 4), dtype=np.float32)
        self.region_masks = np.ones((len(df), num_regions), dtype=np.float32)
        self.region_confs = np.ones((len(df), num_regions), dtype=np.float32)

        has_masks = self.semantic_masks_dir is not None and self.semantic_masks_dir.exists()
        if has_masks:
            print(f"--> Loading semantic masks from {self.semantic_masks_dir / split}...")
            for idx in range(len(df)):
                mask_file = self.semantic_masks_dir / split / f"{idx:06d}.npz"
                if mask_file.exists():
                    try:
                        with np.load(mask_file, allow_pickle=False) as npz:
                            box = npz["bboxes"].astype(np.float32)
                            x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
                            valid = np.isfinite(box).all(axis=1) & (x2 > x1 + 1.0) & (y2 > y1 + 1.0)
                            self.bboxes[idx] = np.clip(box, 0.0, 47.0)
                            self.region_masks[idx] = valid.astype(np.float32)
                            area = np.clip((x2 - x1) * (y2 - y1), 1.0, None) / (48.0 * 48.0)
                            self.region_confs[idx] = np.clip(0.5 + 0.5 * area, 0.0, 1.0) * self.region_masks[idx]
                    except Exception:
                        self.bboxes[idx, :] = [0.0, 0.0, 47.0, 47.0]
                        self.region_confs[idx] = 0.15
                else:
                    self.bboxes[idx, :] = [0.0, 0.0, 47.0, 47.0]
                    self.region_confs[idx] = 0.15
        else:
            self.bboxes[:, :] = [0.0, 0.0, 47.0, 47.0]

    def __len__(self):
        return len(self.labels)


def augment_sample_np(image, label, bboxes, mask, conf):
    """Synchronized NumPy augmentation function executed inside tf.numpy_function."""
    # 1. Synchronized Horizontal Flip (50% prob)
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        flipped_boxes = bboxes.copy()
        flipped_boxes[:, 0] = 47.0 - bboxes[:, 2]
        flipped_boxes[:, 2] = 47.0 - bboxes[:, 0]

        # Swap symmetric pairs: 1<->2, 4<->5, 7<->8
        swap = [0, 2, 1, 3, 5, 4, 6, 8, 7]
        bboxes = flipped_boxes[swap]
        mask = mask[swap]
        conf = conf[swap]

    # 2. Synchronized Random Affine (50% prob)
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-10.0, 10.0)
        tx = np.random.uniform(-4.0, 4.0)
        ty = np.random.uniform(-4.0, 4.0)
        scale = np.random.uniform(0.9, 1.1)

        # Simple 2D affine on image
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = 23.5, 23.5

        new_boxes = bboxes.copy()
        for r in range(9):
            if mask[r] == 0:
                continue
            x1, y1, x2, y2 = bboxes[r]
            corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            dx = corners[:, 0] - cx
            dy = corners[:, 1] - cy
            x_new = dx * scale * cos_t + dy * scale * sin_t + cx + tx
            y_new = -dx * scale * sin_t + dy * scale * cos_t + cy + ty
            x1_n = np.clip(np.min(x_new), 0.0, 47.0)
            y1_n = np.clip(np.min(y_new), 0.0, 47.0)
            x2_n = np.clip(np.max(x_new), 0.0, 47.0)
            y2_n = np.clip(np.max(y_new), 0.0, 47.0)

            if (x2_n - x1_n < 2.0) or (y2_n - y1_n < 2.0):
                mask[r] = 0.0
                conf[r] = 0.0
            else:
                new_boxes[r] = [x1_n, y1_n, x2_n, y2_n]
        bboxes = new_boxes

    return image.astype(np.float32), label, bboxes.astype(np.float32), mask.astype(np.float32), conf.astype(np.float32)


def create_tf_dataloader(
    data_path: str,
    split: str = "train",
    batch_size: int = 64,
    semantic_masks_dir: Optional[str] = None,
    is_training: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create optimized tf.data.Dataset for training or validation."""
    raw_ds = FER2013TFDataset(data_path, split=split, semantic_masks_dir=semantic_masks_dir)

    ds = tf.data.Dataset.from_tensor_slices((
        raw_ds.images,
        raw_ds.labels,
        raw_ds.bboxes,
        raw_ds.region_masks,
        raw_ds.region_confs,
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(raw_ds), 5000), reshuffle_each_iteration=True)

    if is_training:
        def _py_aug(img, lbl, box, msk, cnf):
            return tf.numpy_function(
                func=augment_sample_np,
                inp=[img, lbl, box, msk, cnf],
                Tout=[tf.float32, tf.int32, tf.float32, tf.float32, tf.float32]
            )
        ds = ds.map(_py_aug, num_parallel_calls=tf.data.AUTOTUNE)

    # Set explicit static shapes
    def _set_shapes(img, lbl, box, msk, cnf):
        img.set_shape([48, 48, 1])
        lbl.set_shape([])
        box.set_shape([9, 4])
        msk.set_shape([9])
        cnf.set_shape([9])
        return {"images": img, "bboxes": box, "region_mask": msk, "region_confidence": cnf}, lbl

    ds = ds.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
