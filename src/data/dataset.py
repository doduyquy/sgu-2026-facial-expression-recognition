import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT

def create_fer2013_dataset(data_path, split="train", transforms=None, semantic_masks_dir=None, num_regions=9, use_semantic_manifest=True):
    data_split_path = os.path.join(data_path, f"{split}.csv")
    data = pd.read_csv(data_split_path, usecols=[0, 1])
    semantic_masks_dir = Path(semantic_masks_dir) if semantic_masks_dir else None

    semantic_manifest = None
    if semantic_masks_dir is not None and use_semantic_manifest:
        manifest_path = semantic_masks_dir / f"semantic_manifest_{split}.csv"
        if manifest_path.exists():
            semantic_manifest = pd.read_csv(manifest_path)

    def generator():
        for index in range(len(data)):
            emotion, pixels = data.iloc[index].values
            label = int(emotion)

            image_vec = np.fromstring(pixels, sep=' ', dtype=np.uint8)
            image_np = image_vec.reshape((48, 48))

            if semantic_masks_dir is not None:
                mask_path = semantic_masks_dir / split / f"{int(index):06d}.npz"

                detect_success = True
                fallback_used = False
                variant_used = "unknown"
                
                if semantic_manifest is not None and index < len(semantic_manifest):
                    manifest_row = semantic_manifest.iloc[index]
                    if "success" in manifest_row:
                        detect_success = bool(manifest_row["success"])
                    if "fallback_used" in manifest_row:
                        fallback_used = bool(manifest_row["fallback_used"])
                    if "variant_used" in manifest_row:
                        variant_used = str(manifest_row["variant_used"])

                if mask_path.exists():
                    with np.load(mask_path, allow_pickle=False) as npz:
                        bboxes = npz["bboxes"].astype(np.float32)
                else:
                    fallback_used = True
                    detect_success = False
                    bboxes = np.zeros((num_regions, 4), dtype=np.float32)
                    bboxes[:, 0] = 0.0
                    bboxes[:, 1] = 0.0
                    bboxes[:, 2] = 47.0
                    bboxes[:, 3] = 47.0

                x1 = bboxes[:, 0]
                y1 = bboxes[:, 1]
                x2 = bboxes[:, 2]
                y2 = bboxes[:, 3]
                finite_mask = np.isfinite(bboxes).all(axis=1)
                order_mask = (x2 > x1) & (y2 > y1)
                size_mask = ((x2 - x1) >= 2.0) & ((y2 - y1) >= 2.0)
                region_mask = (finite_mask & order_mask & size_mask).astype(np.float32)

                if detect_success:
                    width = np.clip(x2 - x1, 1.0, None)
                    height = np.clip(y2 - y1, 1.0, None)
                    area = (width * height) / float(48 * 48)
                    region_confidence = np.clip(0.5 + 0.5 * area, 0.0, 1.0).astype(np.float32)
                else:
                    region_confidence = (0.15 * region_mask).astype(np.float32)

                if split == "train" and np.random.rand() < 0.5:
                    image_np = np.flip(image_np, axis=1)

                    flipped_bboxes = bboxes.copy()
                    flipped_bboxes[:, 0] = 47.0 - bboxes[:, 2]
                    flipped_bboxes[:, 2] = 47.0 - bboxes[:, 0]

                    swap_pairs = [(1, 2), (4, 5), (7, 8)]
                    for i, j in swap_pairs:
                        tmp = flipped_bboxes[i].copy()
                        flipped_bboxes[i] = flipped_bboxes[j]
                        flipped_bboxes[j] = tmp

                        region_mask[i], region_mask[j] = region_mask[j], region_mask[i]
                        region_confidence[i], region_confidence[j] = region_confidence[j], region_confidence[i]

                    bboxes = flipped_bboxes

                if split == "train" and np.random.rand() < 0.5:
                    angle_deg = np.random.uniform(-10.0, 10.0)
                    tx = np.random.uniform(-4.8, 4.8)
                    ty = np.random.uniform(-4.8, 4.8)
                    scale = np.random.uniform(0.9, 1.1)

                    theta = np.radians(angle_deg)
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    cx, cy = 23.5, 23.5

                    matrix = np.array([
                        [scale * cos_t, scale * sin_t, tx + cx - scale*(cx*cos_t + cy*sin_t)],
                        [-scale * sin_t, scale * cos_t, ty + cy - scale*(-cx*sin_t + cy*cos_t)],
                        [0, 0, 1]
                    ])
                    inv_matrix = np.linalg.inv(matrix)
                    
                    pil_img = Image.fromarray(image_np)
                    pil_img = pil_img.transform(
                        (48, 48), Image.AFFINE, data=inv_matrix.flatten()[:6], resample=Image.BILINEAR
                    )
                    image_np = np.array(pil_img)

                    new_bboxes = bboxes.copy()
                    for r in range(num_regions):
                        if region_mask[r] == 0:
                            continue
                        x1, y1, x2, y2 = bboxes[r]
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
                            region_mask[r] = 0.0
                            region_confidence[r] = 0.0
                        else:
                            new_bboxes[r] = [x1_n, y1_n, x2_n, y2_n]
                    bboxes = new_bboxes

                image_np = image_np[..., np.newaxis] # H, W, 1
                yield {
                    "image": image_np,
                    "label": np.int64(label),
                    "bboxes": bboxes,
                    "region_mask": region_mask,
                    "region_confidence": region_confidence,
                    "detect_success": detect_success,
                    "fallback_used": fallback_used
                }
            else:
                image_np = image_np[..., np.newaxis] # H, W, 1
                yield {
                    "image": image_np,
                    "label": np.int64(label)
                }

    if semantic_masks_dir is not None:
        output_signature = {
            "image": tf.TensorSpec(shape=(48, 48, 1), dtype=tf.uint8),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64),
            "bboxes": tf.TensorSpec(shape=(num_regions, 4), dtype=tf.float32),
            "region_mask": tf.TensorSpec(shape=(num_regions,), dtype=tf.float32),
            "region_confidence": tf.TensorSpec(shape=(num_regions,), dtype=tf.float32),
            "detect_success": tf.TensorSpec(shape=(), dtype=tf.bool),
            "fallback_used": tf.TensorSpec(shape=(), dtype=tf.bool),
        }
    else:
        output_signature = {
            "image": tf.TensorSpec(shape=(48, 48, 1), dtype=tf.uint8),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64)
        }

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if transforms is not None:
        ds = ds.map(lambda d: {**d, "image": transforms(d["image"])}, num_parallel_calls=tf.data.AUTOTUNE)

    return ds