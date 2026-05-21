"""
Research-grade MediaPipe semantic region extraction for FER2013.

Per sample output:
    {
        "masks": (6, 48, 48),
        "bboxes": (6, 4),
        "landmarks": (468, 2),
        "success": bool,
        "fallback_used": bool,
    }

Regions:
    0 -> forehead
    1 -> left_eye
    2 -> right_eye
    3 -> nose
    4 -> mouth
    5 -> cheek

Features:
    - Upscale 48 -> 256
    - CLAHE enhancement
    - Sharpen enhancement
    - MediaPipe FaceMesh
    - Retry detection
    - Flip retry
    - Gaussian semantic masks
    - Fallback template masks
    - Bounding boxes
    - Save .npz
    - Visualization support
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "Please install mediapipe:\n"
        "pip install mediapipe"
    ) from exc


REGION_GROUPS = {
    "forehead": [10, 67, 69, 104, 108, 109, 151, 297, 299, 332, 337, 338],
    "left_eye": [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163],
    "right_eye": [263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390],
    "nose": [1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 327],
    "mouth": [13, 14, 17, 37, 39, 40, 61, 78, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    "cheek": [50, 101, 118, 205, 206, 207, 280, 330, 347, 425, 426],
}

REGION_ORDER = ["forehead", "left_eye", "right_eye", "nose", "mouth", "cheek"]

DEFAULT_DATA_DIR = "dataset/fer13-split"
DEFAULT_OUTPUT_DIR = "dataset/semantic_masks"
DEFAULT_VIS_DIR = "dataset/semantic_vis"

MASK_SIZE = 48
UPSCALE_SIZE = 256
SIGMA = 4.0
SAVE_DTYPE = np.float32
MIN_DETECTION_CONFIDENCE = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Research-grade MediaPipe semantic region extraction for FER2013")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vis_dir", type=str, default=DEFAULT_VIS_DIR)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--vis_samples", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--detection_size", type=int, default=UPSCALE_SIZE)
    parser.add_argument("--min_detection_confidence", type=float, default=MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def import_face_mesh():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh
    from mediapipe.python.solutions import face_mesh

    return face_mesh


def build_face_mesh(min_detection_confidence=MIN_DETECTION_CONFIDENCE):
    face_mesh_mod = import_face_mesh()
    return face_mesh_mod.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=float(min_detection_confidence),
    )


def parse_pixels(pixels):
    values = np.fromstring(str(pixels), sep=" ", dtype=np.uint8)
    if values.size != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, got {values.size}")
    return values.reshape(48, 48)


def resize_gray(gray, size):
    if gray.shape == (size, size):
        return gray.astype(np.uint8)
    try:
        resample = cv2.INTER_CUBIC
    except AttributeError:
        resample = cv2.INTER_LINEAR
    return cv2.resize(gray.astype(np.uint8), (size, size), interpolation=resample)


def to_rgb(gray):
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def preprocess_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray.astype(np.uint8))


def preprocess_sharpen(gray):
    blurred = cv2.GaussianBlur(gray.astype(np.uint8), (0, 0), 3.0)
    return cv2.addWeighted(gray.astype(np.uint8), 1.5, blurred, -0.5, 0)


def preprocess_gamma(gray, gamma):
    arr = gray.astype(np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), float(gamma))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def preprocess_flip(gray):
    return cv2.flip(gray.astype(np.uint8), 1)


def preprocess_variant(gray, variant):
    if variant == "raw":
        return gray.astype(np.uint8), False
    if variant == "clahe":
        return preprocess_clahe(gray), False
    if variant == "sharpen":
        return preprocess_sharpen(gray), False
    if variant == "gamma_0_70":
        return preprocess_gamma(gray, 0.70), False
    if variant == "gamma_0_55":
        return preprocess_gamma(gray, 0.55), False
    if variant == "flip_raw":
        return preprocess_flip(gray), True
    if variant == "flip_clahe":
        return preprocess_flip(preprocess_clahe(gray)), True
    if variant == "flip_sharpen":
        return preprocess_flip(preprocess_sharpen(gray)), True
    if variant == "flip_gamma_0_70":
        return preprocess_flip(preprocess_gamma(gray, 0.70)), True
    raise ValueError(f"Unknown variant: {variant}")


def gray_to_input(gray, detection_size):
    upscaled = resize_gray(gray, int(detection_size))
    return to_rgb(upscaled), upscaled


def detect_face(face_mesh, gray, detection_size):
    image_rgb, _ = gray_to_input(gray, detection_size)
    result = face_mesh.process(image_rgb)
    if not result.multi_face_landmarks:
        return None
    return result.multi_face_landmarks[0]


def face_to_landmarks(face):
    points = np.zeros((468, 2), dtype=np.float32)
    landmarks = face.landmark[:468]
    count = min(len(landmarks), 468)
    for index in range(count):
        points[index, 0] = float(landmarks[index].x) * (MASK_SIZE - 1)
        points[index, 1] = float(landmarks[index].y) * (MASK_SIZE - 1)
    if count < 468:
        points[count:, :] = 0.0
    return points


def gaussian_mask(center_x, center_y, size=MASK_SIZE, sigma=SIGMA):
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    dist_sq = (xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2
    mask = np.exp(-dist_sq / (2.0 * float(sigma) ** 2))
    mask = mask / max(float(mask.max()), 1e-8)
    return mask.astype(SAVE_DTYPE)


def build_region_masks(landmark_points):
    masks = np.zeros((len(REGION_ORDER), MASK_SIZE, MASK_SIZE), dtype=SAVE_DTYPE)
    for region_index, region_name in enumerate(REGION_ORDER):
        indices = REGION_GROUPS[region_name]
        region_points = landmark_points[np.asarray(indices, dtype=np.int32)]
        valid = np.isfinite(region_points).all(axis=1)
        if not np.any(valid):
            continue
        region_points = region_points[valid]
        center_x = float(region_points[:, 0].mean())
        center_y = float(region_points[:, 1].mean())
        masks[region_index] = gaussian_mask(center_x, center_y)
    return masks


def fallback_template_masks():
    masks = np.zeros((len(REGION_ORDER), MASK_SIZE, MASK_SIZE), dtype=SAVE_DTYPE)
    masks[0, 0:10, 12:36] = 1.0
    masks[1, 10:20, 6:20] = 1.0
    masks[2, 10:20, 28:42] = 1.0
    masks[3, 18:32, 18:30] = 1.0
    masks[4, 30:42, 12:36] = 1.0
    masks[5, 18:38, 4:44] = 1.0
    return masks


def masks_to_bboxes(masks, threshold=0.2):
    bboxes = []
    for mask in masks:
        ys, xs = np.where(mask > float(threshold))
        if len(xs) == 0:
            bboxes.append([0, 0, MASK_SIZE - 1, MASK_SIZE - 1])
            continue
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())
        bboxes.append([x1, y1, x2, y2])
    return np.asarray(bboxes, dtype=SAVE_DTYPE)


def landmarks_to_bboxes(landmarks):
    points = np.asarray(landmarks, dtype=np.float32)
    bboxes = []
    for region_name in REGION_ORDER:
        indices = REGION_GROUPS[region_name]
        region_points = points[np.asarray(indices, dtype=np.int32)]
        valid = np.isfinite(region_points).all(axis=1)
        if not np.any(valid):
            bboxes.append([0, 0, MASK_SIZE - 1, MASK_SIZE - 1])
            continue
        region_points = region_points[valid]
        x1 = int(np.clip(np.floor(region_points[:, 0].min() - 2), 0, MASK_SIZE - 1))
        y1 = int(np.clip(np.floor(region_points[:, 1].min() - 2), 0, MASK_SIZE - 1))
        x2 = int(np.clip(np.ceil(region_points[:, 0].max() + 2), 0, MASK_SIZE - 1))
        y2 = int(np.clip(np.ceil(region_points[:, 1].max() + 2), 0, MASK_SIZE - 1))
        bboxes.append([x1, y1, x2, y2])
    return np.asarray(bboxes, dtype=SAVE_DTYPE)


def visualize(gray, masks, save_path):
    canvas = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colors = [
        (255, 216, 76),
        (48, 213, 200),
        (88, 142, 255),
        (190, 115, 255),
        (255, 86, 97),
        (96, 204, 112),
    ]
    for region_index, color in enumerate(colors):
        mask = cv2.resize(masks[region_index].astype(np.float32), (48, 48), interpolation=cv2.INTER_LINEAR)
        overlay = np.zeros_like(canvas, dtype=np.float32)
        overlay[:, :] = color
        alpha = np.clip(mask[..., None], 0.0, 1.0) * 0.45
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - alpha) + overlay * alpha, 0.0, 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)


def detect_with_retry(face_mesh, gray, detection_size):
    variants = [
        ("raw", gray),
        ("clahe", preprocess_clahe(gray)),
        ("sharpen", preprocess_sharpen(gray)),
        ("gamma_0_70", preprocess_gamma(gray, 0.70)),
        ("gamma_0_55", preprocess_gamma(gray, 0.55)),
        ("flip_raw", preprocess_flip(gray)),
        ("flip_clahe", preprocess_flip(preprocess_clahe(gray))),
        ("flip_sharpen", preprocess_flip(preprocess_sharpen(gray))),
        ("flip_gamma_0_70", preprocess_flip(preprocess_gamma(gray, 0.70))),
    ]

    for variant_name, variant_gray in variants:
        image_rgb, _ = gray_to_input(variant_gray, detection_size)
        face = detect_face(face_mesh, variant_gray, detection_size)
        if face is None:
            continue

        landmarks = face.landmark[:468]
        points = np.zeros((468, 2), dtype=np.float32)
        count = min(len(landmarks), 468)
        for index in range(count):
            x = float(landmarks[index].x)
            y = float(landmarks[index].y)
            if variant_name.startswith("flip_"):
                x = 1.0 - x
            points[index, 0] = x * (MASK_SIZE - 1)
            points[index, 1] = y * (MASK_SIZE - 1)

        return points, variant_name, False

    return None, None, True


def process_sample(face_mesh, pixels, detection_size):
    gray = parse_pixels(pixels)
    landmarks, retry_variant, failed = detect_with_retry(face_mesh, gray, detection_size)

    if failed:
        masks = fallback_template_masks()
        bboxes = masks_to_bboxes(masks)
        landmarks = np.zeros((468, 2), dtype=SAVE_DTYPE)
        success = False
        fallback_used = True
        variant_used = "fallback_template"
    else:
        masks = build_region_masks(landmarks)
        bboxes = landmarks_to_bboxes(landmarks)
        success = True
        fallback_used = False
        variant_used = retry_variant or "raw"

    return {
        "masks": masks.astype(SAVE_DTYPE),
        "bboxes": bboxes.astype(SAVE_DTYPE),
        "landmarks": landmarks.astype(SAVE_DTYPE),
        "success": success,
        "fallback_used": fallback_used,
        "variant_used": variant_used,
        "gray": gray,
    }


def process_split(csv_path, output_dir, vis_dir=None, vis_samples=20, max_samples=None, detection_size=UPSCALE_SIZE, min_detection_confidence=MIN_DETECTION_CONFIDENCE, log_every=1000):
    df = pd.read_csv(csv_path, usecols=["emotion", "pixels"])
    split_name = Path(csv_path).stem
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    vis_root = Path(vis_dir) if vis_dir else None
    if vis_root is not None:
        vis_root.mkdir(parents=True, exist_ok=True)

    total = len(df) if max_samples is None else min(len(df), int(max_samples))
    df = df.iloc[:total].copy()

    face_mesh = build_face_mesh(min_detection_confidence=min_detection_confidence)
    records = []
    success_count = 0
    fail_count = 0

    try:
        for idx, row in df.iterrows():
            result = process_sample(face_mesh, row["pixels"], detection_size=detection_size)
            save_path = split_dir / f"{int(idx):06d}.npz"
            np.savez_compressed(
                save_path,
                masks=result["masks"],
                bboxes=result["bboxes"],
                landmarks=result["landmarks"],
                success=np.asarray(result["success"], dtype=np.bool_),
                fallback_used=np.asarray(result["fallback_used"], dtype=np.bool_),
                variant_used=np.asarray(result["variant_used"]),
            )

            if result["success"]:
                success_count += 1
            else:
                fail_count += 1

            records.append(
                {
                    "sample_id": int(idx),
                    "emotion": int(row["emotion"]),
                    "success": bool(result["success"]),
                    "fallback_used": bool(result["fallback_used"]),
                    "variant_used": result["variant_used"],
                    "save_path": str(save_path),
                }
            )

            if vis_root is not None and len(records) <= int(vis_samples):
                vis_path = vis_root / f"{split_name}_{int(idx):06d}.png"
                visualize(result["gray"], result["masks"], str(vis_path))

            if (len(records) % int(log_every)) == 0:
                print(f"[{split_name}] {len(records)}/{len(df)} success={success_count} fail={fail_count}")
    finally:
        face_mesh.close()

    summary = {
        "split": split_name,
        "total": int(len(df)),
        "success": int(success_count),
        "fail": int(fail_count),
        "success_rate": float(success_count / max(len(df), 1)),
        "output_dir": str(split_dir),
        "mask_shape": [len(REGION_ORDER), MASK_SIZE, MASK_SIZE],
    }

    pd.DataFrame(records).to_csv(Path(output_dir) / f"semantic_manifest_{split_name}.csv", index=False, encoding="utf-8-sig")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / f"semantic_summary_{split_name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("===================================")
    print(f"Split: {split_name}")
    print(f"Success: {success_count}")
    print(f"Fail: {fail_count}")
    print(f"Success Rate: {100.0 * success_count / max(len(df), 1):.2f}%")
    print("===================================")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        process_split(
            csv_path=csv_path,
            output_dir=output_dir,
            vis_dir=vis_dir,
            vis_samples=args.vis_samples,
            max_samples=args.max_samples,
            detection_size=args.detection_size,
            min_detection_confidence=args.min_detection_confidence,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()