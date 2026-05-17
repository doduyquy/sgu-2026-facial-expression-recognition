"""
Precompute MediaPipe Face Mesh region masks for FER2013.

Output layout:
    outputs/mediapipe_region_masks/
      train/000000.npy
      val/000000.npy
      test/000000.npy

Each file stores [6, mask_size, mask_size] masks. The default mask size is 7
because ConvNeXt-Tiny final visual tokens are 7x7 for 224x224 inputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGION_LANDMARK_GROUPS = {
    "forehead": [10, 67, 69, 104, 108, 109, 151, 297, 299, 332, 337, 338],
    "left_eye": [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163],
    "right_eye": [263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390],
    "nose": [1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 327],
    "mouth": [13, 14, 17, 37, 39, 40, 61, 78, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    "chin": [136, 148, 149, 150, 152, 172, 176, 288, 361, 365, 377, 378, 379, 397, 400],
}
REGION_ORDER = ["forehead", "left_eye", "right_eye", "nose", "mouth", "chin"]


def import_mediapipe_face_mesh():
    version = "not installed"
    module_path = "unknown"
    try:
        import mediapipe as mp

        version = getattr(mp, "__version__", "unknown")
        module_path = getattr(mp, "__file__", "unknown")
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return mp.solutions.face_mesh
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for this script. On Kaggle, install a legacy "
            "wheel first, for example: pip install --force-reinstall "
            "'mediapipe==0.10.14'"
        ) from exc

    try:
        from mediapipe.python.solutions import face_mesh

        return face_mesh
    except ImportError as exc:
        raise ImportError(
            "MediaPipe FaceMesh legacy API is required for this script. "
            f"Current mediapipe version: {version}; module path: {module_path}. "
            "Neither 'mediapipe.solutions.face_mesh' nor "
            "'mediapipe.python.solutions.face_mesh' is available. On Kaggle, "
            "restart the session and reinstall with: pip install "
            "--force-reinstall 'mediapipe==0.10.14'"
        ) from exc


def pixels_to_rgb(pixels):
    image_vec = np.fromstring(pixels, sep=" ", dtype=np.uint8)
    gray = image_vec.reshape(48, 48)
    return np.repeat(gray[..., None], 3, axis=2)


def resize_nearest(image, size):
    from PIL import Image

    pil = Image.fromarray(image)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    pil = pil.resize((size, size), resample)
    return np.asarray(pil)


def uniform_masks(num_regions, mask_size, dtype):
    return np.ones((num_regions, mask_size, mask_size), dtype=dtype)


def render_gaussian_masks(landmarks, mask_size, sigma, dtype):
    yy, xx = np.meshgrid(
        np.arange(mask_size, dtype=np.float32),
        np.arange(mask_size, dtype=np.float32),
        indexing="ij",
    )
    masks = np.zeros((len(REGION_ORDER), mask_size, mask_size), dtype=np.float32)

    for region_idx, region_name in enumerate(REGION_ORDER):
        indices = REGION_LANDMARK_GROUPS[region_name]
        pts = np.asarray([[landmarks[i].x, landmarks[i].y] for i in indices], dtype=np.float32)
        pts = np.clip(pts, 0.0, 1.0)
        cx = float(pts[:, 0].mean() * (mask_size - 1))
        cy = float(pts[:, 1].mean() * (mask_size - 1))
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = np.exp(-dist_sq / (2.0 * sigma ** 2))
        mask = mask / max(float(mask.max()), 1e-8)
        masks[region_idx] = mask

    return masks.astype(dtype)


def detect_masks(face_mesh, pixels, mask_size, sigma, dtype, detection_size):
    image = pixels_to_rgb(pixels)
    if detection_size and detection_size != 48:
        image = resize_nearest(image, detection_size)

    result = face_mesh.process(image)
    if not result.multi_face_landmarks:
        return uniform_masks(len(REGION_ORDER), mask_size, dtype), True

    face = result.multi_face_landmarks[0]
    masks = render_gaussian_masks(face.landmark, mask_size=mask_size, sigma=sigma, dtype=dtype)
    return masks, False


def process_split(face_mesh, csv_path, output_dir, args, dtype):
    df = pd.read_csv(csv_path, usecols=["emotion", "pixels"])
    split_dir = output_dir / csv_path.stem
    split_dir.mkdir(parents=True, exist_ok=True)

    total = len(df) if args.max_samples is None else min(len(df), int(args.max_samples))
    fallback_count = 0
    saved_count = 0
    skipped_count = 0

    for row_index, row in df.iloc[:total].iterrows():
        mask_path = split_dir / f"{int(row_index):06d}.npy"
        if mask_path.exists() and not args.overwrite:
            skipped_count += 1
            continue

        masks, fallback = detect_masks(
            face_mesh=face_mesh,
            pixels=row["pixels"],
            mask_size=args.mask_size,
            sigma=args.sigma,
            dtype=dtype,
            detection_size=args.detection_size,
        )
        if fallback:
            fallback_count += 1
        np.save(mask_path, masks)
        saved_count += 1

        if saved_count % args.log_every == 0:
            print(
                f"[{csv_path.stem}] saved={saved_count}, skipped={skipped_count}, "
                f"fallback={fallback_count}, last={mask_path.name}"
            )

    return {
        "split": csv_path.stem,
        "total_requested": int(total),
        "saved": int(saved_count),
        "skipped": int(skipped_count),
        "fallback_uniform": int(fallback_count),
        "fallback_rate_saved": float(fallback_count / max(saved_count, 1)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/fer13-split")
    parser.add_argument("--output-dir", type=str, default="outputs/mediapipe_region_masks")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--mask-size", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=1.25)
    parser.add_argument("--detection-size", type=int, default=192)
    parser.add_argument("--save-dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.save_dtype == "float16" else np.float32

    face_mesh_solution = import_mediapipe_face_mesh()
    face_mesh = face_mesh_solution.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
    )

    summaries = []
    try:
        for split in args.splits:
            csv_path = data_dir / f"{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing split CSV: {csv_path}")
            summaries.append(process_split(face_mesh, csv_path, output_dir, args, dtype))
    finally:
        face_mesh.close()

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "mask_shape": [len(REGION_ORDER), args.mask_size, args.mask_size],
        "save_dtype": args.save_dtype,
        "sigma": args.sigma,
        "detection_size": args.detection_size,
        "splits": summaries,
    }
    summary_path = output_dir / "mediapipe_mask_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved summary:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
