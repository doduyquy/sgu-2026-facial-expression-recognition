"""
Precompute MediaPipe Face Mesh region masks for RAF-DB ImageFolder data.

Expected input layout:
    DATASET/
      train/1/*.jpg
      train/2/*.jpg
      ...
      test/7/*.jpg

Output layout:
    outputs/rafdb_mediapipe_region_masks/
      train/1/image.jpg.npy
      test/7/image.jpg.npy

Each file stores [6, mask_size, mask_size] masks. The default mask size is 7
because the proposed ConvNeXt-Tiny region-attention branch uses a 7x7 visual
token grid for 224x224 inputs.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.precompute_face_parsing_region_masks import (  # noqa: E402
    geometry_templates,
)
from scripts.precompute_mediapipe_region_masks import (  # noqa: E402
    REGION_ORDER,
    import_mediapipe_face_mesh,
    render_gaussian_masks,
    uniform_masks,
)
from scripts.train_rafdb_imagefolder import (  # noqa: E402
    CLASS_FOLDERS,
    RAFDB_FOLDER_TO_NAME,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_dataset_root(raw_root):
    root_value = str(raw_root)
    if root_value.lower() != "auto":
        root = Path(root_value)
        if not root.exists():
            raise FileNotFoundError(f"Configured RAF-DB root does not exist: {root}")
        return root

    candidates = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.glob("*/DATASET"))
        candidates.extend(input_root.glob("*/*/DATASET"))

    candidates.extend(Path.cwd().glob("*/DATASET"))
    candidates.extend(Path.cwd().glob("DATASET"))

    valid = [
        path
        for path in candidates
        if (path / "train").is_dir() and (path / "test").is_dir()
    ]
    if not valid:
        searched = "/kaggle/input/*/DATASET, /kaggle/input/*/*/DATASET, ./DATASET"
        raise FileNotFoundError(f"Could not auto-find RAF-DB DATASET root. Searched: {searched}")

    valid = sorted(set(path.resolve() for path in valid), key=lambda p: str(p))
    print(f"--> Auto-found RAF-DB root: {valid[0]}")
    return valid[0]


def validate_split(root, split):
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing RAF-DB split directory: {split_dir}")

    folders = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if folders != CLASS_FOLDERS:
        raise ValueError(f"{split} folders must be exactly {CLASS_FOLDERS}, got {folders}")


def iter_images(root, split):
    split_dir = root / split
    for class_folder in CLASS_FOLDERS:
        class_dir = split_dir / class_folder
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path


def read_rgb_image(image_path, detection_size):
    image = Image.open(image_path).convert("RGB")
    if detection_size and image.size != (detection_size, detection_size):
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        image = image.resize((detection_size, detection_size), resample)
    return np.asarray(image, dtype=np.uint8)


def detect_image_masks(face_mesh, image_path, args, dtype):
    if face_mesh is None:
        return geometry_templates(args.mask_size, args.mask_size).astype(dtype), "geometry_only"

    image = read_rgb_image(image_path, args.detection_size)
    result = face_mesh.process(np.ascontiguousarray(image))
    if not result.multi_face_landmarks:
        return uniform_masks(len(REGION_ORDER), args.mask_size, dtype), "uniform_fallback"

    face = result.multi_face_landmarks[0]
    masks = render_gaussian_masks(
        face.landmark,
        mask_size=args.mask_size,
        sigma=args.sigma,
        dtype=dtype,
    )
    return masks, "mediapipe_detected"


def mask_path_for_image(root, output_dir, split, image_path):
    relative = image_path.relative_to(root / split)
    return output_dir / split / relative.parent / f"{relative.name}.npy"


def process_split(face_mesh, root, output_dir, split, args, dtype, writer):
    validate_split(root, split)
    image_paths = list(iter_images(root, split))
    if args.max_samples is not None:
        image_paths = image_paths[: int(args.max_samples)]

    saved_count = 0
    skipped_count = 0
    fallback_count = 0

    for image_path in image_paths:
        mask_path = mask_path_for_image(root, output_dir, split, image_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        if mask_path.exists() and not args.overwrite:
            skipped_count += 1
            mask_mode = "skipped"
        else:
            masks, mask_mode = detect_image_masks(face_mesh, image_path, args, dtype)
            if mask_mode != "mediapipe_detected":
                fallback_count += 1
            np.save(mask_path, masks)
            saved_count += 1

        class_folder = image_path.relative_to(root / split).parts[0]
        writer.writerow(
            {
                "split": split,
                "class_folder": class_folder,
                "class_name": RAFDB_FOLDER_TO_NAME[class_folder],
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_mode": mask_mode,
                "fallback_uniform": int(mask_mode == "uniform_fallback"),
            }
        )

        processed = saved_count + skipped_count
        if processed % args.log_every == 0:
            print(
                f"[{split}] processed={processed}/{len(image_paths)}, "
                f"saved={saved_count}, skipped={skipped_count}, fallback={fallback_count}"
            )

    return {
        "split": split,
        "total_images": int(len(image_paths)),
        "saved": int(saved_count),
        "skipped": int(skipped_count),
        "fallback_uniform": int(fallback_count),
        "fallback_rate_saved": float(fallback_count / max(saved_count, 1)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/rafdb_mediapipe_region_masks")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--mask-size", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=1.25)
    parser.add_argument("--detection-size", type=int, default=224)
    parser.add_argument("--save-dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    root = resolve_dataset_root(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.save_dtype == "float16" else np.float32

    face_mesh = None
    mask_source = "geometry_only"
    if not args.geometry_only:
        try:
            face_mesh_solution = import_mediapipe_face_mesh()
            face_mesh = face_mesh_solution.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=args.min_detection_confidence,
            )
            mask_source = "mediapipe"
        except ImportError as exc:
            print(f"--> MediaPipe unavailable; falling back to geometry-only masks. Reason: {exc}")

    manifest_path = output_dir / "rafdb_mediapipe_mask_manifest.csv"
    summaries = []
    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
            fieldnames = [
                "split",
                "class_folder",
                "class_name",
                "image_path",
                "mask_path",
                "mask_mode",
                "fallback_uniform",
            ]
            writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
            writer.writeheader()
            for split in args.splits:
                summaries.append(process_split(face_mesh, root, output_dir, split, args, dtype, writer))
    finally:
        if face_mesh is not None:
            face_mesh.close()

    summary = {
        "data_root": str(root),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "mask_shape": [len(REGION_ORDER), args.mask_size, args.mask_size],
        "region_order": REGION_ORDER,
        "save_dtype": args.save_dtype,
        "sigma": args.sigma,
        "detection_size": args.detection_size,
        "mask_source": mask_source,
        "splits": summaries,
    }
    summary_path = output_dir / "rafdb_mediapipe_mask_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved manifest:", manifest_path)
    print("Saved summary:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
