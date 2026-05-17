"""
Export FER2013 train images into MediaPipe detection success/failure folders.

The MediaPipe mask precompute script writes uniform all-one masks when face
detection fails. This exporter uses that convention to split train images into:
    outputs/mediapipe_train_detection_folders/detect_success/
    outputs/mediapipe_train_detection_folders/detect_failed/

It also saves per-image mask preview sheets into:
    outputs/mediapipe_train_detection_folders/detect_success_preview/
    outputs/mediapipe_train_detection_folders/detect_failed_preview/
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

REGION_ORDER = ["forehead", "left_eye", "right_eye", "nose", "mouth", "chin"]
REGION_COLORS = np.asarray(
    [
        [255, 216, 76],   # forehead
        [48, 213, 200],   # left eye
        [88, 142, 255],   # right eye
        [190, 115, 255],  # nose
        [255, 86, 97],    # mouth
        [96, 204, 112],   # chin
    ],
    dtype=np.float32,
) / 255.0


def repo_root():
    return Path(__file__).resolve().parents[1]


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Split FER2013 train images by MediaPipe detection result."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=root / "dataset" / "fer13-split" / "train.csv",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=root
        / "mediapipe_region_masks"
        / "mediapipe_region_masks"
        / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "mediapipe_train_detection_folders",
    )
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--preview-alpha", type=float, default=0.55)
    parser.add_argument("--preview-padding", type=int, default=8)
    parser.add_argument("--preview-title-height", type=int, default=38)
    parser.add_argument(
        "--skip-mask-previews",
        action="store_true",
        help="Only export plain images and CSV/JSON logs.",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def pixels_to_image(pixels):
    image_vec = np.fromstring(pixels, sep=" ", dtype=np.uint8)
    if image_vec.size != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, got {image_vec.size}")
    return image_vec.reshape(48, 48)


def resize_image(image, size):
    if size == 48:
        return image
    try:
        import cv2

        return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        from PIL import Image

        try:
            resample = Image.Resampling.NEAREST
        except AttributeError:
            resample = Image.NEAREST
        return np.asarray(Image.fromarray(image).resize((size, size), resample))


def write_png_gray(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Could not encode PNG: {path}")
        encoded.tofile(str(path))
    except ImportError:
        from PIL import Image

        Image.fromarray(image).save(path)


def rgb_float_to_uint8(image):
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def gray_to_rgb_float(image):
    return np.repeat((image.astype(np.float32) / 255.0)[..., None], 3, axis=2)


def resize_float_image(image, size):
    if image.shape[:2] == (size, size):
        return image.astype(np.float32)
    try:
        import cv2

        return cv2.resize(
            image.astype(np.float32),
            (size, size),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
    except ImportError:
        from PIL import Image

        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        return np.asarray(
            Image.fromarray(image.astype(np.float32), mode="F").resize((size, size), resample),
            dtype=np.float32,
        )


def normalize_mask(mask):
    return np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)


def colorize_single_mask(base_rgb, mask, color, alpha):
    mask = normalize_mask(mask)[..., None]
    return np.clip(base_rgb * (1.0 - alpha * mask) + color * alpha * mask, 0.0, 1.0)


def build_combined_overlay(base_rgb, masks, alpha):
    overlay = base_rgb.copy()
    coverage = np.zeros(base_rgb.shape[:2], dtype=np.float32)
    for region_idx, color in enumerate(REGION_COLORS):
        mask = normalize_mask(masks[region_idx])
        coverage = np.maximum(coverage, mask)
        overlay = np.clip(
            overlay * (1.0 - alpha * mask[..., None]) + color * alpha * mask[..., None],
            0.0,
            1.0,
        )
    return overlay, coverage


def draw_title(canvas_rgb, title, x=4, y=13):
    try:
        import cv2

        for line_idx, line in enumerate(title.split("\n")):
            cv2.putText(
                canvas_rgb,
                line,
                (x, y + line_idx * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        return canvas_rgb
    except ImportError:
        from PIL import Image, ImageDraw, ImageFont

        pil = Image.fromarray(canvas_rgb)
        draw = ImageDraw.Draw(pil)
        font = ImageFont.load_default()
        for line_idx, line in enumerate(title.split("\n")):
            draw.text((x, 4 + line_idx * 13), line, fill=(20, 20, 20), font=font)
        return np.asarray(pil)


def make_tile(image_rgb, title, tile_size, title_height):
    tile = np.full((tile_size + title_height, tile_size, 3), 255, dtype=np.uint8)
    image_uint8 = rgb_float_to_uint8(image_rgb)
    if image_uint8.shape[:2] != (tile_size, tile_size):
        try:
            import cv2

            image_uint8 = cv2.resize(
                image_uint8,
                (tile_size, tile_size),
                interpolation=cv2.INTER_LINEAR,
            )
        except ImportError:
            from PIL import Image

            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.BILINEAR
            image_uint8 = np.asarray(
                Image.fromarray(image_uint8).resize((tile_size, tile_size), resample)
            )
    tile = draw_title(tile, title)
    tile[title_height:title_height + tile_size, :tile_size] = image_uint8
    return tile


def write_png_rgb(path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError(f"Could not encode PNG: {path}")
        encoded.tofile(str(path))
    except ImportError:
        from PIL import Image

        Image.fromarray(image_rgb).save(path)


def save_mask_preview(path, image_gray, masks, label_title, reason, alpha, padding, title_height):
    tile_size = int(image_gray.shape[0])
    base_rgb = gray_to_rgb_float(image_gray)
    row_tiles = [make_tile(base_rgb, label_title, tile_size, title_height)]

    if masks is None:
        row_tiles.append(make_tile(base_rgb, f"no mask\n{reason}", tile_size, title_height))
    else:
        masks_resized = np.stack(
            [resize_float_image(mask, tile_size) for mask in masks],
            axis=0,
        )
        masks_resized = np.clip(masks_resized, 0.0, 1.0)
        combined, coverage = build_combined_overlay(base_rgb, masks_resized, alpha)
        row_tiles.append(
            make_tile(combined, f"all regions\nmax={coverage.max():.2f}", tile_size, title_height)
        )

        for region_idx, region_name in enumerate(REGION_ORDER):
            overlay = colorize_single_mask(
                base_rgb,
                masks_resized[region_idx],
                REGION_COLORS[region_idx],
                alpha,
            )
            row_tiles.append(
                make_tile(
                    overlay,
                    f"{region_name}\nmax={masks_resized[region_idx].max():.2f}",
                    tile_size,
                    title_height,
                )
            )

    tile_height = tile_size + title_height
    canvas_width = padding + len(row_tiles) * (tile_size + padding)
    canvas_height = padding + tile_height + padding
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    for col_idx, tile in enumerate(row_tiles):
        x = padding + col_idx * (tile_size + padding)
        canvas[padding:padding + tile_height, x:x + tile_size] = tile
    write_png_rgb(path, canvas)


def classify_mask(mask_path, atol):
    if not mask_path.exists():
        return "detect_failed", "missing_mask", None

    masks = np.load(mask_path)
    if masks.ndim != 3 or masks.shape[0] != len(REGION_ORDER):
        return "detect_failed", f"invalid_shape_{masks.shape}", None

    if np.allclose(masks, 1.0, atol=atol):
        return "detect_failed", "uniform_fallback", masks.astype(np.float32)

    return "detect_success", "non_uniform_mask", masks.astype(np.float32)


def relative_or_absolute(path, start):
    try:
        return str(path.relative_to(start))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    if not args.train_csv.exists():
        raise FileNotFoundError(args.train_csv)
    if not args.mask_dir.exists():
        raise FileNotFoundError(args.mask_dir)

    success_dir = args.output_dir / "detect_success"
    failed_dir = args.output_dir / "detect_failed"
    success_preview_dir = args.output_dir / "detect_success_preview"
    failed_preview_dir = args.output_dir / "detect_failed_preview"
    manifest_path = args.output_dir / "train_mediapipe_detection_manifest.csv"
    summary_path = args.output_dir / "train_mediapipe_detection_summary.json"
    success_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_mask_previews:
        success_preview_dir.mkdir(parents=True, exist_ok=True)
        failed_preview_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "total": 0,
        "detect_success": 0,
        "detect_failed": 0,
        "missing_mask": 0,
        "uniform_fallback": 0,
        "invalid_mask": 0,
        "mask_previews": 0,
    }

    with args.train_csv.open("r", encoding="utf-8", newline="") as f_in, manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "row_index",
                "emotion",
                "emotion_name",
                "status",
                "reason",
                "image_file",
                "preview_file",
                "mask_file",
            ],
        )
        writer.writeheader()

        for row_index, row in enumerate(reader):
            if args.limit is not None and row_index >= args.limit:
                break

            emotion = int(row["emotion"])
            emotion_name = EMOTION_DICT.get(emotion, str(emotion))
            mask_path = args.mask_dir / f"{row_index:06d}.npy"
            status, reason, masks = classify_mask(mask_path, args.atol)
            out_dir = success_dir if status == "detect_success" else failed_dir
            out_path = out_dir / f"{row_index:06d}_{emotion_name}.png"
            preview_dir = (
                success_preview_dir if status == "detect_success" else failed_preview_dir
            )
            preview_path = preview_dir / f"{row_index:06d}_{emotion_name}_mask_preview.png"

            image = pixels_to_image(row["pixels"])
            image = resize_image(image, args.image_size)
            write_png_gray(out_path, image)
            if not args.skip_mask_previews:
                save_mask_preview(
                    path=preview_path,
                    image_gray=image,
                    masks=masks,
                    label_title=f"idx {row_index}\n{emotion_name}",
                    reason=reason,
                    alpha=float(args.preview_alpha),
                    padding=int(args.preview_padding),
                    title_height=int(args.preview_title_height),
                )
                counts["mask_previews"] += 1

            counts["total"] += 1
            counts[status] += 1
            if reason == "missing_mask":
                counts["missing_mask"] += 1
            elif reason == "uniform_fallback":
                counts["uniform_fallback"] += 1
            elif reason.startswith("invalid_shape_"):
                counts["invalid_mask"] += 1

            writer.writerow(
                {
                    "row_index": row_index,
                    "emotion": emotion,
                    "emotion_name": emotion_name,
                    "status": status,
                    "reason": reason,
                    "image_file": str(out_path),
                    "preview_file": str(preview_path) if not args.skip_mask_previews else "",
                    "mask_file": str(mask_path),
                }
            )

            if counts["total"] % args.log_every == 0:
                print(
                    f"processed={counts['total']} "
                    f"success={counts['detect_success']} "
                    f"failed={counts['detect_failed']} "
                    f"previews={counts['mask_previews']}"
                )

    summary = {
        "train_csv": str(args.train_csv),
        "mask_dir": str(args.mask_dir),
        "output_dir": str(args.output_dir),
        "success_dir": str(success_dir),
        "failed_dir": str(failed_dir),
        "success_preview_dir": str(success_preview_dir) if not args.skip_mask_previews else None,
        "failed_preview_dir": str(failed_preview_dir) if not args.skip_mask_previews else None,
        "manifest": str(manifest_path),
        "image_size": int(args.image_size),
        "mask_previews_enabled": not args.skip_mask_previews,
        "counts": counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
