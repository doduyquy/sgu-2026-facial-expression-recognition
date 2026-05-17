"""
Visualize precomputed MediaPipe region masks for FER2013.

The script reads FER2013 split CSV rows and matching mask .npy files, then
saves a contact-sheet PNG with:
    original image | combined colored overlay | 6 individual region overlays
"""

import argparse
import csv
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save visual previews of MediaPipe FER region masks."
    )
    parser.add_argument("--data-dir", type=str, default="dataset/fer13-split")
    parser.add_argument("--mask-dir", type=str, default="outputs/mediapipe_region_masks")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="outputs/mediapipe_mask_previews")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--display-size", type=int, default=112)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--title-height", type=int, default=38)
    return parser.parse_args()


def find_split_data_dir(data_dir):
    requested = Path(data_dir)
    if (requested / "train.csv").exists():
        return requested

    candidates = [
        Path("/kaggle/input/fer13-split"),
        Path("/kaggle/input/fer13-split/doduyquynii"),
        Path("/kaggle/input/datasets/doduyquynii/fer13-split"),
        Path("/kaggle/input/datasets/doduyquynii"),
        Path("dataset/fer13-split"),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            print(f"--> Using discovered data-dir: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Could not find train.csv/val.csv/test.csv under data-dir: {data_dir}"
    )


def find_mask_dir(mask_dir, split):
    requested = Path(mask_dir)
    if (requested / split).exists():
        return requested

    candidates = [
        Path("/kaggle/working/outputs/mediapipe_region_masks"),
        Path("/kaggle/input/fer2013-mediapipe-region-masks/mediapipe_region_masks"),
        Path("/kaggle/working/fer2013-mediapipe-region-masks-input/mediapipe_region_masks"),
        Path("outputs/mediapipe_region_masks"),
    ]
    for candidate in candidates:
        if (candidate / split).exists():
            print(f"--> Using discovered mask-dir: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Could not find mask split folder '{split}' under mask-dir: {mask_dir}. "
        "Run scripts/precompute_mediapipe_region_masks.py first."
    )


def read_split_csv(data_dir, split):
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "emotion" not in reader.fieldnames or "pixels" not in reader.fieldnames:
            raise ValueError(
                f"Expected CSV columns 'emotion' and 'pixels' in {csv_path}."
            )
        for original_idx, row in enumerate(reader):
            rows.append(
                {
                    "emotion": int(row["emotion"]),
                    "pixels": row["pixels"],
                    "original_idx": original_idx,
                }
            )
    return rows


def pixels_to_gray(pixels):
    image_vec = np.fromstring(pixels, sep=" ", dtype=np.uint8)
    return image_vec.reshape(48, 48)


def get_bilinear_resample():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for mask visualization. Install it with: "
            "pip install pillow"
        ) from exc

    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def resize_float(image, size):
    from PIL import Image

    mode = get_bilinear_resample()
    pil = Image.fromarray(image.astype(np.float32), mode="F")
    return np.asarray(pil.resize((size, size), mode), dtype=np.float32)


def resize_uint8(image, size):
    from PIL import Image

    mode = get_bilinear_resample()
    return np.asarray(Image.fromarray(image).resize((size, size), mode), dtype=np.uint8)


def normalize_mask(mask):
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


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
            overlay * (1.0 - alpha * mask[..., None])
            + color * alpha * mask[..., None],
            0.0,
            1.0,
        )
    return overlay, coverage


def load_sample(rows, mask_dir, split, index, display_size):
    if index < 0 or index >= len(rows):
        raise IndexError(f"Index {index} is outside split '{split}' with {len(rows)} rows.")

    row = rows[index]
    original_idx = int(row["original_idx"])
    mask_path = mask_dir / split / f"{original_idx:06d}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask file: {mask_path}")

    gray = pixels_to_gray(row["pixels"])
    gray_resized = resize_uint8(gray, display_size)
    base_rgb = np.repeat((gray_resized.astype(np.float32) / 255.0)[..., None], 3, axis=2)

    masks = np.load(mask_path).astype(np.float32)
    if masks.ndim != 3 or masks.shape[0] != len(REGION_ORDER):
        raise ValueError(f"Expected mask shape [6,H,W], got {masks.shape} at {mask_path}")

    masks_resized = np.stack(
        [resize_float(mask, display_size) for mask in masks],
        axis=0,
    )
    masks_resized = np.clip(masks_resized, 0.0, 1.0)

    return {
        "index": index,
        "original_idx": original_idx,
        "label": row["emotion"],
        "mask_path": mask_path,
        "base_rgb": base_rgb,
        "masks": masks_resized,
    }


def select_indices(rows, mask_dir, split, requested_indices, start, count):
    if requested_indices:
        return requested_indices

    selected = []
    cursor = max(0, start)
    while cursor < len(rows) and len(selected) < count:
        mask_path = mask_dir / split / f"{cursor:06d}.npy"
        if mask_path.exists():
            selected.append(cursor)
        cursor += 1

    if not selected:
        raise FileNotFoundError(
            f"No mask .npy files found for split '{split}' under {mask_dir / split}."
        )
    return selected


def rgb_float_to_pil(image):
    from PIL import Image

    image = np.clip(image, 0.0, 1.0)
    return Image.fromarray((image * 255.0).astype(np.uint8), mode="RGB")


def make_tile(image, title, tile_size, title_height):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (tile_size, tile_size + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for line_idx, line in enumerate(title.split("\n")):
        draw.text((4, 4 + line_idx * 13), line, fill=(20, 20, 20), font=font)
    canvas.paste(
        image.resize((tile_size, tile_size), get_bilinear_resample()),
        (0, title_height),
    )
    return canvas


def save_preview(samples, output_path, alpha, padding, title_height):
    from PIL import Image

    columns = ["image", "combined"] + REGION_ORDER
    tile_size = samples[0]["base_rgb"].shape[0]
    tile_width = tile_size
    tile_height = tile_size + title_height
    canvas_width = padding + len(columns) * (tile_width + padding)
    canvas_height = padding + len(samples) * (tile_height + padding)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    for row_idx, sample in enumerate(samples):
        base_rgb = sample["base_rgb"]
        masks = sample["masks"]
        label_name = EMOTION_DICT.get(sample["label"], str(sample["label"]))
        combined, coverage = build_combined_overlay(base_rgb, masks, alpha)

        row_tiles = [
            make_tile(
                rgb_float_to_pil(base_rgb),
                f"idx {sample['index']}\n{label_name}",
                tile_size=tile_size,
                title_height=title_height,
            ),
            make_tile(
                rgb_float_to_pil(combined),
                f"all regions\nmax={coverage.max():.2f}",
                tile_size=tile_size,
                title_height=title_height,
            ),
        ]

        for region_idx, region_name in enumerate(REGION_ORDER):
            overlay = colorize_single_mask(
                base_rgb,
                masks[region_idx],
                REGION_COLORS[region_idx],
                alpha,
            )
            row_tiles.append(
                make_tile(
                    rgb_float_to_pil(overlay),
                    f"{region_name}\nmax={masks[region_idx].max():.2f}",
                    tile_size=tile_size,
                    title_height=title_height,
                )
            )

        y = padding + row_idx * (tile_height + padding)
        for col_idx, tile in enumerate(row_tiles):
            x = padding + col_idx * (tile_width + padding)
            canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def save_region_legend(output_path, padding=8):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    swatch = 18
    row_height = 26
    width = 260
    height = padding + len(REGION_ORDER) * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, region_name in enumerate(REGION_ORDER):
        y = padding + idx * row_height
        color = tuple((REGION_COLORS[idx] * 255).astype(np.uint8).tolist())
        draw.rectangle((padding, y, padding + swatch, y + swatch), fill=color)
        draw.text(
            (padding + swatch + 8, y + 2),
            f"{idx}: {region_name}",
            fill=(20, 20, 20),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main():
    args = parse_args()
    data_dir = find_split_data_dir(args.data_dir)
    mask_dir = find_mask_dir(args.mask_dir, args.split)
    rows = read_split_csv(data_dir, args.split)
    indices = select_indices(
        rows=rows,
        mask_dir=mask_dir,
        split=args.split,
        requested_indices=args.indices,
        start=args.start,
        count=args.count,
    )
    samples = [
        load_sample(rows, mask_dir, args.split, index, args.display_size)
        for index in indices
    ]

    output_dir = Path(args.output_dir)
    output_name = args.output_name or f"{args.split}_mediapipe_mask_preview.png"
    output_path = save_preview(
        samples=samples,
        output_path=output_dir / output_name,
        alpha=float(args.alpha),
        padding=int(args.padding),
        title_height=int(args.title_height),
    )
    legend_path = save_region_legend(output_dir / "mediapipe_region_color_legend.png")

    print(f"--> Data dir: {data_dir}")
    print(f"--> Mask dir: {mask_dir}")
    print(f"--> Visualized indices: {indices}")
    print(f"--> Saved preview: {output_path}")
    print(f"--> Saved legend: {legend_path}")


if __name__ == "__main__":
    main()
