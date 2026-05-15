"""
Precompute 6 FER region masks from a pretrained face-parsing model.

This script is designed for Kaggle:
1. Export FER2013 CSV rows to temporary PNG images.
2. Optionally run the `face_parsing` CLI to produce parsing label maps.
3. Convert 19-class parsing maps into 6 region masks:
   forehead, left_eye, right_eye, nose, mouth, chin.
4. Save masks as .npy files plus manifest/summary artifacts.

If no face-parsing checkpoint/package is available, use --geometry-only to
produce a weak no-checkpoint bootstrap mask set. That fallback is not U-Net,
but it lets the region-mask dataloader/model be smoke-tested.
"""

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DEFAULT_CKPT_URL = (
    "https://huggingface.co/bes-dev/face_parsing/resolve/main/79999_iter.pth"
)
SPLITS = ("train", "val", "test")
REGION_NAMES = ("forehead", "left_eye", "right_eye", "nose", "mouth", "chin")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute FER2013 face-parsing region masks."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/unet_region_masks")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-url", type=str, default=DEFAULT_CKPT_URL)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-regions", type=int, default=6)
    parser.add_argument("--face-parsing-cmd", type=str, default="face_parsing")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--preview-count", type=int, default=24)
    return parser.parse_args()


def resolve_data_path(path):
    required = {"train.csv", "val.csv", "test.csv"}
    root = Path(path)
    if root.is_dir() and required.issubset({p.name for p in root.iterdir()}):
        return root

    for current in root.rglob("*"):
        if current.is_dir() and required.issubset({p.name for p in current.iterdir()}):
            print(f"--> Data path not exact; using discovered split folder: {current}")
            return current

    raise FileNotFoundError(f"Could not find train.csv/val.csv/test.csv under {path}")


def download_checkpoint(url, output_dir):
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "79999_iter.pth"
    if ckpt_path.exists():
        return ckpt_path

    print(f"--> Download checkpoint: {url}")
    urllib.request.urlretrieve(url, ckpt_path)
    return ckpt_path


def pixels_to_image(pixels, image_size):
    vec = np.fromstring(pixels, sep=" ", dtype=np.uint8)
    image = Image.fromarray(vec.reshape(48, 48), mode="L")
    return image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)


def export_split_images(data_path, split, image_dir, image_size):
    df = pd.read_csv(data_path / f"{split}.csv", usecols=[0, 1])
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_idx, row in df.iterrows():
        image = pixels_to_image(row.iloc[1], image_size=image_size)
        image_path = image_dir / f"{row_idx:06d}.png"
        image.save(image_path)
        rows.append(
            {
                "split": split,
                "row_index": int(row_idx),
                "label": int(row.iloc[0]),
                "image_path": str(image_path),
            }
        )
    return pd.DataFrame(rows)


def run_face_parsing(face_parsing_cmd, checkpoint, image_dir, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        face_parsing_cmd,
        "--ckpt",
        str(checkpoint),
        "--img_path",
        str(image_dir),
        "--res_path",
        str(result_dir),
    ]
    print("--> Run:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_parsing_map(result_dir, row_index):
    stem = f"{row_index:06d}"
    candidates = [
        result_dir / f"parsing_{stem}.png",
        result_dir / f"{stem}.png",
        result_dir / "parsing" / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(result_dir.rglob(f"*{stem}*.png"))
    for match in matches:
        if "parsing" in match.name.lower() or match.parent.name.lower() == "parsing":
            return match
    return None


def ellipse_mask(height, width, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    mask = np.exp(-0.5 * dist).astype(np.float32)
    mask[mask < 0.05] = 0.0
    if mask.max() > 0:
        mask = mask / mask.max()
    return mask


def geometry_templates(height, width):
    return np.stack(
        [
            ellipse_mask(height, width, 0.50, 0.25, 0.32, 0.16),
            ellipse_mask(height, width, 0.35, 0.40, 0.16, 0.09),
            ellipse_mask(height, width, 0.65, 0.40, 0.16, 0.09),
            ellipse_mask(height, width, 0.50, 0.53, 0.15, 0.16),
            ellipse_mask(height, width, 0.50, 0.70, 0.24, 0.10),
            ellipse_mask(height, width, 0.50, 0.83, 0.28, 0.12),
        ],
        axis=0,
    ).astype(np.float32)


def load_parsing_label_map(path, image_size):
    parsing = Image.open(path)
    if parsing.mode != "L":
        parsing = parsing.convert("L")
    if parsing.size != (image_size, image_size):
        parsing = parsing.resize((image_size, image_size), Image.NEAREST)
    arr = np.array(parsing, dtype=np.uint8)
    if arr.max() > 18:
        raise ValueError(
            f"Parsing map {path} does not look like label ids 0..18; max={arr.max()}"
        )
    return arr


def parsing_to_region_masks(parsing, image_size):
    height, width = parsing.shape
    templates = geometry_templates(height, width)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    skin = parsing == 1
    left_eye = np.isin(parsing, [2, 4])
    right_eye = np.isin(parsing, [3, 5])
    nose = parsing == 10
    mouth = np.isin(parsing, [11, 12, 13])
    forehead = skin & (yy < 0.45)
    chin = skin & (yy > 0.62)

    hard_masks = [forehead, left_eye, right_eye, nose, mouth, chin]
    masks = []
    fallback_regions = []
    for idx, hard_mask in enumerate(hard_masks):
        mask = hard_mask.astype(np.float32)
        if mask.sum() < 8:
            mask = templates[idx]
            fallback_regions.append(REGION_NAMES[idx])
        masks.append(mask)

    masks = np.stack(masks, axis=0).astype(np.float32)
    masks = soften_masks(masks, image_size=image_size)
    return masks, fallback_regions


def soften_masks(masks, image_size):
    soft = []
    for mask in masks:
        img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.max() > 0:
            arr = arr / arr.max()
        soft.append(arr)
    return np.stack(soft, axis=0).astype(np.float32)


def save_preview(manifest_df, output_path, max_items=24):
    if max_items <= 0 or manifest_df.empty:
        return

    samples = manifest_df.head(max_items)
    tile_w, tile_h = 224, 224
    cols = 4
    rows = int(np.ceil(len(samples) / cols))
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")

    colors = [
        (255, 80, 80),
        (80, 160, 255),
        (80, 220, 160),
        (255, 200, 70),
        (220, 100, 255),
        (100, 100, 100),
    ]

    for i, row in enumerate(samples.itertuples(index=False)):
        image = Image.open(row.image_path).convert("RGB").resize((tile_w, tile_h))
        masks = np.load(row.mask_path)
        overlay = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        for region_idx, color in enumerate(colors):
            mask = Image.fromarray((masks[region_idx] * 180).astype(np.uint8), mode="L")
            mask = mask.resize((tile_w, tile_h), Image.BILINEAR)
            color_layer = Image.new("RGBA", (tile_w, tile_h), (*color, 0))
            color_layer.putalpha(mask)
            overlay = Image.alpha_composite(overlay, color_layer)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h
        canvas.paste(image, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_masks_for_split(
    split,
    rows_df,
    output_dir,
    result_dir,
    image_size,
    geometry_only=False,
):
    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    fallback_count = 0
    missing_count = 0
    for row in rows_df.itertuples(index=False):
        mask_path = split_out / f"{int(row.row_index):06d}.npy"
        quality = "ok"
        fallback_regions = []

        if geometry_only:
            masks = geometry_templates(image_size, image_size)
            quality = "geometry_only"
            fallback_regions = list(REGION_NAMES)
        else:
            parsing_path = find_parsing_map(result_dir, int(row.row_index))
            if parsing_path is None:
                masks = geometry_templates(image_size, image_size)
                quality = "missing_parsing_fallback"
                fallback_regions = list(REGION_NAMES)
                missing_count += 1
            else:
                try:
                    parsing = load_parsing_label_map(parsing_path, image_size=image_size)
                    masks, fallback_regions = parsing_to_region_masks(parsing, image_size=image_size)
                    if fallback_regions:
                        quality = "partial_fallback"
                except Exception as exc:
                    print(f"[WARN] {split} row={row.row_index}: {exc}; using geometry fallback.")
                    masks = geometry_templates(image_size, image_size)
                    quality = "parse_error_fallback"
                    fallback_regions = list(REGION_NAMES)
                    missing_count += 1

        if fallback_regions:
            fallback_count += 1

        np.save(mask_path, masks.astype(np.float32))
        manifest_rows.append(
            {
                "split": split,
                "row_index": int(row.row_index),
                "label": int(row.label),
                "image_path": row.image_path,
                "mask_path": str(mask_path),
                "mask_quality_flag": quality,
                "fallback_regions": "|".join(fallback_regions),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / f"manifest_{split}.csv", index=False)
    return manifest, {"fallback_samples": fallback_count, "missing_or_error": missing_count}


def main():
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_regions != 6:
        raise ValueError("This script currently supports exactly 6 FER region masks.")

    checkpoint = args.checkpoint
    if args.download_checkpoint and checkpoint is None and not args.geometry_only:
        checkpoint = str(download_checkpoint(args.checkpoint_url, output_dir))
    if not args.geometry_only and checkpoint is None:
        raise ValueError(
            "Provide --checkpoint, use --download-checkpoint, or pass --geometry-only."
        )

    temp_root = output_dir / "_tmp_face_parsing"
    image_root = temp_root / "images"
    result_root = temp_root / "raw_results"
    preview_root = output_dir / "preview"

    summary = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "geometry_only": bool(args.geometry_only),
        "image_size": int(args.image_size),
        "regions": list(REGION_NAMES),
        "splits": {},
    }

    all_manifests = []
    for split in SPLITS:
        print(f"\n=== {split} ===")
        split_image_dir = image_root / split
        rows_df = export_split_images(data_path, split, split_image_dir, args.image_size)

        split_result_dir = result_root / split
        if not args.geometry_only:
            run_face_parsing(
                args.face_parsing_cmd,
                checkpoint=checkpoint,
                image_dir=split_image_dir,
                result_dir=split_result_dir,
            )

        manifest, split_summary = build_masks_for_split(
            split=split,
            rows_df=rows_df,
            output_dir=output_dir,
            result_dir=split_result_dir,
            image_size=args.image_size,
            geometry_only=args.geometry_only,
        )
        save_preview(
            manifest,
            preview_root / f"{split}_mask_preview.png",
            max_items=args.preview_count,
        )
        all_manifests.append(manifest)
        summary["splits"][split] = {
            "samples": int(len(manifest)),
            **split_summary,
        }
        print(f"--> Saved {len(manifest)} masks to {output_dir / split}")

    pd.concat(all_manifests, ignore_index=True).to_csv(
        output_dir / "manifest_all.csv",
        index=False,
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)

    print("\nDone.")
    print(f"--> Mask directory: {output_dir}")
    print(f"--> Summary: {output_dir / 'summary.json'}")
    print(f"--> Preview: {preview_root}")


if __name__ == "__main__":
    main()

