import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.emotions_dict import EMOTION_DICT
from src.data.landmarks import LandmarkExtractor
from src.utils.config import load_config


def build_parser():
    parser = argparse.ArgumentParser(description="Export FER samples with landmark visualization.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to fer13-split folder.")
    parser.add_argument("--config", type=str, default="resnet", help="Config name to load when data_path is omitted.")
    parser.add_argument("--env", type=str, default="kaggle", choices=["local", "kaggle"], help="Environment profile.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to export.")
    parser.add_argument("--output_dir", type=str, default="outputs/landmark_preview", help="Folder to save output images.")
    parser.add_argument("--landmark_indexes", type=int, nargs="*", default=None, help="Optional FaceMesh indexes.")
    parser.add_argument("--force_template_fallback", action="store_true", help="Use template landmarks when detector fails.")
    return parser


def resolve_csv_path(base_path, split):
    base = Path(base_path)
    candidates = [
        base / f"{split}.csv",
        base / "fer13-split" / f"{split}.csv",
        base / "fer13-split" / "fer13-split" / f"{split}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_pixels(pixel_str):
    image_vec = np.fromstring(pixel_str, sep=" ", dtype=np.uint8)
    return image_vec.reshape((48, 48))


def template_landmarks(num_points):
    base = np.array(
        [
            [0.32, 0.36],
            [0.68, 0.36],
            [0.50, 0.45],
            [0.36, 0.62],
            [0.64, 0.62],
            [0.50, 0.78],
            [0.50, 0.53],
            [0.28, 0.56],
            [0.72, 0.56],
            [0.32, 0.70],
            [0.68, 0.70],
            [0.50, 0.88],
        ],
        dtype=np.float32,
    )
    if num_points <= base.shape[0]:
        points = base[:num_points].copy()
    else:
        extra = np.tile(base[-1:], (num_points - base.shape[0], 1))
        points = np.concatenate([base, extra], axis=0)
    mask = np.ones((num_points,), dtype=np.float32)
    return points, mask


def draw_landmarks(gray_image, points, mask, true_label, save_path, is_fallback=False):
    h, w = gray_image.shape
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(gray_image, cmap="gray")

    valid = mask > 0.5
    if valid.any():
        xs = points[valid, 0] * (w - 1)
        ys = points[valid, 1] * (h - 1)
        color = "gold" if is_fallback else "lime"
        ax.scatter(xs, ys, c=color, s=18)
        src = "fallback" if is_fallback else "detected"
        ax.set_title(f"{EMOTION_DICT[int(true_label)]} | lm={int(valid.sum())} | {src}")
    else:
        ax.set_title(f"{EMOTION_DICT[int(true_label)]} | lm=0 (not detected)")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    if data_path is None:
        cfg = load_config(model=args.config, env=args.env)
        data_path = cfg.get("data_path", None)
        if output_dir == "outputs/landmark_preview":
            base_out = cfg.get("output_dir", "outputs")
            output_dir = str(Path(base_out) / "landmark_preview")

    if data_path is None:
        raise ValueError("Cannot resolve data_path. Pass --data_path or ensure config has data_path.")

    csv_path = resolve_csv_path(data_path, args.split)
    if csv_path is None:
        raise FileNotFoundError(f"Cannot find {args.split}.csv under: {data_path}")

    out_dir = Path(output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, usecols=[0, 1])
    num_samples = min(args.num_samples, len(df))

    extractor = LandmarkExtractor(
        enabled=True,
        backend="mediapipe",
        landmark_indexes=args.landmark_indexes,
    )

    detected_count = 0
    fallback_count = 0

    for i in range(num_samples):
        label, pixels = df.iloc[i].values
        img = parse_pixels(pixels)
        points, mask = extractor.extract_points_with_mask(img)
        used_fallback = False

        if (mask > 0.5).any():
            detected_count += 1
        elif args.force_template_fallback:
            n_points = len(args.landmark_indexes) if args.landmark_indexes is not None else len(extractor.landmark_indexes)
            points, mask = template_landmarks(n_points)
            fallback_count += 1
            used_fallback = True

        save_path = out_dir / f"sample_{i:03d}_label_{int(label)}.png"
        draw_landmarks(img, points, mask, label, str(save_path), is_fallback=used_fallback)

    print(f"Saved {num_samples} images to: {out_dir}")
    print(f"Landmark detected on {detected_count}/{num_samples} samples")
    if args.force_template_fallback:
        print(f"Template fallback used on {fallback_count}/{num_samples} samples")


if __name__ == "__main__":
    main()
