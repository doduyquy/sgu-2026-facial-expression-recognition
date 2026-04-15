import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Export FER images after Sobel edge extraction.")
    parser.add_argument("--config", type=str, default="resnet", help="Config name in configs/*.yaml")
    parser.add_argument("--env", type=str, default="kaggle", choices=["local", "kaggle"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=-1, help="How many images to export. -1 means all.")
    parser.add_argument("--start_index", type=int, default=0, help="Start row index in split csv.")
    parser.add_argument("--output_dir", type=str, default="outputs/sobel_images")
    return parser.parse_args()


def sobel_edge(image_2d: np.ndarray) -> np.ndarray:
    # image_2d: uint8 grayscale (H, W)
    x = torch.from_numpy(image_2d).float().unsqueeze(0).unsqueeze(0) / 255.0

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).unsqueeze(1)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=torch.float32,
    ).unsqueeze(1)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
    edge = edge / edge.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)

    edge_np = edge.squeeze(0).squeeze(0).numpy()
    edge_np = np.clip(edge_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return edge_np


def make_side_by_side(original: np.ndarray, edge: np.ndarray) -> Image.Image:
    h, w = original.shape
    canvas = Image.new("L", (w * 2, h))
    canvas.paste(Image.fromarray(original), (0, 0))
    canvas.paste(Image.fromarray(edge), (w, 0))
    return canvas


def main():
    args = parse_args()
    cfg = load_config(model=args.config, env=args.env)

    data_path = cfg["kaggle"]["data_path"] if args.env == "kaggle" else cfg["local"]["data_path"]
    csv_path = Path(data_path) / f"{args.split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Split csv not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=[0, 1])
    total = len(df)

    start = max(args.start_index, 0)
    if start >= total:
        raise ValueError(f"start_index={start} is out of range. Total rows: {total}")

    if args.num_samples < 0:
        end = total
    else:
        end = min(start + args.num_samples, total)

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for idx in range(start, end):
        label, pixels = df.iloc[idx].values

        img_vec = np.fromstring(pixels, sep=" ", dtype=np.uint8)
        img_np = img_vec.reshape((48, 48))
        edge_np = sobel_edge(img_np)

        name = f"sample_{idx:05d}_label_{int(label)}"
        original_path = out_dir / f"{name}_orig.png"
        edge_path = out_dir / f"{name}_sobel.png"
        pair_path = out_dir / f"{name}_pair.png"

        Image.fromarray(img_np).save(original_path)
        Image.fromarray(edge_np).save(edge_path)
        make_side_by_side(img_np, edge_np).save(pair_path)

        metadata_rows.append([idx, int(label), str(original_path), str(edge_path), str(pair_path)])

    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "original_path", "sobel_path", "pair_path"])
        writer.writerows(metadata_rows)

    print(f"Exported {len(metadata_rows)} samples to: {out_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
