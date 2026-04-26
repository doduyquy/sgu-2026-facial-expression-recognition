"""Visualize selected pixel motif centers/bboxes for one sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="artifacts/pixel_motif_dataset_v2")
    p.add_argument("--split", default="test")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out_path", default=None)
    p.add_argument("--image_size", type=int, default=48)
    args = p.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    data_dir = Path(args.data_dir)
    samples = _torch_load(data_dir / f"{args.split}_pixel_motif.pt")
    sample = samples[int(args.index)]
    image_size = int(args.image_size)
    canvas = np.zeros((image_size, image_size), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{args.split}[{args.index}] graph_id={sample['graph_id']} label={sample['label']}")

    mask = torch.as_tensor(sample["mask"]).bool()
    bboxes = torch.as_tensor(sample["bbox"]).float()
    centers = torch.as_tensor(sample["centers"]).float()
    matched = torch.as_tensor(sample["matched_class"]).long()
    scores = torch.as_tensor(sample["match_scores"]).float()
    colors = ["tab:red", "tab:purple", "tab:brown", "tab:green", "tab:blue", "tab:orange", "tab:gray"]
    for i in torch.where(mask)[0].tolist():
        x0, y0, x1, y1 = bboxes[i].tolist()
        x0 *= image_size - 1
        x1 *= image_size - 1
        y0 *= image_size - 1
        y1 *= image_size - 1
        cls = int(matched[i].item())
        color = colors[cls % len(colors)]
        rect = plt.Rectangle((x0, y0), max(1, x1 - x0 + 1), max(1, y1 - y0 + 1), fill=False, color=color, linewidth=1.2)
        ax.add_patch(rect)
        cx = float(centers[i, 0] * (image_size - 1))
        cy = float(centers[i, 1] * (image_size - 1))
        ax.scatter([cx], [cy], s=12, color=color)
        ax.text(cx, cy, f"{cls}:{scores[i]:.2f}", color=color, fontsize=6)
    ax.set_xlim(-1, image_size)
    ax.set_ylim(image_size, -1)
    ax.axis("off")
    fig.tight_layout()

    out_path = Path(args.out_path) if args.out_path else data_dir / f"{args.split}_{args.index}_pixel_motif_evidence.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
