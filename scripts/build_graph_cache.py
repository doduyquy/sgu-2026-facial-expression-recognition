"""
scripts/build_graph_cache.py — Build graph cache từ CSV gốc FER-2013.

Usage:
    python scripts/build_graph_cache.py \
      --train_csv data/fer13-split/train.csv \
      --val_csv   data/fer13-split/val.csv \
      --test_csv  data/fer13-split/test.csv \
      --save_dir  outputs/graph_cache
"""
import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graph.graph_config import GraphConfig
from src.graph.image_to_graph import ImageGraphBuilder
from src.data.fer_split_dataset import FERSplitDataset


def build_and_save_split(csv_path: str, split_name: str, save_path: str, config: GraphConfig):
    dataset = FERSplitDataset(
        csv_path=csv_path,
        split_name=split_name,
        image_size=config.image_size,
    )

    print(f"\n=== Split: {split_name} ===")
    print(dataset.summary())

    builder = ImageGraphBuilder(config)
    graphs  = []

    for sample in tqdm(dataset, desc=f"Building {split_name} graphs"):
        g = builder.build_graph(
            image=sample["image"],
            label=sample["label"],
            image_id=sample["id"],
            split_name=sample["split"],
            usage=sample["usage"],
        )
        graphs.append(g)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(graphs, save_path)
    print(f"Saved {len(graphs)} graphs → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv",   type=str, required=True)
    parser.add_argument("--test_csv",  type=str, required=True)
    parser.add_argument("--save_dir",  type=str, required=True)
    # Graph config override (optional)
    parser.add_argument("--connectivity",  type=int,   default=8)
    parser.add_argument("--image_size",    type=int,   default=48)
    args = parser.parse_args()

    config = GraphConfig(
        image_size=args.image_size,
        connectivity=args.connectivity,
        normalize_pixels=True,
        node_features=["intensity", "x_norm", "y_norm"],
        edge_features=["dx", "dy", "dist", "delta_intensity", "intensity_similarity"],
        intensity_similarity_alpha=1.0,
        save_image_in_graph=False,
    )

    build_and_save_split(args.train_csv, "train", os.path.join(args.save_dir, "train_graphs.pt"), config)
    build_and_save_split(args.val_csv,   "val",   os.path.join(args.save_dir, "val_graphs.pt"),   config)
    build_and_save_split(args.test_csv,  "test",  os.path.join(args.save_dir, "test_graphs.pt"),  config)


if __name__ == "__main__":
    main()
