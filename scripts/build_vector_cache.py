"""
scripts/build_vector_cache.py — Pre-compute graph vectors từ CSV.

Thay vì lưu PixelGraph đầy đủ (~24GB RAM), script này tính thẳng
graph-level vectors (mean/std/max của node_features) và lưu dưới dạng
tensor nhỏ gọn.

Kết quả:
    train_vectors.pt  → {'x': Tensor[N, 9], 'y': Tensor[N]}  (~1 MB)
    val_vectors.pt    → ...
    test_vectors.pt   → ...

Usage:
    python scripts/build_vector_cache.py \\
        --train_csv data/fer13-split/train.csv \\
        --val_csv   data/fer13-split/val.csv \\
        --test_csv  data/fer13-split/test.csv \\
        --save_dir  outputs/vector_cache
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graph.graph_config import GraphConfig
from src.graph.image_to_graph import ImageGraphBuilder
from src.features.graph_vectorizer import GraphVectorizer
from src.data.fer_split_dataset import FERSplitDataset


def build_vector_split(
    csv_path: str,
    split_name: str,
    save_path: str,
    graph_cfg: GraphConfig,
    vectorizer: GraphVectorizer,
):
    dataset = FERSplitDataset(
        csv_path=csv_path,
        split_name=split_name,
        image_size=graph_cfg.image_size,
    )

    print(f"\n=== Split: {split_name}  ({len(dataset)} samples) ===")
    builder = ImageGraphBuilder(graph_cfg)

    all_x = []
    all_y = []

    for sample in tqdm(dataset, desc=f"Building {split_name} vectors"):
        g = builder.build_graph(
            image=sample["image"],
            label=sample["label"],
            image_id=sample["id"],
            split_name=sample["split"],
            usage=sample["usage"],
        )
        vec = vectorizer.transform(g)   # [D]
        all_x.append(vec)
        all_y.append(sample["label"])

    X = torch.tensor(np.stack(all_x), dtype=torch.float32)  # [N, D]
    Y = torch.tensor(all_y, dtype=torch.long)                # [N]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"x": X, "y": Y, "split": split_name}, save_path)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"Saved: {save_path}  |  X={tuple(X.shape)}  |  Y={tuple(Y.shape)}  |  {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv",   type=str, required=True)
    parser.add_argument("--test_csv",  type=str, required=True)
    parser.add_argument("--save_dir",  type=str, required=True)
    # Graph config
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--image_size",   type=int, default=48)
    args = parser.parse_args()

    graph_cfg = GraphConfig(
        image_size=args.image_size,
        connectivity=args.connectivity,
        normalize_pixels=True,
        node_features=["intensity", "x_norm", "y_norm"],
        edge_features=["dx", "dy", "dist", "delta_intensity", "intensity_similarity"],
        intensity_similarity_alpha=1.0,
        save_image_in_graph=False,
    )
    vectorizer = GraphVectorizer(use_mean=True, use_std=True, use_max=True)

    print(f"Graph config: connectivity={graph_cfg.connectivity}, "
          f"node_features={graph_cfg.node_features}")
    print(f"Vector dim  : {vectorizer.infer_output_dim(len(graph_cfg.node_features))}")

    build_vector_split(args.train_csv, "train",
                       os.path.join(args.save_dir, "train_vectors.pt"), graph_cfg, vectorizer)
    build_vector_split(args.val_csv,   "val",
                       os.path.join(args.save_dir, "val_vectors.pt"),   graph_cfg, vectorizer)
    build_vector_split(args.test_csv,  "test",
                       os.path.join(args.save_dir, "test_vectors.pt"),  graph_cfg, vectorizer)

    print("\n=== All splits done! ===")


if __name__ == "__main__":
    main()
