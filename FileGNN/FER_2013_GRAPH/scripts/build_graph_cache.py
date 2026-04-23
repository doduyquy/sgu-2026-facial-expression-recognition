import os
import argparse
import torch
from tqdm import tqdm

from configs.graph_config import GraphConfig
from data.fer_split_dataset import FERSplitDataset
from graph.image_to_graph import ImageGraphBuilder


def build_and_save_split(csv_path: str, split_name: str, save_path: str, config: GraphConfig):
    dataset = FERSplitDataset(
        csv_path=csv_path,
        split_name=split_name,
        image_size=config.image_size,
    )

    print(f"\n=== Split: {split_name} ===")
    print(dataset.summary())

    builder = ImageGraphBuilder(config)
    graphs = []

    for sample in tqdm(dataset, desc=f"Building {split_name} graphs"):
        graph = builder.build_graph(
            image=sample["image"],
            label=sample["label"],
            image_id=sample["id"],
            split_name=sample["split"],
            usage=sample["usage"],
        )
        graphs.append(graph)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(graphs, save_path)
    print(f"Đã lưu {len(graphs)} graph samples vào: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    config = GraphConfig(
        image_size=48,
        connectivity=8,
        normalize_pixels=True,
        node_features=[
            "intensity",
            "x_norm",
            "y_norm",
        ],
        edge_features=[
            "dx",
            "dy",
            "dist",
            "delta_intensity",
            "intensity_similarity",
        ],
        intensity_similarity_alpha=1.0,
        save_image_in_graph=False,
    )

    build_and_save_split(
        csv_path=args.train_csv,
        split_name="train",
        save_path=os.path.join(args.save_dir, "train_graphs.pt"),
        config=config,
    )

    build_and_save_split(
        csv_path=args.val_csv,
        split_name="val",
        save_path=os.path.join(args.save_dir, "val_graphs.pt"),
        config=config,
    )

    build_and_save_split(
        csv_path=args.test_csv,
        split_name="test",
        save_path=os.path.join(args.save_dir, "test_graphs.pt"),
        config=config,
    )


if __name__ == "__main__":
    main()