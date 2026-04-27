"""Debug a candidate-attention batch and model forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataloader import build_dataloader
from src.models import get_model
from src.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="learnable_slot_candidate_motif_gnn")
    p.add_argument("--candidate_attention_dir", default="artifacts/candidate_attention_dataset_v1")
    p.add_argument("--graph_repo_path", default="artifacts/graph_repo")
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    config = load_config(args.config, "kaggle")
    config["dataloader_mode"] = "candidate_attention"
    config["candidate_attention_dataset_path"] = args.candidate_attention_dir
    config.setdefault("data", {})["batch_size"] = int(args.batch_size)
    config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    config["data"]["num_workers"] = 0
    config["num_workers"] = 0
    config.setdefault("logging", {})["use_wandb"] = False

    train_loader, _, _, input_dim = build_dataloader(config, graph_repo_path=args.graph_repo_path)
    batch = next(iter(train_loader))
    print("=" * 80)
    print("Candidate Attention Debug Batch")
    print("=" * 80)
    for key in [
        "candidate_x",
        "candidate_mask",
        "candidate_centers",
        "candidate_bbox",
        "candidate_radius",
        "candidate_edge_index",
        "candidate_edge_attr",
        "candidate_edge_valid",
        "candidate_node_indices",
        "candidate_node_mask",
        "y",
    ]:
        value = batch.get(key)
        if torch.is_tensor(value):
            print(f"{key:<28}: {tuple(value.shape)} {value.dtype}")

    model = get_model(config["model"]["name"], config=config, input_dim=input_dim)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    print("-" * 80)
    print(f"logits                    : {tuple(out['logits'].shape)}")
    print(f"candidate_attention       : {tuple(out['candidate_attention'].shape)}")
    if out.get("class_slot_attention") is not None:
        print(f"class_slot_attention      : {tuple(out['class_slot_attention'].shape)}")
    if out.get("combined_candidate_attention") is not None:
        print(f"combined_candidate_attention: {tuple(out['combined_candidate_attention'].shape)}")
    print(f"slot_embeddings           : {tuple(out['slot_embeddings'].shape)}")
    print(f"aux_loss                  : {tuple(out['aux_loss'].shape)}")
    assert tuple(out["logits"].shape) == (int(args.batch_size), int(config["model"].get("num_classes", 7)))
    if out.get("combined_candidate_attention") is not None:
        assert tuple(out["combined_candidate_attention"].shape) == (
            int(args.batch_size),
            int(config["model"].get("num_classes", 7)),
            int(out["candidate_attention"].shape[-1]),
        )
    assert tuple(out["aux_loss"].shape) == ()
    print("OK")


if __name__ == "__main__":
    main()
