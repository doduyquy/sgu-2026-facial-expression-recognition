"""Sanity-check one HierarchicalMotifGNN batch."""

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
from src.training.losses import build_loss
from src.utils.config import load_config
from src.utils.seed import set_seed


def _config_name(value: str) -> str:
    path = Path(value)
    return path.stem if path.suffix in {".yaml", ".yml"} else value


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hierarchical_motif_gnn")
    parser.add_argument("--env", type=str, default="kaggle", choices=["local", "kaggle"])
    parser.add_argument("--graph_repo_path", type=str, default=None)
    parser.add_argument("--pixel_motif_dataset_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(_config_name(args.config), args.env)
    config["dataloader_mode"] = config.get("data", {}).get("mode", "pixel_motif")
    config.setdefault("data", {})["return_subgraph_tensors"] = True
    config["data"]["num_workers"] = int(args.num_workers)
    config["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        config["data"]["batch_size"] = int(args.batch_size)
    if args.pixel_motif_dataset_path is not None:
        config["pixel_motif_dataset_path"] = args.pixel_motif_dataset_path
    if args.graph_repo_path is not None:
        config["graph_repo_path"] = args.graph_repo_path

    set_seed(config.get("seed", {}).get("random_seed", 42))
    device = _resolve_device()
    graph_repo_path = config.get("graph_repo_path", "artifacts/graph_repo")

    train_loader, _, _, input_dim = build_dataloader(config=config, graph_repo_path=graph_repo_path)
    batch = next(iter(train_loader))
    model = get_model(config["model"]["name"], config=config, input_dim=input_dim).to(device)
    criterion = build_loss(config).to(device)

    batch = _move_batch_to_device(batch, device)
    logits = model(batch)
    y = batch.get("y", batch.get("label")).long()
    loss = criterion(logits, y)
    loss.backward()

    print("Batch shape sanity:")
    for key in ["x", "sub_x", "sub_adj", "sub_node_mask", "mask", "match_scores", "matched_class"]:
        value = batch[key]
        print(f"  {key:<15}: {tuple(value.shape)}")
    print(f"  logits         : {tuple(logits.shape)}")
    print(f"  loss           : {float(loss.detach().cpu()):.6f}")

    valid_node_counts = batch["sub_node_mask"].sum(dim=-1)
    valid_subgraphs = batch["mask"].bool()
    checks = {
        "valid subgraphs have nodes": bool((valid_node_counts[valid_subgraphs] > 0).all().item()),
        "sub_adj finite": bool(torch.isfinite(batch["sub_adj"]).all().item()),
        "sub_x finite": bool(torch.isfinite(batch["sub_x"]).all().item()),
        "logits finite": bool(torch.isfinite(logits).all().item()),
    }
    print("Checks:")
    for name, ok in checks.items():
        print(f"  {name:<28}: {ok}")
    if not all(checks.values()):
        raise RuntimeError(f"Sanity check failed: {checks}")


if __name__ == "__main__":
    main()
