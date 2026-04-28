"""Debug one D4A full-graph batch, forward pass, and backward pass."""

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


def _stats(name: str, value: torch.Tensor) -> None:
    v = value.detach().float()
    print(
        f"{name:<28}: min={v.min().item():.6f} max={v.max().item():.6f} "
        f"mean={v.mean().item():.6f} std={v.std(unbiased=False).item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_graph_adaptive_motif_slot_gnn_d4a")
    parser.add_argument("--env", default="kaggle", choices=["local", "kaggle"])
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(_config_name(args.config), args.env)
    config["dataloader_mode"] = "full_graph"
    config.setdefault("data", {})["mode"] = "full_graph"
    config["data"]["batch_size"] = int(args.batch_size)
    config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    config["data"]["num_workers"] = int(args.num_workers)
    config["num_workers"] = int(args.num_workers)
    config.setdefault("logging", {})["use_wandb"] = False
    if args.graph_repo_path is not None:
        config["graph_repo_path"] = args.graph_repo_path

    set_seed(config.get("seed", {}).get("random_seed", 42))
    device = _resolve_device()
    graph_repo_path = config.get("graph_repo_path", "artifacts/graph_repo")
    train_loader, _, _, input_dim = build_dataloader(config, graph_repo_path=graph_repo_path)
    batch = next(iter(train_loader))

    print("=" * 80)
    print("D4A Full Graph Debug Batch")
    print("=" * 80)
    for key in ["node_features", "x", "edge_index", "edge_attr", "node_mask", "y"]:
        value = batch.get(key)
        if torch.is_tensor(value):
            print(f"{key:<28}: {tuple(value.shape)} {value.dtype}")
    _stats("node_features", batch["node_features"])
    _stats("edge_attr", batch["edge_attr"])

    model = get_model(config["model"]["name"], config=config, input_dim=input_dim).to(device)
    criterion = build_loss(config).to(device)
    batch = _move_batch_to_device(batch, device)
    out = model(batch)
    logits = out["logits"] if isinstance(out, dict) else out
    y = batch.get("y", batch.get("label")).long()
    loss = criterion(logits, y)
    if isinstance(out, dict) and out.get("aux_loss") is not None:
        loss = loss + out["aux_loss"]
    loss.backward()

    print("-" * 80)
    print(f"logits                    : {tuple(logits.shape)}")
    if isinstance(out, dict):
        print(f"slot_assignments          : {tuple(out['slot_assignments'].shape)}")
        print(f"slot_embeddings           : {tuple(out['slot_embeddings'].shape)}")
        print(f"slot_gates                : {tuple(out['slot_gates'].shape)}")
        print(f"null_mass                 : {float(out['null_mass'].detach().cpu()):.6f}")
        print(f"active_slot_count_soft    : {float(out['active_slot_count_soft'].detach().cpu()):.6f}")
        print(f"assignment_entropy        : {float(out['assignment_entropy'].detach().cpu()):.6f}")
    print(f"loss                      : {float(loss.detach().cpu()):.6f}")

    B = int(args.batch_size)
    num_slots = int(config["model"].get("num_slots", 32))
    hidden_dim = int(config["model"].get("hidden_dim", 128))
    num_classes = int(config["model"].get("num_classes", 7))
    assert tuple(logits.shape) == (B, num_classes)
    assert tuple(out["slot_assignments"].shape) == (B, 2304, num_slots + 1)
    assert tuple(out["slot_embeddings"].shape) == (B, num_slots, hidden_dim)
    assert tuple(out["slot_gates"].shape) == (B, num_slots)
    assert torch.isfinite(logits).all()
    print("OK: D4A full graph forward/backward shapes are valid")


if __name__ == "__main__":
    main()
