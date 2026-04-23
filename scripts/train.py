"""
scripts/train.py — Entry point huấn luyện GNN FER-2013.

Đọc từ canonical graph repository (chunks), không dùng *_graphs.pt kiểu cũ.

Kaggle workflow:
    1. Upload artifacts/graph_repo/ lên Kaggle (dataset: fer-graph-repo)
    2. Set graph_repo_path trong base.yaml → /kaggle/input/fer-graph-repo/graph_repo
    3. Chạy: python -m scripts.train --config mlp_baseline --env kaggle

Local workflow:
    1. Build repo trước: python scripts/build_graph_repository.py ...
    2. Chạy: python -m scripts.train --config mlp_baseline --env local
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import load_checkpoints
from src.data.dataloader import build_dataloader
from src.models import get_model
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.evaluation.evaluator import evaluate_and_show


# -----------------------------------------------------------------------

def main():
    print("\n\t\t--> GNN FER-2013 Training <--\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mlp_baseline",
                        help="Tên file config (không có .yaml), vd: mlp_baseline")
    parser.add_argument("--env", type=str, default="kaggle",
                        choices=["local", "kaggle"])
    parser.add_argument("--dataloader_mode", type=str, default="graph_vector",
                        choices=["graph_vector", "resolved"],
                        help="graph_vector: MLP baseline | resolved: GNN (future)")
    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device}")

    # ── Config ──
    config = load_config(args.config, args.env)
    config["dataloader_mode"] = args.dataloader_mode
    set_seed(config["seed"].get("random_seed", 42))

    # ── Paths từ flat-merged env config ──
    root_path       = config.get("root_path", ".")
    graph_repo_path = config.get("graph_repo_path", "artifacts/graph_repo")

    print(f"--- root_path       : {root_path}")
    print(f"--- graph_repo_path : {graph_repo_path}")

    # ── DataLoaders từ graph repository ──
    train_loader, val_loader, test_loader, input_dim = build_dataloader(
        config=config,
        graph_repo_path=graph_repo_path,
    )

    # ── Model ──
    model = get_model(
        name=config["model"]["name"],
        config=config,
        input_dim=input_dim,
    )
    print(f"--- Model: {config['model']['name']} | input_dim={input_dim}")
    print(f"--- Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss / Optimizer / Scheduler ──
    criterion = build_loss(config=config)
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    # ── Checkpoint path ──
    run_name  = f"{config['model']['name']}_{datetime.now().strftime('%d%m%Y_%H%M')}"
    ckpt_dir  = os.path.join(root_path, "outputs", "checkpoints", config['model']['name'])
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Train ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        run_name=run_name,
        save_dir=ckpt_path,
    )
    trainer.fit()

    # ── Evaluate trên test set với best checkpoint ──
    print("\n" + "=" * 55)
    print("Evaluate on TEST SET with best checkpoint")
    print("=" * 55)
    load_checkpoints(model, optimizer, ckpt_path, device)

    eval_dir = os.path.join(root_path, "outputs", "figures")
    os.makedirs(eval_dir, exist_ok=True)
    evaluate_and_show(model, test_loader, device, eval_dir)

    print("\n\t\tDONE!\n")


if __name__ == "__main__":
    main()
