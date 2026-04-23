"""
scripts/train.py — Entry point huấn luyện GNN FER-2013.

Kaggle workflow:
    1. Add pre-built .pt graph cache dataset vào notebook
    2. Chạy: python -m scripts.train --config mlp_baseline --env kaggle

Local workflow:
    1. Build cache trước: python scripts/build_graph_cache.py ...
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
    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device}")

    # ── Config ──
    config = load_config(args.config, args.env)
    set_seed(config["seed"].get("random_seed", 42))

    # ── Paths từ flat-merged env config ──
    # env.yaml merge flat keys vào top-level config
    root_path        = config.get("root_path", ".")
    graph_cache_path = config.get("graph_cache_path", "outputs/graph_cache")

    print(f"--- root_path        : {root_path}")
    print(f"--- graph_cache_path : {graph_cache_path}")

    # ── Kiểm tra graph cache tồn tại ──
    for split in ["train", "val", "test"]:
        pt_file = os.path.join(graph_cache_path, f"{split}_graphs.pt")
        if not os.path.exists(pt_file):
            raise FileNotFoundError(
                f"\n[ERROR] Không tìm thấy graph cache: {pt_file}\n"
                f"Kaggle: Hãy add dataset .pt vào notebook và set đúng graph_cache_path trong base.yaml.\n"
                f"Local : Chạy scripts/build_graph_cache.py trước.\n"
            )

    # ── DataLoaders từ .pt cache ──
    train_loader, val_loader, test_loader, input_dim = build_dataloader(config, graph_cache_path)

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
    run_name = f"{config['model']['name']}_{datetime.now().strftime('%d%m%Y_%H%M')}"
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
