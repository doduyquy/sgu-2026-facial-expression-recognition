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

def flush_stdio() -> None:
    """
    Flush stdout/stderr trước khi DataLoader workers được tạo.

    Trên Linux/Kaggle, khi dùng multiprocessing kiểu fork, worker có thể kế thừa
    stdout buffer chưa flush của process cha và làm một số log đầu chương trình
    bị in lặp lại.
    """
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass

def resolve_device() -> torch.device:
    """
    Chọn device an toàn cho môi trường Kaggle/local.

    Với một số runtime Kaggle, `torch.cuda.is_available()` vẫn trả về True
    dù GPU quá cũ so với binary PyTorch hiện tại (ví dụ Tesla P100 = sm_60,
    nhưng wheel chỉ hỗ trợ sm_70+). Khi đó cần fallback CPU sớm để tránh
    crash ở bước forward đầu tiên.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())

        if supported_arches and current_arch not in supported_arches:
            device_name = torch.cuda.get_device_name(0)
            print(
                "[WARN] CUDA runtime detected but GPU is incompatible with the current "
                "PyTorch build.\n"
                f"       GPU             : {device_name} ({current_arch})\n"
                f"       Supported archs : {', '.join(sorted(supported_arches))}\n"
                "       Fallback device : cpu"
            )
            return torch.device("cpu")
    except Exception as exc:
        print(f"[WARN] Cannot validate CUDA compatibility ({exc}). Fallback to CPU.")
        return torch.device("cpu")

    return torch.device("cuda")

def main():
    print("\n\t\t--> GNN FER-2013 Training <--\n", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mlp_baseline",
                        help="Ten file config (khong co .yaml), vd: mlp_baseline")
    parser.add_argument("--env", type=str, default="kaggle",
                        choices=["local", "kaggle"])
    parser.add_argument("--dataloader_mode", type=str, default=None,
                        help="Override dataloader mode tu config: "
                             "graph_vector | subgraph_descriptor | resolved")
    parser.add_argument("--graph_repo_path", type=str, default=None,
                        help="Override graph_repo_path tu env.yaml. "
                             "Dung khi path tren Kaggle khac voi gia tri mac dinh trong config. "
                             "Vi du: /kaggle/input/datasets/username/fer-graph-repo/artifacts/graph_repo")
    parser.add_argument("--subgraph_dataset_path", type=str, default=None,
                        help="Override subgraph_dataset_path tu env.yaml. "
                             "Dung cho mode precomputed_subgraph_graph khi dataset tren Kaggle "
                             "nam o path khac gia tri mac dinh trong config.")
    args = parser.parse_args()

    # ── Device ──
    device = resolve_device()
    print(f"--- Device: {device}", flush=True)

    # ── Config ──
    config = load_config(args.config, args.env)
    config["dataloader_mode"] = (
        args.dataloader_mode
        if args.dataloader_mode is not None
        else config.get("data", {}).get("mode", "graph_vector")
    )
    if args.subgraph_dataset_path is not None:
        config["subgraph_dataset_path"] = args.subgraph_dataset_path
    set_seed(config["seed"].get("random_seed", 42))

    # ── Paths ──
    root_path       = config.get("root_path", ".")
    graph_repo_path = config.get("graph_repo_path", "artifacts/graph_repo")

    # CLI override co uu tien cao hon env.yaml
    if args.graph_repo_path is not None:
        graph_repo_path = args.graph_repo_path
        print(f"--- graph_repo_path : {graph_repo_path}  [CLI override]", flush=True)
    else:
        print(f"--- graph_repo_path : {graph_repo_path}  [from env.yaml]", flush=True)

    subgraph_dataset_path = config.get("subgraph_dataset_path")
    if subgraph_dataset_path is not None:
        source = "CLI override" if args.subgraph_dataset_path is not None else "from env.yaml"
        print(f"--- subgraph_dataset_path : {subgraph_dataset_path}  [{source}]", flush=True)

    print(f"--- root_path       : {root_path}", flush=True)
    flush_stdio()

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
    print(f"--- Model: {config['model']['name']} | input_dim={input_dim}", flush=True)
    print(f"--- Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

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
    flush_stdio()
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
