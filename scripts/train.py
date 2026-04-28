"""
scripts/train.py — Entry point huấn luyện GNN FER-2013.

Đọc từ canonical graph repository (chunks), không dùng *_graphs.pt kiểu cũ.

Kaggle workflow:
    1. Upload artifacts/graph_repo/ lên Kaggle (dataset: fer-graph-repo)
    2. Set graph_repo_path trong env.yaml → /kaggle/input/fer-graph-repo/graph_repo
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
from src.training.losses import compute_class_weights
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
                             "graph_vector | subgraph_descriptor | resolved | "
                             "precomputed_subgraph_graph | motif_filtered | pixel_motif | "
                             "candidate_attention | full_graph")
    parser.add_argument("--graph_repo_path", type=str, default=None,
                        help="Override graph_repo_path tu env.yaml. "
                             "Dung khi path tren Kaggle khac voi gia tri mac dinh trong config. "
                             "Vi du: /kaggle/input/fer-graph-repo/graph_repo")
    parser.add_argument("--subgraph_dataset_path", type=str, default=None,
                        help="Override subgraph_dataset_path tu env.yaml. "
                             "Dung cho mode precomputed_subgraph_graph khi dataset tren Kaggle "
                             "nam o path khac gia tri mac dinh trong config.")
    parser.add_argument("--motif_filtered_dataset_path", type=str, default=None,
                        help="Override motif_filtered_dataset_path tu env.yaml. "
                             "Dung cho mode motif_filtered tren local/Kaggle.")
    parser.add_argument("--pixel_motif_dataset_path", type=str, default=None,
                        help="Override pixel_motif_dataset_path tu env.yaml. "
                             "Dung cho mode pixel_motif tren local/Kaggle.")
    parser.add_argument("--candidate_attention_dataset_path", type=str, default=None,
                        help="Override candidate_attention_dataset_path tu env/config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs de sanity-test nhanh.")
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Smoke/debug: limit train batches per epoch. Omit for full train.")
    parser.add_argument("--max_val_batches", type=int, default=None,
                        help="Smoke/debug: limit validation batches per epoch. Omit for full validation.")
    parser.add_argument("--max_test_batches", type=int, default=None,
                        help="Smoke/debug: limit test batches during final evaluation. Omit for full test.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Tat WandB cho local smoke test.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Optional experiment name for run logging.")
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
    if args.motif_filtered_dataset_path is not None:
        config["motif_filtered_dataset_path"] = args.motif_filtered_dataset_path
    if args.pixel_motif_dataset_path is not None:
        config["pixel_motif_dataset_path"] = args.pixel_motif_dataset_path
    if args.candidate_attention_dataset_path is not None:
        config["candidate_attention_dataset_path"] = args.candidate_attention_dataset_path
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.max_train_batches is not None:
        config.setdefault("training", {})["max_train_batches"] = int(args.max_train_batches)
    if args.max_val_batches is not None:
        config.setdefault("training", {})["max_val_batches"] = int(args.max_val_batches)
    if args.max_test_batches is not None:
        config.setdefault("training", {})["max_test_batches"] = int(args.max_test_batches)
    elif (
        config.setdefault("training", {}).get("max_test_batches") is None
        and (
            config["training"].get("max_train_batches") is not None
            or config["training"].get("max_val_batches") is not None
        )
    ):
        fallback_test_batches = config["training"].get("max_val_batches", config["training"].get("max_train_batches"))
        config["training"]["max_test_batches"] = fallback_test_batches
        print(
            f"WARNING/SMOKE: max_test_batches not provided; using {fallback_test_batches} "
            "because train/val batch limits are active.",
            flush=True,
        )
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
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

    motif_filtered_dataset_path = config.get(
        "motif_filtered_dataset_path",
        config.get("data", {}).get("motif_filtered_dataset_path"),
    )
    if motif_filtered_dataset_path is not None:
        source = "CLI override" if args.motif_filtered_dataset_path is not None else "from config/env"
        print(f"--- motif_filtered_dataset_path : {motif_filtered_dataset_path}  [{source}]", flush=True)

    pixel_motif_dataset_path = config.get(
        "pixel_motif_dataset_path",
        config.get("data", {}).get("pixel_motif_dataset_path"),
    )
    if pixel_motif_dataset_path is not None:
        source = "CLI override" if args.pixel_motif_dataset_path is not None else "from config/env"
        print(f"--- pixel_motif_dataset_path : {pixel_motif_dataset_path}  [{source}]", flush=True)

    candidate_attention_dataset_path = config.get(
        "candidate_attention_dataset_path",
        config.get("data", {}).get("candidate_attention_dataset_path"),
    )
    if candidate_attention_dataset_path is not None:
        source = "CLI override" if args.candidate_attention_dataset_path is not None else "from config/env"
        print(f"--- candidate_attention_dataset_path : {candidate_attention_dataset_path}  [{source}]", flush=True)

    print(f"--- root_path       : {root_path}", flush=True)
    _log_run_config(config)
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

    eval_dir = os.path.join(root_path, "outputs", "figures", config["model"]["name"], run_name)
    os.makedirs(eval_dir, exist_ok=True)
    # log checkpoint lên wandb
    evaluate_and_show(model, test_loader, device, eval_dir, config=config)

    if config.get("logging", {}).get("use_wandb", False):
        try:
            from src.utils.logger_wandb import save_model_to_wandb
            save_model_to_wandb(ckpt_path, model_name=run_name)
        except Exception as exc:
            print(f"[WARN] Cannot upload checkpoint to WandB: {exc}")

    print("\n\t\tDONE!\n")


def _log_run_config(config: dict) -> None:
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    loss_cfg = config.get("loss", {}) or {}
    opt_cfg = config.get("optimizer", {}) or {}
    train_cfg = config.get("training", {}) or {}
    sched_cfg = config.get("scheduler", {}) or {}
    seed_cfg = config.get("seed", {}) or {}
    lr = opt_cfg.get("lr", train_cfg.get("lr", 0.001))
    weight_decay = opt_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0001))
    optimizer_name = opt_cfg.get("name", train_cfg.get("optimizer", "adam"))
    scheduler_name = sched_cfg.get("name", train_cfg.get("scheduler", "reduce_lr_on_plateau"))
    class_weight_power = loss_cfg.get("class_weight_power", train_cfg.get("class_weight_power", 1.0))
    class_counts = loss_cfg.get("class_counts")
    actual_weights = None
    if loss_cfg.get("use_class_weights", False) and class_counts is not None:
        actual_weights = compute_class_weights(class_counts, power=float(class_weight_power)).tolist()
    normalize_candidate_x = bool(data_cfg.get("normalize_candidate_x", data_cfg.get("normalize_x", False)))

    print("--- Run config", flush=True)
    print(f"--- experiment name          : {config.get('experiment_name', '<direct-train>')}", flush=True)
    print(f"--- data recipe              : {data_cfg.get('recipe', data_cfg.get('name'))}", flush=True)
    print(f"--- model name               : {model_cfg.get('name')}", flush=True)
    print(f"--- node_dim                 : {model_cfg.get('node_dim')}", flush=True)
    print(f"--- edge_dim                 : {model_cfg.get('edge_dim', data_cfg.get('edge_dim'))}", flush=True)
    print(f"--- hidden_dim               : {model_cfg.get('hidden_dim')}", flush=True)
    print(f"--- seed                     : {seed_cfg.get('random_seed', 42)}", flush=True)
    print(f"--- lr                       : {lr}", flush=True)
    print(f"--- optimizer                : {optimizer_name}", flush=True)
    print(f"--- scheduler                : {scheduler_name}", flush=True)
    print(f"--- weight_decay             : {weight_decay}", flush=True)
    print(f"--- class_weight_power       : {class_weight_power}", flush=True)
    print(f"--- actual class weights     : {actual_weights}", flush=True)
    print(f"--- normalize_candidate_x    : {normalize_candidate_x}", flush=True)
    print(f"--- global_candidate_pooling : {model_cfg.get('use_global_candidate_pooling')}", flush=True)
    print(f"--- global_pooling_type      : {model_cfg.get('global_pooling_type')}", flush=True)
    print(f"--- num_slots                : {model_cfg.get('num_slots')}", flush=True)
    print(f"--- slot_iterations          : {model_cfg.get('slot_iterations')}", flush=True)
    print(f"--- max_train_batches        : {train_cfg.get('max_train_batches')}", flush=True)
    print(f"--- max_val_batches          : {train_cfg.get('max_val_batches')}", flush=True)
    print(f"--- max_test_batches         : {train_cfg.get('max_test_batches')}", flush=True)


if __name__ == "__main__":
    main()
