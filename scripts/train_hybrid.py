"""
train_hybrid.py — Two-phase training for Hybrid CNN + GNN + Motif FER model
============================================================================
Phase 1  (phase1_epochs):
  • Freeze GNN, MotifLayer, NodeSelector
  • Train CNN encoder + Classifier only  (LR = config lr)
  • Objective: pre-warm CNN to ~65% accuracy before GNN gets gradients

Phase 2  (remaining epochs):
  • Unfreeze all parameters
  • LR drops to lr_phase2 (default 1e-4)
  • Full end-to-end: CE + 0.2 * motif_diversity_loss

Usage (Kaggle / local):
    python scripts/train_hybrid.py --config configs/motif_config.yaml --env kaggle
    python scripts/train_hybrid.py --config configs/motif_config.yaml --env local
"""

import os
import sys
import argparse
import torch
import wandb
from pathlib import Path
from datetime import datetime

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config     import load_config
from src.utils.seed       import set_seed
from src.utils.logger_wandb import init_wandb, save_model_to_wandb
from src.utils.checkpoint  import load_checkpoints
from src.utils.data_stats  import get_class_distribution

from src.data.dataloader   import build_dataloader
from src.models            import get_model
from src.training.trainer  import Trainer
from src.training.losses   import build_loss
from src.training.optimizer import build_optimizer, build_scheduler
from src.evaluation.evaluator import evaluate_and_show

# ── Check PyG ─────────────────────────────────────────────────────────────────
try:
    import torch_geometric  # noqa: F401
    print("[✓] torch_geometric available")
except ImportError:
    print("[!] torch_geometric not found — falling back to dense kNN.\n"
          "    Install: pip install torch_geometric")


# ══════════════════════════════════════════════════════════════════════════════
def build_phase_trainer(model, train_loader, val_loader, config,
                        device, run_name, save_dir, lr_override=None, start_epoch=0):
    """
    Build a Trainer with optionally overridden LR.
    Rebuilds optimizer so Phase-2 gets a fresh Adam at lr_phase2.
    """
    # Temporarily override lr in config dict for build_optimizer
    if lr_override is not None:
        config = {**config}   # shallow copy
        config['training'] = {**config['training'], 'lr': lr_override}

    criterion = build_loss(config)
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

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
        save_dir=save_dir,
        start_epoch=start_epoch,
    )
    return trainer


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN+GNN+Motif FER Trainer")
    parser.add_argument("--config", type=str, default="configs/motif_config.yaml")
    parser.add_argument("--env",    type=str, default="local",
                        choices=["local", "kaggle"])
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    config = load_config(args.config, args.env)
    set_seed(config['seed'].get('random_seed', 42))

    platform = config['env']['platform']
    if platform == 'kaggle':
        data_path = config['kaggle']['data_path']
        root_path = config['kaggle']['root_path']
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        data_path = config['local']['data_path']
        root_path = config['local']['root_path']

    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    model_name = config['model'].get('name', 'motif_graph_fer')
    run_name   = f"{model_name}_{timestamp}"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloader(config, data_path)

    # ── Class weights ─────────────────────────────────────────────────────────
    trainset_path = os.path.join(data_path, "train.csv")
    dist = get_class_distribution(trainset_path)
    class_counts  = torch.tensor(dist.values, dtype=torch.float)

    use_cw = config['training'].get('use_class_weights', True)
    cw_mode = config['training'].get('class_weight_mode', 'sqrt_inverse')
    if use_cw:
        if cw_mode == 'sqrt_inverse':
            class_weights = 1.0 / torch.sqrt(class_counts)
        else:
            class_weights = 1.0 / class_counts
        class_weights = (class_weights / class_weights.sum()).to(device)
        print(f"[Class weights ({cw_mode})] {class_weights.cpu().numpy().round(4)}")
    else:
        class_weights = None

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(name=model_name, config=config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model_name} — {n_params:,} parameters")

    # ── Checkpoint path ───────────────────────────────────────────────────────
    ckpt_dir  = os.path.join(root_path, f"outputs/checkpoints/{model_name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{run_name}_best.pth")

    # ── WandB init ────────────────────────────────────────────────────────────
    if config['logging'].get('use_wandb', True):
        init_wandb(config=config, run_name=run_name)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — CNN Warmup
    # ══════════════════════════════════════════════════════════════════════════
    phase1_epochs = int(config['training'].get('phase1_epochs', 5))
    lr_phase1     = float(config['training'].get('lr', 1e-3))
    lr_phase2     = float(config['training'].get('lr_phase2', 1e-4))

    if phase1_epochs > 0:
        print(f"\n{'='*60}")
        print(f"  PHASE 1 — CNN Warmup ({phase1_epochs} epochs, LR={lr_phase1})")
        print(f"  Frozen: gnn_encoder, motif_layer, node_selector")
        print(f"  Trainable: cnn_encoder, classifier")
        print(f"{'='*60}\n")

        model.freeze_for_phase1()

        # Build config copy for phase-1 (no cosine scheduler, just constant LR)
        p1_config = {**config,
                     'training': {**config['training'],
                                  'epochs':    phase1_epochs,
                                  'patience':  phase1_epochs + 1,   # no early stop
                                  'scheduler': 'none',
                                  'lr':         lr_phase1}}

        p1_trainer = build_phase_trainer(
            model, train_loader, val_loader,
            p1_config, device,
            run_name=f"{run_name}_p1",
            save_dir=save_path,
            lr_override=None,
            start_epoch=0,
        )
        p1_trainer.fit()
        print("[Phase 1] Done.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Full End-to-End
    # ══════════════════════════════════════════════════════════════════════════
    total_epochs  = int(config['training'].get('epochs', 60))
    phase2_epochs = max(1, total_epochs - phase1_epochs)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Full End-to-End ({phase2_epochs} epochs, LR={lr_phase2})")
    print(f"  Loss: CE + {config['model'].get('motif_div_weight', 0.2)} × motif_diversity")
    print(f"{'='*60}\n")

    model.unfreeze_all()

    p2_config = {**config,
                 'training': {**config['training'],
                               'epochs': phase2_epochs,
                               'lr':      lr_phase2}}

    p2_trainer = build_phase_trainer(
        model, train_loader, val_loader,
        p2_config, device,
        run_name=run_name,
        save_dir=save_path,
        lr_override=lr_phase2,
        start_epoch=phase1_epochs,
    )
    p2_trainer.fit()
    print("[Phase 2] Done.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION on test set
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  EVALUATION — Test Set")
    print("=" * 60)

    load_checkpoints(model, None, save_path, device)

    eval_dir = os.path.join(root_path, "outputs/figures")
    os.makedirs(eval_dir, exist_ok=True)

    testset_path = os.path.join(data_path, "test.csv")
    evaluate_and_show(model, test_loader, testset_path, device, eval_dir)

    # ── Upload checkpoint to WandB ────────────────────────────────────────────
    if config['logging'].get('use_wandb', True):
        print("\n[WandB] Uploading best checkpoint...")
        save_model_to_wandb(save_path)
        wandb.finish()

    print("\n\t\t✅ DONE!\n")


if __name__ == "__main__":
    main()
