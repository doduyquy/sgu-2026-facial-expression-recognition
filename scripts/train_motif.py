import os
import argparse
import yaml
import torch
from pathlib import Path
import sys

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataloader import get_dataloaders
from src.models import get_model
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.losses import build_loss
from src.training.motif_trainer import MotifTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/motif_config.yaml")
    parser.add_argument("--save_dir", type=str, default="experiments/motif_run")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Training on {device}")

    # 1. Dataloaders
    train_loader, val_loader, _ = get_dataloaders(config)

    # 2. Model
    model = get_model(config['model']['name'], config=config)
    
    # 3. Loss, Optimizer, Scheduler
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # 4. Trainer
    trainer = MotifTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        save_dir=args.save_dir
    )

    # 5. Fit
    print("--> Starting Motif-based Training Pipeline...")
    trainer.fit()

if __name__ == "__main__":
    main()
