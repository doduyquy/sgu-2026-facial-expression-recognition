import os
import sys
import argparse
from pathlib import Path
import torch

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.dataloader import build_dataloader
from src.models import get_model
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer, build_scheduler
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--env", default="local")
    parser.add_argument("--total-epochs", type=int, default=None,
                        help="Override total epochs (if checkpoint lacks epochs_total)")
    parser.add_argument("--non-interactive", action='store_true', help="Don't prompt; abort on missing fields")
    parser.add_argument("--save-path", default=None, help="Optional path to save resumed checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault('WANDB_MODE', 'online')

    config = load_config(args.config, args.env)

    data_path = config['local'].get('data_path', '../dataset') if args.env == 'local' else config['kaggle'].get('data_path', '/kaggle/input')
    train_loader, val_loader, _ = build_dataloader(config=config, data_path=data_path)

    model = get_model(name=config['model']['name'], config=config)
    model = model.to(device)

    criterion = build_loss(config=config, class_weights=None)
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    run_name = f"resume_{config['model'].get('name','model')}"
    default_save = args.save_path or os.path.join(config.get('path', {}).get('root', './'), f"outputs/checkpoints/{config['model'].get('name')}/{run_name}_best.pth")
    os.makedirs(os.path.dirname(default_save), exist_ok=True)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, default_save)

    # load checkpoint safely
    print(f"Loading checkpoint {args.ckpt} -> device {device}")
    ckpt = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt and optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: failed to load optimizer state: {e}")
            # try to restore lr if present in ckpt optimizer groups
            og = ckpt.get('optimizer_state_dict', {}).get('param_groups', None)
            if og and len(og) > 0:
                lr = og[0].get('lr', None)
                if lr is not None:
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                    print(f"Set optimizer lr from ckpt to {lr}")
    else:
        # No optimizer_state_dict key - still try to set lr from checkpoint if possible
        og = ckpt.get('optimizer_state_dict', {}).get('param_groups', None)
        if og and len(og) > 0 and optimizer is not None:
            lr = og[0].get('lr', None)
            if lr is not None:
                for g in optimizer.param_groups:
                    g['lr'] = lr
                print(f"Set optimizer lr from ckpt to {lr} (no optimizer_state_dict load)")
    if 'scheduler_state_dict' in ckpt and scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: failed to load scheduler state: {e}")
    else:
        if scheduler is not None:
            # try to approximate scheduler resume: set last_epoch to saved_epoch
            try:
                saved_epoch_tmp = int(ckpt.get('epoch', 0))
                if hasattr(scheduler, 'last_epoch'):
                    scheduler.last_epoch = saved_epoch_tmp
                    print(f"Set scheduler.last_epoch = {saved_epoch_tmp} (approx resume)")
                else:
                    print("Scheduler has no attribute 'last_epoch' to set; continuing without full restore.")
            except Exception:
                pass
            # prompt user whether to continue without scheduler or abort
            if not args.non_interactive:
                ans = input("Checkpoint missing scheduler_state_dict. Continue without restoring scheduler? [y/N]: ")
                if ans.strip().lower() not in ('y', 'yes'):
                    print("Aborting resume (scheduler required).")
                    return
                else:
                    print("Continuing without restoring scheduler_state_dict.")
            else:
                print("Non-interactive mode: continuing without scheduler_state_dict.")

    saved_epoch = int(ckpt.get('epoch', 0))
    ckpt_total = ckpt.get('epochs_total', None)
    # Determine total epochs: prefer CLI, then ckpt, then config; if still missing prompt user
    total_epochs = args.total_epochs or ckpt_total or config['training'].get('epochs', None)
    if total_epochs is None:
        if not args.non_interactive:
            v = input("Checkpoint missing total epochs. Enter desired total epochs (or blank to abort): ")
            if v.strip() == "":
                print("Aborting resume: total epochs unspecified.")
                return
            try:
                total_epochs = int(v.strip())
            except Exception:
                print("Invalid integer. Aborting.")
                return
        else:
            print("Non-interactive mode and no total epochs available. Aborting.")
            return
    start_epoch = saved_epoch + 1

    # WandB resume/init if run id present in checkpoint
    wandb_id = ckpt.get('wandb_run_id', None)
    if wandb_id is not None:
        try:
            wandb.init(
                project=config['logging'].get('project_name', 'FER2013'),
                entity=config['logging'].get('wandb_entity', None),
                id=wandb_id,
                resume="allow",
                name=run_name,
                config=config,
            )
            print(f"Resuming WandB run id={wandb_id}")
        except Exception as e:
            print(f"Warning: failed to init/resume wandb run {wandb_id}: {e}")
    else:
        try:
            if config.get('logging', {}).get('use_wandb', True):
                wandb.init(
                    project=config['logging'].get('project_name', 'FER2013'),
                    entity=config['logging'].get('wandb_entity', None),
                    name=run_name,
                    config=config,
                    resume="allow",
                )
        except Exception:
            pass

    print(f"Checkpoint epoch={saved_epoch}, total_in_ckpt={ckpt_total}, resuming start_epoch={start_epoch}, total_epochs={total_epochs}")

    # Manual resume loop (uses Trainer.train_one_epoch / validate)
    for ep in range(start_epoch, int(total_epochs)):
        trainer._current_epoch = ep
        train_loss, train_acc = trainer.train_one_epoch()
        val_loss, val_acc = trainer.validate()
        print(f"Epoch {ep+1}/{total_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # scheduler step (mirror typical fit logic)
        if trainer.scheduler is not None:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(val_loss)
            else:
                trainer.scheduler.step()

        # save checkpoint including scheduler + total epochs so future resumes are exact
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
            'epoch': ep,
            'epochs_total': int(total_epochs)
        }, trainer.path_save_ckpt)
        print("Saved checkpoint:", trainer.path_save_ckpt)


if __name__ == '__main__':
    main()
