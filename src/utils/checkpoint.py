import torch
import os


def save_checkpoint(path, model, optimizer, epoch, scheduler=None, epochs_total=None, trainer_runtime=None, wandb_run_id=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        try:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass
    if epochs_total is not None:
        ckpt["epochs_total"] = epochs_total
    if trainer_runtime is not None:
        ckpt["trainer_runtime"] = trainer_runtime
    if wandb_run_id is not None:
        ckpt["wandb_run_id"] = wandb_run_id

    # RNG states
    try:
        ckpt["rng_state"] = torch.get_rng_state()
    except Exception:
        ckpt["rng_state"] = None
    try:
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    except Exception:
        ckpt["cuda_rng_state_all"] = None

    torch.save(ckpt, path)


def load_checkpoints(model, optimizer, checkpoint_path, device, scheduler=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Not found file {checkpoint_path}")

    print(f"--> Loading ckpt from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    # load weight -> model
    model.load_state_dict(ckpt['model_state_dict'])
    # load optimizer if present
    if 'optimizer_state_dict' in ckpt and optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            print("\t-!- Warning: failed to load optimizer state_dict cleanly.")

    # try load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception:
            print("\t-!- Warning: failed to load scheduler state_dict cleanly.")

    epoch = ckpt.get('epoch', 0)
    epochs_total = ckpt.get('epochs_total', None)

    return epoch, epochs_total

