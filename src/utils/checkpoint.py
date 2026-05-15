import torch
import os

def save_checkpoint():
    pass

def load_checkpoints(model, optimizer, checkpoint_path, device, load_optimizer=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Not found file {checkpoint_path}")

    print(f"--> Loading ckpt from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    # load weight -> model
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(ckpt['model_state_dict'])
    # load optimizer and return current checkpoint
    if load_optimizer:
        if optimizer is None:
            raise ValueError("optimizer must be provided when load_optimizer=True")
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    return ckpt['epoch']

