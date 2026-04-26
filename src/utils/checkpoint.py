import torch
import os

def save_checkpoint():
    pass

def load_checkpoints(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Not found file {checkpoint_path}")

    print(f"--> Loading ckpt from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path)
    # load weight -> model
    model.load_state_dict(ckpt['model_state_dict'])
    # load optimizer and return current checkpoint
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    return ckpt['epoch']

