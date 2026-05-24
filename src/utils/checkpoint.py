import torch
import os

def save_checkpoint():
    pass


def load_partial_state_dict(model, checkpoint_state):
    model_state = model.state_dict()
    filtered = {}

    skipped = []
    for k, v in checkpoint_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[PartialLoad] loaded keys: {len(filtered)}")
    print(f"[PartialLoad] skipped shape-mismatch keys: {len(skipped)}")
    for k in skipped[:20]:
        print("  skipped:", k)
    print("[PartialLoad] missing:", len(missing), "unexpected:", len(unexpected))
    return missing, unexpected, skipped

def load_checkpoints(model, optimizer, checkpoint_path, device, partial=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Not found file {checkpoint_path}")

    print(f"--> Loading ckpt from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if partial:
        load_partial_state_dict(model, ckpt["model_state_dict"])
        print("[PartialLoad] optimizer state is skipped because model shape changed.")
    else:
        # load weight -> model
        model.load_state_dict(ckpt['model_state_dict'])
        # load optimizer and return current checkpoint
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    return ckpt['epoch']

