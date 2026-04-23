import torch
import os


def save_checkpoint(model, optimizer, epoch, path: str, **extra):
    """Lưu checkpoint vào path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        **extra,
    }, path)
    print(f"--> Checkpoint saved: {path}")


def load_checkpoints(model, optimizer, checkpoint_path: str, device):
    """
    Load checkpoint từ path.

    Args:
        model:           PyTorch model (sẽ load weights vào)
        optimizer:       optimizer (sẽ load state vào)
        checkpoint_path: đường dẫn file .pth
        device:          torch.device

    Returns:
        epoch: epoch đã save
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint không tồn tại: {checkpoint_path}")

    print(f"--> Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", 0)
    val_f1 = ckpt.get("best_val_macro_f1", None)

    if val_f1 is not None:
        print(f"--> Restored ep={epoch+1}  best_val_macro_f1={val_f1:.4f}")
    else:
        print(f"--> Restored ep={epoch+1}")

    return epoch
