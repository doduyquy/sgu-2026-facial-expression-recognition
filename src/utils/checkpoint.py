import os
from collections import OrderedDict

import torch


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _has_module_prefix(state_dict):
    return any(key.startswith("module.") for key in state_dict.keys())


def _strip_module_prefix(state_dict):
    return OrderedDict(
        (key[len("module."):] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def _add_module_prefix(state_dict):
    return OrderedDict(
        (key if key.startswith("module.") else f"module.{key}", value)
        for key, value in state_dict.items()
    )


def _match_state_dict_to_model(model, state_dict):
    model_is_dp = isinstance(model, torch.nn.DataParallel)
    state_dict_has_module = _has_module_prefix(state_dict)

    if model_is_dp and not state_dict_has_module:
        return _add_module_prefix(state_dict)

    if not model_is_dp and state_dict_has_module:
        return _strip_module_prefix(state_dict)

    return state_dict


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    model_to_save = unwrap_model(model)

    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path,
    )


def load_checkpoints(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Not found file {checkpoint_path}")

    print(f"--> Loading ckpt from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state_dict = _match_state_dict_to_model(model, ckpt["model_state_dict"])

    model.load_state_dict(model_state_dict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt["epoch"]
