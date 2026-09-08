import math

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau


def build_adamw_param_groups(model, weight_decay, backbone_lr=None, head_lr=None, exclude_norm_bias=True):
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm,
                  nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    no_decay = {id(p) for module in model.modules() if isinstance(module, norm_types)
                for p in module.parameters(recurse=False)}
    grouped = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        branch = "backbone" if name.startswith("backbone.") else "head"
        exempt = exclude_norm_bias and (id(param) in no_decay or name.endswith((".bias", ".layer_scale")) or param.ndim <= 1)
        key = branch, exempt
        grouped.setdefault(key, []).append(param)
    groups = []
    for (branch, exempt), params in grouped.items():
        group = {"params": params, "weight_decay": 0.0 if exempt else weight_decay,
                 "name": f"{branch}_{'no_decay' if exempt else 'decay'}"}
        lr = backbone_lr if branch == "backbone" else head_lr
        if lr is not None:
            group["lr"] = lr
        groups.append(group)
    return groups


def build_optimizer(model, cfg):
    lr = cfg.get("lr", 3e-4)
    groups = build_adamw_param_groups(model, cfg.get("weight_decay", 0.002),
                                     cfg.get("backbone_lr", lr), cfg.get("head_lr", lr),
                                     cfg.get("no_weight_decay_norm_bias", True))
    return AdamW(groups, lr=lr, weight_decay=0.0)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    epochs = int(cfg.get("epochs", 60))
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError("Training needs at least one epoch and one full batch")
    kind = cfg.get("scheduler", "warmup_cosine")
    eta_min = float(cfg.get("eta_min", 1e-6))
    if kind in ("warmup_cosine", "cosine_annealing"):
        total = epochs * steps_per_epoch
        warmup = min(int(cfg.get("warmup_epochs", 0)) * steps_per_epoch, total - 1)
        if warmup < 0:
            raise ValueError("warmup_epochs must be nonnegative")
        start = float(cfg.get("warmup_start_factor", 0.1))
        if not 0 < start <= 1:
            raise ValueError("warmup_start_factor must be in (0, 1]")
        functions = []
        for group in optimizer.param_groups:
            if not 0 <= eta_min <= group["lr"]:
                raise ValueError("eta_min must be between zero and each group's LR")
            floor = eta_min / group["lr"]
            def schedule(step, floor=floor):
                if warmup and step < warmup:
                    return start + (1 - start) * step / warmup
                progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
                return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
            functions.append(schedule)
        return LambdaLR(optimizer, functions), "batch"
    if kind == "cosine_annealing_warm_restart":
        return CosineAnnealingWarmRestarts(optimizer, T_0=cfg.get("T_0", 30),
                                          T_mult=cfg.get("T_mult", 2), eta_min=eta_min), "restart"
    if kind == "reduce_lr_on_plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                 patience=cfg.get("lr_patience", 4), min_lr=eta_min), "metric"
    raise ValueError(f"Unsupported scheduler: {kind}")
