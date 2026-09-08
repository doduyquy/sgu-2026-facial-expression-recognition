import copy
import math

import torch
from torch import nn


class ModelEMA:
    """EMA parameters AND floating buffers, with first-update copy and decay ramp."""

    def __init__(self, model, decay=0.999, warmup_updates=2000):
        if not 0 <= decay < 1 or warmup_updates < 0:
            raise ValueError("Invalid EMA decay or warmup_updates")
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = self.decay
        if self.warmup_updates:
            decay *= 1 - math.exp(-self.num_updates / self.warmup_updates)
        if self.num_updates == 1:
            decay = 0.0
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            value = source[name].detach()
            if target.is_floating_point():
                target.mul_(decay).add_(value, alpha=1 - decay)
            else:
                target.copy_(value)


@torch.no_grad()
def recalibrate_bn(model, loader, device, max_batches=0):
    """Update only BN statistics using deterministic TRAIN images, never validation/test.

    Dropout, stochastic depth and TTA stay disabled. Restore all module modes even
    on failure; restore old statistics too if calibration cannot finish.
    """
    if max_batches < 0:
        raise ValueError("max_batches must be nonnegative")
    if getattr(loader.dataset, "split", "train") != "train":
        raise ValueError("BN calibration must use the training split")
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    if not bns:
        return 0
    modes = [(m, m.training) for m in model.modules()]
    momenta = [m.momentum for m in bns]
    backup = [{k: v.detach().clone() for k, v in m.named_buffers()} for m in bns]
    batches = 0
    try:
        model.eval()
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(images.to(device, non_blocking=True), use_tta=False)
            batches += 1
        if not batches:
            raise ValueError("BN calibration loader is empty")
    except Exception:
        for bn, buffers in zip(bns, backup):
            for key, value in bn.named_buffers():
                value.copy_(buffers[key])
        raise
    finally:
        for bn, momentum in zip(bns, momenta):
            bn.momentum = momentum
        for module, training in modes:
            module.training = training
    return batches
