import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from fads_scn.training.trainer import (
    ModelEMA,
    build_adamw_param_groups,
    build_warmup_cosine_scheduler,
    recalibrate_batch_norm,
)


class TinyDifferentialLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


class TinyBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x, use_tta=False):
        return self.dropout(self.bn(x))


def test_differential_lr_and_weight_decay_groups():
    model = TinyDifferentialLRModel()
    groups = build_adamw_param_groups(
        model,
        weight_decay=0.002,
        backbone_lr=5e-5,
        head_lr=3e-4,
    )
    groups_by_name = {group["group_name"]: group for group in groups}

    assert groups_by_name["backbone_decay"]["lr"] == 5e-5
    assert groups_by_name["backbone_no_decay"]["lr"] == 5e-5
    assert groups_by_name["head_decay"]["lr"] == 3e-4
    assert groups_by_name["head_no_decay"]["lr"] == 3e-4
    assert groups_by_name["backbone_decay"]["weight_decay"] == 0.002
    assert groups_by_name["head_decay"]["weight_decay"] == 0.002
    assert groups_by_name["backbone_no_decay"]["weight_decay"] == 0.0
    assert groups_by_name["head_no_decay"]["weight_decay"] == 0.0

    grouped_ids = [id(param) for group in groups for param in group["params"]]
    trainable_ids = [id(param) for param in model.parameters() if param.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)


def test_warmup_cosine_preserves_group_lrs_and_reaches_eta_min():
    model = TinyDifferentialLRModel()
    optimizer = AdamW(
        build_adamw_param_groups(model, 0.002, backbone_lr=5e-5, head_lr=3e-4),
        lr=3e-4,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        start_factor=0.1,
        eta_min=1e-6,
    )

    initial_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert math.isclose(initial_lrs["backbone_decay"], 5e-6, rel_tol=1e-6)
    assert math.isclose(initial_lrs["head_decay"], 3e-5, rel_tol=1e-6)

    for _ in range(2):
        optimizer.step()
        scheduler.step()
    warm_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert math.isclose(warm_lrs["backbone_decay"], 5e-5, rel_tol=1e-6)
    assert math.isclose(warm_lrs["head_decay"], 3e-4, rel_tol=1e-6)

    for _ in range(8):
        optimizer.step()
        scheduler.step()
    final_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert math.isclose(final_lrs["backbone_decay"], 1e-6, rel_tol=1e-6)
    assert math.isclose(final_lrs["head_decay"], 1e-6, rel_tol=1e-6)


def test_ema_decay_ramp_and_buffer_copy():
    model = TinyBNModel()
    ema = ModelEMA(model, decay=0.5, warmup_updates=3)

    with torch.no_grad():
        model.bn.weight.fill_(2.0)
        model.bn.running_mean.fill_(3.0)
    ema.update(model)
    assert ema.current_decay() == 0.0
    assert torch.equal(ema.module.bn.weight, model.bn.weight)
    assert torch.equal(ema.module.bn.running_mean, model.bn.running_mean)

    ema.update(model)
    assert math.isclose(ema.current_decay(), 0.25)
    ema.update(model)
    assert math.isclose(ema.current_decay(), 0.5)


def test_bn_recalibration_uses_data_and_leaves_model_in_eval_mode():
    model = TinyBNModel()
    model.train()
    inputs = torch.full((16, 2), 5.0)
    loader = DataLoader(TensorDataset(inputs), batch_size=4, shuffle=False)

    processed = recalibrate_batch_norm(model, loader, torch.device("cpu"))

    assert processed == 4
    assert torch.allclose(model.bn.running_mean, torch.full((2,), 5.0), atol=1e-5)
    assert model.training is False
    assert model.bn.training is False
    assert model.dropout.training is False
