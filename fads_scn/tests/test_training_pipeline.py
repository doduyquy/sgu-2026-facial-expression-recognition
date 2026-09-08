import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fads_scn.data.dataset import build_dataloaders, transforms_from_config, build_clean_train_loader
from fads_scn.evaluation.evaluator import evaluate_model
from fads_scn.losses.scn_loss import SCNLoss
from fads_scn.runtime import build_model, load_config, apply_overrides, load_checkpoint_model
from fads_scn.training.ema import ModelEMA, recalibrate_bn
from fads_scn.training.optimizer import build_optimizer, build_scheduler
from fads_scn.training.trainer import AttentiveSCNTrainer, selection_key
from fads_scn.calibrate_logit_bias import search_best_logit_bias
from fads_scn.models.spatial_attention import MultiHeadSpatialAttention


@pytest.fixture(autouse=True)
def fast_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def synthetic_outputs():
    labels = torch.arange(7)
    logits = torch.zeros(7, 7, requires_grad=True)
    with torch.no_grad():
        logits[labels, labels] = torch.linspace(-2, 4, 7)
    return {"logits": logits, "alpha": torch.linspace(0.1, 1, 7).view(-1, 1).requires_grad_()}, labels


@pytest.mark.parametrize("scn,weight,epoch,warmup", [(False, 0.1, 10, 0), (True, 0, 10, 0), (True, 0.1, 0, 8)])
def test_disabled_scn_and_warmup_are_plain_ce(scn, weight, epoch, warmup):
    outputs, labels = synthetic_outputs()
    criterion = SCNLoss(use_scn=scn, rank_loss_weight=weight, label_smoothing=0, div_loss_weight=0)
    result = criterion(outputs, labels, current_epoch=epoch, rank_warmup_epochs=warmup)
    expected = F.cross_entropy(outputs["logits"], labels)
    torch.testing.assert_close(result["loss"], expected)
    result["loss"].backward()
    assert outputs["alpha"].grad is None
    assert not result["scn_active"]


def test_rank_is_independent_of_class_weights():
    outputs, labels = synthetic_outputs()
    a = SCNLoss(class_weights=None)(outputs, labels, current_epoch=10)
    b = SCNLoss(class_weights=torch.tensor([4., .1, 2., 1., .5, 3., .2]))(outputs, labels, current_epoch=10)
    assert torch.equal(a["noisy_mask"], b["noisy_mask"])
    torch.testing.assert_close(a["rank_loss"], b["rank_loss"])
    assert not torch.isclose(a["base_ce"], b["base_ce"])


def test_mixup_is_exact_mixture_without_scn_weighting():
    outputs, labels = synthetic_outputs()
    second = labels.roll(1)
    result = SCNLoss(label_smoothing=0)(outputs, labels, targets_b=second, lam=.7, current_epoch=10)
    expected = .7 * F.cross_entropy(outputs["logits"], labels) + .3 * F.cross_entropy(outputs["logits"], second)
    torch.testing.assert_close(result["loss"], expected)
    assert not result["scn_active"]


def test_classwise_ranking_skips_singletons():
    outputs, _ = synthetic_outputs()
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 3])
    result = SCNLoss(rank_mode="classwise")(outputs, labels, current_epoch=10)
    assert not result["ranked_mask"][-2:].any()
    assert result["noisy_mask"][:3].sum() == 1
    assert result["noisy_mask"][3:5].sum() == 1


def test_validation_nll_is_batch_and_alpha_independent():
    class Lookup(nn.Module):
        def forward(self, x, use_tta=False):
            return {"logits": x[:, :7], "alpha": x[:, 7:]}
    outputs, labels = synthetic_outputs()
    ds = TensorDataset(torch.cat((outputs["logits"].detach(), outputs["alpha"].detach()), 1), labels)
    def poison(*args):
        raise AssertionError("Validation must not call training criterion")
    small = evaluate_model(Lookup(), DataLoader(ds, batch_size=1), "cpu", criterion=poison)
    large = evaluate_model(Lookup(), DataLoader(ds, batch_size=7), "cpu", criterion=poison)
    assert small["nll"] == pytest.approx(large["nll"], abs=1e-6)
    assert small["accuracy"] == large["accuracy"]


class BNProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.bn = nn.BatchNorm1d(1)
        self.drop = nn.Dropout(.9)
        self.observed = []

    def forward(self, x, use_tta=None):
        self.observed.append((self.training, self.drop.training, use_tta))
        return self.drop(self.bn(self.linear(x)))


def test_ema_copies_first_update_and_averages_buffers():
    raw = BNProbe()
    ema = ModelEMA(raw, decay=.5, warmup_updates=0)
    with torch.no_grad():
        raw.linear.weight.fill_(3)
        raw.bn.running_mean.fill_(7.5)
    ema.update(raw)
    assert ema.module.linear.weight.item() == 3
    with torch.no_grad():
        raw.linear.weight.fill_(5)
        raw.bn.running_mean.fill_(12.5)
        raw.bn.num_batches_tracked.fill_(2)
    ema.update(raw)
    assert ema.module.linear.weight.item() == 4
    assert ema.module.bn.running_mean.item() == 10
    assert ema.module.bn.num_batches_tracked.item() == 2
    assert all(not p.requires_grad for p in ema.module.parameters())


def test_bn_recalibration_uses_only_bn_training_and_restores_modes():
    model = BNProbe().train()
    with torch.no_grad():
        model.linear.weight.fill_(2)
    ds = TensorDataset(torch.tensor([[1.], [2.], [3.], [4.]]))
    ds.split = "train"
    before = model.linear.weight.clone()
    assert recalibrate_bn(model, DataLoader(ds, batch_size=4), "cpu") == 1
    assert model.bn.running_mean.item() == pytest.approx(5)
    assert model.observed == [(False, False, False)]
    assert model.training and model.drop.training and model.bn.training
    torch.testing.assert_close(model.linear.weight, before)
    ds.split = "val"
    with pytest.raises(ValueError, match="training split"):
        recalibrate_bn(model, DataLoader(ds, batch_size=4), "cpu")


def test_differential_lr_and_warmup_cosine_cover_all_parameters():
    model = nn.Module()
    model.backbone = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    model.backbone.register_parameter("layer_scale", nn.Parameter(torch.ones(2, 1, 1)))
    model.head = nn.Linear(2, 7)
    cfg = {"epochs": 4, "warmup_epochs": 1, "backbone_lr": 5e-5, "head_lr": 3e-4, "eta_min": 1e-6}
    opt = build_optimizer(model, cfg)
    params = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(params) == len(set(params)) == len(list(model.parameters()))
    scale_group = next(g for g in opt.param_groups if any(p is model.backbone.layer_scale for p in g["params"]))
    assert scale_group["weight_decay"] == 0
    scheduler, interval = build_scheduler(opt, cfg, 2)
    assert interval == "batch"
    initial = [g["lr"] for g in opt.param_groups]
    assert min(initial) == pytest.approx(5e-6)
    for _ in range(2):
        opt.step(); scheduler.step()
    assert max(g["lr"] for g in opt.param_groups) == pytest.approx(3e-4)
    for _ in range(5):
        opt.step(); scheduler.step()
    assert all(g["lr"] == pytest.approx(1e-6) for g in opt.param_groups)


@pytest.fixture
def csv_config(tmp_path):
    for split in ("train", "val"):
        pixels = [" ".join([str(50 + i * 10)] * (48 * 48)) for i in range(14)]
        pd.DataFrame({"emotion": np.arange(14) % 7, "pixels": pixels}).to_csv(tmp_path / f"{split}.csv", index=False)
    return {"seed": {"random_seed": 42}, "model": {"backbone": "resnet18", "in_channels": 1,
            "num_classes": 7, "embed_dim": 32, "num_attn_heads": 2, "use_pretrained": False},
            "data": {"data_path": str(tmp_path), "batch_size": 7, "num_workers": 0, "input_size": 48},
            "training": {"epochs": 2, "patience": 2, "warmup_epochs": 1, "use_ema": True,
                         "ema_bn_mode": "recalibrate", "output_dir": str(tmp_path / "outputs")},
            "scn": {"use_scn": False}}


def test_data_transforms_and_independent_loader_rng(csv_config):
    cfg = copy.deepcopy(csv_config)
    cfg["model"]["in_channels"] = 3
    cfg["data"].update(input_size=96, normalization="imagenet")
    image = Image.fromarray(np.full((48, 48), 128, dtype=np.uint8))
    assert transforms_from_config(cfg)(image).shape == (3, 96, 96)
    def order(extra_eval):
        train, val, test = build_dataloaders(cfg)
        assert test is None  # No test.csv exists, and training does not need it.
        list(train)
        if extra_eval:
            list(val)
        return torch.cat([indices for _, _, indices in train])
    assert torch.equal(order(False), order(True))
    clean = build_clean_train_loader(cfg)
    assert torch.equal(next(iter(clean))[0], next(iter(clean))[0])


@pytest.mark.parametrize("variant", ["scn_convnext.yaml", "scn_convnext_light.yaml", "scn_convnext_native.yaml"])
def test_convnext_variants_forward_backward_and_dynamic_tta(variant):
    cfg, _ = load_config(Path("fads_scn/configs") / variant)
    cfg["model"]["use_pretrained"] = False
    model = build_model(cfg)
    channels = cfg["model"]["in_channels"]
    size = 64 if cfg["model"]["backbone_mode"] == "native" else 48
    x = torch.randn(2, channels, size, size)
    model.train()
    outputs = model(x, use_tta=False)
    loss = outputs["logits"].square().mean()
    loss.backward()
    assert torch.isfinite(model.global_proj[2].weight.grad).all()
    if variant.endswith("native.yaml"):
        assert model.spatial_attention is None and model.latent_graph is None
        assert model.backbone.features[0][0].stride == (4, 4)
        assert model.backbone.pretrained_norm is not None
    model.eval()
    with torch.no_grad():
        assert model(x, use_tta="multiscale")["logits"].shape == (2, 7)


def test_light_attention_reduces_parameters():
    dense = MultiHeadSpatialAttention(768, 256, 8)
    light = MultiHeadSpatialAttention(768, 256, 8, attention_type="depthwise")
    assert sum(p.numel() for p in light.attn_conv.parameters()) < sum(p.numel() for p in dense.attn_conv.parameters()) / 10


def test_checkpoint_config_wins_and_graph_off_roundtrip(csv_config, tmp_path):
    cfg = copy.deepcopy(csv_config)
    cfg["model"]["use_latent_graph"] = False
    model = build_model(cfg).eval()
    path = tmp_path / "roundtrip.pth"
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)
    wrong = copy.deepcopy(cfg)
    wrong["model"]["use_latent_graph"] = True
    loaded, effective, _ = load_checkpoint_model(path, fallback_cfg=wrong)
    assert not effective["model"]["use_latent_graph"] and loaded.latent_graph is None
    x = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        torch.testing.assert_close(model(x, use_tta=False)["logits"], loaded(x, use_tta=False)["logits"])


def test_bias_constraints_are_honored_during_refinement():
    names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    logits = 3 * torch.eye(7)
    logits[3, 3], logits[3, 0] = 1, 2
    cfg = {"metric": "acc", "class_names": names, "tune_classes": ["fear"],
           "bias_grid": {"fear": [0., .2]}, "fixed_bias": {"angry": 0., "happy": 0.}}
    result = search_best_logit_bias(logits, torch.arange(7), cfg)
    assert all(value == 0 for i, value in enumerate(result["best_bias"]) if i != 2)
    assert 0 <= result["best_bias"][2] <= .2
    assert result["best_metrics"]["acc"] == pytest.approx(6 / 7)


def test_accuracy_selection_overrides_macro_f1():
    a = {"accuracy": .8, "macro_f1": .6, "nll": 1.}
    b = {"accuracy": .79, "macro_f1": .9, "nll": .5}
    assert selection_key(a) > selection_key(b)


def test_train_save_load_pipeline_without_test_split(csv_config):
    class TinyFER(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4),
                                          nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.classifier, self.gate = nn.Linear(4, 7), nn.Linear(4, 1)
        def forward(self, images, **kwargs):
            features = self.backbone(images)
            return {"logits": self.classifier(features), "alpha": .1 + .9 * self.gate(features).sigmoid()}
    loaders = build_dataloaders(csv_config)
    trainer = AttentiveSCNTrainer(TinyFER(), SCNLoss(use_scn=False), *loaders, cfg=csv_config)
    best = trainer.fit()
    saved = torch.load(best, weights_only=True)
    assert saved["selection_metric"] == "accuracy"
    assert saved["weights_source"] in ("raw", "ema")
    assert best.parent != Path(csv_config["training"]["output_dir"])
    assert not (best.parent / "metrics_test.json").exists()
    lines = (best.parent / "history.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "alpha_by_class" in json.loads(lines[0])["train"]
    result = evaluate_model(trainer.model, loaders[1], "cpu", use_tta=False)
    assert result["accuracy"] == pytest.approx(saved["val_acc"])
    assert result["nll"] == pytest.approx(saved["val_loss"], abs=1e-6)


def test_cli_overrides_reject_unknown_keys():
    cfg, _ = load_config("fads_scn/configs/scn_convnext_ce.yaml")
    apply_overrides(cfg, ["training.use_class_weights=false", "model.use_latent_graph=false"])
    assert not cfg["scn"]["use_scn"] and not cfg["training"]["use_class_weights"]
    with pytest.raises(ValueError, match="Unknown"):
        apply_overrides(cfg, ["training.typo=1"])
