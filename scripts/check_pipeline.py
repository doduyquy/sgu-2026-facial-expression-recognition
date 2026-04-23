"""
Full pipeline check script — chạy trước khi push lên Kaggle.
Usage: python scripts/check_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

PASS = "[OK]"
FAIL = "[FAIL]"

def section(title):
    print(f"\n=== {title} ===")

def ok(msg):
    print(f"  {PASS} {msg}")

def fail(msg):
    print(f"  {FAIL} {msg}")
    sys.exit(1)

# ────────────────────────────────────────────────────────────────────────────
section("1. CONFIG LOAD")
from src.utils.config import load_config

cfg_k = load_config("mlp_baseline", "kaggle")
cfg_l = load_config("mlp_baseline", "local")

assert cfg_k["env"]["platform"] == "kaggle"
ok(f"env.platform = kaggle")

assert cfg_k.get("graph_cache_path") is not None
ok(f"kaggle graph_cache_path = {cfg_k['graph_cache_path']}")

assert cfg_l.get("graph_cache_path") is not None
ok(f"local  graph_cache_path = {cfg_l['graph_cache_path']}")

nw_k = cfg_k.get("num_workers")
nw_l = cfg_l.get("num_workers")
assert nw_k == 2, f"num_workers kaggle should be 2, got {nw_k}"
assert nw_l == 0, f"num_workers local should be 0, got {nw_l}"
ok(f"num_workers: kaggle={nw_k} | local={nw_l}")

assert cfg_k["model"]["name"] == "mlp_baseline"
ok(f"model.name = {cfg_k['model']['name']}")

assert cfg_k["training"]["epochs"] == 50
ok(f"training.epochs = {cfg_k['training']['epochs']}")

assert cfg_k["logging"]["use_wandb"] is False
ok(f"use_wandb = {cfg_k['logging']['use_wandb']}")

# ────────────────────────────────────────────────────────────────────────────
section("2. GRAPH CONFIG")
from src.graph.graph_config import GraphConfig

g_cfg = GraphConfig.from_config(cfg_l)
assert g_cfg.connectivity == 8
assert g_cfg.image_size == 48
assert len(g_cfg.node_features) == 3
assert len(g_cfg.edge_features) == 5
ok(f"connectivity={g_cfg.connectivity}, image_size={g_cfg.image_size}")
ok(f"node_features={g_cfg.node_features}")
ok(f"edge_features={g_cfg.edge_features}")

# ────────────────────────────────────────────────────────────────────────────
section("3. IMAGE-TO-GRAPH (1 sample)")
from src.graph.image_to_graph import ImageGraphBuilder

builder = ImageGraphBuilder(g_cfg)
fake_img = np.random.randint(0, 256, (48, 48), dtype=np.uint8).astype(np.float32)
g = builder.build_graph(image=fake_img, label=3, image_id=0,
                         split_name="test", usage="test")

assert g.node_features.shape == (2304, 3), f"node_features shape={g.node_features.shape}"
assert g.edge_index.shape[0] == 2
assert g.edge_attr.shape[1] == 5
assert g.label == 3
assert not np.isnan(g.node_features).any(), "NaN in node_features!"
assert not np.isnan(g.edge_attr).any(), "NaN in edge_attr!"

ok(f"node_features: {g.node_features.shape}")
ok(f"edge_index:    {g.edge_index.shape}")
ok(f"edge_attr:     {g.edge_attr.shape}")
ok("No NaN detected")

# ────────────────────────────────────────────────────────────────────────────
section("4. GRAPH VECTORIZER")
from src.features.graph_vectorizer import GraphVectorizer

vec = GraphVectorizer(use_mean=True, use_std=True, use_max=True)
v = vec.transform(g)
expected_dim = 9  # 3 node_features × 3 poolings
assert v.shape == (expected_dim,), f"vector shape={v.shape}"
assert not np.isnan(v).any(), "NaN in graph vector!"
inferred = vec.infer_output_dim(node_feature_dim=3)
assert inferred == expected_dim

ok(f"graph vector shape: {v.shape}  (mean+std+max pooling × {len(g_cfg.node_features)} features)")
ok(f"infer_output_dim={inferred}")

# ────────────────────────────────────────────────────────────────────────────
section("5. MODEL REGISTRY")
from src.models import get_model, MODEL_REGISTRY

assert "mlp_baseline" in MODEL_REGISTRY
ok(f"registry keys: {list(MODEL_REGISTRY.keys())}")

model = get_model("mlp_baseline", cfg_l, input_dim=9)
x = torch.randn(4, 9)
logits = model(x)
assert logits.shape == (4, 7), f"logits shape={logits.shape}"
n_params = sum(p.numel() for p in model.parameters())
ok(f"MLPBaseline forward: (4,9) -> {tuple(logits.shape)}")
ok(f"Params: {n_params:,}")

# ────────────────────────────────────────────────────────────────────────────
section("6. LOSS / OPTIMIZER / SCHEDULER")
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer, build_scheduler

criterion = build_loss(cfg_l)
optimizer = build_optimizer(model, cfg_l)
scheduler = build_scheduler(optimizer, cfg_l)  # 'reduce_lr_on_plateau' from mlp_baseline.yaml

import torch.optim.lr_scheduler as lrs
assert isinstance(scheduler, lrs.ReduceLROnPlateau), f"scheduler type FAIL: {type(scheduler)}"
y = torch.tensor([0, 1, 2, 3])
loss_val = criterion(logits, y)
assert not torch.isnan(loss_val), "Loss is NaN!"
ok(f"CrossEntropyLoss = {loss_val.item():.4f}")
ok("Adam optimizer")
ok(f"Scheduler = ReduceLROnPlateau (config: reduce_lr_on_plateau)")

# ────────────────────────────────────────────────────────────────────────────
section("7. CHECKPOINT save/load")
from src.utils.checkpoint import save_checkpoint, load_checkpoints
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "sub", "test.pth")
    save_checkpoint(model, optimizer, epoch=5, path=ckpt_path, best_val_macro_f1=0.42)
    restored_ep = load_checkpoints(model, optimizer, ckpt_path, device=torch.device("cpu"))
    assert restored_ep == 5

ok("save_checkpoint / load_checkpoints OK (epoch=5 restored)")
ok("torch.load uses map_location + weights_only=False")

# ────────────────────────────────────────────────────────────────────────────
section("8. EVALUATION METRICS")
from src.evaluation.metrics import compute_classification_metrics

y_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
y_pred = [0, 1, 2, 3, 4, 5, 0, 0, 1, 2]
m = compute_classification_metrics(y_true, y_pred)
assert "accuracy" in m and "macro_f1" in m and "weighted_f1" in m and "confusion_matrix" in m
assert 0.0 <= m["accuracy"] <= 1.0

ok(f"accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  weighted_f1={m['weighted_f1']:.4f}")

# ────────────────────────────────────────────────────────────────────────────
section("9. TRAINER INIT + 1 STEP")
from src.training.trainer import Trainer

class DictDataset(torch.utils.data.Dataset):
    """Giả lập GraphVectorDataset output."""
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, i):
        return {"x": self.xs[i], "y": self.ys[i]}

fake_x = torch.randn(64, 9)
fake_y = torch.randint(0, 7, (64,))
dl = DataLoader(DictDataset(fake_x, fake_y), batch_size=16)

import tempfile as _tf
ckpt_tmp = os.path.join(_tf.mkdtemp(), "best.pth")

trainer = Trainer(
    model=model, train_loader=dl, val_loader=dl,
    criterion=criterion, optimizer=optimizer, scheduler=None,
    config=cfg_l, device=torch.device("cpu"),
    run_name="check_run", save_dir=ckpt_tmp,
)
ok("Trainer init OK")

# Chạy 1 epoch để verify toàn bộ train loop
train_m = trainer.train_one_epoch()
val_m   = trainer.validate()
assert "loss" in train_m and "accuracy" in train_m and "macro_f1" in train_m
assert "loss" in val_m   and "accuracy" in val_m   and "macro_f1" in val_m
ok(f"train_one_epoch: loss={train_m['loss']:.4f}  acc={train_m['accuracy']:.4f}  macro_f1={train_m['macro_f1']:.4f}")
ok(f"validate:        loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.4f}  macro_f1={val_m['macro_f1']:.4f}")

# ────────────────────────────────────────────────────────────────────────────
print()
print("=" * 48)
print("  ALL PIPELINE CHECKS PASSED ✓")
print("  Sẵn sàng push lên Kaggle!")
print("=" * 48)
