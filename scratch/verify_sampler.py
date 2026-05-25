"""Verify Tasks 1, 2, 3 for WeightedRandomSampler + YAML updates."""
import sys, io, os, tempfile, inspect
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import yaml
import torch
from torch.utils.data import WeightedRandomSampler
from src.data.dataloader import build_dataloader, _build_weighted_sampler

with open('configs/semantic_roi_graph.yaml') as f:
    cfg = yaml.safe_load(f)

t = cfg['training']
m = cfg['model']

# ── Task 1 config ─────────────────────────────────────────────────────────────
print("=== TASK 1: WeightedRandomSampler config ===")
val = t["use_weighted_sampler"]
print(f"  use_weighted_sampler : {val}   (expected True)")
assert val is True, f"FAIL: {val}"
print("  [PASS]")

# ── Task 2 ────────────────────────────────────────────────────────────────────
print()
print("=== TASK 2: Fusion scale ===")
val2 = m["fusion_scale"]
print(f"  fusion_scale : {val2}   (expected 0.5)")
assert val2 == 0.5, f"FAIL fusion_scale={val2}"
print("  [PASS]")

# ── Task 3 ────────────────────────────────────────────────────────────────────
print()
print("=== TASK 3: Macro diversity weight ===")
val3 = t["macro_motif_diversity_weight"]
print(f"  macro_motif_diversity_weight : {val3}   (expected 0.01)")
assert abs(val3 - 0.01) < 1e-9, f"FAIL: {val3}"
print("  [PASS]")

# ── Task 1: sampler unit test ─────────────────────────────────────────────────
print()
print("=== TASK 1: _build_weighted_sampler unit test ===")

# Simulate 100 samples: 70 class-0 (dominant), 10 class-1 (rare), 20 class-2
fake_labels = [0] * 70 + [1] * 10 + [2] * 20

tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
tmp.write("emotion,pixels\n")
for lbl in fake_labels:
    tmp.write(f"{lbl},0 0 0\n")
tmp.close()

class _FakeDataset:
    def __len__(self): return 100

sampler = _build_weighted_sampler(_FakeDataset(), tmp.name)
os.unlink(tmp.name)

assert isinstance(sampler, WeightedRandomSampler), "Not a WeightedRandomSampler"
assert sampler.num_samples == 100
assert sampler.replacement is True

# Draw one epoch and check class distribution
counts = {0: 0, 1: 0, 2: 0}
for idx in sampler:
    counts[fake_labels[int(idx)]] += 1
total = sum(counts.values())
ratios = {k: v / total for k, v in counts.items()}
print(f"  Sampled class ratios: { {k: f'{v:.1%}' for k, v in ratios.items()} }")

# Rare class (10% original) should be drawn much more than 10%
assert ratios[1] > 0.20, f"FAIL: rare class still undersampled at {ratios[1]:.1%}"
print(f"  [PASS] Rare class (10% original) sampled at {ratios[1]:.1%} (expected ~33%)")

# ── Task 1: DataLoader wiring ─────────────────────────────────────────────────
print()
print("=== TASK 1: DataLoader shuffle=False wiring ===")
src = inspect.getsource(build_dataloader)
assert "shuffle=train_shuffle" in src, "FAIL: shuffle not parameterised"
assert "sampler=train_sampler" in src, "FAIL: sampler not passed"
assert "train_shuffle  = False" in src or "train_shuffle = False" in src, \
    "FAIL: shuffle not disabled when sampler enabled"
print("  [PASS] shuffle=False correctly set when sampler is active")

print()
print("=" * 52)
print("ALL CHECKS PASSED")
print("=" * 52)
print()
print("To verify on Kaggle, add after first batch:")
print("  unique, counts = labels.unique(return_counts=True)")
print("  print(dict(zip(unique.tolist(), counts.tolist())))")
print("  # Expect each class ~8-10 times in batch of 64")
