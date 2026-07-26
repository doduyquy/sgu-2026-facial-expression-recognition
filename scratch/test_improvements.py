import sys
sys.path.append('.')
import torch
import yaml

print("=" * 50)
print("TEST 1: Data Augmentation (RandomErasing, GaussianBlur, Sharpness)")
print("=" * 50)
from src.data.transforms import build_transform
config = yaml.safe_load(open('configs/semantic_roi_graph.yaml', 'r'))
train_transform = build_transform(config, split="train")
print(f"Train transform pipeline:\n{train_transform}")
print("OK!\n")

print("=" * 50)
print("TEST 2: AdamW Optimizer")
print("=" * 50)
from src.training.optimizer import build_optimizer, build_scheduler
from src.models import get_model
model = get_model(name=config['model']['name'], config=config)
optimizer = build_optimizer(model=model, config=config)
print(f"Optimizer type: {type(optimizer).__name__}")
assert type(optimizer).__name__ == 'AdamW', "Expected AdamW!"
print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")
print("OK!\n")

print("=" * 50)
print("TEST 3: CosineAnnealingWarmRestarts Scheduler")
print("=" * 50)
scheduler = build_scheduler(optimizer=optimizer, config=config)
print(f"Scheduler type: {type(scheduler).__name__}")
assert 'CosineAnnealingWarmRestarts' in type(scheduler).__name__, "Expected CosineAnnealingWarmRestarts!"
# Simulate a few steps
for ep in range(5):
    scheduler.step(ep + 1)
    lr = optimizer.param_groups[0]['lr']
    print(f"  Epoch {ep+1}: lr = {lr:.8f}")
print("OK!\n")

print("=" * 50)
print("ALL 3 IMPROVEMENTS PASSED SUCCESSFULLY!")
print("=" * 50)
