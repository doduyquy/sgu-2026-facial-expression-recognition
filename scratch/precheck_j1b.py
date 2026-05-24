"""
Pre-train sanity check for Scenario J1b.

Verifies:
  1. structure_head_scale ≈ 0.02 at init (eval, no grad)
  2. max_diff logits vs logits_motif < 1e-6  (fusion_scale=0.0)
  3. structure_head_scale <= 0.10 after one train step (cap holds under gradient)
  4. Gradients flow to structure_aware_head parameters
  5. Backward does not explode (loss finite, grad norm finite)

Run from repo root:
  .venv\\Scripts\\python.exe scratch/precheck_j1b.py
"""

import sys, math
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.models.semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Build model from same config as YAML ─────────────────────────────────────
cfg = SemanticRoiGraphConfig(
    num_classes=7,
    num_regions=9,
    feature_dim=256,
    semantic_state_dim=128,
    semantic_latent_dim=256,
    semantic_attn_heads=4,
    hyperedge_count=4,
    router_hidden_dim=256,
    cross_region_compositions=8,
    macro_motifs_per_class=4,
    micro_motifs_per_region=8,
    fusion_scale=0.0,
    use_pretrained=False,          # fast check; swap to True for real run
    enable_program_logit_calibrator=False,
    # J1b
    enable_structure_aware_head=True,
    structure_head_hidden_dim=128,
    structure_head_dropout=0.15,
    structure_head_init_scale=0.02,
    structure_head_max_scale=0.10,
    structure_head_use_region_context=True,
    structure_head_use_macro_context=False,
)

model = SemanticROIGraphFER(cfg).to(DEVICE)

# ── Dummy batch (B=8, mimics real loader) ────────────────────────────────────
B = 8
images = torch.randn(B, 1, 48, 48, device=DEVICE)
bboxes = torch.zeros(B, 9, 4, device=DEVICE)
bboxes[:, :, 2] = 24.0
bboxes[:, :, 3] = 24.0
labels = torch.randint(0, 7, (B,), device=DEVICE)

# ════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Init state (eval, no grad)
# ════════════════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    out = model(images, bboxes)

scale_init = out["structure_head_scale"]
scale_init_f = scale_init.detach().float().item() if torch.is_tensor(scale_init) else float(scale_init)
max_diff     = (out["logits"] - out["logits_motif"]).abs().max().item()
init_corr    = (out["logits_motif"] - out["logits_program_raw"]).abs().mean().item()

print("\n── CHECK 1: Init state (eval) ──────────────────────────────────────────")
print(f"  structure_head_scale : {scale_init_f:.4f}   (expected ~0.02)")
print(f"  max_diff logits vs logits_motif : {max_diff:.2e}   (expected < 1e-6)")
print(f"  mean |corrected - raw| : {init_corr:.2e}   (expected ~0)")

ok1a = 0.01 < scale_init_f < 0.04
ok1b = max_diff < 1e-6
ok1c = init_corr < 1e-5

print(f"  [{'PASS' if ok1a else 'FAIL'}] scale ≈ 0.02")
print(f"  [{'PASS' if ok1b else 'FAIL'}] logits == logits_motif (fusion_scale=0)")
print(f"  [{'PASS' if ok1c else 'FAIL'}] correction ≈ 0 at init")

# ════════════════════════════════════════════════════════════════════════════
# CHECK 2 — One train step: cap holds + gradients flow
# ════════════════════════════════════════════════════════════════════════════
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
optimizer.zero_grad()

out_train = model(images, bboxes)
loss = F.cross_entropy(out_train["logits"], labels)
loss.backward()

# Check grad norm (should be finite and not explode)
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = math.sqrt(total_norm)

# Check gradient on structure_aware_head
head_grad_norm = 0.0
head = model.structure_aware_head
if head is not None:
    for p in head.parameters():
        if p.grad is not None:
            head_grad_norm += p.grad.data.norm(2).item() ** 2
    head_grad_norm = math.sqrt(head_grad_norm)

scale_raw_grad = None
if head is not None and head.raw_residual_scale.grad is not None:
    scale_raw_grad = head.raw_residual_scale.grad.item()

# Take optimizer step
optimizer.step()

# After step: scale must still be <= max_scale
model.eval()
with torch.no_grad():
    out_post = model(images, bboxes)
scale_post = out_post["structure_head_scale"]
scale_post_f = scale_post.detach().float().item() if torch.is_tensor(scale_post) else float(scale_post)

print("\n── CHECK 2: After one train step ───────────────────────────────────────")
print(f"  CE loss              : {loss.item():.4f}   (expected finite)")
print(f"  Global grad norm     : {total_norm:.4f}   (expected finite, not huge)")
print(f"  Head grad norm       : {head_grad_norm:.4f}   (expected > 0 → grads flow)")
print(f"  raw_scale grad       : {scale_raw_grad}   (expected not None)")
print(f"  scale after step     : {scale_post_f:.4f}   (expected <= 0.10)")

ok2a = math.isfinite(loss.item())
ok2b = math.isfinite(total_norm) and total_norm < 1e4
ok2c = head_grad_norm > 0
ok2d = scale_raw_grad is not None
ok2e = scale_post_f <= 0.1001

print(f"  [{'PASS' if ok2a else 'FAIL'}] loss is finite")
print(f"  [{'PASS' if ok2b else 'FAIL'}] grad norm finite & < 1e4  ({total_norm:.1f})")
print(f"  [{'PASS' if ok2c else 'FAIL'}] head gradients flow  ({head_grad_norm:.4f})")
print(f"  [{'PASS' if ok2d else 'FAIL'}] raw_scale has gradient")
print(f"  [{'PASS' if ok2e else 'FAIL'}] scale <= 0.10 after step  ({scale_post_f:.4f})")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
all_checks = [ok1a, ok1b, ok1c, ok2a, ok2b, ok2c, ok2d, ok2e]
passed = sum(all_checks)
total  = len(all_checks)

print(f"\n{'='*60}")
print(f"RESULT: {passed}/{total} checks passed")
if passed == total:
    print("✓  All pre-train checks PASSED — safe to launch full training.")
else:
    failed = [i+1 for i, ok in enumerate(all_checks) if not ok]
    print(f"✗  Failed checks: {failed}  — investigate before full training.")
print('='*60)
