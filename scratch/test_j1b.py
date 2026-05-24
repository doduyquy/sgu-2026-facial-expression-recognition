"""Smoke test for Scenario J1b: Capped Region-Only StructureAwareResidualHead."""
import sys
import torch
sys.path.insert(0, ".")

from src.models.semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig

# ── J1b config ──────────────────────────────────────────────────────────────
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
    use_pretrained=False,
    enable_program_logit_calibrator=False,
    # J1b
    enable_structure_aware_head=True,
    structure_head_hidden_dim=128,
    structure_head_dropout=0.15,
    structure_head_init_scale=0.02,
    structure_head_max_scale=0.10,
    structure_head_use_region_context=True,
    structure_head_use_macro_context=False,   # region-only
)

model = SemanticROIGraphFER(cfg)
model.eval()

B = 4
images = torch.randn(B, 1, 48, 48)
bboxes = torch.zeros(B, 9, 4)
bboxes[:, :, 2] = 20.0
bboxes[:, :, 3] = 20.0

with torch.no_grad():
    outputs = model(images, bboxes)

# ── Shape assertions ─────────────────────────────────────────────────────────
assert "logits" in outputs, "missing logits"
assert "logits_motif" in outputs, "missing logits_motif"
assert "logits_program_raw" in outputs, "missing logits_program_raw"
assert "structure_head_delta" in outputs, "missing structure_head_delta"
assert "structure_head_scale" in outputs, "missing structure_head_scale"

assert outputs["logits"].shape[-1] == 7, f"bad logits shape {outputs['logits'].shape}"
assert outputs["logits_motif"].shape[-1] == 7
assert outputs["logits_program_raw"].shape[-1] == 7
assert outputs["structure_head_delta"].shape[-1] == 7

# ── Scale assertions ─────────────────────────────────────────────────────────
# fusion_scale=0.0 => logits == logits_motif exactly
max_diff = (outputs["logits"] - outputs["logits_motif"]).abs().max().item()
print(f"[J1b] max_diff logits vs logits_motif : {max_diff:.2e}")
assert max_diff < 1e-6, f"FAIL: logits != logits_motif  (diff={max_diff})"

# scale must be close to init_scale (0.02) at init and hard-capped <= max_scale (0.10)
scale_val = outputs["structure_head_scale"]
scale_f = scale_val.detach().float().item() if torch.is_tensor(scale_val) else float(scale_val)
print(f"[J1b] structure_head_scale            : {scale_f:.4f}  (expected ~0.02, hard cap 0.10)")
assert scale_f <= 0.1001, f"FAIL: scale={scale_f} exceeds max_scale=0.10"
assert 0.01 < scale_f < 0.04, f"FAIL: init scale={scale_f} far from expected ~0.02"

# correction should be near-zero at init (last linear zeroed)
raw_corrected_diff = (outputs["logits_motif"] - outputs["logits_program_raw"]).abs().mean().item()
print(f"[J1b] mean abs corrected - raw        : {raw_corrected_diff:.2e}  (expected ~0)")
assert raw_corrected_diff < 1e-5, f"FAIL: init correction too large ({raw_corrected_diff})"

# ── Verify macro context is OFF ──────────────────────────────────────────────
head = model.structure_aware_head
assert head is not None, "head should be initialised"
assert not head.use_macro_context, "macro context should be disabled (J1b)"
assert head.macro_norm is None, "macro_norm should be None when macro context disabled"
print(f"[J1b] use_macro_context               : {head.use_macro_context}  (expected False)")
print(f"[J1b] max_scale attr                  : {head.max_scale:.4f}     (expected 0.10)")

# ── Hard-cap stress test ─────────────────────────────────────────────────────
# Force raw_residual_scale to a very large value; scale should still be <= max_scale
with torch.no_grad():
    head.raw_residual_scale.fill_(100.0)   # sigmoid(100) ≈ 1.0 → scale ≈ max_scale
    dummy_region = torch.randn(1, 9, 256)
    dummy_logits = torch.randn(1, 7)
    _, _, stress_scale = head(dummy_logits, region_tokens=dummy_region)
stress_f = stress_scale.item()
print(f"[J1b] stress-test scale (raw=100)     : {stress_f:.4f}  (expected ~0.10)")
assert stress_f <= 0.1001, f"FAIL: cap broken — scale={stress_f}"

print()
print("=== ALL ASSERTIONS PASSED ===")
print(f"  logits shape     : {outputs['logits'].shape}")
print(f"  delta shape      : {outputs['structure_head_delta'].shape}")
print(f"  init scale       : {scale_f:.4f}")
print(f"  max_scale        : {head.max_scale:.4f}")
print(f"  J1b enabled      : {outputs['structure_head_enabled']}")
print(f"  macro disabled   : {not head.use_macro_context}")
