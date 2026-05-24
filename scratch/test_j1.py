"""Smoke test for Scenario J1: StructureAwareResidualHead."""
import sys
import torch
sys.path.insert(0, ".")

from src.models.semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig

# Build J1 config
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
    # J1
    enable_program_logit_calibrator=False,
    enable_structure_aware_head=True,
    structure_head_hidden_dim=128,
    structure_head_dropout=0.10,
    structure_head_init_scale=0.05,
    structure_head_use_region_context=True,
    structure_head_use_macro_context=True,
)

model = SemanticROIGraphFER(cfg)
model.eval()

B = 2
images = torch.randn(B, 1, 48, 48)
bboxes = torch.zeros(B, 9, 4)
bboxes[:, :, 2] = 20.0
bboxes[:, :, 3] = 20.0

with torch.no_grad():
    outputs = model(images, bboxes)

# --- Key assertions ---
assert "logits" in outputs, "missing logits"
assert "logits_motif" in outputs, "missing logits_motif"
assert "logits_program_raw" in outputs, "missing logits_program_raw"
assert "structure_head_delta" in outputs, "missing structure_head_delta"
assert "structure_head_scale" in outputs, "missing structure_head_scale"

assert outputs["logits"].shape[-1] == 7, "bad logits shape"
assert outputs["logits_motif"].shape[-1] == 7
assert outputs["logits_program_raw"].shape[-1] == 7
assert outputs["structure_head_delta"].shape[-1] == 7

# fusion_scale=0.0 => logits == logits_motif
max_diff = (outputs["logits"] - outputs["logits_motif"]).abs().max().item()
print(f"[J1] max_diff logits vs logits_motif: {max_diff:.2e}")
assert max_diff < 1e-6, f"FAIL: max_diff={max_diff}"

# scale should be ~0.05 at init
scale_val = outputs["structure_head_scale"]
scale_f = scale_val.item() if torch.is_tensor(scale_val) else float(scale_val)
print(f"[J1] structure_head_scale: {scale_f:.4f}")
assert 0.03 < scale_f < 0.08, f"FAIL: scale={scale_f} (expected ~0.05)"

# correction should be near-zero at init (last linear zeroed)
raw_corrected_diff = (outputs["logits_motif"] - outputs["logits_program_raw"]).abs().mean().item()
print(f"[J1] mean abs corrected - raw: {raw_corrected_diff:.2e}")
assert raw_corrected_diff < 1e-5, f"FAIL: raw_corrected_diff={raw_corrected_diff}"

print()
print("=== ALL ASSERTIONS PASSED ===")
print(f"  logits shape:       {outputs['logits'].shape}")
print(f"  delta shape:        {outputs['structure_head_delta'].shape}")
print(f"  scale:              {scale_f:.4f}")
print(f"  max_diff (logits):  {max_diff:.2e}")
print(f"  correction:         {raw_corrected_diff:.2e}")
print(f"  J1 enabled:         {outputs['structure_head_enabled']}")
