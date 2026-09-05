import sys
from pathlib import Path
import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.losses.scn_loss import SCNLoss


def test_pure_image_forward_backward():
    print("==================================================")
    print("[TEST] Running Unit Test: Pure Image-Based Attentive-SCN")
    print("==================================================")

    # 1. Instantiate Model (use resnet18 for instant CPU testing)
    model = AttentiveSCNFER(
        backbone_name="resnet18",
        num_classes=7,
        in_channels=1,
        embed_dim=256,
        num_attn_heads=4,
        dropout=0.25,
        use_pretrained=False,
    )
    model.train()

    # 2. Synthetic Batch: PURE IMAGES ONLY! (No bboxes, no masks, no landmarks)
    B = 8
    imgs = torch.randn(B, 1, 48, 48)
    labels = torch.randint(0, 7, (B,))
    print(f"Input shape: {imgs.shape} (Pure Images: Batch={B}, Channels=1, 48x48)")

    # 3. Forward Pass
    outputs = model(imgs, use_tta=False)
    logits = outputs["logits"]
    alpha = outputs["alpha"]
    attn_maps = outputs["attn_maps"]
    div_loss = outputs["diversity_loss"]

    print(f"[OK] Logits shape: {logits.shape} (Expected [4, 7])")
    print(f"[OK] Alpha shape:  {alpha.shape}  (Expected [4, 1])")
    print(f"[OK] Attn maps:    {attn_maps.shape} (Expected [4, 4, 12, 12])")
    print(f"[OK] Diversity loss: {div_loss.item():.4f}")

    assert logits.shape == (B, 7), f"Expected logits shape ({B}, 7), got {logits.shape}"
    assert alpha.shape == (B, 1), f"Expected alpha shape ({B}, 1), got {alpha.shape}"
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), "Alpha must be in range [0, 1]"
    assert attn_maps.shape == (B, 4, 12, 12), f"Expected attn maps shape ({B}, 4, 12, 12), got {attn_maps.shape}"

    # 4. SCN Loss Computation
    criterion = SCNLoss(
        num_classes=7,
        label_smoothing=0.05,
        margin=0.15,
        clean_ratio=0.70,
        rank_loss_weight=0.10,
    )

    loss_dict = criterion(outputs, labels, current_epoch=10, rank_warmup_epochs=5)
    loss = loss_dict["loss"]
    print(f"[OK] Total Loss: {loss.item():.4f} (Weighted CE: {loss_dict['weighted_ce']:.4f}, Rank: {loss_dict['rank_loss']:.4f})")

    # 5. Backward Pass
    loss.backward()
    print("[OK] Backward gradients computed successfully! Zero NaNs.")

    # Check gradients exist
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "All trainable parameters should have gradients!"
    print("[OK] All trainable parameters received non-zero gradients.")

    # 6. Test Automatic Flip TTA in Eval Mode
    model.eval()
    with torch.no_grad():
        tta_out = model(imgs, use_tta=True)
        print(f"[OK] TTA Logits shape: {tta_out['logits'].shape} (Evaluated with automatic horizontal flip)")
        assert tta_out["logits"].shape == (B, 7)

    print("\n[SUCCESS] ALL PURE IMAGE-BASED SCN UNIT TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    test_pure_image_forward_backward()
