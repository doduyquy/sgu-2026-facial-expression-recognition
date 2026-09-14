import pytest
import torch

from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.models.region_cross_attention import TopKRegionCrossAttention


def test_topk_region_cross_attention_forward_backward_and_spacing():
    batch_size = 2
    module = TopKRegionCrossAttention(
        detail_channels=16,
        context_channels=32,
        embed_dim=16,
        num_regions=4,
        num_heads=4,
        suppression_radius=1,
        dropout=0.0,
    )
    detail = torch.randn(batch_size, 16, 6, 6, requires_grad=True)
    context = torch.randn(batch_size, 32, 4, 4, requires_grad=True)

    output = module(detail, context)

    assert output["features"].shape == (batch_size, 16)
    assert output["indices"].shape == (batch_size, 4)
    assert output["locations"].shape == (batch_size, 4, 2)
    assert output["weights"].shape == (batch_size, 4)
    assert output["attention_maps"].shape == (batch_size, 4, 4, 4)
    assert output["saliency_map"].shape == (batch_size, 6, 6)
    assert torch.allclose(output["weights"].sum(dim=1), torch.ones(batch_size), atol=1e-6)
    assert torch.allclose(
        output["attention_maps"].flatten(2).sum(dim=2),
        torch.ones(batch_size, 4),
        atol=1e-6,
    )

    rows = torch.div(output["indices"], 6, rounding_mode="floor")
    columns = output["indices"].remainder(6)
    for first in range(4):
        for second in range(first + 1, 4):
            chebyshev_distance = torch.maximum(
                (rows[:, first] - rows[:, second]).abs(),
                (columns[:, first] - columns[:, second]).abs(),
            )
            assert torch.all(chebyshev_distance > 1)

    output["features"].square().mean().backward()
    assert detail.grad is not None
    assert context.grad is not None
    assert module.saliency[-1].weight.grad is not None
    assert module.cross_attention.in_proj_weight.grad is not None


def test_convnext_region_model_forward_and_tta():
    batch_size = 1
    model = AttentiveSCNFER(
        backbone_name="convnext_tiny",
        num_classes=7,
        in_channels=1,
        embed_dim=32,
        num_attn_heads=4,
        use_latent_graph=True,
        use_spatial_attention=True,
        use_region_cross_attention=True,
        num_regions=6,
        region_attn_heads=4,
        region_suppression_radius=1,
        region_gate_init=0.05,
        dropout=0.0,
        use_pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        inputs = torch.randn(batch_size, 1, 48, 48)
        output = model(inputs, use_tta=False)
        tta_output = model(inputs, use_tta=True)

    assert output["logits"].shape == (batch_size, 7)
    assert output["features"].shape == (batch_size, 32)
    assert output["region_indices"].shape == (batch_size, 6)
    assert output["region_locations"].shape == (batch_size, 6, 2)
    assert output["region_weights"].shape == (batch_size, 6)
    assert output["region_attention_maps"].shape == (batch_size, 6, 12, 12)
    assert output["region_saliency_map"].shape == (batch_size, 12, 12)
    assert torch.allclose(output["region_gate"], torch.full((32,), 0.05), atol=1e-6)
    assert tta_output["logits"].shape == (batch_size, 7)
    assert tta_output["region_indices"].shape == (batch_size, 6)


def test_region_attention_rejects_unsupported_backbone():
    with pytest.raises(ValueError, match="requires ConvNeXt"):
        AttentiveSCNFER(
            backbone_name="resnet18",
            embed_dim=32,
            num_attn_heads=4,
            use_region_cross_attention=True,
            use_pretrained=False,
        )
