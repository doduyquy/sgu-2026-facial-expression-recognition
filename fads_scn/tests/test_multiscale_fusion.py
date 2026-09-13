import pytest
import torch

from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.models.multiscale_fusion import (
    AdaptiveBranchFusion,
    MultiScaleDualPooling,
)


def test_multiscale_pool_and_adaptive_fusion_backward():
    batch_size, embed_dim = 2, 16
    feature_channels = (8, 16, 24, 32)
    spatial_sizes = (16, 8, 4, 4)
    feature_maps = [
        torch.randn(batch_size, channels, size, size, requires_grad=True)
        for channels, size in zip(feature_channels, spatial_sizes)
    ]

    multiscale = MultiScaleDualPooling(
        feature_channels=feature_channels,
        embed_dim=embed_dim,
        dropout=0.0,
        se_reduction=4,
    )
    adaptive_fusion = AdaptiveBranchFusion(embed_dim=embed_dim, dropout=0.0)

    f_multi, scale_weights = multiscale(feature_maps)
    f_global = torch.randn(batch_size, embed_dim, requires_grad=True)
    f_local_graph = torch.randn(batch_size, embed_dim, requires_grad=True)
    fused, fusion_weights = adaptive_fusion((f_global, f_local_graph, f_multi))

    assert f_multi.shape == (batch_size, embed_dim)
    assert fused.shape == (batch_size, embed_dim)
    assert scale_weights.shape == (batch_size, 4)
    assert fusion_weights.shape == (batch_size, 3)
    assert torch.allclose(scale_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    assert torch.allclose(fusion_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    fused.square().mean().backward()
    assert all(feature_map.grad is not None for feature_map in feature_maps)
    assert adaptive_fusion.router[-1].weight.grad is not None


def test_convnext_m1_forward_exposes_fusion_diagnostics():
    batch_size = 1
    model = AttentiveSCNFER(
        backbone_name="convnext_tiny",
        num_classes=7,
        in_channels=1,
        embed_dim=32,
        num_attn_heads=4,
        use_latent_graph=True,
        use_spatial_attention=True,
        use_multiscale_fusion=True,
        dropout=0.0,
        use_pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(batch_size, 1, 48, 48), use_tta=False)

    assert outputs["logits"].shape == (batch_size, 7)
    assert outputs["features"].shape == (batch_size, 32)
    assert outputs["adj_matrix"].shape == (batch_size, 4, 4)
    assert outputs["scale_weights"].shape == (batch_size, 4)
    assert outputs["fusion_weights"].shape == (batch_size, 3)
    assert torch.allclose(outputs["scale_weights"].sum(dim=1), torch.ones(batch_size), atol=1e-6)
    assert torch.allclose(outputs["fusion_weights"].sum(dim=1), torch.ones(batch_size), atol=1e-6)


def test_m1_rejects_backbones_without_stage_extraction():
    with pytest.raises(ValueError, match="requires a ConvNeXt backbone"):
        AttentiveSCNFER(
            backbone_name="resnet18",
            embed_dim=32,
            num_attn_heads=4,
            use_multiscale_fusion=True,
            use_pretrained=False,
        )
