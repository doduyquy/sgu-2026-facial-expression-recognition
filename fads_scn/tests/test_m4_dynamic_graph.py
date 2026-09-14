import sys
from pathlib import Path

import torch


repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.models.mask_guided_dynamic_graph import (
    MaskGuidedDynamicRegionGraph,
)


def test_m4_branch_shapes_and_sparse_adjacency():
    batch_size, channels, height, width = 2, 64, 12, 12
    branch = MaskGuidedDynamicRegionGraph(
        in_channels=channels,
        embed_dim=32,
        num_nodes=8,
        num_classes=7,
        graph_depth=3,
        top_k=3,
        edge_heads=4,
        dropout=0.0,
    )
    outputs = branch(torch.randn(batch_size, channels, height, width))

    assert outputs["residual_logits"].shape == (batch_size, 7)
    assert outputs["graph_feature"].shape == (batch_size, 32)
    assert outputs["attn_maps"].shape == (batch_size, 8, height, width)
    assert outputs["mask_map"].shape == (batch_size, 1, height, width)
    assert outputs["adj_matrix"].shape == (batch_size, 8, 8)
    assert outputs["adj_matrices"].shape == (batch_size, 3, 8, 8)
    assert outputs["class_attention"].shape == (batch_size, 7, 8)
    assert outputs["region_geometry"].shape == (batch_size, 8, 5)

    spatial_sums = outputs["attn_maps"].sum(dim=(-1, -2))
    assert torch.allclose(spatial_sums, torch.ones_like(spatial_sums), atol=1e-5)

    adjacency = outputs["adj_matrix"]
    assert torch.allclose(
        adjacency.sum(dim=-1), torch.ones_like(adjacency.sum(dim=-1)), atol=1e-5
    )
    assert (adjacency > 0).sum(dim=-1).max().item() == 3
    assert torch.allclose(
        outputs["class_attention"].sum(dim=-1),
        torch.ones(batch_size, 7),
        atol=1e-5,
    )


def test_m4_full_model_forward_backward():
    batch_size = 2
    model = AttentiveSCNFER(
        backbone_name="resnet18",
        num_classes=7,
        in_channels=1,
        embed_dim=64,
        use_latent_graph=False,
        use_spatial_attention=False,
        use_m4_graph=True,
        m4_num_nodes=8,
        m4_graph_depth=2,
        m4_top_k=3,
        m4_edge_heads=4,
        m4_gate_init=-2.0,
        dropout=0.0,
        use_pretrained=False,
    )
    model.train()
    outputs = model(torch.randn(batch_size, 1, 48, 48), use_tta=False)

    assert outputs["logits"].shape == (batch_size, 7)
    assert outputs["base_logits"].shape == (batch_size, 7)
    assert outputs["residual_logits"].shape == (batch_size, 7)
    assert outputs["graph_gate"].shape == (batch_size, 7)
    assert outputs["alpha"].shape == (batch_size, 1)
    assert outputs["adj_matrix"].shape == (batch_size, 8, 8)
    assert model.spatial_attention is None
    assert model.latent_graph is None

    expected_gate = torch.full_like(outputs["graph_gate"], torch.sigmoid(torch.tensor(-2.0)))
    assert torch.allclose(outputs["graph_gate"], expected_gate, atol=1e-6)

    loss = (
        outputs["logits"].square().mean()
        + outputs["alpha"].mean()
        + 0.01 * outputs["diversity_loss"]
        + 0.01 * outputs["sparsity_loss"]
    )
    assert torch.isfinite(loss)
    loss.backward()

    assert model.backbone.conv1.weight.grad is not None
    assert model.m4_graph.mask_refiner.input_proj[0].weight.grad is not None
    assert model.m4_graph.tokenizer.node_queries.grad is not None
    assert model.m4_graph.graph_blocks[0].edge_encoder[0].weight.grad is not None
    assert model.m4_graph.readout.class_queries.grad is not None
    assert model.m4_gate.net[-1].weight.grad is not None


def test_m4_flip_tta_keeps_diagnostics():
    model = AttentiveSCNFER(
        backbone_name="resnet18",
        num_classes=7,
        in_channels=1,
        embed_dim=32,
        use_latent_graph=False,
        use_spatial_attention=False,
        use_m4_graph=True,
        m4_num_nodes=8,
        m4_graph_depth=1,
        m4_top_k=3,
        m4_edge_heads=4,
        dropout=0.0,
        use_pretrained=False,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 1, 48, 48), use_tta=True)

    assert outputs["logits"].shape == (1, 7)
    assert outputs["base_logits"].shape == (1, 7)
    assert outputs["residual_logits"].shape == (1, 7)
    assert outputs["graph_gate"].shape == (1, 7)
    assert outputs["mask_map"].shape[-2:] == (12, 12)


if __name__ == "__main__":
    test_m4_branch_shapes_and_sparse_adjacency()
    test_m4_full_model_forward_backward()
    test_m4_flip_tta_keeps_diagnostics()
    print("ALL M4 DYNAMIC GRAPH TESTS PASSED")
