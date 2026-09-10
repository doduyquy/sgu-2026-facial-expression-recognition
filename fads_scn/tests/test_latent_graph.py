import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.models.latent_graph import LatentGraphReasoner
from fads_scn.models.attentive_scn_model import AttentiveSCNFER
from fads_scn.losses.scn_loss import SCNLoss


def test_latent_graph_reasoner_standalone():
    print("Testing LatentGraphReasoner Standalone...")
    B, M, D, H, W = 4, 8, 256, 12, 12
    reasoner = LatentGraphReasoner(embed_dim=D, num_nodes=M, dropout=0.1)

    node_tokens = torch.randn(B, M, D)
    attn_maps = torch.softmax(torch.randn(B, M, H, W).view(B, M, -1), dim=-1).view(B, M, H, W)

    f_graph, adj_matrix, sparsity_loss = reasoner(node_tokens, attn_maps)

    assert f_graph.shape == (B, D), f"Expected f_graph shape {(B, D)}, got {f_graph.shape}"
    assert adj_matrix.shape == (B, M, M), f"Expected adj_matrix shape {(B, M, M)}, got {adj_matrix.shape}"
    assert sparsity_loss.ndim == 0, f"Expected scalar sparsity_loss, got {sparsity_loss.shape}"

    # Verify adjacency row sum = 1.0 (valid probability distribution)
    row_sums = adj_matrix.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Adjacency rows do not sum to 1.0!"
    print("  [PASS] LatentGraphReasoner standalone forward & stochastic matrix verified.")


def test_sparsity_loss_penalizes_diffuse_edges():
    M = 4
    uniform_adj = torch.full((2, M, M), 1.0 / M)
    peaked_adj = torch.zeros(2, M, M)
    peaked_adj[:, :, 0] = 1.0

    uniform_loss = LatentGraphReasoner.edge_entropy_loss(uniform_adj)
    peaked_loss = LatentGraphReasoner.edge_entropy_loss(peaked_adj)

    assert uniform_loss > peaked_loss, "Uniform adjacency should receive a higher sparsity penalty"
    assert peaked_loss.item() < 1e-6, "One-hot adjacency should have near-zero entropy penalty"
    print("  [PASS] Edge entropy sparsity loss direction verified.")


def test_reliability_sparse_graph_is_topk_and_reliability_gated():
    """The proposed graph retains self plus exactly at most k dynamic neighbours."""
    B, M, D, H, W = 3, 5, 16, 4, 4
    topk = 2
    reasoner = LatentGraphReasoner(
        embed_dim=D,
        num_nodes=M,
        hidden_dim=32,
        dropout=0.0,
        graph_mode="reliability_sparse",
        topk=topk,
        self_loop_bias=1.0,
    )
    node_tokens = torch.randn(B, M, D)
    attn_maps = torch.softmax(torch.randn(B, M, H * W), dim=-1).view(B, M, H, W)
    global_features = torch.randn(B, D)
    reliability = torch.tensor([[0.10], [0.50], [0.90]])

    f_graph, adj_matrix, sparsity_loss = reasoner(
        node_tokens,
        attn_maps,
        global_features=global_features,
        reliability=reliability,
    )

    assert f_graph.shape == (B, D)
    assert torch.isfinite(sparsity_loss)
    assert torch.allclose(adj_matrix.sum(dim=-1), torch.ones(B, M), atol=1e-6)
    assert torch.all(torch.diagonal(adj_matrix, dim1=-2, dim2=-1) > 0)
    active_edges = (adj_matrix > 1e-12).sum(dim=-1)
    assert torch.all(active_edges <= topk + 1), "Each node may use only self plus top-k neighbours"
    assert reasoner.last_graph_gain.shape == (B, 1)
    assert torch.all((reasoner.last_graph_gain > 0) & (reasoner.last_graph_gain < 1))

    f_graph.mean().backward()
    assert reasoner.reliability_gate[0].weight.grad is not None
    assert reasoner.q_proj.weight.grad is not None
    print("  [PASS] Reliability-gated sparse top-k graph verified.")


def test_sparse_graph_is_a_valid_ablation_without_reliability_inputs():
    reasoner = LatentGraphReasoner(
        embed_dim=8,
        num_nodes=4,
        hidden_dim=16,
        dropout=0.0,
        graph_mode="sparse",
        topk=1,
    )
    tokens = torch.randn(2, 4, 8)
    maps = torch.softmax(torch.randn(2, 4, 9), dim=-1).view(2, 4, 3, 3)
    _, adjacency, _ = reasoner(tokens, maps)

    assert torch.allclose(adjacency.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    assert torch.all((adjacency > 1e-12).sum(dim=-1) <= 2)
    assert torch.allclose(reasoner.last_graph_gain, torch.ones(2, 1))


def test_attentive_scn_with_graph_gradient_flow():
    print("Testing AttentiveSCNFER with Latent Dynamic Graph & Gradient Flow...")
    B = 4
    x = torch.randn(B, 1, 48, 48)
    targets = torch.randint(0, 7, (B,))

    model = AttentiveSCNFER(
        backbone_name="resnet18",
        num_classes=7,
        in_channels=1,
        embed_dim=256,
        num_attn_heads=8,
        use_latent_graph=True,
        use_pretrained=False,
    )
    criterion = SCNLoss(num_classes=7, margin=0.15, clean_ratio=0.70, rank_loss_weight=0.1)

    model.train()
    outputs = model(x, use_tta=False)

    assert outputs["logits"].shape == (B, 7), f"Logits shape mismatch: {outputs['logits'].shape}"
    assert outputs["alpha"].shape == (B, 1), f"Alpha shape mismatch: {outputs['alpha'].shape}"
    assert outputs["adj_matrix"].shape == (B, 8, 8), f"Adj matrix shape mismatch: {outputs['adj_matrix'].shape}"
    assert outputs["sparsity_loss"] is not None

    loss_dict = criterion(outputs, targets)
    loss = loss_dict["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()

    # Check gradients in LatentGraphReasoner
    assert model.latent_graph.q_proj.weight.grad is not None, "q_proj grad is None!"
    assert model.latent_graph.geo_scale.grad is not None, "geo_scale grad is None!"
    assert model.latent_graph.readout_gate[0].weight.grad is not None, "readout_gate grad is None!"
    
    # Check gradients in Backbone
    conv1_grad = list(model.backbone.parameters())[0].grad
    assert conv1_grad is not None, "Backbone conv1 grad is None!"
    print("  [PASS] Full gradient flow through backbone, spatial attention, latent graph, and SCN head verified.")


def test_tta_and_eval_mode():
    print("Testing Eval Mode and Test-Time Augmentation (TTA)...")
    B = 2
    x = torch.randn(B, 1, 48, 48)

    model = AttentiveSCNFER(
        backbone_name="resnet18",
        num_classes=7,
        in_channels=1,
        embed_dim=256,
        num_attn_heads=8,
        use_latent_graph=True,
        use_pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        out_tta = model(x, use_tta=True)
        assert out_tta["logits"].shape == (B, 7)
        assert out_tta["alpha"].shape == (B, 1)
        assert out_tta["adj_matrix"].shape == (B, 8, 8)
    print("  [PASS] TTA inference verified.")


def test_toggle_latent_graph_off():
    print("Testing Toggle use_latent_graph=False...")
    B = 2
    x = torch.randn(B, 1, 48, 48)

    model = AttentiveSCNFER(
        backbone_name="resnet18",
        num_classes=7,
        in_channels=1,
        embed_dim=256,
        num_attn_heads=8,
        use_latent_graph=False,
        use_pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        out = model(x, use_tta=False)
        assert out["logits"].shape == (B, 7)
        assert out["adj_matrix"] is None
    print("  [PASS] Clean degradation when use_latent_graph=False verified.")


if __name__ == "__main__":
    test_latent_graph_reasoner_standalone()
    test_sparsity_loss_penalizes_diffuse_edges()
    test_attentive_scn_with_graph_gradient_flow()
    test_tta_and_eval_mode()
    test_toggle_latent_graph_off()
    print("\nALL LATENT GRAPH REASONER TESTS PASSED SUCCESSFULLY!")
