from src.models.mlp_baseline import MLPBaseline
from src.models.subgraph_mlp_baseline import SubgraphMLPBaseline
from src.models.subgraph_gnn_baseline import SubgraphGNNBaseline
from src.models.motif_guided_mlp import MotifGuidedMLP
from src.models.motif_guided_gnn import MotifGuidedGNN

"""
Hi, tạo model mới thì:
    1. Tạo file src/models/model_name.py
    2. Thêm vào MODEL_REGISTRY ở dưới
    3. Tạo file configs/model_name.yaml để set config cho nó.

Roadmap:
    - mlp_baseline:            graph vector + MLP (hiện tại)
    - subgraph_mlp_baseline:   subgraph descriptor + MLP (precomputed)
    - subgraph_gnn_baseline:   subgraph descriptor + GraphSAGE GNN (precomputed)
    - gcn:                     GCN trên full pixel graph (future)
    - graphsage:               GraphSAGE trên full pixel graph (future)
"""

MODEL_REGISTRY = {
    "mlp_baseline": lambda config, input_dim, **kw: MLPBaseline(
        input_dim=input_dim,
        num_classes=config["data"].get("num_classes", 7),
        hidden_dims=tuple(config["model"].get("hidden_dims", [64, 32])),
        dropout=config["model"].get("dropout", 0.2),
    ),
    "subgraph_mlp_baseline": lambda config, input_dim, **kw: SubgraphMLPBaseline(
        input_dim=input_dim,
        num_classes=config["data"].get("num_classes", 7),
        hidden_dims=tuple(config["model"].get("hidden_dims", [64, 32])),
        dropout=config["model"].get("dropout", 0.2),
    ),
    "subgraph_gnn_baseline": lambda config, input_dim, **kw: SubgraphGNNBaseline(
        input_dim=input_dim,
        num_classes=config["data"].get("num_classes", 7),
        hidden_dims=tuple(config["model"].get("hidden_dims", [128, 64])),
        dropout=config["model"].get("dropout", 0.2),
        gnn_layers=config["model"].get("gnn_layers", 2),
    ),
    "motif_guided_mlp": lambda config, input_dim, **kw: MotifGuidedMLP(
        input_dim=input_dim,
        hidden_dim=config["model"].get("hidden_dim", 128),
        num_classes=config["model"].get("num_classes", config["data"].get("num_classes", 7)),
        dropout=config["model"].get("dropout", 0.3),
        use_motif_score_vector=config["model"].get("use_motif_score_vector", True),
        use_match_score_feature=config["model"].get("use_match_score_feature", True),
        use_match_score_weighting=config["model"].get("use_match_score_weighting", True),
    ),
    "motif_guided_gnn": lambda config, input_dim, **kw: MotifGuidedGNN(
        input_dim=input_dim,
        hidden_dim=config["model"].get("hidden_dim", 128),
        gnn_hidden_dim=config["model"].get("gnn_hidden_dim", config["model"].get("hidden_dim", 128)),
        num_layers=config["model"].get("num_layers", 2),
        num_classes=config["model"].get("num_classes", config["data"].get("num_classes", 7)),
        dropout=config["model"].get("dropout", 0.3),
        use_edge_attr=config["model"].get("use_edge_attr", False),
        edge_attr_dim=config["model"].get("edge_attr_dim", config.get("data", {}).get("edge_attr_dim", 3)),
        use_motif_score_vector=config["model"].get("use_motif_score_vector", True),
        use_match_score_feature=config["model"].get("use_match_score_feature", True),
        use_match_score_weighting=config["model"].get("use_match_score_weighting", True),
        pooling=config["model"].get("pooling", "motif_attention"),
    ),
}


def get_model(name: str, config: dict, **kwargs):
    """
    Factory function: tạo model theo tên trong config.

    Args:
        name:      key trong MODEL_REGISTRY (= config['model']['name'])
        config:    full config dict
        **kwargs:  extra args (vd: input_dim cho MLP)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](config=config, **kwargs)
