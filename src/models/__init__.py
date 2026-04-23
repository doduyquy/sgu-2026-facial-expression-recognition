from src.models.mlp_baseline import MLPBaseline

"""
Hi, tạo model mới thì:
    1. Tạo file src/models/model_name.py
    2. Thêm vào MODEL_REGISTRY ở dưới
    3. Tạo file configs/model_name.yaml để set config cho nó.

Roadmap:
    - mlp_baseline:  graph vector + MLP (hiện tại)
    - gcn:           GCN trên full pixel graph
    - graphsage:     GraphSAGE trên full pixel graph
    - subgraph_mlp:  subgraph descriptor + MLP
"""

MODEL_REGISTRY = {
    "mlp_baseline": lambda config, input_dim, **kw: MLPBaseline(
        input_dim=input_dim,
        num_classes=config["data"].get("num_classes", 7),
        hidden_dims=tuple(config["model"].get("hidden_dims", [64, 32])),
        dropout=config["model"].get("dropout", 0.2),
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
