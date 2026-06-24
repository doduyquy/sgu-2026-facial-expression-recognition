
from .semantic_roi_graph import SemanticROIGraphFER, SemanticRoiGraphConfig

def get_model(name, config):
    """
    Model factory.
    Supported: 'motif_graph_fer'  — Hybrid CNN + GATConv + Motif model.
               'semantic_roi_graph_fer' — Semantic ROI Graph FER model.
    """
    if name == 'semantic_roi_graph_fer':
        model_cfg = config.get('model').copy()
        model_cfg.pop('name', None)
        allowed_keys = set(SemanticRoiGraphConfig.__annotations__.keys())
        model_cfg = {key: value for key, value in model_cfg.items() if key in allowed_keys}
        cfg = SemanticRoiGraphConfig(**model_cfg)
        model = SemanticROIGraphFER(cfg)
        model.training_cfg = config.get('training', {})
        return model
    else:
        raise ValueError(
            f"Model '{name}' not found. "
            "Supported models: ['motif_graph_fer', 'semantic_roi_graph_fer']"
        )