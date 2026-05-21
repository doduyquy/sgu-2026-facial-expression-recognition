from .motif_graph_fer import MotifGraphModel
from .semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig

def get_model(name, config):
    """
    Model factory.
    Supported: 'motif_graph_fer'  — Hybrid CNN + GATConv + Motif model.
    """
    if name == 'motif_graph_fer':
        model_config = config['model'].copy()
        if 'training' in config:
            model_config['clip_embedding_path'] = config['training'].get('clip_embedding_path', None)
        return MotifGraphModel(model_config)
    if name == 'semantic_roi_graph_fer':
        model_cfg = config.get('model', {})
        cfg = SemanticRoiGraphConfig(**model_cfg)
        return SemanticROIGraphFER(cfg)
    else:
        raise ValueError(
            f"Model '{name}' not found. "
            "Supported models: ['motif_graph_fer', 'semantic_roi_graph_fer']"
        )