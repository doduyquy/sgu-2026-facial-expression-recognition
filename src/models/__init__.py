from .motif_graph_fer import MotifGraphModel
from .vgg_fer import Vgg
from .fer_advanced_model import FERAdvancedModel

def get_model(name, config):
    """
    Model factory.
    Supported: 
      - 'motif_graph_fer'  — Hybrid CNN + GATConv + Motif model.
      - 'vgg'              — VGG-based model for FER2013
      - 'fer_advanced'     — Advanced model with VGG backbone + Region Attention + Graph + Motif
    """
    if name == 'motif_graph_fer':
        return MotifGraphModel(config['model'])
    elif name == 'vgg':
        return Vgg(drop=config.get('dropout', 0.2))
    elif name == 'fer_advanced':
        model_cfg = config.get('model', {})
        return FERAdvancedModel(
            feat_dim=model_cfg.get('feat_dim', 128),
            num_emotions=7,
            num_regions=model_cfg.get('num_regions', 3),
            num_graph_layers=model_cfg.get('num_graph_layers', 2),
            num_heads=model_cfg.get('num_heads', 4),
            dropout=model_cfg.get('dropout', 0.3),
            use_vgg=model_cfg.get('use_vgg', True)
        )
    else:
        raise ValueError(
            f"Model '{name}' not found. "
            "Supported models: ['motif_graph_fer', 'vgg', 'fer_advanced']"
        )