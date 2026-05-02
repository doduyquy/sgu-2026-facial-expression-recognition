from .motif_graph_fer import MotifGraphModel
from .fer_advanced_model import FERAdvancedModel

def get_model(name, config):
    """
    Model factory.
    Supported: 
      - 'motif_graph_fer'  — Hybrid CNN + GATConv + Motif model.
      - 'fer_advanced'     — Advanced model with VGG backbone + Region Attention + Graph + Motif
    """
    if name == 'motif_graph_fer':
        return MotifGraphModel(config['model'])
    elif name == 'fer_advanced':
        model_cfg = config.get('model', {})
        return FERAdvancedModel(
            feat_dim=model_cfg.get('feat_dim', 128),
            num_emotions=7,
            num_regions=model_cfg.get('num_regions', 3),
            num_graph_layers=model_cfg.get('num_graph_layers', 2),
            num_heads=model_cfg.get('num_heads', 4),
            dropout=model_cfg.get('dropout', 0.3),
            use_vgg=model_cfg.get('use_vgg', True),
            motifs_per_class=model_cfg.get('motifs_per_class', 2),
            attention_temperature=model_cfg.get('attention_temperature', 0.5),
            attention_power=model_cfg.get('attention_power', 2.0),
            region_dropout=model_cfg.get('region_dropout', 0.2),
            graph_use_adj=model_cfg.get('graph_use_adj', True),
        )
    else:
        raise ValueError(
            f"Model '{name}' not found. "
            "Supported models: ['motif_graph_fer', 'fer_advanced']"
        )