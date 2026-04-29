from .motif_graph_fer import MotifGraphModel

def get_model(name, config):
    """
    Model factory.
    Supported: 'motif_graph_fer'  — Hybrid CNN + GATConv + Motif model.
    """
    if name == 'motif_graph_fer':
        return MotifGraphModel(config['model'])
    else:
        raise ValueError(
            f"Model '{name}' not found. "
            "Supported models: ['motif_graph_fer']"
        )