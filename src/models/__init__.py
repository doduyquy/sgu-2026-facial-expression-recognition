from .motif_graph_fer import MotifGraphModel

def get_model(name, config):
    """
    Model factory.
    Supported: 'motif_graph_fer', 'emo_gnp'
    """
    if name in ['motif_graph_fer', 'emo_gnp']:
        return MotifGraphModel(config['model'])
    else:
        raise ValueError(
            f"Model '{name}' not found. "
            "Supported models: ['motif_graph_fer', 'emo_gnp']"
        )