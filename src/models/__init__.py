from .motif_graph_fer import MotifGraphModel

def get_model(name, config):
    """
    Model factory: only MotifGraphModel is kept.
    """
    if name == 'motif_graph_fer':
        return MotifGraphModel(config['model'])
    else:
        raise ValueError(f"Model {name} not found. Only 'motif_graph_fer' is supported.")