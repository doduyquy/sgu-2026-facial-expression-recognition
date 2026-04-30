from .motif_graph_fer import MotifGraphModel
from .semantic_motif_gnn import SemanticMotifGNN

def get_model(name, config):
    """
    Model factory.
    Supported:
        'motif_graph_fer'     — Hybrid CNN + GATConv + Motif model.
        'semantic_motif_gnn'  — Semantic Graph + Structured Motif model.
    """
    if name == 'motif_graph_fer':
        return MotifGraphModel(config['model'])
    elif name == 'semantic_motif_gnn':
        return SemanticMotifGNN(config['model'])
    else:
        raise ValueError(
            f"Model '{name}' not found. "
            "Supported models: ['motif_graph_fer', 'semantic_motif_gnn']"
        )