import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.io import load_graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    graphs, load_mode = load_graphs(args.graph_path)
    print(f"Số graph trong file: {len(graphs)}")
    print(f"Load mode: {load_mode}")

    g = graphs[args.index]

    print("\n=== GRAPH INFO ===")
    print(f"graph_id: {g.graph_id}")
    print(f"label: {g.label}")
    print(f"split: {g.split}")
    print(f"usage: {g.usage}")
    print(f"height x width: {g.height} x {g.width}")

    print("\n=== SHAPES ===")
    print("node_features:", g.node_features.shape)
    print("edge_index:", g.edge_index.shape)
    print("edge_attr:", g.edge_attr.shape)

    print("\n=== FEATURE NAMES ===")
    print("node_feature_names:", g.metadata.get("node_feature_names"))
    print("edge_feature_names:", g.metadata.get("edge_feature_names"))

    print("\n=== CHECK NAN ===")
    print("node_features has nan:", np.isnan(g.node_features).any())
    print("edge_attr has nan:", np.isnan(g.edge_attr).any())

    print("\n=== FIRST 5 NODES ===")
    print(g.node_features[:5])

    print("\n=== FIRST 5 EDGES ===")
    print(g.edge_index[:, :5])

    print("\n=== FIRST 5 EDGE ATTR ===")
    print(g.edge_attr[:5])


if __name__ == "__main__":
    main()
