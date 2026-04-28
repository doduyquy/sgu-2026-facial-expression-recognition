"""Full pixel-graph dataset for D4 adaptive motif slot models."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from data.chunked_graph_dataset import ChunkedGraphDataset
from data.graph_types import ResolvedPixelGraph


class FullGraphDataset(Dataset):
    """Read resolved full 48x48 pixel graphs directly from the graph repository."""

    def __init__(self, repo_root: str, split: str, cache_chunks: int = 1) -> None:
        self.split = split
        self._ds = ChunkedGraphDataset(
            repo_root=repo_root,
            split=split,
            resolve=True,
            cache_chunks=cache_chunks,
        )
        self.shared = self._ds.shared

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        graph: ResolvedPixelGraph = self._ds[idx]
        node_features = graph.node_features.float()
        return {
            "graph_id": int(graph.graph_id),
            "node_features": node_features,
            "x": node_features,
            "edge_index": graph.edge_index.long(),
            "edge_attr": graph.edge_attr.float(),
            "node_mask": torch.ones(graph.num_nodes, dtype=torch.bool),
            "label": torch.tensor(int(graph.label), dtype=torch.long),
            "y": torch.tensor(int(graph.label), dtype=torch.long),
        }

    @property
    def input_dim(self) -> int:
        return int(self._ds.num_node_features)

    @property
    def edge_dim(self) -> int:
        return int(self._ds.num_edge_features)

    @property
    def num_nodes(self) -> int:
        if self.shared is not None:
            return int(self.shared.num_nodes)
        if len(self) == 0:
            return 0
        return int(self[0]["node_features"].shape[0])

    @property
    def num_edges(self) -> int:
        if self.shared is not None:
            return int(self.shared.num_edges)
        if len(self) == 0:
            return 0
        return int(self[0]["edge_index"].shape[1])


def collate_fn_full_graph(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate resolved full graphs while keeping shared edge_index unbatched."""
    node_features = torch.stack([s["node_features"] for s in batch])
    edge_attr = torch.stack([s["edge_attr"] for s in batch])
    node_mask = torch.stack([s["node_mask"] for s in batch])
    labels = torch.stack([s["y"] for s in batch])
    out = {
        "graph_id": torch.tensor([int(s["graph_id"]) for s in batch], dtype=torch.long),
        "node_features": node_features,
        "x": node_features,
        "edge_index": batch[0]["edge_index"],
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "label": labels,
        "y": labels,
    }
    return out
