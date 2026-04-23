"""
Dataset that converts resolved graphs into bags of subgraph descriptors.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset

from data.chunked_graph_dataset import ChunkedGraphDataset
from src.graph.subgraph_descriptor import infer_descriptor_dim, subgraph_to_descriptor
from src.graph.subgraph_generator import generate_candidate_subgraphs


class SubgraphDescriptorDataset(Dataset):
    """
    Return one fixed-size bag of subgraph descriptors per image.

    Output format:
        {
            "x": FloatTensor [K, D],
            "mask": FloatTensor [K],
            "y": LongTensor [],
            "num_valid_subgraphs": LongTensor [],
        }
    """

    def __init__(
        self,
        repo_path: str,
        split: str,
        num_subgraphs: int = 16,
        subgraph_radius: int = 1,
        seed_stride: int = 4,
        max_candidates: int = 64,
        max_nodes_per_subgraph: Optional[int] = None,
        cache_chunks: int = 1,
    ) -> None:
        self._ds = ChunkedGraphDataset(
            repo_root=repo_path,
            split=split,
            resolve=True,
            cache_chunks=cache_chunks,
        )
        self.num_subgraphs = int(num_subgraphs)
        self.subgraph_radius = int(subgraph_radius)
        self.seed_stride = int(seed_stride)
        self.max_candidates = int(max_candidates)
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self._descriptor_dim = infer_descriptor_dim(
            node_feature_dim=self._ds.num_node_features,
            edge_feature_dim=self._ds.num_edge_features,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        graph = self._ds[idx]
        candidates = generate_candidate_subgraphs(
            graph=graph,
            radius=self.subgraph_radius,
            seed_stride=self.seed_stride,
            max_candidates=self.max_candidates,
            max_nodes_per_subgraph=self.max_nodes_per_subgraph,
        )

        descriptors = [
            subgraph_to_descriptor(subgraph)
            for subgraph in candidates[: self.num_subgraphs]
        ]

        valid_count = len(descriptors)
        bag = torch.zeros((self.num_subgraphs, self._descriptor_dim), dtype=torch.float32)
        mask = torch.zeros(self.num_subgraphs, dtype=torch.float32)

        if valid_count > 0:
            stacked = torch.stack(descriptors, dim=0)
            bag[:valid_count] = stacked
            mask[:valid_count] = 1.0

        return {
            "x": bag,
            "mask": mask,
            "y": torch.tensor(graph.label, dtype=torch.long),
            "num_valid_subgraphs": torch.tensor(valid_count, dtype=torch.long),
        }

    @property
    def input_dim(self) -> int:
        return self._descriptor_dim
