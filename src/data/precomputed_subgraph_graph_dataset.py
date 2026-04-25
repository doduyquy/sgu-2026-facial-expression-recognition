"""
src/data/precomputed_subgraph_graph_dataset.py
Dataset đọc file .pt đã precomputed bởi scripts/precompute_subgraph_graph_dataset.py.
Không resolve graph, không sinh subgraph online → cực nhanh.

Output mỗi sample:
    {
        "x"         : FloatTensor [K, D],
        "mask"      : FloatTensor [K],
        "edge_index": LongTensor  [2, E],
        "edge_attr" : FloatTensor [E, 1],
        "centers"   : FloatTensor [K, 2],
        "y"         : LongTensor  [],
        "graph_id"  : int,
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class PrecomputedSubgraphGraphDataset(Dataset):
    """Dataset đọc file precomputed subgraph-level graph (.pt)."""

    def __init__(self, pt_path: str | Path) -> None:
        pt_path = Path(pt_path)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"[PrecomputedSubgraphGraphDataset] File không tồn tại: {pt_path}\n"
                f"  Hãy chạy scripts/precompute_subgraph_graph_dataset.py trước."
            )

        self._samples: List[Dict] = torch.load(pt_path, weights_only=False)
        if len(self._samples) == 0:
            raise RuntimeError(f"File rỗng: {pt_path}")

        s0 = self._samples[0]
        self._num_subgraphs:  int = int(s0["x"].shape[0])
        self._descriptor_dim: int = int(s0["x"].shape[1])

        print(
            f"[PrecomputedSubgraphGraphDataset] Loaded {len(self._samples)} samples "
            f"from {pt_path.name} | K={self._num_subgraphs} | D={self._descriptor_dim}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self._samples[idx]
        return {
            "x"         : s["x"],
            "mask"      : s["mask"],
            "edge_index": s["edge_index"],
            "edge_attr" : s["edge_attr"],
            "centers"   : s["centers"],
            "y"         : torch.tensor(s["label"], dtype=torch.long),
            "graph_id"  : s["graph_id"],
        }

    @property
    def input_dim(self) -> int:
        """Descriptor dimension D."""
        return self._descriptor_dim

    @property
    def num_subgraphs(self) -> int:
        """K — số subgraph nodes mỗi ảnh."""
        return self._num_subgraphs
