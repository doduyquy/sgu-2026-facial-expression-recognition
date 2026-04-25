"""Dataset for motif-filtered image-level subgraph bags."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class MotifFilteredDataset(Dataset):
    """Load precomputed motif-filtered samples for one split."""

    FILENAMES = {
        "train": "train_motif_filtered.pt",
        "val": "val_motif_filtered.pt",
        "test": "test_motif_filtered.pt",
    }

    def __init__(self, data_dir: str | Path, split: str) -> None:
        if split not in self.FILENAMES:
            raise ValueError(f"Unknown split {split!r}; expected one of {list(self.FILENAMES)}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.path = self.data_dir / self.FILENAMES[split]
        if not self.path.exists():
            raise FileNotFoundError(
                f"[MotifFilteredDataset] Missing file: {self.path}\n"
                f"  Run scripts/precompute_motif_filtered_dataset.py first."
            )

        try:
            self._samples: List[Dict] = torch.load(self.path, map_location="cpu", weights_only=False)
        except TypeError:
            self._samples = torch.load(self.path, map_location="cpu")

        if len(self._samples) == 0:
            raise RuntimeError(f"Empty motif-filtered dataset: {self.path}")

        s0 = self._samples[0]
        self._num_subgraphs = int(s0["x"].shape[0])
        self._descriptor_dim = int(s0["x"].shape[1])
        self._edge_attr_dim = int(s0["edge_attr"].shape[1]) if s0["edge_attr"].ndim == 2 else 0

        print(
            f"[MotifFilteredDataset] Loaded {len(self._samples)} samples from {self.path.name} "
            f"| K={self._num_subgraphs} | D={self._descriptor_dim}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self._samples[idx]
        label = torch.tensor(int(s["label"]), dtype=torch.long)
        return {
            "graph_id": int(s["graph_id"]),
            "x": torch.as_tensor(s["x"]).float(),
            "mask": torch.as_tensor(s["mask"]).bool(),
            "centers": torch.as_tensor(s["centers"]).float(),
            "edge_index": torch.as_tensor(s["edge_index"]).long(),
            "edge_attr": torch.as_tensor(s["edge_attr"]).float(),
            "match_scores": torch.as_tensor(s["match_scores"]).float(),
            "matched_class": torch.as_tensor(s["matched_class"]).long(),
            "matched_motif_id": torch.as_tensor(s["matched_motif_id"]).long(),
            "matched_disc_score": torch.as_tensor(s["matched_disc_score"]).float(),
            "motif_score_vector": torch.as_tensor(s["motif_score_vector"]).float(),
            "label": label,
            "y": label,
        }

    @property
    def input_dim(self) -> int:
        return self._descriptor_dim

    @property
    def num_subgraphs(self) -> int:
        return self._num_subgraphs

    @property
    def edge_attr_dim(self) -> int:
        return self._edge_attr_dim
