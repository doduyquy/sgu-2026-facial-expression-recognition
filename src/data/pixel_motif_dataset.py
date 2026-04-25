"""Dataset for pixel-preserving motif-selected image-level samples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class PixelMotifDataset(Dataset):
    """Load pixel-preserving motif dataset V2."""

    FILENAMES = {
        "train": "train_pixel_motif.pt",
        "val": "val_pixel_motif.pt",
        "test": "test_pixel_motif.pt",
    }
    _STATS_CACHE: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(self, data_dir: str | Path, split: str, normalize_x: bool = False) -> None:
        if split not in self.FILENAMES:
            raise ValueError(f"Unknown split {split!r}")
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize_x = bool(normalize_x)
        self.path = self.data_dir / self.FILENAMES[split]
        if not self.path.exists():
            raise FileNotFoundError(
                f"[PixelMotifDataset] Missing file: {self.path}\n"
                f"  Run scripts/precompute_pixel_motif_dataset.py first."
            )
        try:
            self._samples: List[Dict] = torch.load(self.path, map_location="cpu", weights_only=False)
        except TypeError:
            self._samples = torch.load(self.path, map_location="cpu")
        if not self._samples:
            raise RuntimeError(f"Empty dataset: {self.path}")
        s0 = self._samples[0]
        self._num_subgraphs = int(s0["x"].shape[0])
        self._descriptor_dim = int(s0["x"].shape[1])
        self._max_nodes = int(s0["node_indices"].shape[1])
        self._x_mean: torch.Tensor | None = None
        self._x_std: torch.Tensor | None = None
        if self.normalize_x:
            self._x_mean, self._x_std = self._load_or_compute_train_stats()
        print(
            f"[PixelMotifDataset] Loaded {len(self._samples)} samples from {self.path.name} "
            f"| K={self._num_subgraphs} | D={self._descriptor_dim} | max_nodes={self._max_nodes}"
            f" | normalize_x={self.normalize_x}"
        )

    def _load_or_compute_train_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = str(self.data_dir.resolve())
        if cache_key in self._STATS_CACHE:
            return self._STATS_CACHE[cache_key]

        if self.split == "train":
            train_samples = self._samples
        else:
            train_path = self.data_dir / self.FILENAMES["train"]
            try:
                train_samples = torch.load(train_path, map_location="cpu", weights_only=False)
            except TypeError:
                train_samples = torch.load(train_path, map_location="cpu")

        total = torch.zeros(self._descriptor_dim, dtype=torch.float64)
        total_sq = torch.zeros(self._descriptor_dim, dtype=torch.float64)
        count = 0
        for sample in train_samples:
            x = torch.as_tensor(sample["x"]).float()
            mask = torch.as_tensor(sample.get("mask", torch.ones(x.shape[0], dtype=torch.bool))).bool()
            xv = x[mask]
            if xv.numel() == 0:
                continue
            total += xv.double().sum(dim=0)
            total_sq += xv.double().pow(2).sum(dim=0)
            count += int(xv.shape[0])
        if count <= 0:
            raise RuntimeError(f"Cannot compute x normalization stats from {self.data_dir}")
        mean = (total / count).float()
        var = (total_sq / count).float() - mean.pow(2)
        std = var.clamp_min(0.0).sqrt().clamp_min(1e-6)
        self._STATS_CACHE[cache_key] = (mean, std)
        return mean, std

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self._samples[idx]
        label = torch.tensor(int(s["label"]), dtype=torch.long)
        x = torch.as_tensor(s["x"]).float()
        if self.normalize_x:
            x = (x - self._x_mean) / self._x_std
        return {
            "graph_id": int(s["graph_id"]),
            "x": x,
            "mask": torch.as_tensor(s["mask"]).bool(),
            "centers": torch.as_tensor(s["centers"]).float(),
            "bbox": torch.as_tensor(s["bbox"]).float(),
            "selected_indices": torch.as_tensor(s["selected_indices"]).long(),
            "node_indices": torch.as_tensor(s["node_indices"]).long(),
            "node_mask": torch.as_tensor(s["node_mask"]).bool(),
            "edge_index": torch.as_tensor(s["edge_index"]).long(),
            "edge_attr": torch.as_tensor(s["edge_attr"]).float(),
            "match_scores": torch.as_tensor(s["match_scores"]).float(),
            "matched_class": torch.as_tensor(s["matched_class"]).long(),
            "matched_motif_id": torch.as_tensor(s["matched_motif_id"]).long(),
            "matched_disc_score": torch.as_tensor(s["matched_disc_score"]).float(),
            "motif_score_vector": torch.as_tensor(s["motif_score_vector"]).float(),
            "coverage_cell": torch.as_tensor(s["coverage_cell"]).long(),
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
    def max_nodes(self) -> int:
        return self._max_nodes
