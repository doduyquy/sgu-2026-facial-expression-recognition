"""Dataset for candidate-level learnable slot attention models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CandidateAttentionDataset(Dataset):
    def __init__(self, dataset_path: str | Path, split: str, normalize_x: bool = False) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.samples = _torch_load(self.dataset_path / f"{split}_candidate_attention.pt")
        self.meta = _torch_load(self.dataset_path / "meta.pt") if (self.dataset_path / "meta.pt").exists() else {}
        self.normalize_x = bool(normalize_x)
        self._mean = None
        self._std = None
        if self.normalize_x and self.samples:
            xs = []
            for sample in self.samples[: min(2048, len(self.samples))]:
                mask = torch.as_tensor(sample["candidate_mask"]).bool()
                x = torch.as_tensor(sample["candidate_x"]).float()
                if mask.any():
                    xs.append(x[mask])
            if xs:
                all_x = torch.cat(xs, dim=0)
                self._mean = all_x.mean(dim=0)
                self._std = all_x.std(dim=0, unbiased=False).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def input_dim(self) -> int:
        if self.meta.get("descriptor_dim"):
            return int(self.meta["descriptor_dim"])
        if not self.samples:
            return 0
        return int(self.samples[0]["candidate_x"].shape[-1])

    @property
    def max_candidates(self) -> int:
        if self.meta.get("max_candidates"):
            return int(self.meta["max_candidates"])
        if not self.samples:
            return 0
        return int(self.samples[0]["candidate_x"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        out = {
            "graph_id": int(sample["graph_id"]),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "y": torch.tensor(int(sample["label"]), dtype=torch.long),
        }
        for key, value in sample.items():
            if key in {"graph_id", "label"}:
                continue
            out[key] = value.clone() if torch.is_tensor(value) else value
        if self.normalize_x and self._mean is not None:
            out["candidate_x"] = (out["candidate_x"].float() - self._mean) / self._std
        return out


def collate_fn_candidate_attention(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "graph_id": torch.tensor([int(s["graph_id"]) for s in batch], dtype=torch.long),
        "label": torch.stack([s["label"] for s in batch]),
        "y": torch.stack([s["y"] for s in batch]),
    }
    fixed_keys = [
        "candidate_x",
        "candidate_mask",
        "candidate_centers",
        "candidate_bbox",
        "candidate_radius",
        "candidate_coverage_cell",
        "candidate_node_indices",
        "candidate_node_mask",
    ]
    for key in fixed_keys:
        if key in batch[0]:
            out[key] = torch.stack([s[key] for s in batch])

    edge_indices = [s["candidate_edge_index"] for s in batch]
    edge_attrs = [s.get("candidate_edge_attr") for s in batch]
    edge_valids = [s["candidate_edge_valid"] for s in batch]
    e_max = max(int(e.shape[1]) for e in edge_indices) if edge_indices else 0
    e_max = max(e_max, 1)
    attr_dim = int(edge_attrs[0].shape[1]) if edge_attrs and edge_attrs[0] is not None and edge_attrs[0].ndim == 2 else 0
    edge_index = torch.zeros((len(batch), 2, e_max), dtype=torch.long)
    edge_attr = torch.zeros((len(batch), e_max, attr_dim), dtype=torch.float32)
    edge_valid = torch.zeros((len(batch), e_max), dtype=torch.bool)
    for i, ei in enumerate(edge_indices):
        e = int(ei.shape[1])
        if e > 0:
            edge_index[i, :, :e] = ei
            edge_valid[i, :e] = edge_valids[i].bool()[:e]
            if attr_dim > 0 and edge_attrs[i] is not None:
                edge_attr[i, :e, :] = edge_attrs[i].float()

    out["candidate_edge_index"] = edge_index
    out["candidate_edge_attr"] = edge_attr
    out["candidate_edge_valid"] = edge_valid
    # Alias used by generic Trainer for sample count only; model consumes candidate_x.
    out["x"] = out["candidate_x"]
    return out

