"""Dataset for candidate-level learnable slot attention models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

SCALER_STATS_FILENAME = "candidate_x_scaler_stats.pt"


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _split_file(dataset_path: str | Path, split: str) -> Path:
    return Path(dataset_path) / f"{split}_candidate_attention.pt"


def _candidate_x_sum_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    sum_x = None
    sumsq_x = None
    for sample in samples:
        mask = torch.as_tensor(sample["candidate_mask"]).bool()
        x = torch.as_tensor(sample["candidate_x"]).float()
        if mask.any():
            values = x[mask]
            count += int(values.shape[0])
            cur_sum = values.sum(dim=0)
            cur_sumsq = (values * values).sum(dim=0)
            sum_x = cur_sum if sum_x is None else sum_x + cur_sum
            sumsq_x = cur_sumsq if sumsq_x is None else sumsq_x + cur_sumsq
    return {"count": count, "sum": sum_x, "sumsq": sumsq_x}


def compute_candidate_x_scaler_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit candidate_x scaler on valid candidates from the provided train samples."""
    stats = _candidate_x_sum_stats(samples)
    if stats["count"] == 0 or stats["sum"] is None or stats["sumsq"] is None:
        raise ValueError("Cannot fit candidate_x scaler: no valid train candidates found.")
    count = int(stats["count"])
    mean = stats["sum"] / count
    var = (stats["sumsq"] / count - mean * mean).clamp_min(0.0)
    std = var.sqrt().clamp_min(1e-6)
    return {
        "mean": mean.cpu(),
        "std": std.cpu(),
        "source": "train_only",
        "num_valid_candidates": count,
        "descriptor_dim": int(mean.numel()),
        "std_min_clamp": 1e-6,
    }


def compute_candidate_x_scaler_from_train(
    dataset_path: str | Path,
    *,
    train_samples: list[dict[str, Any]] | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Fit candidate_x scaler from the train split only and optionally persist it."""
    dataset_path = Path(dataset_path)
    samples = train_samples if train_samples is not None else _torch_load(_split_file(dataset_path, "train"))
    stats = compute_candidate_x_scaler_from_samples(samples)
    stats["split"] = "train"
    stats["artifact_path"] = str(dataset_path)
    if save:
        _torch_save(stats, dataset_path / SCALER_STATS_FILENAME)
    return stats


def load_candidate_x_scaler_stats(dataset_path: str | Path) -> dict[str, Any] | None:
    path = Path(dataset_path) / SCALER_STATS_FILENAME
    if not path.exists():
        return None
    stats = _torch_load(path)
    stats["stats_path"] = str(path)
    return stats


def apply_candidate_x_normalization(candidate_x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (candidate_x.float() - mean) / std.clamp_min(1e-6)


class CandidateAttentionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        normalize_x: bool = False,
        scaler_stats: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.samples = _torch_load(_split_file(self.dataset_path, split))
        self.meta = _torch_load(self.dataset_path / "meta.pt") if (self.dataset_path / "meta.pt").exists() else {}
        self.normalize_x = bool(normalize_x)
        self._mean = None
        self._std = None
        self._scaler_stats = None
        self._scaler_stats_path = self.dataset_path / SCALER_STATS_FILENAME
        if self.normalize_x:
            stats = scaler_stats or load_candidate_x_scaler_stats(self.dataset_path)
            if stats is None:
                stats = compute_candidate_x_scaler_from_train(
                    self.dataset_path,
                    train_samples=self.samples if self.split == "train" else None,
                    save=True,
                )
                stats["stats_path"] = str(self._scaler_stats_path)
            self._scaler_stats = stats
            self._mean = torch.as_tensor(stats["mean"]).float()
            self._std = torch.as_tensor(stats["std"]).float().clamp_min(1e-6)
        self._log_scaling_info()

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
            out["candidate_x"] = apply_candidate_x_normalization(out["candidate_x"], self._mean, self._std)
        return out

    @property
    def descriptor_scaling_source(self) -> str:
        if not self.normalize_x:
            return "disabled"
        if self._mean is None or self._std is None:
            return "enabled, but no valid candidates found"
        return str(self._scaler_stats.get("source", "train_only")) if self._scaler_stats else "train_only"

    @property
    def scaler_stats_path(self) -> str:
        if self._scaler_stats and self._scaler_stats.get("stats_path"):
            return str(self._scaler_stats["stats_path"])
        return str(self._scaler_stats_path)

    @property
    def descriptor_storage(self) -> str:
        return str(self.meta.get("descriptor_storage", self.meta.get("candidate_x_storage", "raw_unmarked")))

    @property
    def geometry_scaling_enabled(self) -> bool:
        return bool(self.meta.get("geometry_normalized", True))

    def _log_scaling_info(self) -> None:
        print(
            f"--- [{self.split}] descriptor_scaling: {'enabled' if self.normalize_x else 'disabled'}",
            flush=True,
        )
        print(f"--- [{self.split}] descriptor mean/std source: {self.descriptor_scaling_source}", flush=True)
        print(f"--- [{self.split}] scaler file path: {self.scaler_stats_path if self.normalize_x else 'none'}", flush=True)
        if self.normalize_x and self._mean is not None and self._std is not None:
            mean5 = [round(float(v), 6) for v in self._mean[:5]]
            std5 = [round(float(v), 6) for v in self._std[:5]]
            print(f"--- [{self.split}] candidate_x train_mean first5: {mean5}", flush=True)
            print(f"--- [{self.split}] candidate_x train_std first5 : {std5}", flush=True)
            if self.split in {"val", "test"}:
                print(f"--- [{self.split}] using train-only candidate_x scaler (no {self.split} stats fit)", flush=True)
        print(f"--- [{self.split}] candidate_x storage: {self.descriptor_storage}", flush=True)
        if not self.normalize_x and self.descriptor_storage not in {"raw", "raw_unmarked"}:
            print(
                f"[warn] [{self.split}] normalize_candidate_x=false but artifact marks "
                f"candidate_x storage as {self.descriptor_storage!r}; rebuild a raw descriptor artifact "
                "for a true no-scale run.",
                flush=True,
            )
        print(
            f"--- [{self.split}] geometry_scaling: "
            f"{'enabled' if self.geometry_scaling_enabled else 'disabled'}",
            flush=True,
        )
        print(
            f"--- [{self.split}] edge_attr scaling: normalized spatial dx/dy/dist from normalized centers; "
            "edge_type unscaled",
            flush=True,
        )


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
