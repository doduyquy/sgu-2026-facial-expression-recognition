"""Audit candidate_x scaling protocol for candidate_attention_dataset_v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.candidate_attention_dataset import (
    SCALER_STATS_FILENAME,
    apply_candidate_x_normalization,
    compute_candidate_x_scaler_from_train,
    load_candidate_x_scaler_stats,
)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _empty_accum() -> dict[str, Any]:
    return {
        "count": 0,
        "sum": None,
        "sumsq": None,
        "min": None,
        "max": None,
    }


def _update_accum(acc: dict[str, Any], values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    values = values.float()
    acc["count"] += int(values.shape[0])
    cur_sum = values.sum(dim=0)
    cur_sumsq = (values * values).sum(dim=0)
    cur_min = values.min()
    cur_max = values.max()
    acc["sum"] = cur_sum if acc["sum"] is None else acc["sum"] + cur_sum
    acc["sumsq"] = cur_sumsq if acc["sumsq"] is None else acc["sumsq"] + cur_sumsq
    acc["min"] = cur_min if acc["min"] is None else torch.minimum(acc["min"], cur_min)
    acc["max"] = cur_max if acc["max"] is None else torch.maximum(acc["max"], cur_max)


def _finish_accum(acc: dict[str, Any]) -> dict[str, Any]:
    if acc["count"] == 0:
        return {"count": 0}
    count = int(acc["count"])
    mean = acc["sum"] / count
    var = (acc["sumsq"] / count - mean * mean).clamp_min(0.0)
    std = var.sqrt()
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": float(acc["min"].item()),
        "max": float(acc["max"].item()),
        "scalar_mean": float(mean.mean().item()),
        "scalar_std": float(std.mean().item()),
    }


def _split_stats(samples: list[dict[str, Any]], mean: torch.Tensor | None = None, std: torch.Tensor | None = None) -> dict[str, Any]:
    acc = _empty_accum()
    for sample in samples:
        mask = torch.as_tensor(sample["candidate_mask"]).bool()
        x = torch.as_tensor(sample["candidate_x"]).float()
        if not mask.any():
            continue
        values = x[mask]
        if mean is not None and std is not None:
            values = apply_candidate_x_normalization(values, mean, std)
        _update_accum(acc, values)
    return _finish_accum(acc)


def _print_stats(prefix: str, stats: dict[str, Any]) -> None:
    if not stats.get("count"):
        print(f"{prefix}: empty")
        return
    mean5 = [round(float(v), 6) for v in stats["mean"][:5]]
    std5 = [round(float(v), 6) for v in stats["std"][:5]]
    print(
        f"{prefix}: valid={stats['count']} "
        f"mean={stats['scalar_mean']:.6f} std={stats['scalar_std']:.6f} "
        f"min={stats['min']:.6f} max={stats['max']:.6f}"
    )
    print(f"{prefix}: mean first5={mean5}")
    print(f"{prefix}: std  first5={std5}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="artifacts/candidate_attention_dataset_v1")
    parser.add_argument("--write_missing_scaler", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.pt"
    meta = _torch_load(meta_path) if meta_path.exists() else {}
    descriptor_storage = str(meta.get("descriptor_storage", meta.get("candidate_x_storage", "raw_unmarked")))

    print("=" * 80)
    print("Candidate X Scaling Audit")
    print("=" * 80)
    print(f"artifact path       : {data_dir}")
    print(f"descriptor_storage  : {descriptor_storage}")
    print(f"descriptor_scaling  : {meta.get('descriptor_scaling', 'unknown')}")
    print(f"scaler file path    : {data_dir / SCALER_STATS_FILENAME}")

    samples_by_split: dict[str, list[dict[str, Any]]] = {}
    raw_stats_by_split: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        split_path = data_dir / f"{split}_candidate_attention.pt"
        if not split_path.exists():
            print(f"WARN: missing split file: {split_path}")
            continue
        samples = _torch_load(split_path)
        samples_by_split[split] = samples
        print(f"{split:<5} samples      : {len(samples)}")
        raw_stats_by_split[split] = _split_stats(samples)
        _print_stats(f"{split:<5} raw", raw_stats_by_split[split])

    stats = load_candidate_x_scaler_stats(data_dir)
    status = "PASS"
    if stats is None:
        status = "WARN"
        print(f"WARN: missing {SCALER_STATS_FILENAME}")
        if "train" not in samples_by_split:
            print("FAIL: cannot compute transient train-only scaler because train split is missing")
            return
        stats = compute_candidate_x_scaler_from_train(
            data_dir,
            train_samples=samples_by_split["train"],
            save=bool(args.write_missing_scaler),
        )
        print(f"transient scaler source: {stats.get('source')}")
        if args.write_missing_scaler:
            print(f"wrote missing scaler -> {data_dir / SCALER_STATS_FILENAME}")

    source = str(stats.get("source", "unknown"))
    mean = torch.as_tensor(stats["mean"]).float()
    std = torch.as_tensor(stats["std"]).float().clamp_min(1e-6)
    print(f"scaler source       : {source}")
    print(f"scaler valid cand.  : {stats.get('num_valid_candidates', 'unknown')}")
    print(f"train_mean first5   : {[round(float(v), 6) for v in mean[:5]]}")
    print(f"train_std first5    : {[round(float(v), 6) for v in std[:5]]}")

    if source != "train_only":
        status = "FAIL"
        print(f"FAIL: scaler source is {source!r}, expected 'train_only'")

    for split, samples in samples_by_split.items():
        after = _split_stats(samples, mean=mean, std=std)
        _print_stats(f"{split:<5} after train-scaler", after)

    if descriptor_storage not in {"raw", "raw_unmarked"}:
        status = "WARN" if status == "PASS" else status
        print(
            "WARN: artifact metadata does not mark candidate_x as raw; "
            "rebuild a raw descriptor artifact before interpreting no-scale/scaling ablations."
        )

    train_raw = raw_stats_by_split.get("train")
    if train_raw and abs(train_raw["scalar_mean"]) < 0.05 and 0.9 < train_raw["scalar_std"] < 1.1:
        status = "WARN" if status == "PASS" else status
        print(
            "WARN: train raw stats look already standardized. "
            "If this was not intentional, rebuild candidate_x from raw descriptors."
        )

    if status == "PASS":
        print("PASS: train-only candidate_x scaler is used for train/val/test")
    elif status == "WARN":
        print("WARN: audit completed with warnings; see messages above")
    else:
        print("FAIL: split-local or invalid candidate_x scaler protocol detected")


if __name__ == "__main__":
    main()
