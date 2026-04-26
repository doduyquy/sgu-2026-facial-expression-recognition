"""Config-driven experiment runner."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.artifact_builder import ensure_pixel_motif_artifacts, resolve_artifact_paths, run_command


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_experiment_config(config_name_or_path: str) -> tuple[dict[str, Any], Path]:
    """Load an experiment config by name or YAML path."""
    raw = Path(config_name_or_path)
    candidates: list[Path]
    if raw.suffix in {".yaml", ".yml"}:
        candidates = [raw, ROOT_DIR / raw]
    else:
        candidates = [
            ROOT_DIR / "configs" / "experiments" / f"{config_name_or_path}.yaml",
            ROOT_DIR / "configs" / f"{config_name_or_path}.yaml",
            raw,
        ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise ValueError(f"Experiment config must be a mapping: {path}")
            return cfg, path.resolve()
    raise FileNotFoundError(f"Cannot find experiment config {config_name_or_path!r}. Tried: {candidates}")


def debug_hierarchical_batch(train_cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.debug_hierarchical_batch",
        "--config",
        str(train_cfg.get("config", "hierarchical_motif_gnn")),
        "--env",
        str(train_cfg.get("env", "kaggle")),
        "--pixel_motif_dataset_path",
        str(paths["pixel_motif_dir"]),
        "--graph_repo_path",
        str(paths["graph_repo"]),
        "--batch_size",
        str(train_cfg.get("debug_batch_size", 2)),
        "--num_workers",
        "0",
    ]
    run_command(cmd)


def train_model(train_cfg: dict[str, Any], paths: dict[str, Path], *, epochs: int | None, no_wandb: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(train_cfg["config"]),
        "--env",
        str(train_cfg.get("env", "kaggle")),
        "--pixel_motif_dataset_path",
        str(paths["pixel_motif_dir"]),
        "--graph_repo_path",
        str(paths["graph_repo"]),
        "--epochs",
        str(epochs if epochs is not None else train_cfg.get("epochs", 80)),
    ]
    if no_wandb or bool(train_cfg.get("no_wandb", False)):
        cmd.append("--no_wandb")
    run_command(cmd)


def zip_outputs(outputs_cfg: dict[str, Any], experiment_name: str) -> None:
    if not bool(outputs_cfg.get("zip_outputs", True)):
        return
    outputs_dir = ROOT_DIR / "outputs"
    if not outputs_dir.exists():
        print(f"[zip] Skip because outputs directory is missing: {outputs_dir}", flush=True)
        return
    zip_name = str(outputs_cfg.get("zip_name", f"{experiment_name}_outputs.zip"))
    zip_path = Path(outputs_cfg.get("zip_path", Path("/kaggle/working") / zip_name))
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=outputs_dir)
    print(f"[zip] Created {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MB)", flush=True)


def run_experiment(
    config_name_or_path: str,
    *,
    csv_root: str | Path | None = None,
    out_root: str | Path | None = None,
    pixel_motif_dir: str | Path | None = None,
    epochs: int | None = None,
    smoke: bool = False,
    build_only: bool = False,
    train_only: bool = False,
    debug_only: bool = False,
    no_wandb: bool = False,
    no_skip_existing: bool = False,
) -> None:
    """Run one experiment from config."""
    cfg, cfg_path = load_experiment_config(config_name_or_path)
    experiment_cfg = dict(cfg.get("experiment", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    train_cfg = dict(cfg.get("training", {}) or {})
    outputs_cfg = dict(cfg.get("outputs", {}) or {})
    experiment_name = str(experiment_cfg.get("name", cfg_path.stem))

    if sum(bool(v) for v in [build_only, train_only, debug_only]) > 1:
        raise ValueError("--build_only, --train_only, and --debug_only are mutually exclusive")

    if smoke:
        data_cfg["smoke"] = True
    if no_skip_existing:
        data_cfg["skip_existing"] = False
    if pixel_motif_dir is not None:
        data_cfg["pixel_motif_dir"] = str(pixel_motif_dir)
    elif out_root is not None:
        # Keep local/Kaggle artifact overrides coherent; don't keep a config's
        # absolute pixel_motif_dir when the caller moves artifact_root.
        data_cfg.pop("pixel_motif_dir", None)

    paths = resolve_artifact_paths(data_cfg, out_root_override=out_root)

    print("=" * 100, flush=True)
    print(f"Experiment : {experiment_name}", flush=True)
    print(f"Config     : {cfg_path}", flush=True)
    print(f"Model cfg  : {train_cfg.get('config')}", flush=True)
    print(f"out_root   : {paths['out_root']}", flush=True)
    print(f"graph_repo : {paths['graph_repo']}", flush=True)
    print(f"dataset    : {paths['pixel_motif_dir']}", flush=True)
    print("=" * 100, flush=True)

    if not train_only and not debug_only:
        paths = ensure_pixel_motif_artifacts(data_cfg, csv_root=csv_root, out_root=out_root)

    if build_only:
        print("Build-only mode complete.", flush=True)
        return

    if bool(train_cfg.get("debug_batch", train_cfg.get("debug_hierarchical_batch", False))):
        debug_hierarchical_batch(train_cfg, paths)
    if debug_only:
        print("Debug-only mode complete.", flush=True)
        return

    if bool(train_cfg.get("enabled", True)):
        train_model(train_cfg, paths, epochs=epochs, no_wandb=no_wandb)
        zip_outputs(outputs_cfg, experiment_name)
    else:
        print("Training disabled by experiment config.", flush=True)
