"""End-to-end Pixel Motif experiment runner: CSV -> artifacts -> train -> evaluate."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
REQUIRED_CSV_FILES = {"train.csv", "val.csv", "test.csv"}


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 100, flush=True)
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    print("=" * 100, flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return data


def _resolve_experiment_path(value: str) -> Path:
    raw = Path(value)
    candidates = []
    if raw.suffix in {".yaml", ".yml"}:
        candidates.append(raw)
        candidates.append(ROOT_DIR / raw)
    else:
        candidates.extend(
            [
                ROOT_DIR / "configs" / "experiments" / f"{value}.yaml",
                ROOT_DIR / "configs" / f"{value}.yaml",
                raw,
            ]
        )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Cannot find experiment config {value!r}. Tried: {candidates}")


def _find_csv_root(search_root: Path = Path("/kaggle/input")) -> Path:
    for dirname, _dirs, files in os.walk(search_root):
        path = Path(dirname)
        if REQUIRED_CSV_FILES.issubset(set(files)):
            return path
    raise FileNotFoundError(f"Could not find {sorted(REQUIRED_CSV_FILES)} under {search_root}")


def _csv_root_from_config(build_cfg: dict[str, Any], cli_csv_root: str | None) -> Path:
    if cli_csv_root:
        return Path(cli_csv_root).resolve()
    value = build_cfg.get("csv_root", "auto")
    if value is None or str(value).lower() == "auto":
        return _find_csv_root()
    return Path(value).resolve()


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _pipeline_command(
    build_cfg: dict[str, Any],
    csv_root: Path,
    out_root: Path,
    pixel_motif_dir: Path,
    cli_smoke: bool,
    cli_no_skip_existing: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_pixel_motif_v2_pipeline.py",
        "--stage",
        str(build_cfg.get("stage", "all")),
        "--csv_root",
        str(csv_root),
        "--out_root",
        str(out_root),
        "--pixel_motif_dir",
        str(pixel_motif_dir),
        "--edge_attr_mode",
        str(build_cfg.get("edge_attr_mode", "spatial")),
        "--chunk_size",
        str(build_cfg.get("chunk_size", 500)),
        "--connectivity",
        str(build_cfg.get("connectivity", 8)),
        "--max_candidates",
        str(build_cfg.get("max_candidates", 128)),
        "--seed_stride",
        str(build_cfg.get("seed_stride", 4)),
        "--radii",
        *[str(v) for v in _as_list(build_cfg.get("radii", [1, 2]))],
        "--coverage_grid",
        *[str(v) for v in _as_list(build_cfg.get("coverage_grid", [4, 4]))],
        "--log_every",
        str(build_cfg.get("log_every", 1000)),
        "--num_motifs_per_class",
        str(build_cfg.get("num_motifs_per_class", 16)),
        "--max_subgraphs_per_class",
        str(build_cfg.get("max_subgraphs_per_class", 50000)),
        "--alpha",
        str(build_cfg.get("alpha", 0.5)),
        "--seed",
        str(build_cfg.get("seed", 42)),
        "--num_exemplars",
        str(build_cfg.get("num_exemplars", 5)),
        "--top_k",
        str(build_cfg.get("top_k", 32)),
        "--knn_k",
        str(build_cfg.get("knn_k", 4)),
        "--beta",
        str(build_cfg.get("beta", 0.5)),
        "--gamma",
        str(build_cfg.get("gamma", 0.25)),
        "--eta",
        str(build_cfg.get("eta", 0.05)),
        "--diversity_sigma",
        str(build_cfg.get("diversity_sigma", 0.12)),
    ]
    if bool(build_cfg.get("skip_existing", True)) and not cli_no_skip_existing:
        cmd.append("--skip_existing")
    if bool(build_cfg.get("smoke", False)) or cli_smoke:
        cmd.extend(["--smoke", "--smoke_samples", str(build_cfg.get("smoke_samples", 100))])
    return cmd


def _debug_command(train_cfg: dict[str, Any], pixel_motif_dir: Path, graph_repo: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.debug_hierarchical_batch",
        "--config",
        str(train_cfg.get("config", "hierarchical_motif_gnn")),
        "--env",
        str(train_cfg.get("env", "kaggle")),
        "--pixel_motif_dataset_path",
        str(pixel_motif_dir),
        "--graph_repo_path",
        str(graph_repo),
        "--batch_size",
        str(train_cfg.get("debug_batch_size", 2)),
        "--num_workers",
        "0",
    ]


def _train_command(
    train_cfg: dict[str, Any],
    pixel_motif_dir: Path,
    graph_repo: Path,
    cli_epochs: int | None,
    cli_no_wandb: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(train_cfg["config"]),
        "--env",
        str(train_cfg.get("env", "kaggle")),
        "--pixel_motif_dataset_path",
        str(pixel_motif_dir),
        "--graph_repo_path",
        str(graph_repo),
        "--epochs",
        str(cli_epochs if cli_epochs is not None else train_cfg.get("epochs", 80)),
    ]
    if bool(train_cfg.get("no_wandb", False)) or cli_no_wandb:
        cmd.append("--no_wandb")
    return cmd


def _zip_outputs(outputs_cfg: dict[str, Any], experiment_name: str) -> None:
    if not bool(outputs_cfg.get("zip_outputs", True)):
        return
    outputs_dir = ROOT_DIR / "outputs"
    if not outputs_dir.exists():
        print(f"[zip] Skip because outputs directory is missing: {outputs_dir}", flush=True)
        return
    zip_name = str(outputs_cfg.get("zip_name", f"{experiment_name}_outputs.zip"))
    zip_path = Path("/kaggle/working") / zip_name
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=outputs_dir)
    print(f"[zip] Created {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MB)", flush=True)


def main() -> None:
    os.chdir(ROOT_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="hierarchical_motif_gnn_c")
    parser.add_argument("--csv_root", default=None, help="Override CSV root. Use when not on Kaggle.")
    parser.add_argument("--out_root", default=None, help="Override artifact root.")
    parser.add_argument("--pixel_motif_dir", default=None, help="Override final pixel motif dataset directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--smoke", action="store_true", help="Force smoke build.")
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--debug_only", action="store_true", help="Run hierarchical debug batch and skip training.")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_skip_existing", action="store_true")
    args = parser.parse_args()

    exp_path = _resolve_experiment_path(args.experiment)
    exp_cfg = _load_yaml(exp_path)
    experiment_name = str(exp_cfg.get("experiment", {}).get("name", exp_path.stem))
    build_cfg = dict(exp_cfg.get("build", {}) or {})
    train_cfg = dict(exp_cfg.get("train", {}) or {})
    outputs_cfg = dict(exp_cfg.get("outputs", {}) or {})

    out_root = Path(args.out_root or build_cfg.get("out_root", "/kaggle/working/artifacts")).resolve()
    graph_repo = out_root / "graph_repo"
    default_dataset_name = (
        "pixel_motif_dataset_v2_rich_edges"
        if build_cfg.get("edge_attr_mode") == "rich"
        else "pixel_motif_dataset_v2"
    )
    configured_pixel_dir = build_cfg.get("pixel_motif_dir")
    if args.pixel_motif_dir is not None:
        pixel_motif_dir = Path(args.pixel_motif_dir).resolve()
    elif args.out_root is not None:
        # Local smoke tests often override out_root; keep all generated artifacts together.
        pixel_motif_dir = (out_root / default_dataset_name).resolve()
    elif configured_pixel_dir is not None:
        pixel_motif_dir = Path(configured_pixel_dir).resolve()
    else:
        pixel_motif_dir = (out_root / default_dataset_name).resolve()

    print("=" * 100, flush=True)
    print(f"Experiment : {experiment_name}", flush=True)
    print(f"Config     : {exp_path}", flush=True)
    print(f"out_root   : {out_root}", flush=True)
    print(f"graph_repo : {graph_repo}", flush=True)
    print(f"dataset    : {pixel_motif_dir}", flush=True)
    print("=" * 100, flush=True)

    if sum(bool(v) for v in [args.train_only, args.build_only, args.debug_only]) > 1:
        raise ValueError("--train_only, --build_only, and --debug_only are mutually exclusive")

    if not args.train_only and not args.debug_only:
        csv_root = _csv_root_from_config(build_cfg, args.csv_root)
        _run(_pipeline_command(build_cfg, csv_root, out_root, pixel_motif_dir, args.smoke, args.no_skip_existing))

    if args.build_only:
        print("Build-only mode complete.", flush=True)
        return

    if bool(train_cfg.get("enabled", True)):
        if bool(train_cfg.get("debug_hierarchical_batch", False)):
            _run(_debug_command(train_cfg, pixel_motif_dir, graph_repo))
        if args.debug_only:
            print("Debug-only mode complete.", flush=True)
            return
        _run(_train_command(train_cfg, pixel_motif_dir, graph_repo, args.epochs, args.no_wandb))
        _zip_outputs(outputs_cfg, experiment_name)
    else:
        print("Training disabled by experiment config.", flush=True)


if __name__ == "__main__":
    main()
