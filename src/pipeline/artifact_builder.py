"""Pixel motif artifact builder API.

This module is the stable data-pipeline surface for experiments. It calls the
atomic build/inspect scripts directly and does not depend on notebook logic or
another orchestration script.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SPLITS = ["train", "val", "test"]
STAGE_ORDER = ["graph_repo", "candidates", "motif_bank", "motif_dataset"]
REQUIRED_CSV_FILES = {"train.csv", "val.csv", "test.csv"}


def run_command(cmd: list[str]) -> None:
    """Run one command with visible boundaries."""
    print("\n" + "=" * 100, flush=True)
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    print("=" * 100, flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def find_csv_root(search_root: str | Path = "/kaggle/input") -> Path:
    """Find a folder containing train.csv, val.csv, and test.csv."""
    root = Path(search_root)
    for dirname, _dirs, files in os.walk(root):
        if REQUIRED_CSV_FILES.issubset(set(files)):
            return Path(dirname)
    raise FileNotFoundError(f"Could not find {sorted(REQUIRED_CSV_FILES)} under {root}")


def resolve_csv_root(value: str | Path | None, search_root: str | Path = "/kaggle/input") -> Path:
    if value is None or str(value).lower() == "auto":
        return find_csv_root(search_root)
    path = Path(value)
    missing = [name for name in sorted(REQUIRED_CSV_FILES) if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"CSV root {path} is missing: {missing}")
    return path


def resolve_artifact_paths(data_cfg: dict[str, Any], out_root_override: str | Path | None = None) -> dict[str, Path]:
    """Resolve canonical artifact paths from a data config."""
    data_cfg = normalize_data_config(data_cfg)
    out_root = Path(out_root_override or data_cfg.get("artifact_root", "/kaggle/working/artifacts"))
    edge_attr_mode = str(data_cfg.get("edge_attr_mode", "spatial"))
    default_dataset = "pixel_motif_dataset_v2_rich_edges" if edge_attr_mode == "rich" else "pixel_motif_dataset_v2"
    pixel_motif_dir = Path(data_cfg.get("pixel_motif_dir", out_root / default_dataset))
    return {
        "out_root": out_root,
        "graph_repo": out_root / "graph_repo",
        "candidate_dir": out_root / "pixel_candidate_subgraphs_v2",
        "motif_bank_dir": out_root / "pixel_motif_bank_v2",
        "pixel_motif_dir": pixel_motif_dir,
    }


def normalize_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten template-style data sections into the builder's internal option map."""
    out = dict(data_cfg)
    for section in ["graph", "candidates", "motif_bank", "motif_dataset"]:
        nested = out.pop(section, None)
        if isinstance(nested, dict):
            out.update(nested)
    return out


def has_graph_repo(path: Path) -> bool:
    return (
        (path / "manifest.pt").exists()
        and (path / "shared" / "shared_graph.pt").exists()
        and all(any((path / split).glob("chunk_*.pt")) for split in SPLITS)
    )


def has_candidate_dataset(path: Path) -> bool:
    return (path / "meta.pt").exists() and all((path / f"{split}_pixel_candidates.pt").exists() for split in SPLITS)


def has_motif_bank(path: Path) -> bool:
    return (path / "pixel_motif_bank.pt").exists()


def has_pixel_motif_dataset(path: Path) -> bool:
    return (path / "meta.pt").exists() and all((path / f"{split}_pixel_motif.pt").exists() for split in SPLITS)


def resolve_stages(stage: str) -> list[str]:
    if stage == "all":
        return list(STAGE_ORDER)
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGE_ORDER + ['all']}")
    return [stage]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _csv_path(csv_root: Path, split: str) -> Path:
    path = csv_root / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_graph_repo(data_cfg: dict[str, Any], csv_root: Path, graph_repo: Path, skip_existing: bool) -> None:
    if skip_existing and has_graph_repo(graph_repo):
        print(f"[skip] graph_repo exists: {graph_repo}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/build_graph_repository.py",
        "--train_csv",
        str(_csv_path(csv_root, "train")),
        "--val_csv",
        str(_csv_path(csv_root, "val")),
        "--test_csv",
        str(_csv_path(csv_root, "test")),
        "--repo_root",
        str(graph_repo),
        "--chunk_size",
        str(data_cfg.get("chunk_size", 500)),
        "--connectivity",
        str(data_cfg.get("connectivity", 8)),
    ]
    if skip_existing:
        cmd.append("--skip_existing")
    run_command(cmd)


def build_candidates(data_cfg: dict[str, Any], graph_repo: Path, candidate_dir: Path, skip_existing: bool) -> None:
    if skip_existing and has_candidate_dataset(candidate_dir):
        print(f"[skip] candidates exist: {candidate_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/precompute_pixel_candidate_subgraphs.py",
        "--repo_root",
        str(graph_repo),
        "--out_dir",
        str(candidate_dir),
        "--max_candidates",
        str(data_cfg.get("max_candidates", 128)),
        "--seed_stride",
        str(data_cfg.get("seed_stride", 4)),
        "--radii",
        *[str(v) for v in _as_list(data_cfg.get("radii", [1, 2]))],
        "--coverage_grid",
        *[str(v) for v in _as_list(data_cfg.get("coverage_grid", [4, 4]))],
        "--log_every",
        str(data_cfg.get("log_every", 1000)),
    ]
    if bool(data_cfg.get("smoke", False)):
        cmd.extend(["--max_samples_per_split", str(data_cfg.get("smoke_samples", 100))])
    run_command(cmd)
    run_command([sys.executable, "scripts/inspect_pixel_candidate_subgraphs.py", "--data_dir", str(candidate_dir)])


def build_motif_bank(data_cfg: dict[str, Any], candidate_dir: Path, motif_bank_dir: Path, skip_existing: bool) -> None:
    if skip_existing and has_motif_bank(motif_bank_dir):
        print(f"[skip] motif bank exists: {motif_bank_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/build_pixel_motif_bank.py",
        "--input_dir",
        str(candidate_dir),
        "--out_dir",
        str(motif_bank_dir),
        "--num_motifs_per_class",
        str(data_cfg.get("num_motifs_per_class", 16)),
        "--max_subgraphs_per_class",
        str(data_cfg.get("max_subgraphs_per_class", 50000)),
        "--alpha",
        str(data_cfg.get("alpha", 0.5)),
        "--seed",
        str(data_cfg.get("seed", 42)),
        "--num_exemplars",
        str(data_cfg.get("num_exemplars", 5)),
    ]
    run_command(cmd)
    run_command(
        [
            sys.executable,
            "scripts/inspect_pixel_motif_bank.py",
            "--motif_bank_path",
            str(motif_bank_dir / "pixel_motif_bank.pt"),
        ]
    )


def build_pixel_motif_dataset(
    data_cfg: dict[str, Any],
    candidate_dir: Path,
    motif_bank_dir: Path,
    pixel_motif_dir: Path,
    skip_existing: bool,
) -> None:
    if skip_existing and has_pixel_motif_dataset(pixel_motif_dir):
        print(f"[skip] pixel motif dataset exists: {pixel_motif_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/precompute_pixel_motif_dataset.py",
        "--candidate_dir",
        str(candidate_dir),
        "--motif_bank_path",
        str(motif_bank_dir / "pixel_motif_bank.pt"),
        "--out_dir",
        str(pixel_motif_dir),
        "--top_k",
        str(data_cfg.get("top_k", 32)),
        "--knn_k",
        str(data_cfg.get("knn_k", 4)),
        "--beta",
        str(data_cfg.get("beta", 0.5)),
        "--gamma",
        str(data_cfg.get("gamma", 0.25)),
        "--eta",
        str(data_cfg.get("eta", 0.05)),
        "--diversity_sigma",
        str(data_cfg.get("diversity_sigma", 0.12)),
        "--edge_attr_mode",
        str(data_cfg.get("edge_attr_mode", "spatial")),
    ]
    run_command(cmd)
    run_command([sys.executable, "scripts/inspect_pixel_motif_dataset.py", "--data_dir", str(pixel_motif_dir)])
    run_command(
        [
            sys.executable,
            "scripts/audit_pixel_motif_dataset.py",
            "--data_dir",
            str(pixel_motif_dir),
            "--splits",
            *SPLITS,
        ]
    )


def print_artifact_summary(paths: Iterable[Path]) -> None:
    print("\nArtifacts:", flush=True)
    for path in paths:
        if not path.exists():
            print(f"  MISSING {path}", flush=True)
            continue
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2
        print(f"  {path} ({size:.2f} MB)", flush=True)


def ensure_pixel_motif_artifacts(
    data_cfg: dict[str, Any],
    *,
    csv_root: str | Path | None = None,
    out_root: str | Path | None = None,
) -> dict[str, Path]:
    """Build or reuse pixel motif artifacts and return resolved paths."""
    data_cfg = normalize_data_config(data_cfg)
    paths = resolve_artifact_paths(data_cfg, out_root_override=out_root)
    resolved_csv_root = resolve_csv_root(csv_root or data_cfg.get("csv_root", "auto"))
    skip_existing = bool(data_cfg.get("skip_existing", True))
    stages = resolve_stages(str(data_cfg.get("stage", "all")))

    print(f"Stages: {stages}", flush=True)
    print(f"CSV root: {resolved_csv_root}", flush=True)
    print(f"Artifact root: {paths['out_root']}", flush=True)

    for stage in stages:
        if stage == "graph_repo":
            build_graph_repo(data_cfg, resolved_csv_root, paths["graph_repo"], skip_existing)
        elif stage == "candidates":
            build_candidates(data_cfg, paths["graph_repo"], paths["candidate_dir"], skip_existing)
        elif stage == "motif_bank":
            build_motif_bank(data_cfg, paths["candidate_dir"], paths["motif_bank_dir"], skip_existing)
        elif stage == "motif_dataset":
            build_pixel_motif_dataset(
                data_cfg,
                paths["candidate_dir"],
                paths["motif_bank_dir"],
                paths["pixel_motif_dir"],
                skip_existing,
            )

    print_artifact_summary(
        [paths["graph_repo"], paths["candidate_dir"], paths["motif_bank_dir"], paths["pixel_motif_dir"]]
    )
    return paths
