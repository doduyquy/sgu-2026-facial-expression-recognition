"""Pixel motif artifact builder API.

This module is the stable data-pipeline surface for experiments. It calls the
atomic build/inspect scripts directly and does not depend on notebook logic or
another orchestration script.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SPLITS = ["train", "val", "test"]
STAGE_ORDER = ["graph_repo", "candidates", "motif_bank", "motif_dataset"]
REQUIRED_CSV_FILES = {"train.csv", "val.csv", "test.csv"}

# Keys in manifest that must match config when loading from artifact.
_MANIFEST_CHECK_KEYS = {
    "node_feature_dim": ("graph", "node_feature_dim"),
    "connectivity": ("graph", "connectivity"),
    "descriptor_dim": ("motif", "descriptor_dim"),
    "top_k": ("pixel_motif_dataset", "top_k"),
    "nmax": ("pixel_motif_dataset", "nmax"),
    "has_node_indices": ("pixel_motif_dataset", "has_node_indices"),
    "has_node_mask": ("pixel_motif_dataset", "has_node_mask"),
}


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


def _check_sub_x_cache(path: Path) -> bool:
    """Check if V3 cache tensors (sub_x, sub_adj) exist in first sample."""
    try:
        import torch as _torch
        pt = path / "train_pixel_motif.pt"
        if not pt.exists():
            return False
        data = _torch.load(pt, map_location="cpu", weights_only=False)
        if isinstance(data, list) and len(data) > 0:
            s = data[0]
            return "sub_x" in s and "sub_adj" in s
    except Exception:
        pass
    return False


def has_hierarchical_cache(path: Path) -> bool:
    """Check if V3 hierarchical cache is complete."""
    return has_pixel_motif_dataset(path) and _check_sub_x_cache(path)


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


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _probe_nmax(pixel_motif_dir: Path) -> int | None:
    """Best-effort: read nmax from meta.pt if available."""
    try:
        import torch
        meta_path = pixel_motif_dir / "meta.pt"
        if not meta_path.exists():
            return None
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict):
            return int(meta.get("nmax") or meta.get("Nmax") or 0) or None
    except Exception:
        pass
    return None


def _probe_descriptor_dim(pixel_motif_dir: Path) -> int | None:
    """Best-effort: read descriptor_dim from meta.pt if available."""
    try:
        import torch
        meta_path = pixel_motif_dir / "meta.pt"
        if not meta_path.exists():
            return None
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict):
            return int(meta.get("descriptor_dim") or 0) or None
    except Exception:
        pass
    return None


def _check_node_indices(pixel_motif_dir: Path) -> bool:
    """Check a sample from train_pixel_motif.pt for node_indices key."""
    try:
        import torch
        pt = pixel_motif_dir / "train_pixel_motif.pt"
        if not pt.exists():
            return False
        data = torch.load(pt, map_location="cpu", weights_only=False)
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            return "node_indices" in sample
        if isinstance(data, dict):
            return "node_indices" in data
    except Exception:
        pass
    return False


def _check_node_mask(pixel_motif_dir: Path) -> bool:
    try:
        import torch
        pt = pixel_motif_dir / "train_pixel_motif.pt"
        if not pt.exists():
            return False
        data = torch.load(pt, map_location="cpu", weights_only=False)
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            return "node_mask" in sample
        if isinstance(data, dict):
            return "node_mask" in data
    except Exception:
        pass
    return False


def write_manifest(
    out_root: Path,
    data_cfg: dict[str, Any],
    experiment_name: str,
    pixel_motif_dir: Path,
) -> Path:
    """Write manifest.json after a successful artifact build."""
    nmax = _probe_nmax(pixel_motif_dir)
    descriptor_dim = _probe_descriptor_dim(pixel_motif_dir)
    has_node_indices = _check_node_indices(pixel_motif_dir)
    has_node_mask = _check_node_mask(pixel_motif_dir)

    manifest = {
        "artifact_version": "pixel_motif_v2",
        "experiment_name": experiment_name,
        "created_from": "csv",
        "graph": {
            "image_size": 48,
            "node_feature_dim": 7,
            "connectivity": int(data_cfg.get("connectivity", 8)),
        },
        "candidate": {
            "seed_stride": int(data_cfg.get("seed_stride", 4)),
            "radii": _as_list(data_cfg.get("radii", [1, 2])),
            "max_candidates": int(data_cfg.get("max_candidates", 128)),
            "nmax": nmax,
        },
        "motif": {
            "num_motifs_per_class": int(data_cfg.get("num_motifs_per_class", 16)),
            "descriptor_dim": descriptor_dim,
        },
        "pixel_motif_dataset": {
            "top_k": int(data_cfg.get("top_k", 32)),
            "nmax": nmax,
            "descriptor_dim": descriptor_dim,
            "has_node_indices": has_node_indices,
            "has_node_mask": has_node_mask,
            "has_sub_x_cache": _check_sub_x_cache(pixel_motif_dir),
            "has_sub_adj_cache": _check_sub_x_cache(pixel_motif_dir),
            "sub_x_dtype": "float32",
            "sub_adj_dtype": "uint8",
            "edge_attr_mode": str(data_cfg.get("edge_attr_mode", "spatial")),
        },
        "compatible_models": ["motif_guided_gnn", "hierarchical_motif_gnn"],
    }

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] Written: {manifest_path}", flush=True)
    return manifest_path


def read_manifest(out_root: Path) -> dict[str, Any]:
    """Read manifest.json from artifact root. Raises FileNotFoundError if missing."""
    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {out_root}. "
            "Run with mode=build_and_train first to build and save artifacts."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(
    out_root: Path,
    data_cfg: dict[str, Any],
    require_node_indices: bool = False,
    require_node_mask: bool = False,
) -> dict[str, Any]:
    """Read and validate manifest against current data config. Returns manifest."""
    manifest = read_manifest(out_root)
    pmd = manifest.get("pixel_motif_dataset", {})
    errors: list[str] = []

    # Check top_k
    expected_top_k = int(data_cfg.get("top_k", 32))
    if pmd.get("top_k") != expected_top_k:
        errors.append(f"top_k: manifest={pmd.get('top_k')} vs config={expected_top_k}")

    # Check nmax if configured
    expected_nmax = data_cfg.get("nmax")
    if expected_nmax is not None and pmd.get("nmax") is not None:
        if int(pmd["nmax"]) != int(expected_nmax):
            errors.append(f"nmax: manifest={pmd.get('nmax')} vs config={expected_nmax}")

    # Check descriptor_dim if configured
    expected_dim = data_cfg.get("descriptor_dim")
    if expected_dim is not None and pmd.get("descriptor_dim") is not None:
        if int(pmd["descriptor_dim"]) != int(expected_dim):
            errors.append(f"descriptor_dim: manifest={pmd.get('descriptor_dim')} vs config={expected_dim}")

    # Check node_indices requirement
    if require_node_indices and not pmd.get("has_node_indices", False):
        errors.append("has_node_indices=False but model requires node_indices (HierarchicalMotifGNN). Rebuild artifact.")

    if require_node_mask and not pmd.get("has_node_mask", False):
        errors.append("has_node_mask=False but model requires node_mask. Rebuild artifact.")

    if errors:
        raise ValueError(
            f"Artifact manifest validation failed ({out_root}/manifest.json):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    print(f"[manifest] Validated OK: {out_root / 'manifest.json'}", flush=True)
    return manifest


def load_artifacts_from_input(
    artifact_input_path: str | Path,
    out_root: str | Path,
) -> dict[str, Path]:
    """Copy/symlink artifacts from /kaggle/input/<dataset>/artifacts -> out_root.

    On Kaggle, /kaggle/input is read-only so we copy to /kaggle/working.
    Returns resolved paths dict (same schema as resolve_artifact_paths).
    """
    src = Path(artifact_input_path)
    dst = Path(out_root)

    if not src.exists():
        raise FileNotFoundError(f"artifact_input_path not found: {src}")

    print(f"[load_artifacts] Copying {src} -> {dst}", flush=True)
    if dst.exists():
        # Remove stale artifacts to avoid mixing versions
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[load_artifacts] Done.", flush=True)

    # Return paths using the same schema as resolve_artifact_paths
    return {
        "out_root": dst,
        "graph_repo": dst / "graph_repo",
        "candidate_dir": dst / "pixel_candidate_subgraphs_v2",
        "motif_bank_dir": dst / "pixel_motif_bank_v2",
        "pixel_motif_dir": dst / "pixel_motif_dataset_v2",
    }


def build_hierarchical_cache(
    data_cfg: dict[str, Any],
    pixel_motif_dir: Path,
    graph_repo: Path,
    out_dir: Path,
    skip_existing: bool,
) -> None:
    """Precompute V3 hierarchical subgraph tensors (sub_x, sub_adj, sub_node_mask)."""
    if skip_existing and has_hierarchical_cache(out_dir):
        print(f"[skip] hierarchical cache exists: {out_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/precompute_hierarchical_motif_dataset.py",
        "--pixel_motif_dataset_path", str(pixel_motif_dir),
        "--graph_repo_path", str(graph_repo),
        "--out_dir", str(out_dir),
        "--log_every", str(data_cfg.get("log_every", 500)),
    ]
    run_command(cmd)


def zip_artifacts(out_root: Path, zip_path: Path) -> None:
    """Zip the entire artifacts directory for download/publishing."""
    if not out_root.exists():
        print(f"[zip_artifacts] Skipping — artifacts dir missing: {out_root}", flush=True)
        return
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out_root.parent, base_dir=out_root.name)
    print(f"[zip_artifacts] Created {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MB)", flush=True)


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

    # Optional V3 hierarchical cache — only when build_hierarchical_cache: true in data config
    if bool(data_cfg.get("build_hierarchical_cache", False)):
        v2_dir = paths["pixel_motif_dir"]  # source: V2
        v3_dir = paths["out_root"] / "pixel_motif_dataset_v3_hierarchical"
        build_hierarchical_cache(data_cfg, v2_dir, paths["graph_repo"], v3_dir, skip_existing)
        paths["pixel_motif_dir"] = v3_dir  # switch downstream to V3
        print(f"[pipeline] Using V3 hierarchical cache: {v3_dir}", flush=True)

    print_artifact_summary(
        [paths["graph_repo"], paths["candidate_dir"], paths["motif_bank_dir"], paths["pixel_motif_dir"]]
    )
    return paths
