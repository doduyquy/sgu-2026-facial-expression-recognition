"""Run Pixel-preserving Motif V2 stages on local/Kaggle.

This script is intentionally a thin orchestrator around the existing scripts.
It is useful on Kaggle where local machines should not build heavy artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


STAGE_ORDER = ["graph_repo", "candidates", "motif_bank", "motif_dataset"]
SPLITS = ["train", "val", "test"]


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 100, flush=True)
    print("RUN:", " ".join(cmd), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(cmd, check=True)


def _has_graph_repo(path: Path) -> bool:
    return (
        (path / "manifest.pt").exists()
        and (path / "shared" / "shared_graph.pt").exists()
        and all(any((path / split).glob("chunk_*.pt")) for split in SPLITS)
    )


def _has_candidate_dataset(path: Path) -> bool:
    return (path / "meta.pt").exists() and all(
        (path / f"{split}_pixel_candidates.pt").exists() for split in SPLITS
    )


def _has_motif_bank(path: Path) -> bool:
    return (path / "pixel_motif_bank.pt").exists()


def _has_pixel_motif_dataset(path: Path) -> bool:
    return (path / "meta.pt").exists() and all(
        (path / f"{split}_pixel_motif.pt").exists() for split in SPLITS
    )


def _csv_path(csv_root: Path, split: str) -> Path:
    path = csv_root / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split}.csv at {path}. Expected csv_root with train.csv, val.csv, test.csv."
        )
    return path


def _resolve_stages(stage: str) -> list[str]:
    if stage == "all":
        return list(STAGE_ORDER)
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGE_ORDER + ['all']}")
    return [stage]


def _stage_graph_repo(args, graph_repo: Path) -> None:
    if args.skip_existing and _has_graph_repo(graph_repo):
        print(f"[skip] graph_repo exists: {graph_repo}", flush=True)
        return
    if args.csv_root is None:
        raise ValueError("--csv_root is required for stage graph_repo/all")
    csv_root = Path(args.csv_root)
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
        str(args.chunk_size),
        "--connectivity",
        str(args.connectivity),
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    _run(cmd)


def _stage_candidates(args, graph_repo: Path, candidate_dir: Path) -> None:
    if args.skip_existing and _has_candidate_dataset(candidate_dir):
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
        str(args.max_candidates),
        "--seed_stride",
        str(args.seed_stride),
        "--radii",
        *[str(r) for r in args.radii],
        "--coverage_grid",
        str(args.coverage_grid[0]),
        str(args.coverage_grid[1]),
        "--log_every",
        str(args.log_every),
    ]
    if args.smoke:
        cmd.extend(["--max_samples_per_split", str(args.smoke_samples)])
    _run(cmd)
    _run([sys.executable, "scripts/inspect_pixel_candidate_subgraphs.py", "--data_dir", str(candidate_dir)])


def _stage_motif_bank(args, candidate_dir: Path, motif_bank_dir: Path) -> None:
    if args.skip_existing and _has_motif_bank(motif_bank_dir):
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
        str(args.num_motifs_per_class),
        "--max_subgraphs_per_class",
        str(args.max_subgraphs_per_class),
        "--alpha",
        str(args.alpha),
        "--seed",
        str(args.seed),
        "--num_exemplars",
        str(args.num_exemplars),
    ]
    _run(cmd)
    _run(
        [
            sys.executable,
            "scripts/inspect_pixel_motif_bank.py",
            "--motif_bank_path",
            str(motif_bank_dir / "pixel_motif_bank.pt"),
        ]
    )


def _stage_motif_dataset(args, candidate_dir: Path, motif_bank_dir: Path, pixel_motif_dir: Path) -> None:
    if args.skip_existing and _has_pixel_motif_dataset(pixel_motif_dir):
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
        str(args.top_k),
        "--knn_k",
        str(args.knn_k),
        "--beta",
        str(args.beta),
        "--gamma",
        str(args.gamma),
        "--eta",
        str(args.eta),
        "--diversity_sigma",
        str(args.diversity_sigma),
        "--edge_attr_mode",
        str(args.edge_attr_mode),
    ]
    _run(cmd)
    _run([sys.executable, "scripts/inspect_pixel_motif_dataset.py", "--data_dir", str(pixel_motif_dir)])
    _run(
        [
            sys.executable,
            "scripts/audit_pixel_motif_dataset.py",
            "--data_dir",
            str(pixel_motif_dir),
            "--splits",
            *SPLITS,
        ]
    )


def _print_summary(paths: Iterable[Path]) -> None:
    print("\nArtifacts:", flush=True)
    for path in paths:
        if not path.exists():
            print(f"  MISSING {path}", flush=True)
            continue
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2
        print(f"  {path} ({size:.2f} MB)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=STAGE_ORDER + ["all"])
    p.add_argument("--csv_root", default=None)
    p.add_argument("--out_root", default="artifacts")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_samples", type=int, default=100)

    p.add_argument("--chunk_size", type=int, default=500)
    p.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    p.add_argument("--max_candidates", type=int, default=128)
    p.add_argument("--seed_stride", type=int, default=4)
    p.add_argument("--radii", nargs="+", type=int, default=[1, 2])
    p.add_argument("--coverage_grid", nargs=2, type=int, default=[4, 4])
    p.add_argument("--log_every", type=int, default=1000)

    p.add_argument("--num_motifs_per_class", type=int, default=16)
    p.add_argument("--max_subgraphs_per_class", type=int, default=50000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_exemplars", type=int, default=5)

    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--knn_k", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--diversity_sigma", type=float, default=0.12)
    p.add_argument("--edge_attr_mode", choices=["spatial", "rich"], default="spatial")
    args = p.parse_args()

    out_root = Path(args.out_root)
    graph_repo = out_root / "graph_repo"
    candidate_dir = out_root / "pixel_candidate_subgraphs_v2"
    motif_bank_dir = out_root / "pixel_motif_bank_v2"
    pixel_motif_dir = out_root / "pixel_motif_dataset_v2"

    stages = _resolve_stages(args.stage)
    print(f"Stages: {stages}", flush=True)
    print(f"out_root: {out_root}", flush=True)

    for stage in stages:
        if stage == "graph_repo":
            _stage_graph_repo(args, graph_repo)
        elif stage == "candidates":
            _stage_candidates(args, graph_repo, candidate_dir)
        elif stage == "motif_bank":
            _stage_motif_bank(args, candidate_dir, motif_bank_dir)
        elif stage == "motif_dataset":
            _stage_motif_dataset(args, candidate_dir, motif_bank_dir, pixel_motif_dir)

    _print_summary([graph_repo, candidate_dir, motif_bank_dir, pixel_motif_dir])
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
