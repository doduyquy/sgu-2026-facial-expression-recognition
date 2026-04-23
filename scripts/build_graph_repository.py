"""
scripts/build_graph_repository.py — Build the full canonical graph repository.

Reads train.csv / val.csv / test.csv, converts each image to a PixelGraphSample,
and saves in chunked format under:

    <repo_root>/
      shared/shared_graph.pt
      train/chunk_000.pt, chunk_001.pt, …
      val/chunk_000.pt, …
      test/chunk_000.pt, …
      manifest.pt

Memory strategy: at most one chunk is kept in RAM at a time during writing.

Usage
-----
    python scripts/build_graph_repository.py \\
        --train_csv data/fer13-split/train.csv \\
        --val_csv   data/fer13-split/val.csv \\
        --test_csv  data/fer13-split/test.csv \\
        --repo_root artifacts/graph_repo

    # With custom chunk size and 4-connectivity
    python scripts/build_graph_repository.py \\
        --train_csv data/fer13-split/train.csv \\
        --val_csv   data/fer13-split/val.csv \\
        --test_csv  data/fer13-split/test.csv \\
        --repo_root artifacts/graph_repo \\
        --chunk_size 1000 \\
        --connectivity 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from configs.graph_config import GraphConfig
from data.raw_fer_dataset import RawFERDataset
from data.raw_types import RawSample
from data.shared_graph_builder import SharedGraphBuilder
from data.canonical_graph_builder import CanonicalGraphBuilder
from data.graph_repository import GraphRepositoryWriter
from data.graph_types import SharedGraphStructure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_graph_repository")


# ===========================================================================
# Args
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build canonical graph repository from FER-2013 CSV splits."
    )
    p.add_argument("--train_csv",    required=True,  help="Path to train.csv")
    p.add_argument("--val_csv",      required=True,  help="Path to val.csv")
    p.add_argument("--test_csv",     required=True,  help="Path to test.csv")
    p.add_argument("--repo_root",    default="artifacts/graph_repo",
                   help="Output repository root (default: artifacts/graph_repo)")
    p.add_argument("--chunk_size",   type=int, default=500,
                   help="Samples per chunk file (default: 500)")
    p.add_argument("--connectivity", type=int, default=8, choices=[4, 8],
                   help="Pixel grid connectivity (default: 8)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip split if its directory already has chunks")
    return p.parse_args()


# ===========================================================================
# Per-split build
# ===========================================================================

def build_split(
    csv_path: str,
    split: str,
    cfg: GraphConfig,
    shared: SharedGraphStructure,
    writer: GraphRepositoryWriter,
    skip_existing: bool,
) -> None:
    """Stream one split from CSV → PixelGraphSamples → chunks on disk."""

    split_dir = Path(cfg.repo_root) / split
    if skip_existing and split_dir.exists() and any(split_dir.glob("chunk_*.pt")):
        log.info("Split '%s' already exists — skipping (--skip_existing)", split)
        return

    log.info("═" * 60)
    log.info("Building split: %s  (csv: %s)", split, csv_path)

    raw_ds = RawFERDataset(csv_path=csv_path, split=split, validate=False)
    raw_ds.print_summary()

    graph_builder = CanonicalGraphBuilder(config=cfg, shared=shared)

    t0 = time.perf_counter()
    with writer.open_split(split) as sw:
        for raw_sample in tqdm(
            raw_ds,
            desc=f"  {split}",
            unit="img",
            dynamic_ncols=True,
        ):
            _validate_raw_sample(raw_sample, cfg)
            pixel_graph = graph_builder.build(raw_sample)
            _validate_pixel_graph(pixel_graph, shared)
            sw.add(pixel_graph)

    elapsed = time.perf_counter() - t0
    n = len(raw_ds)
    log.info(
        "Split '%s' done: %d samples in %.1f s (%.0f img/s)",
        split, n, elapsed, n / elapsed,
    )


# ===========================================================================
# Validation helpers
# ===========================================================================

def _validate_raw_sample(raw: RawSample, cfg: GraphConfig) -> None:
    if raw.image.shape != (cfg.height, cfg.width):
        raise ValueError(
            f"sample_id={raw.sample_id}: expected ({cfg.height},{cfg.width}), "
            f"got {raw.image.shape}"
        )
    if raw.label < 0 or raw.label > 6:
        raise ValueError(
            f"sample_id={raw.sample_id}: label {raw.label} out of [0..6]"
        )


def _validate_pixel_graph(pg, shared: SharedGraphStructure) -> None:
    import torch
    expected_nodes = shared.num_nodes
    expected_edges = shared.num_edges
    if pg.node_features.shape[0] != expected_nodes:
        raise ValueError(
            f"graph_id={pg.graph_id}: node_features has {pg.node_features.shape[0]} rows, "
            f"expected {expected_nodes}"
        )
    if pg.edge_attr_dynamic.shape[0] != expected_edges:
        raise ValueError(
            f"graph_id={pg.graph_id}: edge_attr_dynamic has {pg.edge_attr_dynamic.shape[0]} rows, "
            f"expected {expected_edges}"
        )
    if torch.isnan(pg.node_features).any():
        raise ValueError(f"graph_id={pg.graph_id}: NaN in node_features")
    if torch.isnan(pg.edge_attr_dynamic).any():
        raise ValueError(f"graph_id={pg.graph_id}: NaN in edge_attr_dynamic")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    cfg = GraphConfig(
        connectivity=args.connectivity,
        chunk_size=args.chunk_size,
        repo_root=args.repo_root,
    )
    log.info("GraphConfig: %s", cfg.to_dict())

    # 1. Build shared graph (once)
    log.info("Building SharedGraphStructure …")
    shared = SharedGraphBuilder(cfg).build()
    log.info("  %s", shared)

    writer = GraphRepositoryWriter(repo_root=args.repo_root, config=cfg)

    # 2. Save shared graph
    writer.write_shared(shared)

    # 3. Build each split
    splits: List[Tuple[str, str]] = [
        ("train", args.train_csv),
        ("val",   args.val_csv),
        ("test",  args.test_csv),
    ]

    t_total = time.perf_counter()
    for split, csv_path in splits:
        build_split(
            csv_path=csv_path,
            split=split,
            cfg=cfg,
            shared=shared,
            writer=writer,
            skip_existing=args.skip_existing,
        )

    # 4. Save manifest
    writer.save_manifest()

    elapsed_total = time.perf_counter() - t_total
    log.info("═" * 60)
    log.info("Repository built in %.1f s → %s", elapsed_total, args.repo_root)
    log.info("Ready to upload artifacts/graph_repo/ to Kaggle as a dataset.")


if __name__ == "__main__":
    main()
