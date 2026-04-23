"""
scripts/build_shared_graph.py — Build and save SharedGraphStructure once.

This script computes the shared topology (edge_index + static edge attrs)
for the 48×48 8-neighbor pixel grid and saves it to:

    <repo_root>/shared/shared_graph.pt

The result is reused by every image in the dataset — no need to recompute
per image or per split.

Usage
-----
    python scripts/build_shared_graph.py
    python scripts/build_shared_graph.py --repo_root artifacts/graph_repo
    python scripts/build_shared_graph.py --connectivity 4 --chunk_size 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.graph_config import GraphConfig
from data.shared_graph_builder import SharedGraphBuilder
from data.graph_repository import GraphRepositoryWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_shared_graph")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build SharedGraphStructure for FER-2013 48×48 pixel grid."
    )
    p.add_argument("--repo_root",    default="artifacts/graph_repo",
                   help="Repository root directory (default: artifacts/graph_repo)")
    p.add_argument("--height",       type=int, default=48)
    p.add_argument("--width",        type=int, default=48)
    p.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    p.add_argument("--chunk_size",   type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = GraphConfig(
        height=args.height,
        width=args.width,
        connectivity=args.connectivity,
        chunk_size=args.chunk_size,
        repo_root=args.repo_root,
    )

    log.info("GraphConfig: %s", cfg.to_dict())

    # Build
    builder = SharedGraphBuilder(cfg)
    shared = builder.build()

    log.info("SharedGraphStructure: %s", shared)
    log.info("  edge_index  : %s  dtype=%s", tuple(shared.edge_index.shape), shared.edge_index.dtype)
    log.info("  edge_attr_s : %s  dtype=%s", tuple(shared.edge_attr_static.shape), shared.edge_attr_static.dtype)

    # Save
    writer = GraphRepositoryWriter(repo_root=args.repo_root, config=cfg)
    out_path = writer.write_shared(shared)

    log.info("Done. Shared graph saved to: %s", out_path)

    # Quick sanity check
    import torch
    loaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert loaded.num_edges == shared.num_edges, "Round-trip check failed!"
    log.info("Round-trip sanity check passed ✓")


if __name__ == "__main__":
    main()
