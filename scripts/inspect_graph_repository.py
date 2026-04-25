"""
scripts/inspect_graph_repository.py — Inspect and validate the canonical graph repository.

Checks
------
  1. Shared graph: shape, dtype, feature names
  2. Per-split: number of chunks, total sample count
  3. First sample of each split: node_features / edge_attr_dynamic shapes
  4. Resolved graph: edge_index, full edge_attr shapes, feature names
  5. NaN / Inf checks
  6. Metadata / feature name consistency

Usage
-----
    python scripts/inspect_graph_repository.py
    python scripts/inspect_graph_repository.py --repo_root artifacts/graph_repo
    python scripts/inspect_graph_repository.py --split train --chunk 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.graph_types import PixelGraphSample, SharedGraphStructure, ResolvedPixelGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inspect_graph_repository")

SEP = "─" * 65


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a canonical graph repository."
    )
    p.add_argument("--repo_root", default="artifacts/graph_repo",
                   help="Repository root directory")
    p.add_argument("--split",     default=None,
                   help="Inspect a specific split (default: all splits)")
    p.add_argument("--chunk",     type=int, default=0,
                   help="Chunk index to sample from (default: 0)")
    p.add_argument("--sample",    type=int, default=0,
                   help="Sample index within the chunk (default: 0)")
    return p.parse_args()


# ===========================================================================
# Inspection helpers
# ===========================================================================

def inspect_shared(shared: SharedGraphStructure) -> None:
    print(f"\n{SEP}")
    print("SHARED GRAPH STRUCTURE")
    print(SEP)
    print(f"  Grid          : {shared.height} × {shared.width} = {shared.num_nodes} nodes")
    print(f"  Connectivity  : {shared.connectivity}-neighbor")
    print(f"  Num edges     : {shared.num_edges}")
    print(f"  edge_index    : shape={tuple(shared.edge_index.shape)}  dtype={shared.edge_index.dtype}")
    print(f"  edge_attr_s   : shape={tuple(shared.edge_attr_static.shape)}  dtype={shared.edge_attr_static.dtype}")
    print(f"  Static feats  : {shared.static_feature_names}")

    # Value sanity
    ei = shared.edge_index
    max_node_id = int(ei.max())
    expected_max = shared.num_nodes - 1
    status = "✓" if max_node_id == expected_max else "✗ MISMATCH"
    print(f"  Max node id   : {max_node_id}  (expected {expected_max})  {status}")

    has_nan_ei = torch.isnan(ei.float()).any().item()
    has_nan_ea = torch.isnan(shared.edge_attr_static).any().item()
    print(f"  NaN in ei     : {'YES ✗' if has_nan_ei else 'No ✓'}")
    print(f"  NaN in ea_s   : {'YES ✗' if has_nan_ea else 'No ✓'}")

    cfg_ver = shared.config_dict.get("version", "unknown")
    print(f"  Config version: {cfg_ver}")
    node_feature_names = shared.config_dict.get("node_feature_names", [])
    if node_feature_names:
        print("  Node feats    :")
        for name in node_feature_names:
            print(f"    - {name}")


def inspect_sample(
    sample: PixelGraphSample,
    resolver: GraphResolver,
    label: str = "sample",
) -> None:
    print(f"\n{SEP}")
    print(f"PER-IMAGE SAMPLE  [{label}]")
    print(SEP)
    print(f"  graph_id          : {sample.graph_id}")
    print(f"  label             : {sample.label}")
    print(f"  split / usage     : {sample.split!r} / {sample.usage!r}")
    print(f"  node_features     : shape={tuple(sample.node_features.shape)}  dtype={sample.node_features.dtype}")
    print(f"  edge_attr_dynamic : shape={tuple(sample.edge_attr_dynamic.shape)}  dtype={sample.edge_attr_dynamic.dtype}")
    print("  node_feature_names:")
    for name in sample.node_feature_names:
        print(f"    - {name}")
    print(f"  dynamic_feat_names: {sample.dynamic_feature_names}")

    has_nan_nf = torch.isnan(sample.node_features).any().item()
    has_nan_ea = torch.isnan(sample.edge_attr_dynamic).any().item()
    has_inf_nf = torch.isinf(sample.node_features).any().item()
    has_inf_ea = torch.isinf(sample.edge_attr_dynamic).any().item()
    print(f"  NaN in node_feats : {'YES ✗' if has_nan_nf else 'No ✓'}")
    print(f"  NaN in edge_dyn   : {'YES ✗' if has_nan_ea else 'No ✓'}")
    print(f"  Inf in node_feats : {'YES ✗' if has_inf_nf else 'No ✓'}")
    print(f"  Inf in edge_dyn   : {'YES ✗' if has_inf_ea else 'No ✓'}")

    # Node feature stats
    nf = sample.node_features
    print(f"\n  Node feature stats (min / mean / max):")
    for i, name in enumerate(sample.node_feature_names):
        col = nf[:, i]
        print(f"    {name:15s}: {col.min().item():+.4f} / {col.mean().item():+.4f} / {col.max().item():+.4f}")

    # Dynamic edge attr stats
    ea = sample.edge_attr_dynamic
    print(f"\n  Dynamic edge attr stats (min / mean / max):")
    for i, name in enumerate(sample.dynamic_feature_names):
        col = ea[:, i]
        print(f"    {name:25s}: {col.min().item():+.4f} / {col.mean().item():+.4f} / {col.max().item():+.4f}")

    # Resolved graph
    print(f"\n{SEP}")
    print("RESOLVED GRAPH")
    print(SEP)
    resolved = resolver.resolve(sample)
    print(f"  edge_index  : {tuple(resolved.edge_index.shape)}")
    print(f"  edge_attr   : {tuple(resolved.edge_attr.shape)}  (static+dynamic)")
    print(f"  edge_feats  : {resolved.edge_feature_names}")
    print(f"  has_nan     : {'YES ✗' if resolved.has_nan() else 'No ✓'}")
    has_inf_resolved = torch.isinf(resolved.node_features).any().item() or torch.isinf(resolved.edge_attr).any().item()
    print(f"  has_inf     : {'YES ✗' if has_inf_resolved else 'No ✓'}")
    print(f"  {resolved}")


def inspect_split(
    reader: GraphRepositoryReader,
    shared: SharedGraphStructure,
    resolver: GraphResolver,
    split: str,
    chunk_idx: int,
    sample_idx: int,
) -> None:
    print(f"\n{SEP}")
    print(f"SPLIT: {split.upper()}")
    print(SEP)

    try:
        n_chunks = reader.num_chunks(split)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return

    n_samples = reader.num_samples(split)
    print(f"  Num chunks  : {n_chunks}")
    print(f"  Num samples : {n_samples if n_samples else '(see manifest)'}")

    # Load specified chunk
    chunk = reader.load_chunk(split, chunk_idx)
    print(f"  Chunk[{chunk_idx}] size: {len(chunk)} samples")

    if sample_idx >= len(chunk):
        print(f"  ✗ sample_idx={sample_idx} out of range for chunk size {len(chunk)}")
        return

    sample = chunk[sample_idx]
    inspect_sample(
        sample=sample,
        resolver=resolver,
        label=f"{split}/chunk_{chunk_idx:03d}[{sample_idx}]",
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    print(f"\n{'═'*65}")
    print(f"  GRAPH REPOSITORY INSPECTOR")
    print(f"  repo_root: {args.repo_root}")
    print(f"{'═'*65}")

    reader = GraphRepositoryReader(args.repo_root)
    manifest = reader.load_manifest()
    print(f"\nManifest version : {manifest.get('version', 'N/A')}")
    print(f"Built at         : {manifest.get('built_at', 'N/A')}")
    print(f"Chunk size       : {manifest.get('chunk_size', 'N/A')}")

    # Shared graph
    shared = reader.load_shared()
    inspect_shared(shared)

    resolver = GraphResolver(shared)

    # Which splits to inspect
    splits_to_check = [args.split] if args.split else reader.available_splits()
    if not splits_to_check:
        splits_to_check = ["train", "val", "test"]

    for split in splits_to_check:
        inspect_split(
            reader=reader,
            shared=shared,
            resolver=resolver,
            split=split,
            chunk_idx=args.chunk,
            sample_idx=args.sample,
        )

    print(f"\n{'═'*65}")
    print("Inspection complete.")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
