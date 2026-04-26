"""Precompute V3 hierarchical subgraph tensors (sub_x, sub_adj, sub_node_mask).

Input : pixel_motif_dataset_v2  +  graph_repo
Output: pixel_motif_dataset_v3_hierarchical

Each sample keeps all original keys and adds:
    sub_x          [K, Nmax, 7]   float32
    sub_adj        [K, Nmax, Nmax] uint8   (0/1 adjacency, cast to float in dataloader)
    sub_node_mask  [K, Nmax]       bool

sub_adj is built from a neighbor lookup dict (one-time build from shared edge_index),
NOT from the dense [2304x2304] adjacency matrix.  Cost per sample = K * Nmax * 8 ops.
Processing is done chunk-by-chunk to minimise I/O.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Core tensor builder
# ---------------------------------------------------------------------------

def _build_neighbor_dict(edge_index: torch.Tensor) -> dict[int, list[int]]:
    """Build adjacency neighbor list from shared edge_index. O(M)."""
    neighbors: dict[int, list[int]] = defaultdict(list)
    srcs = edge_index[0].tolist()
    dsts = edge_index[1].tolist()
    for src, dst in zip(srcs, dsts):
        neighbors[int(src)].append(int(dst))
    return neighbors


def build_sub_tensors(
    node_features: torch.Tensor,
    neighbors: dict[int, list[int]],
    node_indices: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded subgraph tensors for K subgraphs via neighbor lookup.

    Returns:
        sub_x          [K, Nmax, F]     float32
        sub_adj        [K, Nmax, Nmax]  uint8
        sub_node_mask  [K, Nmax]        bool
    """
    node_indices = torch.as_tensor(node_indices).long()
    node_mask = torch.as_tensor(node_mask).bool() & node_indices.ge(0)
    K, Nmax = node_indices.shape
    F = int(node_features.shape[1])

    sub_x = torch.zeros(K, Nmax, F, dtype=torch.float32)
    sub_adj = torch.zeros(K, Nmax, Nmax, dtype=torch.uint8)
    sub_node_mask_out = torch.zeros(K, Nmax, dtype=torch.bool)

    for k in range(K):
        valid_mask = node_mask[k]
        if not bool(valid_mask.any()):
            continue
        valid_pos = valid_mask.nonzero(as_tuple=True)[0].tolist()
        global_ids = [int(node_indices[k, li]) for li in valid_pos]
        n = len(global_ids)

        # sub_x: gather node features
        gid_t = torch.tensor(global_ids, dtype=torch.long)
        sub_x[k, :n] = node_features[gid_t]
        sub_node_mask_out[k, :n] = True

        # sub_adj: neighbor lookup  O(n * 8)
        local_map = {gid: li for li, gid in enumerate(global_ids)}
        for li, u in enumerate(global_ids):
            for v in neighbors.get(u, []):
                lv = local_map.get(v)
                if lv is not None:
                    sub_adj[k, li, lv] = 1

    return sub_x, sub_adj, sub_node_mask_out


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(
    split: str,
    pixel_motif_dir: Path,
    graph_repo_path: Path,
    neighbors: dict[int, list[int]],
    out_dir: Path,
    log_every: int = 500,
) -> None:
    src_pt = pixel_motif_dir / f"{split}_pixel_motif.pt"
    print(f"\n[{split}] Loading {src_pt} ...", flush=True)
    samples = torch.load(src_pt, map_location="cpu", weights_only=False)
    print(f"[{split}] {len(samples)} samples loaded.", flush=True)

    # Build graph_id -> sample_idx lookup
    gid_to_idx: dict[int, int] = {int(s["graph_id"]): i for i, s in enumerate(samples)}

    # Copy samples into output list (shallow copy, we will add keys)
    out_samples: list[dict] = [dict(s) for s in samples]

    chunk_dir = graph_repo_path / split
    chunk_files = sorted(chunk_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pt files found in {chunk_dir}")

    processed = 0
    t0 = time.time()

    for chunk_file in chunk_files:
        chunk_data = torch.load(chunk_file, map_location="cpu", weights_only=False)
        for graph_sample in chunk_data:
            gid = int(graph_sample.graph_id)
            idx = gid_to_idx.get(gid)
            if idx is None:
                continue  # graph not in this split's pixel_motif dataset

            s = samples[idx]
            node_features = graph_sample.node_features.float()
            node_indices = torch.as_tensor(s["node_indices"]).long()
            node_mask = torch.as_tensor(s["node_mask"]).bool()

            sub_x, sub_adj, sub_node_mask = build_sub_tensors(
                node_features, neighbors, node_indices, node_mask
            )
            out_samples[idx]["sub_x"] = sub_x            # float32
            out_samples[idx]["sub_adj"] = sub_adj         # uint8
            out_samples[idx]["sub_node_mask"] = sub_node_mask  # bool

            processed += 1
            if processed % log_every == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                remain = (len(samples) - processed) / max(rate, 1e-6)
                print(
                    f"  [{split}] {processed}/{len(samples)}  "
                    f"({rate:.1f} s/s  ETA {remain/60:.1f} min)",
                    flush=True,
                )

    # Sanity check
    missing = [i for i, s in enumerate(out_samples) if "sub_x" not in s]
    if missing:
        print(
            f"  [warn] {len(missing)}/{len(out_samples)} samples missing V3 tensors "
            f"(graph_ids not found in graph_repo chunks).",
            flush=True,
        )

    out_pt = out_dir / f"{split}_pixel_motif.pt"
    print(f"  [{split}] Saving → {out_pt} ...", flush=True)
    torch.save(out_samples, out_pt)
    size_gb = out_pt.stat().st_size / 1024 ** 3
    total_t = time.time() - t0
    print(
        f"  [{split}] Done  {processed}/{len(samples)} samples  "
        f"{size_gb:.2f} GB  {total_t/60:.1f} min",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute V3 hierarchical subgraph tensors from pixel_motif_dataset_v2 + graph_repo."
    )
    parser.add_argument(
        "--pixel_motif_dataset_path",
        required=True,
        help="Path to pixel_motif_dataset_v2 directory.",
    )
    parser.add_argument(
        "--graph_repo_path",
        required=True,
        help="Path to graph_repo directory.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for pixel_motif_dataset_v3_hierarchical.",
    )
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()

    pixel_motif_dir = Path(args.pixel_motif_dataset_path)
    graph_repo_path = Path(args.graph_repo_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shared graph structure and build neighbor dict once
    shared_path = graph_repo_path / "shared" / "shared_graph.pt"
    print(f"Loading shared graph: {shared_path}", flush=True)
    shared = torch.load(shared_path, map_location="cpu", weights_only=False)
    neighbors = _build_neighbor_dict(shared.edge_index)
    num_edges = sum(len(v) for v in neighbors.values())
    print(
        f"Neighbor dict: {len(neighbors)} nodes, {num_edges} directed edges",
        flush=True,
    )

    # Copy meta.pt from V2
    meta_src = pixel_motif_dir / "meta.pt"
    if meta_src.exists():
        shutil.copy2(meta_src, out_dir / "meta.pt")
        print(f"Copied meta.pt → {out_dir / 'meta.pt'}", flush=True)

    t_total = time.time()
    for split in args.splits:
        process_split(
            split, pixel_motif_dir, graph_repo_path, neighbors, out_dir, args.log_every
        )

    total_gb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1024 ** 3
    print(
        f"\nV3 cache complete: {out_dir}  "
        f"({total_gb:.2f} GB total  {(time.time()-t_total)/60:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
