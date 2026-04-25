"""
scripts/precompute_subgraph_graph_dataset.py

OPTIMIZATION KEY:
    Mọi ảnh FER-2013 đều share CÙNG graph topology (shared edge_index, 48x48 grid).
    → Precompute subgraph node-sets và edge-masks CHỈ 1 LẦN.
    → Với mỗi ảnh, chỉ cần index vào node_features → tính descriptor.
    → Nhanh hơn bản gốc ~50-100x.

Usage (PowerShell):
    python scripts/precompute_subgraph_graph_dataset.py --repo_root artifacts/graph_repo_v2 --out_dir artifacts/subgraph_graph_dataset_v2

    # Debug nhanh:
    python scripts/precompute_subgraph_graph_dataset.py --repo_root artifacts/graph_repo_v2 --out_dir artifacts/subgraph_graph_dataset_debug --num_subgraphs 16 --max_candidates 32 --splits train
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.graph_types import PixelGraphSample, SharedGraphStructure
from src.graph.subgraph_descriptor import infer_descriptor_dim


# ═══════════════════════════════════════════════════════════════════════
# Topology helpers — chạy 1 LẦN cho toàn split
# ═══════════════════════════════════════════════════════════════════════

def _build_adj(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
    return adj


def _bfs_nodes(seed: int, radius: int, adj: List[List[int]],
               max_nodes: Optional[int] = None) -> List[int]:
    visited = {seed}
    queue = deque([(seed, 0)])
    result = []
    while queue:
        cur, dist = queue.popleft()
        result.append(cur)
        if max_nodes and len(result) >= max_nodes:
            break
        if dist >= radius:
            continue
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return sorted(result)


def _precompute_topology(
    edge_index: torch.Tensor,  # [2, M]
    num_nodes: int,
    height: int,
    width: int,
    seed_stride: int,
    radius: int,
    max_candidates: int,
    num_subgraphs: int,
) -> List[Dict]:
    """
    Tính ONCE cho toàn split:
    - adjacency list
    - seed nodes
    - BFS node-sets per seed
    - edge masks per seed (để tính edge_attr_sub)

    Trả về list of SubgraphTopo dicts, mỗi dict:
        node_ids   : LongTensor [n_nodes_sub]
        edge_index_sub : LongTensor [2, E_sub]  (local re-indexed)
        edge_mask  : BoolTensor [M]
    """
    adj = _build_adj(edge_index, num_nodes)

    # Seeds
    seeds = []
    for y in range(0, height, seed_stride):
        for x in range(0, width, seed_stride):
            seeds.append(y * width + x)
    seeds = seeds[:max_candidates]

    src_all = edge_index[0]
    dst_all = edge_index[1]

    topos = []
    seen_sigs = set()

    for seed in seeds:
        node_ids_list = _bfs_nodes(seed, radius, adj)
        sig = tuple(node_ids_list)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        node_ids = torch.tensor(node_ids_list, dtype=torch.long)
        local_idx = {int(v): i for i, v in enumerate(node_ids_list)}

        # Edge mask: both endpoints in subgraph
        node_set = set(node_ids_list)
        keep = torch.tensor(
            [int(s.item()) in node_set and int(d.item()) in node_set
             for s, d in zip(src_all, dst_all)],
            dtype=torch.bool,
        )

        ei_sub = edge_index[:, keep]
        if ei_sub.numel() > 0:
            rs = [local_idx[int(v)] for v in ei_sub[0].tolist()]
            rd = [local_idx[int(v)] for v in ei_sub[1].tolist()]
            ei_sub = torch.tensor([rs, rd], dtype=torch.long)
        else:
            ei_sub = torch.empty((2, 0), dtype=torch.long)

        topos.append({
            "node_ids"      : node_ids,
            "edge_index_sub": ei_sub,
            "edge_mask"     : keep,
            "num_nodes"     : len(node_ids_list),
        })

        if len(topos) >= max_candidates:
            break

    # Chọn top num_subgraphs theo subgraph size (diverse coverage)
    # Đơn giản: lấy đều nhau theo stride
    if len(topos) > num_subgraphs:
        step = len(topos) / num_subgraphs
        topos = [topos[int(i * step)] for i in range(num_subgraphs)]

    return topos


# ═══════════════════════════════════════════════════════════════════════
# Descriptor từ features (per-image, rất nhanh)
# ═══════════════════════════════════════════════════════════════════════

def _descriptor_from_features(
    node_features: torch.Tensor,   # [N, d]
    edge_attr: torch.Tensor,        # [M, S+D]
    topo: Dict,
) -> torch.Tensor:
    """Tính descriptor vector [D] cho 1 subgraph của 1 ảnh."""
    nf = node_features[topo["node_ids"]].float()   # [n_sub, d]
    ea = edge_attr[topo["edge_mask"]].float()       # [E_sub, feat]

    # Node stats: mean, std, min, max  → 4*d
    if nf.shape[0] > 0:
        nm = nf.mean(0); ns = nf.std(0, unbiased=False)
        nmin = nf.min(0).values; nmax = nf.max(0).values
    else:
        nm = ns = nmin = nmax = torch.zeros(nf.shape[1])

    n_nodes = float(topo["num_nodes"])
    n_edges = float(topo["edge_index_sub"].shape[1])
    denom   = n_nodes * (n_nodes - 1)
    density = float(n_edges / denom) if denom > 0 else 0.0
    struct  = torch.tensor([n_nodes, n_edges, density])

    # Edge stats: mean, std  → 2*e_feat
    if ea.shape[0] > 0 and ea.shape[1] > 0:
        em = ea.mean(0); es = ea.std(0, unbiased=False)
    else:
        ef = ea.shape[1] if ea.ndim == 2 else 0
        em = es = torch.zeros(ef)

    desc = torch.cat([nm, ns, nmin, nmax, struct, em, es])
    return torch.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════════════
# Spatial KNN edge_index giữa các subgraph
# ═══════════════════════════════════════════════════════════════════════

def _subgraph_center(node_ids: torch.Tensor, node_features: torch.Tensor,
                     height: int, width: int) -> Tuple[float, float]:
    nf = node_features[node_ids].float()
    if nf.shape[1] >= 3:
        return float(nf[:, 1].mean()), float(nf[:, 2].mean())
    seed = int(node_ids[0])
    return (seed % width) / max(width - 1, 1), (seed // width) / max(height - 1, 1)


def _build_knn_edges(centers: torch.Tensor, mask: torch.Tensor,
                     knn_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    K = centers.shape[0]
    valid = torch.where(mask.bool())[0]
    nv = len(valid)
    if nv < 2:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1))

    vc = centers[valid]
    diff = vc.unsqueeze(0) - vc.unsqueeze(1)
    dists = diff.pow(2).sum(-1).sqrt()
    dists.fill_diagonal_(float("inf"))

    k = min(knn_k, nv - 1)
    _, knn = dists.topk(k, dim=1, largest=False)

    src_l, dst_l, d_l = [], [], []
    seen = set()
    for i in range(nv):
        oi = int(valid[i])
        for jl in knn[i].tolist():
            oj = int(valid[jl])
            key = (min(oi, oj), max(oi, oj))
            if key in seen:
                continue
            seen.add(key)
            d = float(dists[i, jl])
            src_l += [oi, oj]; dst_l += [oj, oi]; d_l += [d, d]

    if not src_l:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1))

    ei = torch.tensor([src_l, dst_l], dtype=torch.long)
    ea = torch.tensor(d_l, dtype=torch.float32).unsqueeze(-1)
    ea = ea / ea.max().clamp_min(1e-6)
    return ei, ea


# ═══════════════════════════════════════════════════════════════════════
# Process one split
# ═══════════════════════════════════════════════════════════════════════

def _process_split(
    repo_root: str, split: str,
    num_subgraphs: int, subgraph_radius: int, seed_stride: int,
    max_candidates: int, knn_k: int, log_every: int,
) -> List[dict]:

    # ── Load shared topology ONCE ────────────────────────────────────
    reader   = GraphRepositoryReader(repo_root)
    shared: SharedGraphStructure = reader.load_shared()   # <-- đúng tên
    resolver = GraphResolver(shared)

    height, width = shared.height, shared.width
    num_nodes     = shared.num_nodes

    # Đếm số sample từ manifest
    n = reader.num_samples(split)
    if n is None:
        # Fallback: đếm qua chunk paths
        from data.graph_repository import CHUNK_PATTERN
        chunks = reader.chunk_paths(split)
        # load 1 chunk để lấy size
        first_chunk = torch.load(chunks[0], map_location="cpu", weights_only=False)
        n = (len(chunks) - 1) * len(first_chunk)  # ước tính

    # Infer dims từ sample đầu tiên
    first_raw: PixelGraphSample = next(reader.iter_split(split))
    g0 = resolver.resolve(first_raw)
    node_feat_dim = g0.num_node_features
    edge_feat_dim = g0.num_edge_features
    desc_dim      = infer_descriptor_dim(node_feat_dim, edge_feat_dim)

    print(f"\n[{split}] samples | desc_dim={desc_dim} | K={num_subgraphs} | knn_k={knn_k}")
    print(f"[{split}] Precomputing topology ONCE ... ", end="", flush=True)

    t_topo = time.time()
    topos = _precompute_topology(
        edge_index=shared.edge_index,
        num_nodes=num_nodes,
        height=height, width=width,
        seed_stride=seed_stride,
        radius=subgraph_radius,
        max_candidates=max_candidates,
        num_subgraphs=num_subgraphs,
    )
    print(f"done in {time.time()-t_topo:.1f}s  ({len(topos)} subgraphs)", flush=True)

    K = len(topos)
    samples = []
    t0 = time.time()
    idx = 0

    for raw_sample in reader.iter_split(split):
        graph = resolver.resolve(raw_sample)
        nf = graph.node_features   # [N, d]
        ea = graph.edge_attr       # [M, f]

        x    = torch.zeros((K, desc_dim), dtype=torch.float32)
        mask = torch.zeros(K,             dtype=torch.float32)
        centers = torch.zeros((K, 2),    dtype=torch.float32)

        for k, topo in enumerate(topos):
            x[k]    = _descriptor_from_features(nf, ea, topo)
            mask[k] = 1.0
            cx, cy  = _subgraph_center(topo["node_ids"], nf, height, width)
            centers[k, 0] = cx; centers[k, 1] = cy

        ei, ea_knn = _build_knn_edges(centers, mask, knn_k)

        samples.append({
            "graph_id"  : graph.graph_id,
            "label"     : graph.label,
            "x"         : x,
            "mask"      : mask,
            "edge_index": ei,
            "edge_attr" : ea_knn,
            "centers"   : centers,
        })

        if (idx + 1) % log_every == 0 or (n is not None and (idx + 1) == n):
            elapsed = time.time() - t0
            rate    = (idx + 1) / max(elapsed, 1e-6)
            n_str   = str(n) if n is not None else "?"
            eta_str = f"{(n - idx - 1) / max(rate, 1e-6):5.0f}s" if n is not None else "?"
            print(
                f"  [{split}] {idx+1:6d}/{n_str}  |"
                f"  {elapsed:6.1f}s  |"
                f"  {rate:5.1f} samp/s  |"
                f"  ETA {eta_str}",
                flush=True,
            )
        idx += 1

    print(f"  [{split}] Done {idx} samples in {time.time()-t0:.1f}s", flush=True)
    return samples


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root",        default="artifacts/graph_repo_v2")
    p.add_argument("--out_dir",          default="artifacts/subgraph_graph_dataset_v2")
    p.add_argument("--num_subgraphs",    type=int, default=32)
    p.add_argument("--subgraph_radius",  type=int, default=1)
    p.add_argument("--seed_stride",      type=int, default=4)
    p.add_argument("--max_candidates",   type=int, default=64)
    p.add_argument("--knn_k",            type=int, default=4)
    p.add_argument("--splits",           nargs="+", default=["train", "val", "test"])
    p.add_argument("--log_every",        type=int, default=1000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Precompute Subgraph Graph Dataset  (fast topology-shared)")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
    print("=" * 60)

    reader = GraphRepositoryReader(args.repo_root)
    shared = reader.load_shared()
    shared_cfg = shared.config_dict if isinstance(shared.config_dict, dict) else {}
    meta_split = args.splits[0]
    first_sample: PixelGraphSample = next(reader.iter_split(meta_split))
    resolver = GraphResolver(shared)
    first_graph = resolver.resolve(first_sample)
    descriptor_dim = infer_descriptor_dim(first_graph.num_node_features, first_graph.num_edge_features)

    t_total = time.time()
    for split in args.splits:
        samples = _process_split(
            repo_root=args.repo_root, split=split,
            num_subgraphs=args.num_subgraphs,
            subgraph_radius=args.subgraph_radius,
            seed_stride=args.seed_stride,
            max_candidates=args.max_candidates,
            knn_k=args.knn_k,
            log_every=args.log_every,
        )
        out_path = out_dir / f"{split}_subgraph_graph.pt"
        torch.save(samples, out_path)
        mb = out_path.stat().st_size / 1024 ** 2
        print(f"  Saved → {out_path}  ({mb:.1f} MB)")

    meta = {
        **vars(args),
        "descriptor_dim": descriptor_dim,
        "node_feature_names": list(first_graph.node_feature_names),
        "edge_feature_names": list(first_graph.edge_feature_names),
        "graph_config_version": shared_cfg.get("version", "unknown"),
        "graph_config": shared_cfg,
    }
    torch.save(meta, out_dir / "meta.pt")
    print(f"\n  Total: {time.time()-t_total:.1f}s  |  DONE.")


if __name__ == "__main__":
    main()
