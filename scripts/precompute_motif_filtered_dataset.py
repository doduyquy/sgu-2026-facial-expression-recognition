"""Precompute motif-filtered image-level dataset from subgraph descriptors."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.motif.motif_io import load_motif_bank
from src.motif.motif_matching import select_topk_by_motif
from src.motif.motif_scoring import check_finite_tensor


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_directed_knn_edges(centers: torch.Tensor, knn_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-size directed KNN edges over selected centers."""
    centers = centers.float()
    K = int(centers.shape[0])
    if K <= 1:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float32)

    k = max(1, min(int(knn_k), K - 1))
    diff = centers.unsqueeze(1) - centers.unsqueeze(0)  # [K, K, 2], dst - src at [src, dst]
    dist = diff.pow(2).sum(dim=-1).sqrt()
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)

    src_list, dst_list, attrs = [], [], []
    for src in range(K):
        for dst in nn_idx[src].tolist():
            dx = float(centers[dst, 0] - centers[src, 0])
            dy = float(centers[dst, 1] - centers[src, 1])
            d = float((dx * dx + dy * dy) ** 0.5)
            src_list.append(src)
            dst_list.append(int(dst))
            attrs.append([dx, dy, d])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def _process_split(
    split: str,
    input_dir: Path,
    out_dir: Path,
    motif_bank,
    top_k: int,
    knn_k: int,
    beta: float,
) -> List[dict]:
    in_path = input_dir / f"{split}_subgraph_graph.pt"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input split: {in_path}")

    samples = _torch_load(in_path)
    if not isinstance(samples, list) or len(samples) == 0:
        raise RuntimeError(f"Expected non-empty sample list in {in_path}")

    print("\n" + "=" * 72)
    print(f"[{split}] Precomputing motif-filtered dataset")
    print(f"[{split}] input samples: {len(samples)}")

    out_samples = []
    label_hist = Counter()
    matched_hist = Counter()
    motif_score_sums = defaultdict(lambda: torch.zeros(motif_bank.num_classes))
    motif_score_counts = Counter()

    for idx, sample in enumerate(samples):
        x = torch.as_tensor(sample["x"]).float()
        centers = torch.as_tensor(sample["centers"]).float()
        mask = torch.as_tensor(sample.get("mask", torch.ones(x.shape[0]))).bool()

        selected = select_topk_by_motif(
            x=x,
            centers=centers,
            motif_bank=motif_bank,
            top_k=top_k,
            beta=beta,
            mask=mask,
        )
        edge_index, edge_attr = _build_directed_knn_edges(selected["centers"], knn_k=knn_k)
        if edge_index.numel() > 0:
            max_idx = int(edge_index.max().item())
            min_idx = int(edge_index.min().item())
            if min_idx < 0 or max_idx >= top_k:
                raise RuntimeError(f"[{split}] edge_index out of range at sample {idx}")

        for key in ["x", "centers", "match_scores", "matched_disc_score", "motif_score_vector"]:
            check_finite_tensor(f"{split}[{idx}].{key}", selected[key])
        check_finite_tensor(f"{split}[{idx}].edge_attr", edge_attr)

        label = int(sample["label"])
        valid = selected["mask"].bool()
        for cls in selected["matched_class"][valid].tolist():
            matched_hist[int(cls)] += 1
        label_hist[label] += 1
        motif_score_sums[label] += selected["motif_score_vector"]
        motif_score_counts[label] += 1

        out_samples.append(
            {
                "graph_id": int(sample["graph_id"]),
                "label": label,
                "x": selected["x"],
                "mask": selected["mask"],
                "centers": selected["centers"],
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "match_scores": selected["match_scores"],
                "matched_class": selected["matched_class"],
                "matched_motif_id": selected["matched_motif_id"],
                "matched_disc_score": selected["matched_disc_score"],
                "motif_score_vector": selected["motif_score_vector"],
            }
        )

        if (idx + 1) % 2000 == 0 or (idx + 1) == len(samples):
            print(f"[{split}] {idx + 1:6d}/{len(samples)}", flush=True)

    out_path = out_dir / f"{split}_motif_filtered.pt"
    torch.save(out_samples, out_path)

    s0 = out_samples[0]
    print(f"[{split}] saved: {out_path} ({out_path.stat().st_size / 1024 ** 2:.2f} MB)")
    print(f"[{split}] x shape: {tuple(s0['x'].shape)}")
    print(f"[{split}] edge_index shape: {tuple(s0['edge_index'].shape)}")
    print(f"[{split}] edge_attr shape: {tuple(s0['edge_attr'].shape)}")
    print(f"[{split}] matched_class distribution: {dict(sorted(matched_hist.items()))}")
    print(f"[{split}] label distribution: {dict(sorted(label_hist.items()))}")
    print(f"[{split}] average motif_score_vector by label:")
    for label in sorted(motif_score_counts):
        avg = motif_score_sums[label] / max(1, motif_score_counts[label])
        avg_str = ", ".join(f"{v:.3f}" for v in avg.tolist())
        print(f"  label {label}: [{avg_str}]")

    return out_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="artifacts/subgraph_graph_dataset")
    parser.add_argument("--motif_bank_path", default="artifacts/motif_bank_v1/motif_bank.pt")
    parser.add_argument("--out_dir", default="artifacts/motif_filtered_dataset_v1")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--knn_k", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    motif_bank = load_motif_bank(args.motif_bank_path)

    print("=" * 72)
    print("Precompute Motif-Filtered Dataset")
    print("=" * 72)
    for key, value in vars(args).items():
        print(f"{key:<24}: {value}")
    print(f"descriptor_dim          : {motif_bank.descriptor_dim}")
    print(f"num_classes             : {motif_bank.num_classes}")

    input_meta_path = input_dir / "meta.pt"
    input_meta = _torch_load(input_meta_path) if input_meta_path.exists() else {}

    first_split_samples = None
    for split in args.splits:
        first_split_samples = _process_split(
            split=split,
            input_dir=input_dir,
            out_dir=out_dir,
            motif_bank=motif_bank,
            top_k=args.top_k,
            knn_k=args.knn_k,
            beta=args.beta,
        )

    meta = {
        "descriptor_dim": motif_bank.descriptor_dim,
        "top_k": int(args.top_k),
        "knn_k": int(args.knn_k),
        "beta": float(args.beta),
        "num_classes": motif_bank.num_classes,
        "motif_bank_path": str(args.motif_bank_path),
        "input_dir": str(args.input_dir),
        "node_feature_names": input_meta.get("node_feature_names"),
        "graph_config_version": input_meta.get("graph_config_version"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "class_names": motif_bank.emotion_names,
        "splits": list(args.splits),
    }
    torch.save(meta, out_dir / "meta.pt")
    print("\n[Output]")
    print(f"meta.pt: {out_dir / 'meta.pt'}")
    print("DONE")


if __name__ == "__main__":
    main()
