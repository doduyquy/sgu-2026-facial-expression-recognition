"""Precompute pixel-preserving motif-selected image-level dataset."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.motif.motif_scoring import check_finite_tensor
from src.motif_v2.io import load_pixel_motif_bank
from src.motif_v2.matching import greedy_select_with_coverage
from src.motif_v2.topology import (
    RICH_MOTIF_EDGE_ATTR_NAMES,
    build_directed_knn_edges,
    build_directed_knn_rich_edges,
)


SPATIAL_MOTIF_EDGE_ATTR_NAMES = ["dx", "dy", "dist"]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _pad_selected_nodes(selected_indices: torch.Tensor, topologies: List[dict], top_k: int, max_nodes: int):
    node_indices = torch.full((top_k, max_nodes), -1, dtype=torch.long)
    node_mask = torch.zeros((top_k, max_nodes), dtype=torch.bool)
    for out_idx, cand_idx in enumerate(selected_indices.tolist()):
        if cand_idx < 0:
            continue
        nodes = topologies[int(cand_idx)]["node_indices"].long()
        n = min(max_nodes, int(nodes.numel()))
        node_indices[out_idx, :n] = nodes[:n]
        node_mask[out_idx, :n] = True
    return node_indices, node_mask


def _process_split(
    split: str,
    candidate_dir: Path,
    out_dir: Path,
    bank,
    topologies: List[dict],
    top_k: int,
    knn_k: int,
    beta: float,
    gamma: float,
    eta: float,
    diversity_sigma: float,
    max_nodes: int,
    edge_attr_mode: str,
) -> List[dict]:
    path = candidate_dir / f"{split}_pixel_candidates.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    samples = _torch_load(path)
    out_samples = []
    label_hist = Counter()
    matched_hist = Counter()
    coverage_hist = Counter()
    motif_score_sum = defaultdict(lambda: torch.zeros(bank.num_classes))
    motif_score_count = Counter()

    print("\n" + "=" * 80)
    print(f"[{split}] Pixel motif selection")
    print("=" * 80)
    for idx, sample in enumerate(samples):
        selected = greedy_select_with_coverage(
            x=sample["x"],
            centers=sample["centers"],
            bbox=sample["bbox"],
            coverage_cell=sample["coverage_cell"],
            bank=bank,
            top_k=top_k,
            beta=beta,
            gamma=gamma,
            eta=eta,
            diversity_sigma=diversity_sigma,
            mask=sample.get("mask"),
        )
        if edge_attr_mode == "rich":
            edge_index, edge_attr = build_directed_knn_rich_edges(
                centers=selected["centers"],
                bbox=selected["bbox"],
                descriptors=selected["x"],
                match_scores=selected["match_scores"],
                matched_class=selected["matched_class"],
                matched_motif_id=selected["matched_motif_id"],
                matched_disc_score=selected["matched_disc_score"],
                coverage_cell=selected["coverage_cell"],
                knn_k=knn_k,
            )
        else:
            edge_index, edge_attr = build_directed_knn_edges(selected["centers"], knn_k=knn_k)
        node_indices, node_mask = _pad_selected_nodes(
            selected["selected_indices"],
            topologies=topologies,
            top_k=top_k,
            max_nodes=max_nodes,
        )
        for key in ["x", "centers", "bbox", "match_scores", "matched_disc_score", "motif_score_vector"]:
            check_finite_tensor(f"{split}[{idx}].{key}", selected[key])

        valid = selected["mask"].bool()
        label = int(sample["label"])
        label_hist[label] += 1
        motif_score_sum[label] += selected["motif_score_vector"]
        motif_score_count[label] += 1
        for cls in selected["matched_class"][valid].tolist():
            matched_hist[int(cls)] += 1
        for cell in selected["coverage_cell"][valid].tolist():
            coverage_hist[int(cell)] += 1

        out_samples.append(
            {
                "graph_id": int(sample["graph_id"]),
                "label": label,
                "x": selected["x"],
                "mask": selected["mask"],
                "centers": selected["centers"],
                "bbox": selected["bbox"],
                "selected_indices": selected["selected_indices"],
                "node_indices": node_indices,
                "node_mask": node_mask,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "match_scores": selected["match_scores"],
                "matched_class": selected["matched_class"],
                "matched_motif_id": selected["matched_motif_id"],
                "matched_disc_score": selected["matched_disc_score"],
                "motif_score_vector": selected["motif_score_vector"],
                "coverage_cell": selected["coverage_cell"],
            }
        )
        if (idx + 1) % 2000 == 0 or idx + 1 == len(samples):
            print(f"  [{split}] {idx+1:6d}/{len(samples)}", flush=True)

    out_path = out_dir / f"{split}_pixel_motif.pt"
    torch.save(out_samples, out_path)
    print(f"[{split}] saved -> {out_path} ({out_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"[{split}] label_hist={dict(sorted(label_hist.items()))}")
    print(f"[{split}] matched_hist={dict(sorted(matched_hist.items()))}")
    print(f"[{split}] coverage_hist={dict(sorted(coverage_hist.items()))}")
    print(f"[{split}] avg motif score by label:")
    for label in sorted(motif_score_count):
        avg = motif_score_sum[label] / max(1, motif_score_count[label])
        print(f"  label {label}: " + ", ".join(f"{v:.3f}" for v in avg.tolist()))
    return out_samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_dir", default="artifacts/pixel_candidate_subgraphs_v2")
    p.add_argument("--motif_bank_path", default="artifacts/pixel_motif_bank_v2/pixel_motif_bank.pt")
    p.add_argument("--out_dir", default="artifacts/pixel_motif_dataset_v2")
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--knn_k", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--diversity_sigma", type=float, default=0.12)
    p.add_argument(
        "--edge_attr_mode",
        choices=["spatial", "rich"],
        default="spatial",
        help="spatial keeps [dx,dy,dist]; rich adds bbox/descriptor/motif relation features.",
    )
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = load_pixel_motif_bank(args.motif_bank_path)
    candidate_meta = _torch_load(candidate_dir / "meta.pt")
    topologies = candidate_meta["candidate_topologies"]
    max_nodes = int(candidate_meta["max_nodes_per_candidate"])

    print("=" * 80)
    print("Precompute Pixel-preserving Motif Dataset V2")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k:<24}: {v}")
    print(f"descriptor_dim          : {bank.descriptor_dim}")
    print(f"max_nodes_per_candidate : {max_nodes}")

    split_counts = {}
    for split in args.splits:
        samples = _process_split(
            split=split,
            candidate_dir=candidate_dir,
            out_dir=out_dir,
            bank=bank,
            topologies=topologies,
            top_k=args.top_k,
            knn_k=args.knn_k,
            beta=args.beta,
            gamma=args.gamma,
            eta=args.eta,
            diversity_sigma=args.diversity_sigma,
            max_nodes=max_nodes,
            edge_attr_mode=args.edge_attr_mode,
        )
        split_counts[split] = len(samples)

    edge_attr_names = (
        RICH_MOTIF_EDGE_ATTR_NAMES
        if args.edge_attr_mode == "rich"
        else SPATIAL_MOTIF_EDGE_ATTR_NAMES
    )

    meta = {
        "descriptor_dim": bank.descriptor_dim,
        "top_k": int(args.top_k),
        "knn_k": int(args.knn_k),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "eta": float(args.eta),
        "diversity_sigma": float(args.diversity_sigma),
        "edge_attr_mode": args.edge_attr_mode,
        "edge_attr_names": list(edge_attr_names),
        "edge_attr_dim": len(edge_attr_names),
        "num_classes": bank.num_classes,
        "class_names": bank.emotion_names,
        "candidate_dir": str(candidate_dir),
        "motif_bank_path": str(args.motif_bank_path),
        "max_nodes_per_candidate": max_nodes,
        "height": candidate_meta.get("height"),
        "width": candidate_meta.get("width"),
        "coverage_grid": candidate_meta.get("coverage_grid"),
        "node_feature_names": candidate_meta.get("node_feature_names"),
        "edge_feature_names": candidate_meta.get("edge_feature_names"),
        "graph_config_version": candidate_meta.get("graph_config_version"),
        "split_counts": split_counts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(meta, out_dir / "meta.pt")
    print(f"meta saved -> {out_dir / 'meta.pt'}")
    print("DONE")


if __name__ == "__main__":
    main()
