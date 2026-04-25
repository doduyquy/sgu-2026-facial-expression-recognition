"""Build pixel-preserving motif bank from train candidate subgraphs only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.motif.motif_scoring import (
    check_finite_tensor,
    compute_discriminative_score,
    compute_inter_score,
    compute_intra_score,
    cosine_similarity_matrix,
)
from src.motif_v2.io import save_pixel_motif_bank
from src.motif_v2.types import PixelMotifBank, PixelMotifPrototype


EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cluster_descriptors(desc: torch.Tensor, n_clusters: int, seed: int, batch_size: int):
    n_clusters = max(1, min(int(n_clusters), int(desc.shape[0])))
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=int(batch_size),
            random_state=int(seed),
            n_init=3,
            max_iter=100,
            reassignment_ratio=0.01,
        )
        labels_np = km.fit_predict(desc.numpy())
        return torch.from_numpy(km.cluster_centers_).float(), torch.from_numpy(labels_np).long()
    except Exception as exc:
        print(f"[WARN] MiniBatchKMeans unavailable/failed ({exc}); using torch fallback.")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        init = torch.randperm(desc.shape[0], generator=gen)[:n_clusters]
        centers = desc[init].clone()
        labels = torch.zeros(desc.shape[0], dtype=torch.long)
        for _ in range(50):
            labels = torch.cdist(desc, centers).argmin(dim=1)
            for k in range(n_clusters):
                members = desc[labels == k]
                if members.numel() > 0:
                    centers[k] = members.mean(dim=0)
        return centers.float(), labels.long()


def _collect_sampled_descriptors(
    samples: List[dict],
    num_classes: int,
    max_subgraphs_per_class: int,
    seed: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[Tuple[int, int]]], List[int]]:
    buckets: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}
    refs: Dict[int, List[Tuple[int, int]]] = {c: [] for c in range(num_classes)}
    image_counts = [0 for _ in range(num_classes)]

    for sample_idx, sample in enumerate(samples):
        label = int(sample["label"])
        image_counts[label] += 1
        x = torch.as_tensor(sample["x"]).float()
        mask = torch.as_tensor(sample.get("mask", torch.ones(x.shape[0]))).bool()
        for cand_idx in torch.where(mask)[0].tolist():
            buckets[label].append(x[int(cand_idx)].view(1, -1))
            refs[label].append((sample_idx, int(cand_idx)))

    out_desc: Dict[int, torch.Tensor] = {}
    out_refs: Dict[int, List[Tuple[int, int]]] = {}
    for class_id in range(num_classes):
        desc = torch.cat(buckets[class_id], dim=0) if buckets[class_id] else torch.empty((0, samples[0]["x"].shape[1]))
        ref_list = refs[class_id]
        if max_subgraphs_per_class > 0 and desc.shape[0] > max_subgraphs_per_class:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) + class_id * 1009)
            idx = torch.randperm(desc.shape[0], generator=gen)[:max_subgraphs_per_class]
            out_desc[class_id] = desc[idx].contiguous()
            out_refs[class_id] = [ref_list[int(i)] for i in idx.tolist()]
        else:
            out_desc[class_id] = desc.contiguous()
            out_refs[class_id] = ref_list
    return out_desc, out_refs, image_counts


def _make_exemplars(
    center: torch.Tensor,
    class_desc: torch.Tensor,
    class_refs: List[Tuple[int, int]],
    train_samples: List[dict],
    topologies: List[dict],
    top_n: int,
) -> List[dict]:
    sim = cosine_similarity_matrix(center.view(1, -1), class_desc).squeeze(0)
    k = max(1, min(int(top_n), int(sim.numel())))
    vals, idxs = torch.topk(sim, k=k, largest=True)
    exemplars = []
    for rank, (value, idx) in enumerate(zip(vals.tolist(), idxs.tolist())):
        sample_idx, cand_idx = class_refs[int(idx)]
        sample = train_samples[int(sample_idx)]
        topo = topologies[int(cand_idx)]
        exemplars.append(
            {
                "rank": int(rank),
                "graph_id": int(sample["graph_id"]),
                "label": int(sample["label"]),
                "sample_index": int(sample_idx),
                "subgraph_id": int(cand_idx),
                "candidate_id": int(cand_idx),
                "node_indices": topo["node_indices"].clone().long(),
                "edge_index_sub": topo["edge_index_sub"].clone().long(),
                "center": torch.as_tensor(sample["centers"][cand_idx]).float(),
                "bbox": torch.as_tensor(sample["bbox"][cand_idx]).float(),
                "coverage_cell": int(sample["coverage_cell"][cand_idx]),
                "descriptor": torch.as_tensor(sample["x"][cand_idx]).float(),
                "similarity_to_prototype": float(value),
            }
        )
    return exemplars


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="artifacts/pixel_candidate_subgraphs_v2")
    p.add_argument("--out_dir", default="artifacts/pixel_motif_bank_v2")
    p.add_argument("--num_motifs_per_class", type=int, default=16)
    p.add_argument("--max_subgraphs_per_class", type=int, default=50000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kmeans_batch_size", type=int, default=4096)
    p.add_argument("--oversample_factor", type=int, default=2)
    p.add_argument("--num_exemplars", type=int, default=5)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = input_dir / "train_pixel_candidates.pt"
    if train_path.name != "train_pixel_candidates.pt":
        raise AssertionError("Pixel motif bank must be built from train_pixel_candidates.pt only")
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    print("=" * 80)
    print("Build Pixel-preserving Motif Bank V2")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k:<28}: {v}")

    meta = _torch_load(input_dir / "meta.pt")
    topologies = meta["candidate_topologies"]
    train_samples = _torch_load(train_path)
    descriptor_dim = int(meta["descriptor_dim"])
    num_classes = int(meta.get("num_classes", 7))

    sampled, refs, image_counts = _collect_sampled_descriptors(
        train_samples,
        num_classes=num_classes,
        max_subgraphs_per_class=args.max_subgraphs_per_class,
        seed=args.seed,
    )
    all_sampled = torch.cat([sampled[c] for c in range(num_classes) if sampled[c].numel() > 0], dim=0)
    mean = all_sampled.mean(dim=0)
    std = all_sampled.std(dim=0, unbiased=False).clamp_min(1e-6)
    norm = {c: (sampled[c] - mean) / std for c in range(num_classes)}
    global_norm = (all_sampled - mean) / std

    print(f"train samples={len(train_samples)} | descriptor_dim={descriptor_dim}")
    print(f"image_counts={image_counts}")
    for c in range(num_classes):
        print(f"  class {c}: sampled descriptors={sampled[c].shape[0]}")

    motifs: Dict[int, List[PixelMotifPrototype]] = {}
    all_intra, all_inter, all_disc = [], [], []
    for class_id in range(num_classes):
        class_desc = norm[class_id]
        others = [norm[c] for c in range(num_classes) if c != class_id and norm[c].numel() > 0]
        other_desc = torch.cat(others, dim=0) if others else torch.empty((0, descriptor_dim))
        if class_desc.numel() == 0:
            print(f"[WARN] class {class_id} has no descriptors; creating fallback motifs from global mean.")
            center = global_norm.mean(dim=0)
            intra = 0.0
            inter = compute_inter_score(center, other_desc) if other_desc.numel() > 0 else 0.0
            disc = compute_discriminative_score(intra, inter, alpha=args.alpha)
            fallback = [
                PixelMotifPrototype(
                    motif_id=class_id * args.num_motifs_per_class + local_idx,
                    class_id=int(class_id),
                    prototype=center.clone().detach().cpu(),
                    intra_score=float(intra),
                    inter_score=float(inter),
                    discriminative_score=float(disc),
                    support=0,
                    exemplars=[],
                )
                for local_idx in range(args.num_motifs_per_class)
            ]
            motifs[class_id] = fallback
            all_intra.extend([m.intra_score for m in fallback])
            all_inter.extend([m.inter_score for m in fallback])
            all_disc.extend([m.discriminative_score for m in fallback])
            continue
        n_clusters = min(
            class_desc.shape[0],
            max(args.num_motifs_per_class, args.num_motifs_per_class * args.oversample_factor),
        )
        centers, labels = _cluster_descriptors(
            class_desc,
            n_clusters=n_clusters,
            seed=args.seed + class_id * 7919,
            batch_size=args.kmeans_batch_size,
        )
        supports = torch.bincount(labels, minlength=centers.shape[0])
        candidates: List[PixelMotifPrototype] = []
        for center_idx, center in enumerate(centers):
            intra = compute_intra_score(center, class_desc)
            inter = compute_inter_score(center, other_desc)
            disc = compute_discriminative_score(intra, inter, alpha=args.alpha)
            exemplars = _make_exemplars(
                center=center,
                class_desc=class_desc,
                class_refs=refs[class_id],
                train_samples=train_samples,
                topologies=topologies,
                top_n=args.num_exemplars,
            )
            candidates.append(
                PixelMotifPrototype(
                    motif_id=-1,
                    class_id=int(class_id),
                    prototype=center.detach().cpu(),
                    intra_score=float(intra),
                    inter_score=float(inter),
                    discriminative_score=float(disc),
                    support=int(supports[center_idx].item()),
                    exemplars=exemplars,
                )
            )
        candidates.sort(key=lambda m: m.discriminative_score, reverse=True)
        selected = candidates[: args.num_motifs_per_class]
        base = list(selected)
        while len(selected) < args.num_motifs_per_class:
            src = base[len(selected) % len(base)]
            selected.append(
                PixelMotifPrototype(
                    motif_id=-1,
                    class_id=int(class_id),
                    prototype=src.prototype.clone(),
                    intra_score=src.intra_score,
                    inter_score=src.inter_score,
                    discriminative_score=src.discriminative_score,
                    support=src.support,
                    exemplars=list(src.exemplars),
                )
            )
        for local_idx, motif in enumerate(selected):
            motif.motif_id = int(class_id * args.num_motifs_per_class + local_idx)
            check_finite_tensor(f"motif[{motif.motif_id}].prototype", motif.prototype)
            all_intra.append(motif.intra_score)
            all_inter.append(motif.inter_score)
            all_disc.append(motif.discriminative_score)
        motifs[class_id] = selected

    bank = PixelMotifBank(
        motifs=motifs,
        descriptor_dim=descriptor_dim,
        num_classes=num_classes,
        emotion_names=meta.get("class_names", EMOTION_NAMES),
        config={
            "input_dir": str(input_dir),
            "input_train_file": str(train_path),
            "built_from_split": "train",
            "num_motifs_per_class": int(args.num_motifs_per_class),
            "max_subgraphs_per_class": int(args.max_subgraphs_per_class),
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "descriptor_transform": "standardize",
            "descriptor_mean": mean.tolist(),
            "descriptor_std": std.tolist(),
            "num_exemplars": int(args.num_exemplars),
            "image_counts_per_class": image_counts,
            "candidate_meta": {
                "num_candidates": meta.get("num_candidates"),
                "coverage_grid": meta.get("coverage_grid"),
                "radii": meta.get("radii"),
                "seed_stride": meta.get("seed_stride"),
            },
        },
    )

    bank_path = out_dir / "pixel_motif_bank.pt"
    save_pixel_motif_bank(bank, bank_path)
    torch.save(
        {
            "descriptor_dim": descriptor_dim,
            "num_classes": num_classes,
            "emotion_names": bank.emotion_names,
            "config": bank.config,
            "motifs_per_class": {int(c): len(ms) for c, ms in motifs.items()},
            "score_stats": {
                "intra": (min(all_intra), sum(all_intra) / len(all_intra), max(all_intra)),
                "inter": (min(all_inter), sum(all_inter) / len(all_inter), max(all_inter)),
                "disc": (min(all_disc), sum(all_disc) / len(all_disc), max(all_disc)),
            },
        },
        out_dir / "meta.pt",
    )
    print(f"saved bank -> {bank_path} ({bank_path.stat().st_size / 1024**2:.2f} MB)")
    print("DONE")


if __name__ == "__main__":
    main()
