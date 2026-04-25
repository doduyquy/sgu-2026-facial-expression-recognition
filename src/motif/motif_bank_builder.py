"""Build emotion-specific discriminative prototype motif banks from train split."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.motif.motif_scoring import (
    check_finite_tensor,
    compute_discriminative_score,
    compute_inter_score,
    compute_intra_score,
)
from src.motif.motif_types import MotifBank, MotifPrototype


DEFAULT_EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_train_descriptors(input_dir: str | Path) -> List[dict]:
    """Load only train_subgraph_graph.pt to avoid data leakage."""
    path = Path(input_dir) / "train_subgraph_graph.pt"
    if path.name != "train_subgraph_graph.pt":
        raise AssertionError(f"Motif bank must be built from train split only, got {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing train split: {path}")
    samples = _torch_load(path)
    if not isinstance(samples, list) or len(samples) == 0:
        raise RuntimeError(f"Expected a non-empty list of samples in {path}")
    return samples


def collect_descriptors_by_class(samples: List[dict], num_classes: int = 7) -> Dict[int, torch.Tensor]:
    """
    Collect valid subgraph descriptors from train samples by image label.

    Only x[mask == True] descriptors are used.
    """
    buckets: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}
    descriptor_dim = None

    for idx, sample in enumerate(samples):
        label = int(sample["label"])
        if label < 0 or label >= num_classes:
            raise ValueError(f"Sample {idx} has invalid label {label}")

        x = torch.as_tensor(sample["x"]).float().cpu()
        check_finite_tensor(f"sample[{idx}].x", x)
        if x.ndim != 2:
            raise ValueError(f"sample[{idx}].x must be [K, D], got {tuple(x.shape)}")
        descriptor_dim = int(x.shape[1]) if descriptor_dim is None else descriptor_dim

        mask = sample.get("mask")
        if mask is None:
            valid = torch.ones(x.shape[0], dtype=torch.bool)
        else:
            valid = torch.as_tensor(mask).bool().cpu()
            if valid.ndim != 1 or valid.shape[0] != x.shape[0]:
                raise ValueError(
                    f"sample[{idx}].mask must be [{x.shape[0]}], got {tuple(valid.shape)}"
                )
        if valid.any():
            buckets[label].append(x[valid])

    if descriptor_dim is None:
        raise RuntimeError("Cannot infer descriptor dimension from empty samples")

    result: Dict[int, torch.Tensor] = {}
    for class_id in range(num_classes):
        if buckets[class_id]:
            result[class_id] = torch.cat(buckets[class_id], dim=0).contiguous()
        else:
            result[class_id] = torch.empty((0, descriptor_dim), dtype=torch.float32)
        check_finite_tensor(f"descriptors[class={class_id}]", result[class_id])
    return result


def sample_descriptors_per_class(
    descriptors_by_class: Dict[int, torch.Tensor],
    max_subgraphs_per_class: int,
    seed: int,
) -> Dict[int, torch.Tensor]:
    """Subsample descriptors per class for efficient motif mining."""
    sampled: Dict[int, torch.Tensor] = {}
    for class_id, desc in descriptors_by_class.items():
        desc = desc.float().cpu()
        n = desc.shape[0]
        if max_subgraphs_per_class is None or max_subgraphs_per_class <= 0 or n <= max_subgraphs_per_class:
            sampled[class_id] = desc
            continue
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(class_id) * 1009)
        idx = torch.randperm(n, generator=gen)[:max_subgraphs_per_class]
        sampled[class_id] = desc[idx].contiguous()
    return sampled


def _simple_torch_kmeans(
    descriptors: torch.Tensor,
    n_clusters: int,
    seed: int,
    max_iter: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small CPU fallback for environments without scikit-learn."""
    n = descriptors.shape[0]
    if n == 0:
        raise ValueError("Cannot run kmeans on an empty descriptor tensor")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if n >= n_clusters:
        init_idx = torch.randperm(n, generator=gen)[:n_clusters]
    else:
        init_idx = torch.randint(0, n, (n_clusters,), generator=gen)
    centers = descriptors[init_idx].clone()
    labels = torch.zeros(n, dtype=torch.long)

    for _ in range(max_iter):
        distances = torch.cdist(descriptors.float(), centers.float(), p=2)
        new_labels = distances.argmin(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

        for k in range(n_clusters):
            members = descriptors[labels == k]
            if members.numel() > 0:
                centers[k] = members.mean(dim=0)
            else:
                centers[k] = descriptors[torch.randint(0, n, (1,), generator=gen).item()]

    return centers.float(), labels.long()


def _cluster_descriptors(
    descriptors: torch.Tensor,
    n_clusters: int,
    seed: int,
    kmeans_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = descriptors.shape[0]
    n_clusters = max(1, min(int(n_clusters), int(n)))

    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=int(kmeans_batch_size),
            random_state=int(seed),
            n_init=3,
            max_iter=100,
            reassignment_ratio=0.01,
        )
        labels_np = km.fit_predict(descriptors.numpy())
        centers = torch.from_numpy(km.cluster_centers_).float()
        labels = torch.from_numpy(labels_np).long()
        return centers, labels
    except Exception as exc:
        print(f"[WARN] MiniBatchKMeans unavailable/failed ({exc}); using torch kmeans fallback.")
        return _simple_torch_kmeans(descriptors, n_clusters=n_clusters, seed=seed)


def build_motifs_for_class(
    class_id: int,
    class_descriptors: torch.Tensor,
    other_descriptors: torch.Tensor,
    num_motifs_per_class: int,
    alpha: float,
    seed: int,
    kmeans_batch_size: int,
    oversample_factor: int = 2,
) -> List[MotifPrototype]:
    """Cluster one class and keep the most discriminative prototype centroids."""
    class_descriptors = torch.as_tensor(class_descriptors).float().cpu().contiguous()
    other_descriptors = torch.as_tensor(other_descriptors).float().cpu().contiguous()
    check_finite_tensor(f"class_descriptors[{class_id}]", class_descriptors)
    check_finite_tensor(f"other_descriptors[{class_id}]", other_descriptors)

    n = int(class_descriptors.shape[0])
    if n == 0:
        raise ValueError(f"Class {class_id} has no descriptors; cannot build motifs")

    n_candidates = max(int(num_motifs_per_class), int(num_motifs_per_class) * int(oversample_factor))
    n_candidates = min(n_candidates, n)

    centers, labels = _cluster_descriptors(
        class_descriptors,
        n_clusters=n_candidates,
        seed=int(seed) + int(class_id) * 7919,
        kmeans_batch_size=kmeans_batch_size,
    )
    support_counts = torch.bincount(labels, minlength=centers.shape[0])

    candidates: List[MotifPrototype] = []
    for cluster_idx, center in enumerate(centers):
        intra = compute_intra_score(center, class_descriptors, top_fraction=0.2)
        inter = compute_inter_score(center, other_descriptors, top_fraction=0.2)
        disc = compute_discriminative_score(intra, inter, alpha=alpha)
        candidates.append(
            MotifPrototype(
                motif_id=-1,
                class_id=int(class_id),
                prototype=center.detach().cpu(),
                intra_score=float(intra),
                inter_score=float(inter),
                discriminative_score=float(disc),
                support=int(support_counts[cluster_idx].item()),
                exemplar={"cluster_index": int(cluster_idx)},
            )
        )

    candidates.sort(key=lambda m: m.discriminative_score, reverse=True)
    selected = candidates[: int(num_motifs_per_class)]

    if not selected:
        center = class_descriptors.mean(dim=0)
        intra = compute_intra_score(center, class_descriptors, top_fraction=0.2)
        inter = compute_inter_score(center, other_descriptors, top_fraction=0.2)
        selected = [
            MotifPrototype(
                motif_id=-1,
                class_id=int(class_id),
                prototype=center.detach().cpu(),
                intra_score=float(intra),
                inter_score=float(inter),
                discriminative_score=compute_discriminative_score(intra, inter, alpha=alpha),
                support=n,
                exemplar={"fallback": "class_mean"},
            )
        ]

    # If the class is tiny, repeat the best available motifs so the bank shape is stable.
    base = list(selected)
    repeat_cursor = 0
    while len(selected) < int(num_motifs_per_class):
        src = base[repeat_cursor % len(base)]
        selected.append(
            MotifPrototype(
                motif_id=-1,
                class_id=int(class_id),
                prototype=torch.as_tensor(src.prototype).clone().detach().cpu(),
                intra_score=float(src.intra_score),
                inter_score=float(src.inter_score),
                discriminative_score=float(src.discriminative_score),
                support=int(src.support),
                exemplar={"repeated_from_cluster": src.exemplar},
            )
        )
        repeat_cursor += 1

    for local_idx, motif in enumerate(selected):
        motif.motif_id = int(class_id) * int(num_motifs_per_class) + int(local_idx)
        if motif.exemplar is None:
            motif.exemplar = {}
        motif.exemplar["rank_in_class"] = int(local_idx)

    return selected


def build_motif_bank(
    input_dir: str | Path,
    num_motifs_per_class: int = 16,
    max_subgraphs_per_class: int = 50000,
    alpha: float = 0.5,
    seed: int = 42,
    kmeans_batch_size: int = 4096,
    oversample_factor: int = 2,
    num_classes: int = 7,
    emotion_names: List[str] | None = None,
) -> MotifBank:
    """Build a MotifBank from the train split only."""
    random.seed(int(seed))
    torch.manual_seed(int(seed))

    samples = load_train_descriptors(input_dir)
    input_train_file = str(Path(input_dir) / "train_subgraph_graph.pt")
    raw_counts = [0 for _ in range(num_classes)]
    for sample in samples:
        raw_counts[int(sample["label"])] += 1

    descriptors = collect_descriptors_by_class(samples, num_classes=num_classes)
    sampled = sample_descriptors_per_class(
        descriptors,
        max_subgraphs_per_class=max_subgraphs_per_class,
        seed=seed,
    )

    descriptor_dim = int(next(iter(descriptors.values())).shape[1])
    all_sampled = torch.cat(
        [sampled[c] for c in range(num_classes) if sampled[c].numel() > 0],
        dim=0,
    )
    descriptor_mean = all_sampled.mean(dim=0)
    descriptor_std = all_sampled.std(dim=0, unbiased=False).clamp_min(1e-6)
    sampled_for_mining = {
        c: (sampled[c] - descriptor_mean) / descriptor_std
        for c in range(num_classes)
    }
    print(f"[MotifBank] input_dir={input_dir}")
    print(f"[MotifBank] train samples={len(samples)} | descriptor_dim={descriptor_dim}")
    print("[MotifBank] descriptor transform=standardize(train sampled mean/std)")
    print(f"[MotifBank] image counts/class={raw_counts}")
    print("[MotifBank] descriptor counts/class:")
    for class_id in range(num_classes):
        print(
            f"  class {class_id}: raw={descriptors[class_id].shape[0]} "
            f"sampled={sampled[class_id].shape[0]}"
        )

    motifs: Dict[int, List[MotifPrototype]] = {}
    for class_id in range(num_classes):
        class_desc = sampled_for_mining[class_id]
        others = [
            sampled_for_mining[c]
            for c in range(num_classes)
            if c != class_id and sampled_for_mining[c].numel() > 0
        ]
        other_desc = torch.cat(others, dim=0) if others else torch.empty((0, descriptor_dim))
        motifs[class_id] = build_motifs_for_class(
            class_id=class_id,
            class_descriptors=class_desc,
            other_descriptors=other_desc,
            num_motifs_per_class=num_motifs_per_class,
            alpha=alpha,
            seed=seed,
            kmeans_batch_size=kmeans_batch_size,
            oversample_factor=oversample_factor,
        )

    bank = MotifBank(
        motifs=motifs,
        descriptor_dim=descriptor_dim,
        num_classes=int(num_classes),
        emotion_names=emotion_names or DEFAULT_EMOTION_NAMES,
        config={
            "input_dir": str(input_dir),
            "num_motifs_per_class": int(num_motifs_per_class),
            "max_subgraphs_per_class": int(max_subgraphs_per_class),
            "alpha": float(alpha),
            "seed": int(seed),
            "kmeans_batch_size": int(kmeans_batch_size),
            "oversample_factor": int(oversample_factor),
            "built_from_split": "train",
            "input_train_file": input_train_file,
            "descriptor_transform": "standardize",
            "descriptor_mean": descriptor_mean.tolist(),
            "descriptor_std": descriptor_std.tolist(),
            "image_counts_per_class": raw_counts,
            "descriptor_counts_per_class": {
                int(c): int(descriptors[c].shape[0]) for c in range(num_classes)
            },
            "sampled_descriptor_counts_per_class": {
                int(c): int(sampled[c].shape[0]) for c in range(num_classes)
            },
        },
    )
    return bank
