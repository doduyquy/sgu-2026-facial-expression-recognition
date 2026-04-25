"""Motif matching and motif-guided top-k subgraph selection."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

from src.motif.motif_scoring import check_finite_tensor, cosine_similarity_matrix
from src.motif.motif_types import MotifBank


def _transform_descriptors_for_bank(x: torch.Tensor, motif_bank: MotifBank) -> torch.Tensor:
    """Apply the descriptor transform used when the motif bank was built."""
    transform = motif_bank.config.get("descriptor_transform")
    if transform == "standardize":
        mean = torch.tensor(motif_bank.config["descriptor_mean"], dtype=x.dtype, device=x.device)
        std = torch.tensor(motif_bank.config["descriptor_std"], dtype=x.dtype, device=x.device).clamp_min(1e-6)
        return (x - mean) / std
    return x


def flatten_motif_bank(motif_bank: MotifBank):
    """Flatten per-class motif bank to tensors."""
    prototypes = []
    motif_ids = []
    class_ids = []
    disc_scores = []

    for class_id in range(motif_bank.num_classes):
        for motif in motif_bank.motifs.get(class_id, []):
            proto = torch.as_tensor(motif.prototype).float().view(-1)
            if proto.numel() != motif_bank.descriptor_dim:
                raise ValueError(
                    f"Motif {motif.motif_id} dim={proto.numel()} "
                    f"!= bank descriptor_dim={motif_bank.descriptor_dim}"
                )
            prototypes.append(proto)
            motif_ids.append(int(motif.motif_id))
            class_ids.append(int(motif.class_id))
            disc_scores.append(float(motif.discriminative_score))

    if not prototypes:
        raise ValueError("Motif bank is empty")

    prototypes_t = torch.stack(prototypes, dim=0)
    motif_ids_t = torch.tensor(motif_ids, dtype=torch.long)
    class_ids_t = torch.tensor(class_ids, dtype=torch.long)
    disc_scores_t = torch.tensor(disc_scores, dtype=torch.float32)
    check_finite_tensor("motif_bank.prototypes", prototypes_t)
    check_finite_tensor("motif_bank.disc_scores", disc_scores_t)
    return prototypes_t, motif_ids_t, class_ids_t, disc_scores_t


def match_descriptors_to_motifs(
    x: torch.Tensor,
    motif_bank: MotifBank,
    return_similarity: bool = False,
) -> Dict[str, torch.Tensor]:
    """Match each subgraph descriptor to its nearest prototype motif by cosine similarity."""
    x = torch.as_tensor(x).float()
    if x.ndim != 2:
        raise ValueError(f"Expected x [K, D], got {tuple(x.shape)}")
    if x.shape[1] != motif_bank.descriptor_dim:
        raise ValueError(f"Descriptor dim mismatch: x={x.shape[1]} bank={motif_bank.descriptor_dim}")

    x_for_match = _transform_descriptors_for_bank(x, motif_bank)
    prototypes, motif_ids, class_ids, disc_scores = flatten_motif_bank(motif_bank)
    sim = cosine_similarity_matrix(x_for_match, prototypes)
    best_score, best_idx = sim.max(dim=1)

    result = {
        "best_score": best_score,
        "matched_class": class_ids[best_idx],
        "matched_motif_id": motif_ids[best_idx],
        "matched_disc_score": disc_scores[best_idx],
    }
    if return_similarity:
        result["similarity_matrix"] = sim
    return result


def compute_motif_score_vector(
    x: torch.Tensor,
    motif_bank: MotifBank,
    reduce: str = "max_mean",
    top_k: int = 3,
) -> torch.Tensor:
    """Compute one image-level motif score per emotion class."""
    x = torch.as_tensor(x).float()
    if x.numel() == 0:
        return torch.zeros(motif_bank.num_classes, dtype=torch.float32)

    x_for_match = _transform_descriptors_for_bank(x, motif_bank)
    prototypes, _, class_ids, _ = flatten_motif_bank(motif_bank)
    sim = cosine_similarity_matrix(x_for_match, prototypes)  # [K, M]
    scores = torch.zeros(motif_bank.num_classes, dtype=torch.float32)

    for class_id in range(motif_bank.num_classes):
        motif_mask = class_ids == class_id
        if not motif_mask.any():
            continue
        class_sim = sim[:, motif_mask]
        if reduce == "max":
            scores[class_id] = class_sim.max()
        elif reduce in {"mean_topk", "topk_mean"}:
            flat = class_sim.flatten()
            k = max(1, min(int(top_k), flat.numel()))
            scores[class_id] = torch.topk(flat, k=k, largest=True).values.mean()
        elif reduce == "max_mean":
            per_subgraph = class_sim.max(dim=1).values
            k = max(1, min(int(top_k), per_subgraph.numel()))
            scores[class_id] = torch.topk(per_subgraph, k=k, largest=True).values.mean()
        else:
            raise ValueError(f"Unknown motif score reduce={reduce!r}")

    check_finite_tensor("motif_score_vector", scores)
    return scores


def select_topk_by_motif(
    x: torch.Tensor,
    centers: torch.Tensor,
    motif_bank: MotifBank,
    top_k: int,
    beta: float = 0.5,
    mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Select top-k subgraphs by motif match + discriminative motif score.

    Returns padded tensors and a boolean mask if fewer than top_k valid subgraphs exist.
    """
    x = torch.as_tensor(x).float().cpu()
    centers = torch.as_tensor(centers).float().cpu()
    if x.ndim != 2:
        raise ValueError(f"Expected x [K, D], got {tuple(x.shape)}")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Expected centers [K, 2], got {tuple(centers.shape)}")
    if centers.shape[0] != x.shape[0]:
        raise ValueError(f"centers K={centers.shape[0]} does not match x K={x.shape[0]}")

    if mask is None:
        valid_mask = torch.ones(x.shape[0], dtype=torch.bool)
    else:
        valid_mask = torch.as_tensor(mask).bool().cpu()
        if valid_mask.ndim != 1 or valid_mask.shape[0] != x.shape[0]:
            raise ValueError(f"Expected mask [{x.shape[0]}], got {tuple(valid_mask.shape)}")

    valid_indices = torch.where(valid_mask)[0]
    D = x.shape[1]
    top_k = int(top_k)

    out = {
        "selected_indices": torch.full((top_k,), -1, dtype=torch.long),
        "x": torch.zeros((top_k, D), dtype=torch.float32),
        "mask": torch.zeros((top_k,), dtype=torch.bool),
        "centers": torch.zeros((top_k, 2), dtype=torch.float32),
        "match_scores": torch.zeros((top_k,), dtype=torch.float32),
        "matched_class": torch.full((top_k,), -1, dtype=torch.long),
        "matched_motif_id": torch.full((top_k,), -1, dtype=torch.long),
        "matched_disc_score": torch.zeros((top_k,), dtype=torch.float32),
        "motif_score_vector": torch.zeros((motif_bank.num_classes,), dtype=torch.float32),
    }

    if valid_indices.numel() == 0:
        return out

    x_valid = x[valid_indices]
    centers_valid = centers[valid_indices]
    matches = match_descriptors_to_motifs(x_valid, motif_bank)
    motif_scores = compute_motif_score_vector(x_valid, motif_bank, reduce="max_mean")

    selection_score = matches["best_score"] + float(beta) * matches["matched_disc_score"]
    check_finite_tensor("selection_score", selection_score)
    n_select = min(top_k, x_valid.shape[0])
    _, local_top = torch.topk(selection_score, k=n_select, largest=True)
    original_top = valid_indices[local_top]

    out["selected_indices"][:n_select] = original_top.long()
    out["x"][:n_select] = x_valid[local_top]
    out["mask"][:n_select] = True
    out["centers"][:n_select] = centers_valid[local_top]
    out["match_scores"][:n_select] = matches["best_score"][local_top]
    out["matched_class"][:n_select] = matches["matched_class"][local_top]
    out["matched_motif_id"][:n_select] = matches["matched_motif_id"][local_top]
    out["matched_disc_score"][:n_select] = matches["matched_disc_score"][local_top]
    out["motif_score_vector"] = motif_scores

    for name, value in out.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            check_finite_tensor(f"select_topk_by_motif.{name}", value)
    return out
