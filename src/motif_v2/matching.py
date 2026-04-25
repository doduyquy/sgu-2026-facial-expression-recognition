"""Matching and coverage/diversity selection for pixel-preserving motifs."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from src.motif.motif_scoring import check_finite_tensor, cosine_similarity_matrix
from src.motif_v2.types import PixelMotifBank


def flatten_pixel_motif_bank(bank: PixelMotifBank):
    prototypes, motif_ids, class_ids, disc_scores = [], [], [], []
    for class_id in range(bank.num_classes):
        for motif in bank.motifs.get(class_id, []):
            prototypes.append(torch.as_tensor(motif.prototype).float().view(-1))
            motif_ids.append(int(motif.motif_id))
            class_ids.append(int(motif.class_id))
            disc_scores.append(float(motif.discriminative_score))
    if not prototypes:
        raise ValueError("Empty pixel motif bank")
    return (
        torch.stack(prototypes),
        torch.tensor(motif_ids, dtype=torch.long),
        torch.tensor(class_ids, dtype=torch.long),
        torch.tensor(disc_scores, dtype=torch.float32),
    )


def transform_descriptors(x: torch.Tensor, bank: PixelMotifBank) -> torch.Tensor:
    transform = bank.config.get("descriptor_transform")
    x = torch.as_tensor(x).float()
    if transform == "standardize":
        mean = torch.tensor(bank.config["descriptor_mean"], dtype=x.dtype, device=x.device)
        std = torch.tensor(bank.config["descriptor_std"], dtype=x.dtype, device=x.device).clamp_min(1e-6)
        return (x - mean) / std
    return x


def motif_score_vector_from_similarity(sim: torch.Tensor, class_ids: torch.Tensor, num_classes: int, top_k: int = 3) -> torch.Tensor:
    out = torch.zeros(num_classes, dtype=torch.float32)
    for class_id in range(num_classes):
        mask = class_ids == class_id
        if not mask.any():
            continue
        per_candidate = sim[:, mask].max(dim=1).values
        k = max(1, min(int(top_k), int(per_candidate.numel())))
        out[class_id] = torch.topk(per_candidate, k=k).values.mean()
    return out


def match_candidates(x: torch.Tensor, bank: PixelMotifBank) -> Dict[str, torch.Tensor]:
    prototypes, motif_ids, class_ids, disc_scores = flatten_pixel_motif_bank(bank)
    x_t = transform_descriptors(x, bank)
    sim = cosine_similarity_matrix(x_t, prototypes)
    best_score, best_idx = sim.max(dim=1)
    return {
        "similarity_matrix": sim,
        "best_score": best_score,
        "matched_class": class_ids[best_idx],
        "matched_motif_id": motif_ids[best_idx],
        "matched_disc_score": disc_scores[best_idx],
        "motif_score_vector": motif_score_vector_from_similarity(sim, class_ids, bank.num_classes),
    }


def greedy_select_with_coverage(
    x: torch.Tensor,
    centers: torch.Tensor,
    bbox: torch.Tensor,
    coverage_cell: torch.Tensor,
    bank: PixelMotifBank,
    top_k: int,
    beta: float = 0.5,
    gamma: float = 0.25,
    eta: float = 0.05,
    diversity_sigma: float = 0.12,
    mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    x = torch.as_tensor(x).float()
    centers = torch.as_tensor(centers).float()
    bbox = torch.as_tensor(bbox).float()
    coverage_cell = torch.as_tensor(coverage_cell).long()
    if mask is None:
        valid_mask = torch.ones(x.shape[0], dtype=torch.bool)
    else:
        valid_mask = torch.as_tensor(mask).bool()

    valid = torch.where(valid_mask)[0]
    D = int(x.shape[1])
    top_k = int(top_k)
    out = {
        "selected_indices": torch.full((top_k,), -1, dtype=torch.long),
        "x": torch.zeros((top_k, D), dtype=torch.float32),
        "mask": torch.zeros((top_k,), dtype=torch.bool),
        "centers": torch.zeros((top_k, 2), dtype=torch.float32),
        "bbox": torch.zeros((top_k, 4), dtype=torch.float32),
        "coverage_cell": torch.full((top_k,), -1, dtype=torch.long),
        "match_scores": torch.zeros((top_k,), dtype=torch.float32),
        "matched_class": torch.full((top_k,), -1, dtype=torch.long),
        "matched_motif_id": torch.full((top_k,), -1, dtype=torch.long),
        "matched_disc_score": torch.zeros((top_k,), dtype=torch.float32),
        "motif_score_vector": torch.zeros((bank.num_classes,), dtype=torch.float32),
    }
    if valid.numel() == 0:
        return out

    xv = x[valid]
    cv = centers[valid]
    bv = bbox[valid]
    cellv = coverage_cell[valid]
    matches = match_candidates(xv, bank)
    base = matches["best_score"] + float(beta) * matches["matched_disc_score"]

    selected_local: List[int] = []
    used_cells = set()
    available = torch.ones(xv.shape[0], dtype=torch.bool)
    sigma = max(float(diversity_sigma), 1e-6)

    for _ in range(min(top_k, xv.shape[0])):
        score = base.clone()
        if selected_local:
            selected_centers = cv[torch.tensor(selected_local, dtype=torch.long)]
            dist = torch.cdist(cv, selected_centers).min(dim=1).values
            redundancy = torch.exp(-dist / sigma)
            score = score - float(gamma) * redundancy
        if eta != 0:
            bonus = torch.tensor([0.0 if int(c.item()) in used_cells else float(eta) for c in cellv])
            score = score + bonus
        score[~available] = -1e9
        best = int(score.argmax().item())
        if not bool(available[best]):
            break
        selected_local.append(best)
        used_cells.add(int(cellv[best].item()))
        available[best] = False

    if selected_local:
        idx = torch.tensor(selected_local, dtype=torch.long)
        n = int(idx.numel())
        out["selected_indices"][:n] = valid[idx]
        out["x"][:n] = xv[idx]
        out["mask"][:n] = True
        out["centers"][:n] = cv[idx]
        out["bbox"][:n] = bv[idx]
        out["coverage_cell"][:n] = cellv[idx]
        out["match_scores"][:n] = matches["best_score"][idx]
        out["matched_class"][:n] = matches["matched_class"][idx]
        out["matched_motif_id"][:n] = matches["matched_motif_id"][idx]
        out["matched_disc_score"][:n] = matches["matched_disc_score"][idx]
    out["motif_score_vector"] = matches["motif_score_vector"]

    for name, value in out.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            check_finite_tensor(f"greedy_select.{name}", value)
    return out
