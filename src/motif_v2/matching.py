"""Matching and coverage/diversity selection for pixel-preserving motifs."""

from __future__ import annotations

import math
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


def _motif_scalar(motif, attr: str, metadata_key: str | None = None, default: float = 0.0) -> float:
    value = getattr(motif, attr, None)
    if value is None and metadata_key is not None:
        value = getattr(motif, "metadata", {}).get(metadata_key)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def flatten_pixel_motif_bank_with_metadata(bank: PixelMotifBank):
    prototypes, motif_ids, class_ids, disc_scores = [], [], [], []
    purities, entropy_norms, global_dominance = [], [], []
    for class_id in range(bank.num_classes):
        for motif in bank.motifs.get(class_id, []):
            prototypes.append(torch.as_tensor(motif.prototype).float().view(-1))
            motif_ids.append(int(motif.motif_id))
            class_ids.append(int(motif.class_id))
            disc_scores.append(float(motif.discriminative_score))
            purities.append(_motif_scalar(motif, "class_purity", "class_purity", default=1.0))
            entropy_norm = _motif_scalar(motif, "entropy_norm", "entropy_norm", default=0.0)
            entropy_norms.append(entropy_norm)
            global_dominance.append(
                _motif_scalar(motif, "global_dominance", "global_dominance", default=entropy_norm)
            )
    if not prototypes:
        raise ValueError("Empty pixel motif bank")
    return {
        "prototypes": torch.stack(prototypes),
        "motif_ids": torch.tensor(motif_ids, dtype=torch.long),
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "disc_scores": torch.tensor(disc_scores, dtype=torch.float32),
        "class_purity": torch.tensor(purities, dtype=torch.float32),
        "entropy_norm": torch.tensor(entropy_norms, dtype=torch.float32),
        "global_dominance": torch.tensor(global_dominance, dtype=torch.float32),
    }


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


def match_candidates(
    x: torch.Tensor,
    bank: PixelMotifBank,
    *,
    disc_weight: float = 0.0,
    global_weight: float = 0.0,
    use_adjusted_match: bool = False,
) -> Dict[str, torch.Tensor]:
    flat = flatten_pixel_motif_bank_with_metadata(bank)
    prototypes = flat["prototypes"]
    motif_ids = flat["motif_ids"]
    class_ids = flat["class_ids"]
    disc_scores = flat["disc_scores"]
    x_t = transform_descriptors(x, bank)
    sim = cosine_similarity_matrix(x_t, prototypes)
    if use_adjusted_match:
        adjusted = sim + float(disc_weight) * disc_scores.view(1, -1) - float(global_weight) * flat[
            "global_dominance"
        ].view(1, -1)
        best_adjusted_score, best_idx = adjusted.max(dim=1)
        best_score = sim.gather(1, best_idx.view(-1, 1)).squeeze(1)
        motif_score_matrix = adjusted
    else:
        best_score, best_idx = sim.max(dim=1)
        best_adjusted_score = best_score
        motif_score_matrix = sim
    return {
        "similarity_matrix": sim,
        "best_score": best_score,
        "best_adjusted_score": best_adjusted_score,
        "matched_class": class_ids[best_idx],
        "matched_motif_id": motif_ids[best_idx],
        "matched_disc_score": disc_scores[best_idx],
        "matched_class_purity": flat["class_purity"][best_idx],
        "matched_entropy_norm": flat["entropy_norm"][best_idx],
        "matched_global_dominance": flat["global_dominance"][best_idx],
        "motif_score_vector": motif_score_vector_from_similarity(motif_score_matrix, class_ids, bank.num_classes),
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
    selection_mode: str = "greedy_coverage",
    disc_weight: float | None = None,
    global_weight: float = 0.0,
    coverage_weight: float | None = None,
    redundancy_weight: float | None = None,
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
    if selection_mode == "greedy_discriminative_soft_coverage":
        effective_disc_weight = float(beta if disc_weight is None else disc_weight)
        effective_coverage_weight = float(eta if coverage_weight is None else coverage_weight)
        effective_redundancy_weight = float(gamma if redundancy_weight is None else redundancy_weight)
        matches = match_candidates(
            xv,
            bank,
            disc_weight=effective_disc_weight,
            global_weight=float(global_weight),
            use_adjusted_match=True,
        )
        base = matches["best_adjusted_score"]
    elif selection_mode in {"greedy_coverage", "legacy", "greedy"}:
        effective_disc_weight = float(beta)
        effective_coverage_weight = float(eta)
        effective_redundancy_weight = float(gamma)
        matches = match_candidates(xv, bank)
        base = matches["best_score"] + effective_disc_weight * matches["matched_disc_score"]
    else:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}; expected greedy_coverage or "
            "greedy_discriminative_soft_coverage"
        )

    selected_local: List[int] = []
    used_cells = set()
    cell_counts: Dict[int, int] = {}
    available = torch.ones(xv.shape[0], dtype=torch.bool)
    sigma = max(float(diversity_sigma), 1e-6)
    x_norm = torch.nn.functional.normalize(xv, dim=1)

    for _ in range(min(top_k, xv.shape[0])):
        score = base.clone()
        if selected_local:
            selected_idx = torch.tensor(selected_local, dtype=torch.long)
            if selection_mode == "greedy_discriminative_soft_coverage":
                desc_sim = (x_norm @ x_norm[selected_idx].T).max(dim=1).values
                redundancy = desc_sim.clamp_min(0.0)
            else:
                selected_centers = cv[selected_idx]
                dist = torch.cdist(cv, selected_centers).min(dim=1).values
                redundancy = torch.exp(-dist / sigma)
            score = score - effective_redundancy_weight * redundancy
        if effective_coverage_weight != 0:
            if selection_mode == "greedy_discriminative_soft_coverage":
                bonus = torch.tensor(
                    [
                        effective_coverage_weight / math.sqrt(1.0 + float(cell_counts.get(int(c.item()), 0)))
                        for c in cellv
                    ],
                    dtype=score.dtype,
                )
            else:
                bonus = torch.tensor(
                    [0.0 if int(c.item()) in used_cells else effective_coverage_weight for c in cellv],
                    dtype=score.dtype,
                )
            score = score + bonus
        score[~available] = -1e9
        best = int(score.argmax().item())
        if not bool(available[best]):
            break
        selected_local.append(best)
        selected_cell = int(cellv[best].item())
        used_cells.add(selected_cell)
        cell_counts[selected_cell] = cell_counts.get(selected_cell, 0) + 1
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
