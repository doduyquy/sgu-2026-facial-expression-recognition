"""Audit utilities for selected pixel motif subgraphs."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
DEFAULT_NUM_CLASSES = 7
DEFAULT_NUM_COVERAGE_CELLS = 16

REQUIRED_KEYS = [
    "label",
    "mask",
    "match_scores",
    "matched_class",
    "matched_motif_id",
    "matched_disc_score",
    "motif_score_vector",
    "coverage_cell",
]

OPTIONAL_KEYS = [
    "graph_id",
    "centers",
    "bbox",
    "selected_indices",
    "node_indices",
    "node_mask",
]

PER_SAMPLE_FIELDS = [
    "graph_id",
    "label",
    "k",
    "mask",
    "coverage_cell",
    "matched_class",
    "matched_motif_id",
    "match_score",
    "matched_disc_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "selected_index",
]


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_meta(pixel_motif_dir: Path) -> dict[str, Any]:
    path = pixel_motif_dir / "meta.pt"
    if not path.exists():
        return {}
    meta = torch_load(path)
    return meta if isinstance(meta, dict) else {}


def resolve_splits(split: str) -> list[str]:
    if split == "all":
        return ["train", "val", "test"]
    return [split]


def load_split_samples(pixel_motif_dir: Path, split: str) -> list[dict[str, Any]]:
    path = pixel_motif_dir / f"{split}_pixel_motif.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing pixel motif split artifact: {path}")
    samples = torch_load(path)
    if not isinstance(samples, list):
        raise TypeError(f"Expected {path} to contain a list of samples, got {type(samples)!r}")
    return samples


def validate_samples(samples: list[dict[str, Any]], split: str) -> list[str]:
    warnings: list[str] = []
    if not samples:
        warnings.append(f"[{split}] split is empty")
        return warnings
    first = samples[0]
    missing_required = [key for key in REQUIRED_KEYS if key not in first]
    if missing_required:
        raise KeyError(f"[{split}] missing required keys: {missing_required}")
    missing_optional = [key for key in OPTIONAL_KEYS if key not in first]
    for key in missing_optional:
        warnings.append(f"[{split}] optional key missing: {key}; related CSV fields will be blank")
    return warnings


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.cpu()


def _item_or_blank(tensor: torch.Tensor | None, index: int, col: int | None = None) -> Any:
    if tensor is None:
        return ""
    if tensor.numel() == 0:
        return ""
    try:
        value = tensor[index] if col is None else tensor[index, col]
    except IndexError:
        return ""
    if torch.is_tensor(value):
        if value.numel() != 1:
            return ""
        value = value.item()
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if torch.is_tensor(value):
            return int(value.item())
        return int(value)
    except Exception:
        return default


def _entropy(values: Iterable[int], num_bins: int) -> float:
    counts = [0] * num_bins
    total = 0
    for value in values:
        if 0 <= int(value) < num_bins:
            counts[int(value)] += 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def _label_name(class_id: int, class_names: list[str] | None) -> str:
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    if 0 <= class_id < len(EMOTION_NAMES):
        return EMOTION_NAMES[class_id]
    return str(class_id)


class MotifAuditAccumulator:
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        num_coverage_cells: int = DEFAULT_NUM_COVERAGE_CELLS,
        top_n_motifs: int = 20,
        class_names: list[str] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.num_coverage_cells = num_coverage_cells
        self.top_n_motifs = top_n_motifs
        self.class_names = class_names

        self.num_samples = 0
        self.num_slots_total = 0
        self.num_selected_total = 0
        self.label_hist = Counter()
        self.coverage_by_label = {
            c: Counter() for c in range(num_classes)
        }
        self.matched_by_label = {
            c: Counter() for c in range(num_classes)
        }
        self.motif_stats_by_label: dict[int, dict[int, dict[str, float]]] = {
            c: defaultdict(lambda: {"count": 0, "match_sum": 0.0, "disc_sum": 0.0})
            for c in range(num_classes)
        }
        self.match_score_sum = Counter()
        self.match_score_sq_sum = Counter()
        self.disc_score_sum = Counter()
        self.disc_score_sq_sum = Counter()
        self.score_count = Counter()
        self.motif_score_vector_sum = {
            c: torch.zeros(num_classes, dtype=torch.float64) for c in range(num_classes)
        }
        self.motif_score_vector_count = Counter()
        self.coverage_values_by_label = defaultdict(list)

    def add_sample(self, sample: dict[str, Any]) -> None:
        label = _safe_int(sample["label"])
        self.num_samples += 1
        self.label_hist[label] += 1

        mask = _as_tensor(sample["mask"]).bool()
        matched_class = _as_tensor(sample["matched_class"], torch.long)
        matched_motif_id = _as_tensor(sample["matched_motif_id"], torch.long)
        coverage_cell = _as_tensor(sample["coverage_cell"], torch.long)
        match_scores = _as_tensor(sample["match_scores"], torch.float32)
        disc_scores = _as_tensor(sample["matched_disc_score"], torch.float32)
        motif_score_vector = _as_tensor(sample["motif_score_vector"], torch.float32)

        self.num_slots_total += int(mask.numel())
        valid_indices = torch.where(mask)[0].tolist()
        self.num_selected_total += len(valid_indices)

        if 0 <= label < self.num_classes:
            self.motif_score_vector_sum[label] += motif_score_vector[: self.num_classes].double()
            self.motif_score_vector_count[label] += 1

        for idx in valid_indices:
            cls = _safe_int(matched_class[idx])
            motif_id = _safe_int(matched_motif_id[idx])
            cell = _safe_int(coverage_cell[idx])
            match_score = float(match_scores[idx].item())
            disc_score = float(disc_scores[idx].item())

            self.coverage_by_label[label][cell] += 1
            self.matched_by_label[label][cls] += 1
            self.coverage_values_by_label[label].append(cell)

            stats = self.motif_stats_by_label[label][motif_id]
            stats["count"] += 1
            stats["match_sum"] += match_score
            stats["disc_sum"] += disc_score

            self.match_score_sum[label] += match_score
            self.match_score_sq_sum[label] += match_score * match_score
            self.disc_score_sum[label] += disc_score
            self.disc_score_sq_sum[label] += disc_score * disc_score
            self.score_count[label] += 1

    def coverage_entropy_by_class(self) -> dict[str, float]:
        return {
            str(c): _entropy(self.coverage_values_by_label[c], self.num_coverage_cells)
            for c in range(self.num_classes)
        }

    def top_motifs_by_class(self) -> dict[str, list[dict[str, float | int]]]:
        result: dict[str, list[dict[str, float | int]]] = {}
        for label in range(self.num_classes):
            rows = []
            for motif_id, stats in self.motif_stats_by_label[label].items():
                count = int(stats["count"])
                rows.append(
                    {
                        "matched_motif_id": int(motif_id),
                        "count": count,
                        "avg_match_score": stats["match_sum"] / max(1, count),
                        "avg_disc_score": stats["disc_sum"] / max(1, count),
                    }
                )
            rows.sort(key=lambda row: (-int(row["count"]), int(row["matched_motif_id"])))
            result[str(label)] = rows[: self.top_n_motifs]
        return result

    def warnings(self) -> list[str]:
        messages = []
        for label in range(self.num_classes):
            total = sum(self.matched_by_label[label].values())
            if total == 0:
                messages.append(f"true class {label} has no valid selected motifs")
                continue
            top_class, top_count = self.matched_by_label[label].most_common(1)[0]
            ratio = top_count / total
            if top_class != label and ratio >= 0.40:
                true_name = _label_name(label, self.class_names)
                matched_name = _label_name(top_class, self.class_names)
                messages.append(
                    f"true class {label} ({true_name}) is mostly matched as "
                    f"class {top_class} ({matched_name}): {ratio:.1%}"
                )
        return messages

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_selected_total": self.num_selected_total,
            "num_slots_total": self.num_slots_total,
            "valid_selected_ratio": self.num_selected_total / max(1, self.num_slots_total),
            "label_hist": {str(k): int(v) for k, v in sorted(self.label_hist.items())},
            "coverage_entropy_by_class": self.coverage_entropy_by_class(),
            "matched_class_confusion": {
                str(label): {
                    str(cls): int(self.matched_by_label[label].get(cls, 0))
                    for cls in range(self.num_classes)
                }
                for label in range(self.num_classes)
            },
            "top_motifs_per_class": self.top_motifs_by_class(),
            "warnings": self.warnings(),
        }


def iter_selected_rows(samples_by_split: list[tuple[str, list[dict[str, Any]]]]):
    for split, samples in samples_by_split:
        for sample_idx, sample in enumerate(samples):
            graph_id = sample.get("graph_id", sample_idx)
            label = _safe_int(sample["label"])
            mask = _as_tensor(sample["mask"]).bool()
            coverage = _as_tensor(sample["coverage_cell"], torch.long)
            matched_class = _as_tensor(sample["matched_class"], torch.long)
            matched_motif_id = _as_tensor(sample["matched_motif_id"], torch.long)
            match_scores = _as_tensor(sample["match_scores"], torch.float32)
            disc_scores = _as_tensor(sample["matched_disc_score"], torch.float32)
            bbox = _as_tensor(sample["bbox"], torch.float32) if "bbox" in sample else None
            centers = _as_tensor(sample["centers"], torch.float32) if "centers" in sample else None
            selected_indices = (
                _as_tensor(sample["selected_indices"], torch.long)
                if "selected_indices" in sample
                else None
            )

            for k in range(int(mask.numel())):
                row = {
                    "graph_id": graph_id,
                    "label": label,
                    "k": k,
                    "mask": int(mask[k].item()),
                    "coverage_cell": _item_or_blank(coverage, k),
                    "matched_class": _item_or_blank(matched_class, k),
                    "matched_motif_id": _item_or_blank(matched_motif_id, k),
                    "match_score": _item_or_blank(match_scores, k),
                    "matched_disc_score": _item_or_blank(disc_scores, k),
                    "bbox_x1": _item_or_blank(bbox, k, 0),
                    "bbox_y1": _item_or_blank(bbox, k, 1),
                    "bbox_x2": _item_or_blank(bbox, k, 2),
                    "bbox_y2": _item_or_blank(bbox, k, 3),
                    "center_x": _item_or_blank(centers, k, 0),
                    "center_y": _item_or_blank(centers, k, 1),
                    "selected_index": _item_or_blank(selected_indices, k),
                }
                if len(samples_by_split) > 1:
                    row["split"] = split
                yield row


def write_selected_per_sample_csv(
    path: Path,
    samples_by_split: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    fieldnames = list(PER_SAMPLE_FIELDS)
    if len(samples_by_split) > 1:
        fieldnames.append("split")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(iter_selected_rows(samples_by_split))


def write_matrix_csv(
    path: Path,
    *,
    row_name: str,
    col_prefix: str,
    counters_by_label: dict[int, Counter],
    num_rows: int,
    num_cols: int,
    normalize: bool = False,
) -> None:
    fieldnames = [row_name] + [f"{col_prefix}_{i}" for i in range(num_cols)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in range(num_rows):
            counter = counters_by_label[label]
            total = sum(counter.values())
            row = {row_name: label}
            for col in range(num_cols):
                value = counter.get(col, 0)
                row[f"{col_prefix}_{col}"] = value / total if normalize and total else value
            writer.writerow(row)


def write_top_motifs_csv(path: Path, acc: MotifAuditAccumulator) -> None:
    fieldnames = [
        "true_class",
        "matched_motif_id",
        "count",
        "avg_match_score",
        "avg_disc_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in range(acc.num_classes):
            rows = []
            for motif_id, stats in acc.motif_stats_by_label[label].items():
                count = int(stats["count"])
                rows.append(
                    {
                        "true_class": label,
                        "matched_motif_id": int(motif_id),
                        "count": count,
                        "avg_match_score": stats["match_sum"] / max(1, count),
                        "avg_disc_score": stats["disc_sum"] / max(1, count),
                    }
                )
            rows.sort(key=lambda row: (-int(row["count"]), int(row["matched_motif_id"])))
            writer.writerows(rows[: acc.top_n_motifs])


def write_motif_score_vector_csv(path: Path, acc: MotifAuditAccumulator) -> None:
    fieldnames = ["true_class"] + [f"mean_score_class_{i}" for i in range(acc.num_classes)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in range(acc.num_classes):
            count = max(1, acc.motif_score_vector_count[label])
            mean = acc.motif_score_vector_sum[label] / count
            row = {"true_class": label}
            for cls in range(acc.num_classes):
                row[f"mean_score_class_{cls}"] = float(mean[cls].item())
            writer.writerow(row)


def write_score_stats_csv(path: Path, acc: MotifAuditAccumulator) -> None:
    fieldnames = [
        "true_class",
        "count",
        "match_score_mean",
        "match_score_std",
        "matched_disc_score_mean",
        "matched_disc_score_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in range(acc.num_classes):
            count = int(acc.score_count[label])
            match_mean = acc.match_score_sum[label] / max(1, count)
            disc_mean = acc.disc_score_sum[label] / max(1, count)
            match_var = acc.match_score_sq_sum[label] / max(1, count) - match_mean * match_mean
            disc_var = acc.disc_score_sq_sum[label] / max(1, count) - disc_mean * disc_mean
            writer.writerow(
                {
                    "true_class": label,
                    "count": count,
                    "match_score_mean": match_mean,
                    "match_score_std": math.sqrt(max(0.0, match_var)),
                    "matched_disc_score_mean": disc_mean,
                    "matched_disc_score_std": math.sqrt(max(0.0, disc_var)),
                }
            )


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_summary_txt(path: Path, summary: dict[str, Any], class_names: list[str] | None = None) -> None:
    lines = [
        "Selected Motif Audit Summary",
        "=" * 32,
        f"num_samples: {summary['num_samples']}",
        f"num_selected_total: {summary['num_selected_total']}",
        f"num_slots_total: {summary['num_slots_total']}",
        f"valid_selected_ratio: {summary['valid_selected_ratio']:.6f}",
        "",
        "Coverage entropy by class:",
    ]
    for key, value in summary["coverage_entropy_by_class"].items():
        label = int(key)
        lines.append(f"  {key} {_label_name(label, class_names)}: {value:.6f}")

    lines.extend(["", "Top matched class by true class:"])
    for key, row in summary["matched_class_confusion"].items():
        label = int(key)
        pairs = sorted(((int(cls), int(count)) for cls, count in row.items()), key=lambda x: -x[1])
        total = sum(count for _, count in pairs)
        cls, count = pairs[0] if pairs else (-1, 0)
        ratio = count / max(1, total)
        lines.append(
            f"  true {label} {_label_name(label, class_names)} -> "
            f"matched {cls} {_label_name(cls, class_names)}: {count} ({ratio:.2%})"
        )

    lines.extend(["", "Warnings:"])
    if summary["warnings"]:
        lines.extend(f"  - {warning}" for warning in summary["warnings"])
    else:
        lines.append("  none")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_plots(out_dir: Path, acc: MotifAuditAccumulator) -> list[str]:
    saved: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable; skipped plots ({exc})"]

    def _matrix(counters: dict[int, Counter], rows: int, cols: int) -> list[list[float]]:
        return [[float(counters[r].get(c, 0)) for c in range(cols)] for r in range(rows)]

    def _heatmap(matrix, title: str, xlabel: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks(range(acc.num_classes))
        ax.set_yticklabels([_label_name(i, acc.class_names) for i in range(acc.num_classes)])
        ax.set_xticks(range(len(matrix[0]) if matrix else 0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)
        saved.append(filename)

    coverage_matrix = _matrix(acc.coverage_by_label, acc.num_classes, acc.num_coverage_cells)
    matched_matrix = _matrix(acc.matched_by_label, acc.num_classes, acc.num_classes)
    motif_score_matrix = []
    for label in range(acc.num_classes):
        count = max(1, acc.motif_score_vector_count[label])
        motif_score_matrix.append((acc.motif_score_vector_sum[label] / count).tolist())

    _heatmap(
        coverage_matrix,
        "Coverage Cell Count by True Class",
        "coverage_cell",
        "true_class",
        "coverage_heatmap_per_class.png",
    )
    _heatmap(
        matched_matrix,
        "Matched Class Count by True Class",
        "matched_class",
        "true_class",
        "matched_class_heatmap.png",
    )
    _heatmap(
        motif_score_matrix,
        "Mean Motif Score Vector by True Class",
        "score_class",
        "true_class",
        "motif_score_vector_heatmap.png",
    )

    for label in range(acc.num_classes):
        top_rows = acc.top_motifs_by_class()[str(label)][: min(10, acc.top_n_motifs)]
        if not top_rows:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ids = [str(row["matched_motif_id"]) for row in top_rows]
        counts = [int(row["count"]) for row in top_rows]
        ax.bar(ids, counts)
        ax.set_title(f"Top Motifs for Class {label} {_label_name(label, acc.class_names)}")
        ax.set_xlabel("matched_motif_id")
        ax.set_ylabel("count")
        fig.tight_layout()
        filename = f"top_motifs_class_{label}.png"
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)
        saved.append(filename)

    return saved


def save_overlay_plots(
    *,
    out_dir: Path,
    graph_repo_dir: Path,
    samples_by_split: list[tuple[str, list[dict[str, Any]]]],
    max_overlay_samples: int = 24,
    max_boxes_per_sample: int = 32,
) -> list[str]:
    saved: list[str] = []
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable; skipped overlays ({exc})"]

    try:
        from data.graph_repository import GraphRepositoryReader
    except Exception as exc:
        return [f"graph repository reader unavailable; skipped overlays ({exc})"]

    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    requested: dict[str, dict[int, dict[str, Any]]] = {}
    for split, samples in samples_by_split:
        requested[split] = {}
        for sample in samples:
            if len(requested[split]) >= max_overlay_samples:
                break
            if "bbox" not in sample or "mask" not in sample:
                continue
            requested[split][_safe_int(sample.get("graph_id", len(requested[split])))] = sample

    reader = GraphRepositoryReader(graph_repo_dir)
    for split, sample_map in requested.items():
        if not sample_map:
            continue
        remaining = set(sample_map)
        for graph_sample in reader.iter_split(split):
            graph_id = _safe_int(getattr(graph_sample, "graph_id", -1))
            if graph_id not in remaining:
                continue
            motif_sample = sample_map[graph_id]
            node_features = getattr(graph_sample, "node_features")
            image = node_features[:, 0].float().reshape(48, 48).numpy()
            bbox = _as_tensor(motif_sample["bbox"], torch.float32)
            mask = _as_tensor(motif_sample["mask"]).bool()

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            valid_indices = torch.where(mask)[0].tolist()[:max_boxes_per_sample]
            for rank, idx in enumerate(valid_indices):
                x1 = float(bbox[idx, 0].item()) * 47.0
                y1 = float(bbox[idx, 1].item()) * 47.0
                x2 = float(bbox[idx, 2].item()) * 47.0
                y2 = float(bbox[idx, 3].item()) * 47.0
                rect = patches.Rectangle(
                    (x1, y1),
                    max(1.0, x2 - x1),
                    max(1.0, y2 - y1),
                    linewidth=0.8,
                    edgecolor="tab:red" if rank < 8 else "tab:orange",
                    facecolor="none",
                    alpha=0.75 if rank < 8 else 0.35,
                )
                ax.add_patch(rect)
            filename = f"{split}_graph_{graph_id}_selected_bbox.png"
            fig.tight_layout(pad=0)
            fig.savefig(overlay_dir / filename, dpi=160, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            saved.append(str(Path("overlays") / filename))
            remaining.remove(graph_id)
            if not remaining:
                break
        for graph_id in sorted(remaining):
            saved.append(f"graph_id {graph_id} not found in graph_repo split {split}; skipped overlay")

    return saved
