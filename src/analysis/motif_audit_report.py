"""Analyze generated motif audit reports and produce diagnosis artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required audit file: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _class_name(class_id: int, class_names: list[str] | None = None) -> str:
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    if 0 <= class_id < len(EMOTION_NAMES):
        return EMOTION_NAMES[class_id]
    return str(class_id)


def _detect_num_classes(rows: list[dict[str, str]], prefix: str) -> int:
    if not rows:
        return 0
    cols = [key for key in rows[0] if key.startswith(prefix)]
    return len(cols)


def normalize_matrix_rows(
    rows: list[dict[str, str]],
    *,
    row_key: str,
    value_prefix: str,
) -> tuple[list[dict[str, Any]], dict[int, list[float]]]:
    num_cols = _detect_num_classes(rows, value_prefix)
    out_rows: list[dict[str, Any]] = []
    matrix: dict[int, list[float]] = {}
    for row in rows:
        label = _int(row[row_key])
        counts = [_float(row.get(f"{value_prefix}{i}", 0.0)) for i in range(num_cols)]
        total = sum(counts)
        probs = [count / total if total > 0 else 0.0 for count in counts]
        matrix[label] = probs
        out = {row_key: label}
        for i, value in enumerate(probs):
            out[f"{value_prefix}{i}"] = value
        out_rows.append(out)
    return out_rows, matrix


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def rank_desc(values: list[float], target_idx: int) -> int:
    target = values[target_idx]
    return 1 + sum(1 for value in values if value > target)


def analyze_matched_class(
    matched_rows: list[dict[str, str]],
    *,
    class_names: list[str] | None,
    low_diag_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    normalized_rows, matrix = normalize_matrix_rows(
        matched_rows,
        row_key="true_class",
        value_prefix="matched_class_",
    )
    report_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    low_diag_classes = []
    for label, probs in sorted(matrix.items()):
        diag = probs[label] if label < len(probs) else 0.0
        off = [(idx, value) for idx, value in enumerate(probs) if idx != label]
        top_off_class, top_off_ratio = max(off, key=lambda item: item[1]) if off else (-1, 0.0)
        status = "low_diagonal" if diag < low_diag_threshold else "ok"
        if status != "ok":
            low_diag_classes.append(label)
            warnings.append(
                f"class {label} {_class_name(label, class_names)} has low matched-class diagonal "
                f"{diag:.2%}; strongest off-diagonal is {top_off_class} "
                f"{_class_name(top_off_class, class_names)} at {top_off_ratio:.2%}"
            )
        report_rows.append(
            {
                "true_class": label,
                "true_class_name": _class_name(label, class_names),
                "diagonal_ratio": diag,
                "top_offdiag_class": top_off_class,
                "top_offdiag_class_name": _class_name(top_off_class, class_names),
                "top_offdiag_ratio": top_off_ratio,
                "status": status,
            }
        )
    return normalized_rows, {"rows": report_rows, "low_diag_classes": low_diag_classes}, warnings


def analyze_motif_score_rank(
    score_rows: list[dict[str, str]],
    *,
    class_names: list[str] | None,
    warn_rank_gt: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    out_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    bad_classes = []
    num_classes = _detect_num_classes(score_rows, "mean_score_class_")
    for row in score_rows:
        label = _int(row["true_class"])
        values = [_float(row.get(f"mean_score_class_{i}", 0.0)) for i in range(num_classes)]
        order = sorted(range(num_classes), key=lambda idx: values[idx], reverse=True)
        true_rank = rank_desc(values, label)
        top_class = order[0] if order else -1
        top_score = values[top_class] if top_class >= 0 else 0.0
        true_score = values[label] if label < len(values) else 0.0
        margin_to_top = top_score - true_score
        status = "weak_true_class_rank" if true_rank > warn_rank_gt else "ok"
        if status != "ok":
            bad_classes.append(label)
            warnings.append(
                f"class {label} {_class_name(label, class_names)} motif_score true-class rank is "
                f"{true_rank}; top class is {top_class} {_class_name(top_class, class_names)}"
            )
        out_rows.append(
            {
                "true_class": label,
                "true_class_name": _class_name(label, class_names),
                "true_class_score": true_score,
                "true_class_rank": true_rank,
                "top_score_class": top_class,
                "top_score_class_name": _class_name(top_class, class_names),
                "top_score": top_score,
                "margin_to_top": margin_to_top,
                "rank_order": " ".join(str(idx) for idx in order),
                "status": status,
            }
        )
    return out_rows, {"rows": out_rows, "bad_rank_classes": bad_classes}, warnings


def analyze_coverage(
    coverage_rows: list[dict[str, str]],
    *,
    class_names: list[str] | None,
    high_ratio_threshold: float,
    low_ratio_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    normalized_rows, matrix = normalize_matrix_rows(
        coverage_rows,
        row_key="true_class",
        value_prefix="coverage_cell_",
    )
    warnings: list[str] = []
    report_rows = []
    high_classes = []
    low_classes = []
    num_cells = len(next(iter(matrix.values()))) if matrix else 0
    uniform_entropy = math.log(num_cells) if num_cells > 0 else 1.0
    for label, probs in sorted(matrix.items()):
        ent = entropy(probs)
        ratio = ent / uniform_entropy if uniform_entropy > 0 else 0.0
        max_cell = max(range(len(probs)), key=lambda idx: probs[idx]) if probs else -1
        max_cell_ratio = probs[max_cell] if max_cell >= 0 else 0.0
        if ratio >= high_ratio_threshold:
            status = "too_uniform"
            high_classes.append(label)
            warnings.append(
                f"class {label} {_class_name(label, class_names)} coverage entropy is near uniform "
                f"({ratio:.2%} of max); selection may be too diffuse"
            )
        elif ratio <= low_ratio_threshold:
            status = "collapsed"
            low_classes.append(label)
            warnings.append(
                f"class {label} {_class_name(label, class_names)} coverage entropy is low "
                f"({ratio:.2%} of max); selection may collapse into a few cells"
            )
        else:
            status = "ok"
        report_rows.append(
            {
                "true_class": label,
                "true_class_name": _class_name(label, class_names),
                "coverage_entropy": ent,
                "coverage_entropy_ratio": ratio,
                "max_coverage_cell": max_cell,
                "max_coverage_cell_ratio": max_cell_ratio,
                "status": status,
            }
        )
    return normalized_rows, {"rows": report_rows, "too_uniform_classes": high_classes, "collapsed_classes": low_classes}, warnings


def analyze_global_motifs(
    top_motif_rows: list[dict[str, str]],
    *,
    min_classes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    by_motif: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in top_motif_rows:
        by_motif[_int(row["matched_motif_id"])].append(row)

    suspicious_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for motif_id, rows in sorted(by_motif.items()):
        classes = sorted({_int(row["true_class"]) for row in rows})
        if len(classes) < min_classes:
            continue
        total_count = sum(_int(row["count"]) for row in rows)
        avg_match = sum(_float(row["avg_match_score"]) for row in rows) / max(1, len(rows))
        avg_disc = sum(_float(row["avg_disc_score"]) for row in rows) / max(1, len(rows))
        suspicious_rows.append(
            {
                "matched_motif_id": motif_id,
                "num_true_classes": len(classes),
                "true_classes": " ".join(str(c) for c in classes),
                "total_top_count": total_count,
                "avg_match_score": avg_match,
                "avg_disc_score": avg_disc,
                "status": "suspicious_global_motif",
            }
        )
    suspicious_rows.sort(key=lambda row: (-int(row["num_true_classes"]), -int(row["total_top_count"])))
    if suspicious_rows:
        warnings.append(
            f"{len(suspicious_rows)} motif ids appear in top motifs for >= {min_classes} classes"
        )
    return suspicious_rows, {"rows": suspicious_rows}, warnings


def analyze_score_stats(
    score_stat_rows: list[dict[str, str]],
    *,
    class_names: list[str] | None,
    low_quantile_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows = []
    warnings = []
    match_values = [_float(row["match_score_mean"]) for row in score_stat_rows]
    disc_values = [_float(row["matched_disc_score_mean"]) for row in score_stat_rows]
    mean_match = sum(match_values) / max(1, len(match_values))
    mean_disc = sum(disc_values) / max(1, len(disc_values))
    min_match = min(match_values) if match_values else 0.0
    min_disc = min(disc_values) if disc_values else 0.0
    match_threshold = min_match + low_quantile_tolerance * (mean_match - min_match)
    disc_threshold = min_disc + low_quantile_tolerance * (mean_disc - min_disc)
    weak_classes = []
    for row in score_stat_rows:
        label = _int(row["true_class"])
        match_mean = _float(row["match_score_mean"])
        disc_mean = _float(row["matched_disc_score_mean"])
        weak_match = match_mean <= match_threshold
        weak_disc = disc_mean <= disc_threshold
        status_parts = []
        if weak_match:
            status_parts.append("low_match_score")
        if weak_disc:
            status_parts.append("low_disc_score")
        status = "|".join(status_parts) if status_parts else "ok"
        if status != "ok":
            weak_classes.append(label)
            warnings.append(
                f"class {label} {_class_name(label, class_names)} has weak score stats: {status}"
            )
        rows.append(
            {
                "true_class": label,
                "true_class_name": _class_name(label, class_names),
                "count": _int(row["count"]),
                "match_score_mean": match_mean,
                "match_score_std": _float(row["match_score_std"]),
                "matched_disc_score_mean": disc_mean,
                "matched_disc_score_std": _float(row["matched_disc_score_std"]),
                "status": status,
            }
        )
    return rows, {"rows": rows, "weak_score_classes": weak_classes}, warnings


def build_recommendations(analysis: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    low_diag = set(analysis["matched_class"]["low_diag_classes"])
    bad_rank = set(analysis["motif_score_rank"]["bad_rank_classes"])
    too_uniform = set(analysis["coverage"]["too_uniform_classes"])
    collapsed = set(analysis["coverage"]["collapsed_classes"])
    suspicious = analysis["global_motifs"]["rows"]

    if low_diag and bad_rank:
        recommendations.append(
            "Recommend D2 Graph-aware Motif Bank: matched-class diagonal and motif-score rank are weak."
        )
    if too_uniform:
        recommendations.append(
            "Recommend reducing hard coverage pressure or turning coverage into a soft regularization term."
        )
    if collapsed:
        recommendations.append(
            "Recommend increasing diversity/coverage regularization because selected regions collapse into few cells."
        )
    if suspicious:
        recommendations.append(
            "Recommend strengthening discriminative_score or class-specific prototype filtering in motif bank construction."
        )
    if not low_diag and not bad_rank and not too_uniform and not collapsed and not suspicious:
        recommendations.append(
            "Coverage and motif alignment look acceptable; if model errors persist, try D4 Dynamic Relation Motif GNN or prototype/contrastive losses."
        )
    elif low_diag and not bad_rank:
        recommendations.append(
            "Matched-class alignment is weak even when motif score rank is not catastrophic; inspect motif matching and class-balance in motif bank."
        )
    return recommendations


def write_markdown_report(path: Path, diagnosis: dict[str, Any]) -> None:
    lines = [
        "# Motif Selection Audit Diagnosis",
        "",
        "## Final Diagnosis",
    ]
    for rec in diagnosis["recommendations"]:
        lines.append(f"- {rec}")

    lines.extend(["", "## Warnings"])
    if diagnosis["warnings"]:
        lines.extend(f"- {warning}" for warning in diagnosis["warnings"])
    else:
        lines.append("- No automatic warning triggered.")

    lines.extend(["", "## Matched-Class Diagonal"])
    for row in diagnosis["matched_class"]["rows"]:
        lines.append(
            f"- true {row['true_class']} {row['true_class_name']}: "
            f"diag={row['diagonal_ratio']:.2%}, "
            f"top offdiag={row['top_offdiag_class']} {row['top_offdiag_class_name']} "
            f"({row['top_offdiag_ratio']:.2%}), status={row['status']}"
        )

    lines.extend(["", "## Motif Score Rank"])
    for row in diagnosis["motif_score_rank"]["rows"]:
        lines.append(
            f"- true {row['true_class']} {row['true_class_name']}: "
            f"rank={row['true_class_rank']}, top={row['top_score_class']} "
            f"{row['top_score_class_name']}, margin={row['margin_to_top']:.6f}, "
            f"status={row['status']}"
        )

    lines.extend(["", "## Coverage"])
    for row in diagnosis["coverage"]["rows"]:
        lines.append(
            f"- true {row['true_class']} {row['true_class_name']}: "
            f"entropy_ratio={row['coverage_entropy_ratio']:.2%}, "
            f"max_cell={row['max_coverage_cell']} ({row['max_coverage_cell_ratio']:.2%}), "
            f"status={row['status']}"
        )

    lines.extend(["", "## Suspicious Global Motifs"])
    if diagnosis["global_motifs"]["rows"]:
        for row in diagnosis["global_motifs"]["rows"][:30]:
            lines.append(
                f"- motif {row['matched_motif_id']}: appears in {row['num_true_classes']} classes "
                f"[{row['true_classes']}], total_top_count={row['total_top_count']}"
            )
    else:
        lines.append("- No motif appears in top motifs for enough classes to trigger this rule.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_audit_report(
    audit_dir: Path,
    out_dir: Path,
    *,
    low_diag_threshold: float = 0.18,
    motif_rank_warn_gt: int = 2,
    coverage_high_ratio_threshold: float = 0.96,
    coverage_low_ratio_threshold: float = 0.70,
    global_motif_min_classes: int = 4,
) -> dict[str, Any]:
    summary = read_json(audit_dir / "summary.json")
    class_names = None
    if summary.get("class_names"):
        class_names = list(summary["class_names"])

    matched_rows = read_csv_rows(audit_dir / "matched_class_by_true_class.csv")
    coverage_rows = read_csv_rows(audit_dir / "coverage_by_true_class.csv")
    top_motif_rows = read_csv_rows(audit_dir / "top_motifs_by_true_class.csv")
    motif_score_rows = read_csv_rows(audit_dir / "motif_score_vector_by_true_class.csv")
    score_stat_rows = read_csv_rows(audit_dir / "match_score_stats_by_true_class.csv")

    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_matched, matched_analysis, matched_warnings = analyze_matched_class(
        matched_rows,
        class_names=class_names,
        low_diag_threshold=low_diag_threshold,
    )
    normalized_coverage, coverage_analysis, coverage_warnings = analyze_coverage(
        coverage_rows,
        class_names=class_names,
        high_ratio_threshold=coverage_high_ratio_threshold,
        low_ratio_threshold=coverage_low_ratio_threshold,
    )
    motif_rank_rows, motif_rank_analysis, rank_warnings = analyze_motif_score_rank(
        motif_score_rows,
        class_names=class_names,
        warn_rank_gt=motif_rank_warn_gt,
    )
    suspicious_rows, global_motif_analysis, global_warnings = analyze_global_motifs(
        top_motif_rows,
        min_classes=global_motif_min_classes,
    )
    score_rows, score_analysis, score_warnings = analyze_score_stats(
        score_stat_rows,
        class_names=class_names,
        low_quantile_tolerance=0.25,
    )

    class_bias_rows = []
    score_by_class = {row["true_class"]: row for row in score_rows}
    rank_by_class = {row["true_class"]: row for row in motif_rank_rows}
    coverage_by_class = {row["true_class"]: row for row in coverage_analysis["rows"]}
    for row in matched_analysis["rows"]:
        label = row["true_class"]
        class_bias_rows.append(
            {
                "true_class": label,
                "true_class_name": row["true_class_name"],
                "diagonal_ratio": row["diagonal_ratio"],
                "top_offdiag_class": row["top_offdiag_class"],
                "top_offdiag_class_name": row["top_offdiag_class_name"],
                "top_offdiag_ratio": row["top_offdiag_ratio"],
                "motif_score_true_rank": rank_by_class[label]["true_class_rank"],
                "coverage_entropy_ratio": coverage_by_class[label]["coverage_entropy_ratio"],
                "match_score_mean": score_by_class[label]["match_score_mean"],
                "matched_disc_score_mean": score_by_class[label]["matched_disc_score_mean"],
                "status": ";".join(
                    part
                    for part in [
                        row["status"] if row["status"] != "ok" else "",
                        rank_by_class[label]["status"] if rank_by_class[label]["status"] != "ok" else "",
                        coverage_by_class[label]["status"] if coverage_by_class[label]["status"] != "ok" else "",
                        score_by_class[label]["status"] if score_by_class[label]["status"] != "ok" else "",
                    ]
                    if part
                )
                or "ok",
            }
        )

    write_csv(
        out_dir / "normalized_matched_class_by_true_class.csv",
        normalized_matched,
        list(normalized_matched[0].keys()) if normalized_matched else ["true_class"],
    )
    write_csv(
        out_dir / "normalized_coverage_by_true_class.csv",
        normalized_coverage,
        list(normalized_coverage[0].keys()) if normalized_coverage else ["true_class"],
    )
    write_csv(
        out_dir / "motif_score_rank_by_true_class.csv",
        motif_rank_rows,
        list(motif_rank_rows[0].keys()) if motif_rank_rows else ["true_class"],
    )
    write_csv(
        out_dir / "suspicious_global_motifs.csv",
        suspicious_rows,
        [
            "matched_motif_id",
            "num_true_classes",
            "true_classes",
            "total_top_count",
            "avg_match_score",
            "avg_disc_score",
            "status",
        ],
    )
    write_csv(
        out_dir / "class_bias_report.csv",
        class_bias_rows,
        list(class_bias_rows[0].keys()) if class_bias_rows else ["true_class"],
    )

    diagnosis = {
        "audit_dir": str(audit_dir),
        "thresholds": {
            "low_diag_threshold": low_diag_threshold,
            "motif_rank_warn_gt": motif_rank_warn_gt,
            "coverage_high_ratio_threshold": coverage_high_ratio_threshold,
            "coverage_low_ratio_threshold": coverage_low_ratio_threshold,
            "global_motif_min_classes": global_motif_min_classes,
        },
        "matched_class": matched_analysis,
        "motif_score_rank": motif_rank_analysis,
        "coverage": coverage_analysis,
        "global_motifs": global_motif_analysis,
        "score_stats": score_analysis,
        "warnings": (
            matched_warnings
            + rank_warnings
            + coverage_warnings
            + global_warnings
            + score_warnings
        ),
    }
    diagnosis["recommendations"] = build_recommendations(diagnosis)

    write_json(out_dir / "audit_diagnosis.json", diagnosis)
    write_markdown_report(out_dir / "audit_diagnosis.md", diagnosis)
    return diagnosis

