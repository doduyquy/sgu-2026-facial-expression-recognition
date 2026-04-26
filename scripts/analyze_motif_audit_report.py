"""Analyze D0 motif audit outputs and produce a diagnosis report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.motif_audit_report import analyze_audit_report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze selected motif audit report outputs.")
    p.add_argument("--audit_dir", default="outputs/audit/motif_selection_test")
    p.add_argument("--out_dir", default="outputs/audit/motif_selection_test/report")
    p.add_argument("--low_diag_threshold", type=float, default=0.18)
    p.add_argument("--motif_rank_warn_gt", type=int, default=2)
    p.add_argument("--coverage_high_ratio_threshold", type=float, default=0.96)
    p.add_argument("--coverage_low_ratio_threshold", type=float, default=0.70)
    p.add_argument("--global_motif_min_classes", type=int, default=4)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    diagnosis = analyze_audit_report(
        audit_dir=Path(args.audit_dir),
        out_dir=Path(args.out_dir),
        low_diag_threshold=args.low_diag_threshold,
        motif_rank_warn_gt=args.motif_rank_warn_gt,
        coverage_high_ratio_threshold=args.coverage_high_ratio_threshold,
        coverage_low_ratio_threshold=args.coverage_low_ratio_threshold,
        global_motif_min_classes=args.global_motif_min_classes,
    )

    print("=" * 90)
    print("Motif Audit Diagnosis")
    print("=" * 90)
    print(f"audit_dir : {args.audit_dir}")
    print(f"out_dir   : {args.out_dir}")
    print("\nRecommendations:")
    for rec in diagnosis["recommendations"]:
        print(f"- {rec}")
    print("\nWarnings:")
    if diagnosis["warnings"]:
        for warning in diagnosis["warnings"]:
            print(f"- {warning}")
    else:
        print("- none")
    print("\nSaved:")
    for name in [
        "audit_diagnosis.md",
        "audit_diagnosis.json",
        "normalized_matched_class_by_true_class.csv",
        "normalized_coverage_by_true_class.csv",
        "motif_score_rank_by_true_class.csv",
        "suspicious_global_motifs.csv",
        "class_bias_report.csv",
    ]:
        print(f"- {Path(args.out_dir) / name}")
    print("DONE")


if __name__ == "__main__":
    main()

