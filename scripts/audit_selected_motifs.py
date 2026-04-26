"""Audit selected motif subgraphs in pixel_motif_dataset_v2 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.motif_audit import (
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_COVERAGE_CELLS,
    MotifAuditAccumulator,
    load_meta,
    load_split_samples,
    resolve_splits,
    save_plots,
    save_overlay_plots,
    validate_samples,
    write_matrix_csv,
    write_motif_score_vector_csv,
    write_score_stats_csv,
    write_selected_per_sample_csv,
    write_summary_json,
    write_summary_txt,
    write_top_motifs_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit selected motif subgraphs without training or rebuilding artifacts."
    )
    p.add_argument("--pixel_motif_dir", default="artifacts/pixel_motif_dataset_v2")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--out_dir", default="outputs/audit/motif_selection_test")
    p.add_argument("--top_n_motifs", type=int, default=20)
    p.add_argument(
        "--save_per_sample_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write selected_motifs_per_sample.csv. Enabled by default.",
    )
    p.add_argument(
        "--save_summary_json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.json. Enabled by default.",
    )
    p.add_argument("--save_plots", action="store_true", help="Write PNG plots if matplotlib is available.")
    p.add_argument(
        "--graph_repo_dir",
        default=None,
        help="Optional graph_repo path used only for bbox overlay reconstruction.",
    )
    p.add_argument(
        "--save_overlays",
        action="store_true",
        help="Save selected bbox overlays. Requires --graph_repo_dir and matplotlib.",
    )
    p.add_argument("--max_overlay_samples", type=int, default=24)
    p.add_argument(
        "--normalize_matrices",
        action="store_true",
        help="Write coverage/matched-class matrix values as row percentages instead of counts.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    pixel_motif_dir = Path(args.pixel_motif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(pixel_motif_dir)
    num_classes = int(meta.get("num_classes") or DEFAULT_NUM_CLASSES)
    coverage_grid = meta.get("coverage_grid")
    if coverage_grid and len(coverage_grid) == 2:
        num_coverage_cells = int(coverage_grid[0]) * int(coverage_grid[1])
    else:
        num_coverage_cells = DEFAULT_NUM_COVERAGE_CELLS
    class_names = meta.get("class_names")
    if class_names is not None:
        class_names = list(class_names)

    print("=" * 90)
    print("Audit Selected Pixel Motifs")
    print("=" * 90)
    print(f"pixel_motif_dir    : {pixel_motif_dir}")
    print(f"split              : {args.split}")
    print(f"out_dir            : {out_dir}")
    print(f"num_classes        : {num_classes}")
    print(f"coverage_cells     : {num_coverage_cells}")
    if class_names:
        print(f"class_names        : {class_names}")

    splits = resolve_splits(args.split)
    samples_by_split = []
    warnings: list[str] = []
    for split in splits:
        samples = load_split_samples(pixel_motif_dir, split)
        split_warnings = validate_samples(samples, split)
        warnings.extend(split_warnings)
        samples_by_split.append((split, samples))
        print(f"[{split}] loaded {len(samples)} samples")
        for warning in split_warnings:
            print(f"WARNING: {warning}")

    acc = MotifAuditAccumulator(
        num_classes=num_classes,
        num_coverage_cells=num_coverage_cells,
        top_n_motifs=args.top_n_motifs,
        class_names=class_names,
    )

    for split, samples in samples_by_split:
        total = len(samples)
        for idx, sample in enumerate(samples):
            acc.add_sample(sample)
            if (idx + 1) % 2000 == 0 or idx + 1 == total:
                print(f"[{split}] audited {idx + 1:6d}/{total}", flush=True)

    if args.save_per_sample_csv:
        path = out_dir / "selected_motifs_per_sample.csv"
        write_selected_per_sample_csv(path, samples_by_split)
        print(f"saved: {path}")

    coverage_path = out_dir / "coverage_by_true_class.csv"
    write_matrix_csv(
        coverage_path,
        row_name="true_class",
        col_prefix="coverage_cell",
        counters_by_label=acc.coverage_by_label,
        num_rows=num_classes,
        num_cols=num_coverage_cells,
        normalize=args.normalize_matrices,
    )
    print(f"saved: {coverage_path}")

    matched_path = out_dir / "matched_class_by_true_class.csv"
    write_matrix_csv(
        matched_path,
        row_name="true_class",
        col_prefix="matched_class",
        counters_by_label=acc.matched_by_label,
        num_rows=num_classes,
        num_cols=num_classes,
        normalize=args.normalize_matrices,
    )
    print(f"saved: {matched_path}")

    top_motifs_path = out_dir / "top_motifs_by_true_class.csv"
    write_top_motifs_csv(top_motifs_path, acc)
    print(f"saved: {top_motifs_path}")

    motif_vector_path = out_dir / "motif_score_vector_by_true_class.csv"
    write_motif_score_vector_csv(motif_vector_path, acc)
    print(f"saved: {motif_vector_path}")

    score_stats_path = out_dir / "match_score_stats_by_true_class.csv"
    write_score_stats_csv(score_stats_path, acc)
    print(f"saved: {score_stats_path}")

    summary = acc.summary()
    if warnings:
        summary["artifact_warnings"] = warnings
    if args.save_summary_json:
        summary_path = out_dir / "summary.json"
        write_summary_json(summary_path, summary)
        print(f"saved: {summary_path}")
    summary_txt_path = out_dir / "summary.txt"
    write_summary_txt(summary_txt_path, summary, class_names=class_names)
    print(f"saved: {summary_txt_path}")

    if args.save_plots:
        plot_results = save_plots(out_dir, acc)
        for item in plot_results:
            if item.endswith(".png"):
                print(f"saved plot: {out_dir / item}")
            else:
                print(f"WARNING: {item}")

    if args.save_overlays:
        if not args.graph_repo_dir:
            print("WARNING: --save_overlays requires --graph_repo_dir; skipped overlays")
        else:
            overlay_results = save_overlay_plots(
                out_dir=out_dir,
                graph_repo_dir=Path(args.graph_repo_dir),
                samples_by_split=samples_by_split,
                max_overlay_samples=args.max_overlay_samples,
            )
            for item in overlay_results:
                if item.endswith(".png"):
                    print(f"saved overlay: {out_dir / item}")
                else:
                    print(f"WARNING: {item}")

    print("\nSummary")
    print("-" * 90)
    print(f"num_samples          : {summary['num_samples']}")
    print(f"num_selected_total   : {summary['num_selected_total']}")
    print(f"valid_selected_ratio : {summary['valid_selected_ratio']:.4f}")
    if summary["warnings"]:
        print("warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")
    else:
        print("warnings             : none")
    print("DONE")


if __name__ == "__main__":
    main()
