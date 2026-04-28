"""Thin entrypoint for config-driven experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import run_experiment


def main() -> None:
    os.chdir(ROOT_DIR)
    parser = argparse.ArgumentParser(
        description="Run a pixel motif experiment from config.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "--experiment", dest="config", default="hierarchical_motif_gnn_c")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["build_and_train", "train_from_artifact", "build_only", "train_only", "debug_only"],
        help=(
            "build_and_train      : build artifacts từ CSV rồi train (default)\n"
            "train_from_artifact  : load artifact từ --artifact_input_path rồi train\n"
            "build_only           : chỉ build artifact, không train\n"
            "train_only           : dùng artifact đã có trong working, chỉ train\n"
            "debug_only           : chỉ chạy debug batch"
        ),
    )
    parser.add_argument(
        "--artifact_input_path",
        default=None,
        help="Path tới artifact đã lưu (dùng khi mode=train_from_artifact).\n"
             "Ví dụ: /kaggle/input/fer2013-pixel-motif-v2-spatial-r12-k32-n25/artifacts",
    )
    parser.add_argument(
        "--zip_artifacts",
        action="store_true",
        help="Zip toàn bộ artifacts sau khi build xong (để tải về hoặc publish Kaggle Dataset).",
    )
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--pixel_motif_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_skip_existing", action="store_true")
    # Legacy flags kept for backward compatibility
    parser.add_argument("--build_only", action="store_true", help="(legacy) Dùng --mode build_only thay thế.")
    parser.add_argument("--train_only", action="store_true", help="(legacy) Dùng --mode train_only thay thế.")
    parser.add_argument("--debug_only", action="store_true", help="(legacy) Dùng --mode debug_only thay thế.")
    args = parser.parse_args()

    run_experiment(
        args.config,
        csv_root=args.csv_root,
        out_root=args.out_root,
        pixel_motif_dir=args.pixel_motif_dir,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
        smoke=args.smoke,
        mode=args.mode,
        artifact_input_path=args.artifact_input_path,
        zip_artifacts_after_build=args.zip_artifacts,
        build_only=args.build_only,
        train_only=args.train_only,
        debug_only=args.debug_only,
        no_wandb=args.no_wandb,
        no_skip_existing=args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
