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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--experiment", dest="config", default="hierarchical_motif_gnn_c")
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--pixel_motif_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--debug_only", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_skip_existing", action="store_true")
    args = parser.parse_args()

    run_experiment(
        args.config,
        csv_root=args.csv_root,
        out_root=args.out_root,
        pixel_motif_dir=args.pixel_motif_dir,
        epochs=args.epochs,
        smoke=args.smoke,
        build_only=args.build_only,
        train_only=args.train_only,
        debug_only=args.debug_only,
        no_wandb=args.no_wandb,
        no_skip_existing=args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
