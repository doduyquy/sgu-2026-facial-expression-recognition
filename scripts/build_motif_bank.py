"""Build emotion-specific discriminative prototype motif bank from train split only."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.motif.motif_bank_builder import build_motif_bank
from src.motif.motif_io import save_motif_bank
from src.motif.motif_scoring import check_finite_tensor


def _score_stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return float(t.min()), float(t.mean()), float(t.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="artifacts/subgraph_graph_dataset")
    parser.add_argument("--out_dir", default="artifacts/motif_bank_v1")
    parser.add_argument("--num_motifs_per_class", type=int, default=16)
    parser.add_argument("--max_subgraphs_per_class", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--oversample_factor", type=int, default=2)
    args = parser.parse_args()

    print("=" * 72)
    print("Build Emotion-Specific Discriminative Prototype Motif Bank")
    print("=" * 72)
    for key, value in vars(args).items():
        print(f"{key:<28}: {value}")
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = build_motif_bank(
        input_dir=args.input_dir,
        num_motifs_per_class=args.num_motifs_per_class,
        max_subgraphs_per_class=args.max_subgraphs_per_class,
        alpha=args.alpha,
        seed=args.seed,
        kmeans_batch_size=args.kmeans_batch_size,
        oversample_factor=args.oversample_factor,
        num_classes=7,
    )

    print("\n[MotifBank] Summary")
    print(f"descriptor_dim : {bank.descriptor_dim}")
    print(f"num_classes    : {bank.num_classes}")
    print(f"emotion_names  : {bank.emotion_names}")

    all_intra, all_inter, all_disc = [], [], []
    for class_id in range(bank.num_classes):
        class_motifs = bank.motifs.get(class_id, [])
        print(f"  class {class_id:<2}: {len(class_motifs)} motifs")
        if len(class_motifs) != args.num_motifs_per_class:
            raise RuntimeError(
                f"Class {class_id} has {len(class_motifs)} motifs, "
                f"expected {args.num_motifs_per_class}"
            )
        for motif in class_motifs:
            proto = torch.as_tensor(motif.prototype).float()
            if tuple(proto.shape) != (bank.descriptor_dim,):
                raise RuntimeError(
                    f"Motif {motif.motif_id} prototype shape {tuple(proto.shape)} "
                    f"!= [{bank.descriptor_dim}]"
                )
            check_finite_tensor(f"motif[{motif.motif_id}].prototype", proto)
            all_intra.append(motif.intra_score)
            all_inter.append(motif.inter_score)
            all_disc.append(motif.discriminative_score)

    for name, values in [
        ("intra_score", all_intra),
        ("inter_score", all_inter),
        ("discriminative_score", all_disc),
    ]:
        mn, mean, mx = _score_stats(values)
        print(f"{name:<24}: min={mn:.4f} mean={mean:.4f} max={mx:.4f}")

    bank_path = out_dir / "motif_bank.pt"
    meta_path = out_dir / "meta.pt"
    save_motif_bank(bank, bank_path)
    torch.save(
        {
            "descriptor_dim": bank.descriptor_dim,
            "num_classes": bank.num_classes,
            "emotion_names": bank.emotion_names,
            "config": bank.config,
            "motifs_per_class": {int(c): len(ms) for c, ms in bank.motifs.items()},
            "score_stats": {
                "intra": _score_stats(all_intra),
                "inter": _score_stats(all_inter),
                "discriminative": _score_stats(all_disc),
            },
        },
        meta_path,
    )

    print("\n[Output]")
    print(f"motif_bank.pt : {bank_path} ({bank_path.stat().st_size / 1024 ** 2:.2f} MB)")
    print(f"meta.pt       : {meta_path} ({meta_path.stat().st_size / 1024 ** 2:.2f} MB)")
    print("DONE")


if __name__ == "__main__":
    main()
