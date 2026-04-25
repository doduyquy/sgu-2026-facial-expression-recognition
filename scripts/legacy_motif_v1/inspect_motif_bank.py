"""Inspect a saved motif bank."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.motif.motif_io import load_motif_bank
from src.motif.motif_scoring import check_finite_tensor


def _stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return f"min={t.min().item():.4f} mean={t.mean().item():.4f} max={t.max().item():.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_bank_path", default="artifacts/motif_bank_v1/motif_bank.pt")
    args = parser.parse_args()

    bank = load_motif_bank(args.motif_bank_path)
    print("=" * 72)
    print("Inspect Motif Bank")
    print("=" * 72)
    print(f"path           : {args.motif_bank_path}")
    print(f"descriptor_dim : {bank.descriptor_dim}")
    print(f"num_classes    : {bank.num_classes}")
    print(f"emotion_names  : {bank.emotion_names}")
    print(f"config         : {bank.config}")

    all_intra, all_inter, all_disc = [], [], []
    for class_id in range(bank.num_classes):
        motifs = bank.motifs.get(class_id, [])
        print("\n" + "-" * 72)
        print(f"class {class_id} ({bank.emotion_names[class_id] if class_id < len(bank.emotion_names) else class_id})")
        print(f"num motifs      : {len(motifs)}")
        if motifs:
            proto0 = torch.as_tensor(motifs[0].prototype).float()
            print(f"prototype shape : {tuple(proto0.shape)}")
        intra = [m.intra_score for m in motifs]
        inter = [m.inter_score for m in motifs]
        disc = [m.discriminative_score for m in motifs]
        if motifs:
            print(f"intra_score     : {_stats(intra)}")
            print(f"inter_score     : {_stats(inter)}")
            print(f"disc_score      : {_stats(disc)}")
        top = sorted(motifs, key=lambda m: m.discriminative_score, reverse=True)[:5]
        print("top 5 by discriminative_score:")
        for motif in top:
            proto = torch.as_tensor(motif.prototype).float()
            check_finite_tensor(f"motif[{motif.motif_id}].prototype", proto)
            print(
                f"  id={motif.motif_id:<3} support={motif.support:<6} "
                f"intra={motif.intra_score:.4f} inter={motif.inter_score:.4f} "
                f"disc={motif.discriminative_score:.4f}"
            )
        all_intra.extend(intra)
        all_inter.extend(inter)
        all_disc.extend(disc)

    print("\n" + "=" * 72)
    print("Global score stats")
    print(f"intra_score           : {_stats(all_intra)}")
    print(f"inter_score           : {_stats(all_inter)}")
    print(f"discriminative_score  : {_stats(all_disc)}")
    print("No NaN/Inf detected in prototypes.")


if __name__ == "__main__":
    main()
