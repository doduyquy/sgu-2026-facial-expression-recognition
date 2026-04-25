"""Inspect pixel-preserving motif bank with exemplars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.motif_v2.io import load_pixel_motif_bank


def _stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return f"min={t.min().item():.4f} mean={t.mean().item():.4f} max={t.max().item():.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--motif_bank_path", default="artifacts/pixel_motif_bank_v2/pixel_motif_bank.pt")
    args = p.parse_args()
    bank = load_pixel_motif_bank(args.motif_bank_path)
    print("=" * 80)
    print("Inspect Pixel-preserving Motif Bank V2")
    print("=" * 80)
    print(f"path           : {args.motif_bank_path}")
    print(f"descriptor_dim : {bank.descriptor_dim}")
    print(f"num_classes    : {bank.num_classes}")
    print(f"emotion_names  : {bank.emotion_names}")
    print(f"config         : {bank.config}")

    all_intra, all_inter, all_disc = [], [], []
    for class_id in range(bank.num_classes):
        motifs = bank.motifs.get(class_id, [])
        print("\n" + "-" * 80)
        print(f"class {class_id} {bank.emotion_names[class_id] if class_id < len(bank.emotion_names) else ''}")
        print(f"num motifs: {len(motifs)}")
        if not motifs:
            continue
        intra = [m.intra_score for m in motifs]
        inter = [m.inter_score for m in motifs]
        disc = [m.discriminative_score for m in motifs]
        print(f"intra : {_stats(intra)}")
        print(f"inter : {_stats(inter)}")
        print(f"disc  : {_stats(disc)}")
        print("top 5 motifs:")
        for motif in sorted(motifs, key=lambda m: m.discriminative_score, reverse=True)[:5]:
            proto = torch.as_tensor(motif.prototype).float()
            if not torch.isfinite(proto).all():
                raise ValueError(f"motif {motif.motif_id} prototype has NaN/Inf")
            ex = motif.exemplars[0] if motif.exemplars else {}
            print(
                f"  id={motif.motif_id:<3} support={motif.support:<6} "
                f"intra={motif.intra_score:.4f} inter={motif.inter_score:.4f} "
                f"disc={motif.discriminative_score:.4f} "
                f"ex_graph={ex.get('graph_id')} ex_candidate={ex.get('candidate_id')} "
                f"bbox={tuple(round(float(v), 3) for v in ex.get('bbox', torch.zeros(4)).tolist()) if ex else None}"
            )
        all_intra.extend(intra)
        all_inter.extend(inter)
        all_disc.extend(disc)
    print("\nGlobal:")
    print(f"intra : {_stats(all_intra)}")
    print(f"inter : {_stats(all_inter)}")
    print(f"disc  : {_stats(all_disc)}")
    print("No NaN/Inf detected.")


if __name__ == "__main__":
    main()
