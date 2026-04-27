"""Visualize candidate attentions from D3-Full checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.candidate_attention_dataset import CandidateAttentionDataset, collate_fn_candidate_attention
from src.models import get_model
from src.utils.config import load_config


def _load_graph_images(graph_repo_dir: str | None, split: str) -> dict[int, torch.Tensor]:
    if not graph_repo_dir:
        return {}
    try:
        from data.graph_repository import GraphRepositoryReader
        reader = GraphRepositoryReader(graph_repo_dir)
        images = {}
        for sample in reader.iter_split(split):
            images[int(sample.graph_id)] = sample.node_features[:, 0].float().reshape(48, 48)
        return images
    except Exception as exc:
        print(f"[WARN] cannot load graph images: {exc}")
        return {}


def _draw_overlay(image, bbox, weights, title: str, out_path: Path, top_n: int = 12) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    vals, idxs = torch.topk(weights.float(), k=min(top_n, int(weights.numel())))
    max_val = float(vals.max().item()) if vals.numel() else 1.0
    for value, idx in zip(vals.tolist(), idxs.tolist()):
        x1, y1, x2, y2 = [float(v) * 47.0 for v in bbox[int(idx)].tolist()]
        alpha = 0.2 + 0.75 * (float(value) / max(max_val, 1e-8))
        rect = patches.Rectangle(
            (x1, y1),
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
            linewidth=1.2,
            edgecolor="tab:red",
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_attention_dir", default="artifacts/candidate_attention_dataset_v1")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="learnable_slot_candidate_motif_gnn")
    p.add_argument("--graph_repo_dir", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", default="outputs/attention_vis/d3full")
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--top_n", type=int, default=12)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = CandidateAttentionDataset(args.candidate_attention_dir, args.split, normalize_x=True)
    config = load_config(args.config, "kaggle")
    input_dim = ds.input_dim
    model = get_model(config["model"]["name"], config=config, input_dim=input_dim)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    images = _load_graph_images(args.graph_repo_dir, args.split)

    csv_path = out_dir / "candidate_attention.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "label",
                "pred",
                "view",
                "slot_or_class",
                "candidate_idx",
                "attention",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "center_x",
                "center_y",
            ],
        )
        writer.writeheader()
        for idx in range(min(int(args.max_samples), len(ds))):
            sample = ds[idx]
            batch = collate_fn_candidate_attention([sample])
            with torch.no_grad():
                out = model(batch)
            logits = out["logits"]
            pred = int(logits.argmax(dim=1).item())
            label = int(sample["label"].item())
            graph_id = int(sample["graph_id"])
            mask = sample["candidate_mask"].bool()
            bbox = sample["candidate_bbox"]
            centers = sample["candidate_centers"]
            image = images.get(graph_id, torch.zeros(48, 48))

            slot_scores = out["candidate_attention"][0].max(dim=0).values
            slot_scores = slot_scores.masked_fill(~mask, 0.0)
            views = [("top_slot", -1, slot_scores)]
            class_attn = out.get("class_slot_attention")
            if class_attn is not None:
                cand_attn = out["candidate_attention"][0]
                pred_scores = torch.matmul(class_attn[0, pred], cand_attn)
                true_scores = torch.matmul(class_attn[0, label], cand_attn)
                views.extend([("pred_class", pred, pred_scores), ("true_class", label, true_scores)])

            for view_name, slot_or_class, weights in views:
                weights = weights.masked_fill(~mask, 0.0)
                png = out_dir / f"{args.split}_graph_{graph_id}_{view_name}.png"
                _draw_overlay(image, bbox, weights, f"{view_name} graph={graph_id} y={label} pred={pred}", png, top_n=args.top_n)
                vals, idxs = torch.topk(weights, k=min(args.top_n, int(mask.sum().item())))
                for value, cand_idx in zip(vals.tolist(), idxs.tolist()):
                    c = int(cand_idx)
                    writer.writerow(
                        {
                            "graph_id": graph_id,
                            "label": label,
                            "pred": pred,
                            "view": view_name,
                            "slot_or_class": slot_or_class,
                            "candidate_idx": c,
                            "attention": float(value),
                            "bbox_x1": float(bbox[c, 0]),
                            "bbox_y1": float(bbox[c, 1]),
                            "bbox_x2": float(bbox[c, 2]),
                            "bbox_y2": float(bbox[c, 3]),
                            "center_x": float(centers[c, 0]),
                            "center_y": float(centers[c, 1]),
                        }
                    )
    print(f"saved: {csv_path}")
    print(f"saved PNGs -> {out_dir}")


if __name__ == "__main__":
    main()

