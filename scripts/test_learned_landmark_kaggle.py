import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.data.dataloader import build_dataloader
from src.models import get_model
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize learned landmark heatmaps on Kaggle.")
    parser.add_argument("--config", type=str, default="resnet")
    parser.add_argument("--env", type=str, default="kaggle", choices=["local", "kaggle"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/learned_landmark_test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--show_points", action="store_true", help="Overlay landmark points (default: off)")
    parser.add_argument("--save_per_keypoint", action="store_true", help="Save per-keypoint heatmap grid")
    parser.add_argument("--save_attention_only", action="store_true", help="Save attention-only heatmap (after attention)")
    return parser.parse_args()


def _soft_argmax_pixel_coords(heatmaps_up):
    # heatmaps_up: (K,H,W), returns coords in image pixel space
    keypoints, h, w = heatmaps_up.shape
    flat = heatmaps_up.view(keypoints, -1)
    probs = flat / flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    xs = torch.linspace(0, w - 1, w, device=heatmaps_up.device, dtype=heatmaps_up.dtype)
    ys = torch.linspace(0, h - 1, h, device=heatmaps_up.device, dtype=heatmaps_up.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    x = (probs * grid_x).sum(dim=-1)
    y = (probs * grid_y).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def _region_scores(hm_sum):
    # hm_sum: (H,W), normalized non-negative heatmap
    h, w = hm_sum.shape
    total = hm_sum.sum().clamp(min=1e-6)

    # Approximate FER facial priors on aligned 48x48 crops.
    left_eye = hm_sum[int(0.18 * h):int(0.45 * h), int(0.10 * w):int(0.45 * w)].sum()
    right_eye = hm_sum[int(0.18 * h):int(0.45 * h), int(0.55 * w):int(0.90 * w)].sum()
    nose = hm_sum[int(0.38 * h):int(0.67 * h), int(0.35 * w):int(0.65 * w)].sum()
    mouth = hm_sum[int(0.60 * h):int(0.93 * h), int(0.20 * w):int(0.80 * w)].sum()

    return {
        "left_eye": (left_eye / total).item(),
        "right_eye": (right_eye / total).item(),
        "nose": (nose / total).item(),
        "mouth": (mouth / total).item(),
    }


def save_overlay(
    image_tensor,
    heatmaps,
    save_path,
    title,
    show_points=False,
    per_keypoint_path=None,
    attention_only_path=None,
):
    # image_tensor: (1,H,W) normalized in [-1,1], heatmaps: (K,h,w)
    img = image_tensor.squeeze(0).cpu().numpy()
    img = (img * 0.5) + 0.5
    img = img.clip(0.0, 1.0)

    img_h, img_w = img.shape
    heatmaps_up = F.interpolate(
        heatmaps.unsqueeze(0),
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    coords_img = _soft_argmax_pixel_coords(heatmaps_up)

    hm_sum = heatmaps_up.sum(dim=0).cpu().numpy()
    hm_sum = hm_sum / max(float(hm_sum.max()), 1e-6)

    xs = coords_img[:, 0].cpu().numpy()
    ys = coords_img[:, 1].cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img, cmap="gray")
    ax.imshow(hm_sum, cmap="magma", alpha=0.35)
    if show_points:
        ax.scatter(xs, ys, c="lime", s=18)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if per_keypoint_path is not None:
        k = heatmaps_up.size(0)
        cols = min(4, k)
        rows = (k + cols - 1) // cols
        fig2, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
        axes = axes if isinstance(axes, (list, tuple)) else axes.reshape(rows, cols)
        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax2 = axes[r][c]
            if idx < k:
                hm_k = heatmaps_up[idx].cpu().numpy()
                hm_k = hm_k / max(float(hm_k.max()), 1e-6)
                ax2.imshow(img, cmap="gray")
                ax2.imshow(hm_k, cmap="magma", alpha=0.45)
                if show_points:
                    ax2.scatter([xs[idx]], [ys[idx]], c="lime", s=18)
                ax2.set_title(f"kp={idx}")
            ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(per_keypoint_path, dpi=160, bbox_inches="tight")
        plt.close(fig2)

    if attention_only_path is not None:
        fig3, ax3 = plt.subplots(1, 1, figsize=(4, 4))
        im = ax3.imshow(hm_sum, cmap="magma", vmin=0.0, vmax=1.0)
        ax3.set_title(f"{title} | attention-only")
        ax3.axis("off")
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        fig3.tight_layout()
        fig3.savefig(attention_only_path, dpi=180, bbox_inches="tight")
        plt.close(fig3)

    scores = _region_scores(heatmaps_up.sum(dim=0))
    return scores


def main():
    args = parse_args()
    cfg = load_config(model=args.config, env=args.env)

    # Learned landmark branch does not need external detector.
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg["model"]["name"] = "resnet"
    cfg["model"]["use_learned_landmark_branch"] = True
    cfg["data"]["use_landmarks"] = False

    data_path = cfg["kaggle"]["data_path"] if args.env == "kaggle" else cfg["local"]["data_path"]
    train_loader, val_loader, test_loader = build_dataloader(config=cfg, data_path=data_path)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(name="resnet", config=cfg).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        _ = model(images)
        heatmaps, coords = model.get_landmark_outputs()

    if heatmaps is None or coords is None:
        raise RuntimeError("Model did not produce landmark heatmaps. Check config/model branch.")

    print(f"Raw normalized coords range: min={coords.min().item():.4f}, max={coords.max().item():.4f}")

    n = min(args.num_samples, images.size(0))
    for i in range(n):
        save_path = out_dir / f"sample_{i:03d}_label_{int(labels[i].item())}.png"
        per_k_path = None
        attn_only_path = None
        if args.save_per_keypoint:
            per_k_path = out_dir / f"sample_{i:03d}_label_{int(labels[i].item())}_per_k.png"
        if args.save_attention_only:
            attn_only_path = out_dir / f"sample_{i:03d}_label_{int(labels[i].item())}_attn_only.png"
        title = f"label={int(labels[i].item())} | K={heatmaps.size(1)}"
        region_scores = save_overlay(
            images[i],
            heatmaps[i],
            save_path,
            title,
            show_points=args.show_points,
            per_keypoint_path=per_k_path,
            attention_only_path=attn_only_path,
        )
        print(
            "sample={} label={} region_focus: left_eye={:.3f} right_eye={:.3f} nose={:.3f} mouth={:.3f}".format(
                i,
                int(labels[i].item()),
                region_scores["left_eye"],
                region_scores["right_eye"],
                region_scores["nose"],
                region_scores["mouth"],
            )
        )

    print(f"Saved {n} learned landmark visualizations to: {out_dir}")


if __name__ == "__main__":
    main()