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


def save_overlay(image_tensor, heatmaps, coords, save_path, title):
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
    ax.scatter(xs, ys, c="lime", s=18)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        title = f"label={int(labels[i].item())} | K={heatmaps.size(1)}"
        save_overlay(images[i], heatmaps[i], coords[i], save_path, title)

    print(f"Saved {n} learned landmark visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
