import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import torch

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


def save_overlay(image_tensor, heatmaps, coords, save_path, title):
    # image_tensor: (1,H,W) normalized in [-1,1], heatmaps: (K,h,w), coords: (K,2)
    img = image_tensor.squeeze(0).cpu().numpy()
    img = (img * 0.5) + 0.5
    img = img.clip(0.0, 1.0)

    hm_sum = heatmaps.sum(dim=0).cpu().numpy()
    hm_sum = hm_sum / max(float(hm_sum.max()), 1e-6)

    h, w = img.shape
    xs = coords[:, 0].cpu().numpy() * (w - 1)
    ys = coords[:, 1].cpu().numpy() * (h - 1)

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

    n = min(args.num_samples, images.size(0))
    for i in range(n):
        save_path = out_dir / f"sample_{i:03d}_label_{int(labels[i].item())}.png"
        title = f"label={int(labels[i].item())} | K={heatmaps.size(1)}"
        save_overlay(images[i], heatmaps[i], coords[i], save_path, title)

    print(f"Saved {n} learned landmark visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
