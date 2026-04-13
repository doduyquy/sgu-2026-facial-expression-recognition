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
    parser = argparse.ArgumentParser(description="Test landmark branch only (learnable/input/hybrid).")
    parser.add_argument("--config", type=str, default="resnet", help="Config name in configs/*.yaml")
    parser.add_argument("--env", type=str, default="kaggle", choices=["local", "kaggle"], help="Runtime env")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Batch source split")
    parser.add_argument("--mode", type=str, default="learnable", choices=["learnable", "input", "hybrid"], help="Landmark token mode")
    parser.add_argument("--save_dir", type=str, default="outputs/landmark_branch_test", help="Output directory")
    return parser.parse_args()


def _pick_batch(loader):
    batch = next(iter(loader))
    if len(batch) == 4:
        images, labels, landmarks, landmark_mask = batch
    elif len(batch) == 3:
        images, labels, landmarks = batch
        landmark_mask = None
    else:
        images, labels = batch
        landmarks = None
        landmark_mask = None
    return images, labels, landmarks, landmark_mask


def _save_token_map(token_tensor, save_path, title):
    # token_tensor: (P, D)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    im = ax.imshow(token_tensor.cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Feature dim")
    ax.set_ylabel("Landmark token index")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_input_landmark_preview(landmarks, save_path):
    # landmarks expected shape: (B, P, H, W) for heatmap mode
    if landmarks is None or landmarks.dim() != 4:
        return

    hm = landmarks[0].sum(dim=0)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(hm.cpu().numpy(), cmap="magma")
    ax.set_title("Input landmark heatmap (sum over points)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    cfg = load_config(model=args.config, env=args.env)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})

    cfg["model"]["landmark_token_mode"] = args.mode
    cfg["model"]["use_landmark_cross_fusion"] = True
    cfg["model"]["use_pyramid_multi_scale"] = True

    # For input/hybrid mode, dataloader should provide landmarks.
    cfg["data"]["use_landmarks"] = args.mode in ("input", "hybrid")

    data_path = cfg.get("data_path", None)
    if data_path is None:
        raise ValueError("config has no data_path. Check env config.")

    train_loader, val_loader, test_loader = build_dataloader(config=cfg, data_path=data_path)
    loader_map = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    loader = loader_map[args.split]

    images, labels, landmarks, landmark_mask = _pick_batch(loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(name=cfg["model"]["name"], config=cfg).to(device)
    model.eval()

    images = images.to(device)
    landmarks = landmarks.to(device) if landmarks is not None else None
    landmark_mask = landmark_mask.to(device) if landmark_mask is not None else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        lm_tokens, lm_mask = model._resolve_landmark_tokens(
            landmarks=landmarks,
            landmark_mask=landmark_mask,
            batch_size=images.size(0),
            device=images.device,
            dtype=images.dtype,
        )

        # Build image features only to test pyramid branch independently.
        x = model.relu(model.bn1(model.conv1(images)))
        x = model.pool(x)
        x2 = model.layer2(x)
        x3 = model.layer3(x2)
        x4 = model.layer4(x3)

        xs, xm, xl = model.pyramid_cross_fusion(
            x2,
            x3,
            x4,
            lm_tokens,
            landmark_mask=lm_mask,
        )

        logits = model(images, landmarks, landmark_mask)

    # Save visual diagnostics.
    _save_token_map(lm_tokens[0], save_dir / f"tokens_{args.mode}.png", f"Landmark tokens ({args.mode})")
    _save_input_landmark_preview(landmarks, save_dir / "input_landmark_heatmap.png")

    summary_path = save_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"mode: {args.mode}\n")
        f.write(f"batch_size: {images.size(0)}\n")
        f.write(f"images: {tuple(images.shape)}\n")
        f.write(f"landmarks: {None if landmarks is None else tuple(landmarks.shape)}\n")
        f.write(f"landmark_mask: {None if landmark_mask is None else tuple(landmark_mask.shape)}\n")
        f.write(f"resolved_lm_tokens: {tuple(lm_tokens.shape)}\n")
        f.write(f"resolved_lm_mask: {None if lm_mask is None else tuple(lm_mask.shape)}\n")
        f.write(f"pyramid_xs: {tuple(xs.shape)}\n")
        f.write(f"pyramid_xm: {tuple(xm.shape)}\n")
        f.write(f"pyramid_xl: {tuple(xl.shape)}\n")
        f.write(f"logits: {tuple(logits.shape)}\n")
        f.write(f"tokens_mean: {float(lm_tokens.mean().item()):.6f}\n")
        f.write(f"tokens_std: {float(lm_tokens.std().item()):.6f}\n")

    print(f"Saved diagnostics to: {save_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
