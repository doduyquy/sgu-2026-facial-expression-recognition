"""
Debug GraphNN subgraph pooling assignments.

Example:
    python scripts/debug_graphnn_subgraphs.py ^
        --checkpoint outputs/best_model.pth ^
        --data-path dataset/fer13-split ^
        --split test ^
        --num-correct 10 ^
        --num-wrong 10

The script saves visualizations to outputs/graphnn_subgraphs by default.
Each saved figure shows:
    1. the input image
    2. overlay heatmaps for graph focus
    3. hard assignment map: each patch -> argmax subgraph
    4. one heatmap per subgraph: soft assignment probability
"""

import argparse
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

REGION_NAMES = [
    "forehead",
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "chin",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=str(ROOT / "dataset" / "fer13-split"))
    parser.add_argument("--config", type=str, default="motif_gnn")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-correct", type=int, default=10)
    parser.add_argument("--num-wrong", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    from src.models.GraphNN import MotifGNN

    model = MotifGNN(config, channels=config["data"].get("channels", 1)).to(device)

    if checkpoint_path is None:
        print("[WARN] No --checkpoint provided. Model is random, so correct/wrong is not meaningful.")
        model.eval()
        return model

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Handle common DataParallel prefix.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys: {incompatible.unexpected_keys}")
    model.eval()
    print(f"[OK] Loaded checkpoint: {checkpoint_path}")
    return model


def recover_image_for_display(x):
    """Transforms normalize with mean=0.5, std=0.5, so invert to [0, 1]."""
    img = x.detach().cpu()
    if img.ndim == 3:
        img = img[0]
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)


def get_subgraph_assignments(model, images):
    """
    Returns:
        logits: [B, C]
        S: [B, T, K], soft patch-to-subgraph assignment
        hard: [B, T], argmax subgraph per patch
        probs: [B, C]
        focus_maps: dict[str, Tensor[B, T]]
    """
    nodes = model._pixel_to_nodes(images)
    h, adj = model._encode_graph(nodes, return_adjacency=True)

    region_prior = getattr(model, "region_prior", None)
    subgraphs, _, S = model.subgraph_pool(h, region_prior)

    sg_attn, _ = model.subgraph_attn(subgraphs, subgraphs, subgraphs)
    subgraphs = model.attn_norm(subgraphs + sg_attn)
    graph_repr = subgraphs.mean(dim=1)
    logits = model.classifier(graph_repr)

    probs = F.softmax(logits, dim=-1)
    hard = S.argmax(dim=-1)

    focus_maps = {
        "A2_cos_focus": adjacency_focus(adj["A2_cos"]),
        "A2_hybrid_focus": adjacency_focus(adj["A2_hybrid"]),
        "S_confidence": S.max(dim=-1).values,
    }
    return logits, S, hard, probs, focus_maps


def adjacency_focus(A):
    """
    Convert adjacency [B, T, T] to one node-importance score per patch.
    We remove the diagonal first so the heatmap shows cross-patch attention
    instead of every patch mostly attending to itself.
    """
    T = A.size(-1)
    eye = torch.eye(T, device=A.device, dtype=torch.bool).unsqueeze(0)
    A_no_self = A.masked_fill(eye, 0.0)
    A_no_self = A_no_self / (A_no_self.sum(dim=-1, keepdim=True) + 1e-8)
    return A_no_self.mean(dim=1)


def patch_grid_shape(image_size, window_size, stride):
    grid_h = (image_size - window_size) // stride + 1
    grid_w = (image_size - window_size) // stride + 1
    return grid_h, grid_w


def save_assignment_figure(
    out_path,
    image,
    S_one,
    hard_one,
    grid_h,
    grid_w,
    title,
    focus_maps=None,
):
    K = S_one.shape[-1]
    hard_map = hard_one.reshape(grid_h, grid_w).detach().cpu()
    soft_maps = S_one.reshape(grid_h, grid_w, K).detach().cpu()
    focus_maps = focus_maps or {}

    panel_w = 180
    panel_h = 210
    image_h = 170
    cols = 4
    rows = math.ceil((2 + len(focus_maps) + K) / cols)
    title_h = 34
    canvas = Image.new("RGB", (cols * panel_w, title_h + rows * panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))

    def paste_panel(idx, panel_img, label):
        row = idx // cols
        col = idx % cols
        x = col * panel_w
        y = title_h + row * panel_h
        draw.text((x + 8, y + 8), label, fill=(0, 0, 0))
        panel_img = panel_img.resize((image_h, image_h), resample=Image.Resampling.NEAREST)
        canvas.paste(panel_img, (x + 5, y + 32))

    gray = (image * 255).to(torch.uint8)
    gray_img = Image.new("L", (gray.shape[1], gray.shape[0]))
    gray_img.putdata(gray.flatten().tolist())
    paste_panel(0, gray_img.convert("RGB"), "image")

    next_panel = 1
    for name, values in focus_maps.items():
        heat = values.reshape(grid_h, grid_w).detach().cpu()
        overlay = make_overlay_image(image, heat)
        paste_panel(next_panel, overlay, name)
        next_panel += 1

    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    hard_img = Image.new("RGB", (grid_w, grid_h))
    hard_pixels = [palette[int(v) % len(palette)] for v in hard_map.reshape(-1).tolist()]
    hard_img.putdata(hard_pixels)
    paste_panel(next_panel, hard_img, "argmax subgraph")
    next_panel += 1

    for k in range(K):
        prob = soft_maps[:, :, k].clamp(0.0, 1.0)
        # Simple blue -> yellow heatmap.
        heat_img = Image.new("RGB", (grid_w, grid_h))
        red = (prob * 255).to(torch.uint8).reshape(-1).tolist()
        green = (prob * 220).to(torch.uint8).reshape(-1).tolist()
        blue = ((1.0 - prob) * 255).to(torch.uint8).reshape(-1).tolist()
        heat_img.putdata(list(zip(red, green, blue)))
        region_name = REGION_NAMES[k] if k < len(REGION_NAMES) else f"S{k}"
        paste_panel(next_panel + k, heat_img, f"{region_name} prob")

    canvas.save(out_path)


def normalize_map(values):
    values = values.float()
    v_min = values.min()
    v_max = values.max()
    return (values - v_min) / (v_max - v_min + 1e-8)


def make_overlay_image(image, heat_grid, alpha=0.45):
    """
    Overlay a low-resolution patch heatmap on the original FER image.
    Red/yellow means stronger graph focus for that patch.
    """
    heat_grid = normalize_map(heat_grid).clamp(0.0, 1.0)
    H, W = image.shape

    base = (image * 255).clamp(0, 255).to(torch.uint8)
    base_img = Image.new("L", (W, H))
    base_img.putdata(base.flatten().tolist())
    base_img = base_img.convert("RGB")

    heat_img = Image.new("RGB", (heat_grid.shape[1], heat_grid.shape[0]))
    red = (heat_grid * 255).to(torch.uint8).reshape(-1).tolist()
    green = (heat_grid * 210).to(torch.uint8).reshape(-1).tolist()
    blue = ((1.0 - heat_grid) * 80).to(torch.uint8).reshape(-1).tolist()
    heat_img.putdata(list(zip(red, green, blue)))
    heat_img = heat_img.resize((W, H), resample=Image.Resampling.BILINEAR)

    return Image.blend(base_img, heat_img, alpha=alpha)


def print_sample_log(kind, sample_idx, label, pred, conf, hard_one, S_one, num_subgraphs):
    counts = torch.bincount(hard_one.detach().cpu(), minlength=num_subgraphs)
    mean_probs = S_one.detach().cpu().mean(dim=0)
    label_name = EMOTION_DICT.get(int(label), str(int(label)))
    pred_name = EMOTION_DICT.get(int(pred), str(int(pred)))
    print(
        f"[{kind}] idx={sample_idx:05d} "
        f"true={label_name}({int(label)}) pred={pred_name}({int(pred)}) conf={conf:.3f}"
    )
    print(f"       hard patch counts per subgraph: {counts.tolist()}")
    print(f"       mean soft probability per subgraph: {[round(x, 4) for x in mean_probs.tolist()]}")


def main():
    args = parse_args()
    global torch, F, Image, ImageDraw, DataLoader
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader

    from src.data.dataset import FER2013
    from src.data.transforms import build_transform
    from src.utils.config import load_config

    if args.output_dir is None:
        args.output_dir = "/kaggle/working/graphnn_heatmaps" if args.env == "kaggle" else str(ROOT / "outputs" / "graphnn_subgraphs")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, args.env)
    config["data"]["batch_size"] = args.batch_size
    config["data"]["channels"] = 1

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = load_model(config, args.checkpoint, device)

    transform = build_transform(config, args.split)
    dataset = FER2013(args.data_path, split=args.split, transforms=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    grid_h, grid_w = patch_grid_shape(
        config["data"].get("image_size", 48),
        config["model"].get("window_size", 5),
        config["model"].get("stride", 2),
    )
    expected_nodes = grid_h * grid_w
    num_subgraphs = config["model"].get("num_subgraphs", 6)
    print(f"[INFO] Patch grid: {grid_h} x {grid_w} = {expected_nodes} nodes")
    print(f"[INFO] Need {args.num_correct} correct and {args.num_wrong} wrong samples")

    found_correct = 0
    found_wrong = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, S, hard, probs, focus_maps = get_subgraph_assignments(model, images)
        preds = logits.argmax(dim=1)
        confs = probs.max(dim=1).values

        for b in range(images.size(0)):
            sample_idx = seen + b
            is_correct = bool(preds[b].item() == labels[b].item())

            if is_correct and found_correct >= args.num_correct:
                continue
            if (not is_correct) and found_wrong >= args.num_wrong:
                continue

            kind = "correct" if is_correct else "wrong"
            if is_correct:
                found_correct += 1
                n = found_correct
            else:
                found_wrong += 1
                n = found_wrong

            print_sample_log(
                kind,
                sample_idx,
                labels[b].item(),
                preds[b].item(),
                confs[b].item(),
                hard[b],
                S[b],
                num_subgraphs,
            )

            image = recover_image_for_display(images[b])
            title = (
                f"{kind} #{n} | idx={sample_idx} | "
                f"true={EMOTION_DICT.get(int(labels[b]), labels[b].item())} | "
                f"pred={EMOTION_DICT.get(int(preds[b]), preds[b].item())}"
            )
            out_path = output_dir / f"{kind}_{n:02d}_idx_{sample_idx:05d}.png"
            sample_focus_maps = {
                name: value[b]
                for name, value in focus_maps.items()
            }
            save_assignment_figure(
                out_path,
                image,
                S[b],
                hard[b],
                grid_h,
                grid_w,
                title,
                focus_maps=sample_focus_maps,
            )
            print(f"       saved: {out_path}")

            if found_correct >= args.num_correct and found_wrong >= args.num_wrong:
                print("[DONE] Collected enough samples.")
                return

        seen += images.size(0)

    print(
        "[DONE] Reached end of split. "
        f"Collected correct={found_correct}, wrong={found_wrong}."
    )


if __name__ == "__main__":
    main()
