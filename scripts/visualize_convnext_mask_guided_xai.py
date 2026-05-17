"""Post-training XAI export for ConvNeXt mask-guided region attention.

This script is intended for Kaggle after training finishes. It loads the best
checkpoint, selects correct/wrong samples, and writes one visualization per
sample with:
  1. Grad-CAM from the final ConvNeXt feature map.
  2. MediaPipe landmarks plus the six saved region masks.
  3. Cross-attention heatmaps for the six region tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.precompute_mediapipe_region_masks import REGION_ORDER, import_mediapipe_face_mesh
from src.data.dataset_unet_mask import FER2013WithUNetMasks
from src.data.transforms import build_landmark_transform
from src.models import get_model
from src.utils.config import load_config


EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
REGION_COLORS = np.array(
    [
        [244, 67, 54],
        [255, 152, 0],
        [255, 235, 59],
        [76, 175, 80],
        [3, 169, 244],
        [156, 39, 176],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="convnext_tiny_mask_guided_region_attention_7462_sam_continue_kaggle_dataset",
        help="Config name or YAML path.",
    )
    parser.add_argument("--env", default="kaggle", help="Config environment override, e.g. kaggle/local.")
    parser.add_argument(
        "--checkpoint",
        default="auto-latest",
        help="Checkpoint path, or auto-latest to find the newest *_best.pth for this model.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to visualize.")
    parser.add_argument("--num-correct", type=int, default=50, help="Number of correct predictions to export.")
    parser.add_argument("--num-wrong", type=int, default=50, help="Number of wrong predictions to export.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test sample cap.")
    parser.add_argument("--data-path", default=None, help="Override FER2013 CSV path.")
    parser.add_argument(
        "--mask-dir",
        default="auto",
        help="Override mask root. Use auto to prefer merged retry masks when available.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for PNG/CSV artifacts.")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto.")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal flip TTA for prediction.")
    parser.add_argument("--skip-landmarks", action="store_true", help="Skip MediaPipe landmark overlay.")
    parser.add_argument("--strict-load", action="store_true", help="Fail on missing/unexpected checkpoint keys.")
    return parser.parse_args()


def resolve_path(path_value: Optional[str], *, base: Path = PROJECT_ROOT) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base / path


def resolve_data_path(config: Dict, override: Optional[str]) -> Path:
    candidates: List[Path] = []
    if override:
        candidates.append(resolve_path(override))

    config_path = config.get("paths", {}).get("data_path") or config.get("data", {}).get("data_path")
    if config_path:
        candidates.append(resolve_path(config_path))

    candidates.extend(
        [
            Path("/kaggle/input/datasets/lhongphuc3/fer2013/fer2013.csv"),
            Path("/kaggle/input/fer2013/fer2013.csv"),
            PROJECT_ROOT / "data" / "fer2013.csv",
            PROJECT_ROOT / "dataset" / "fer2013.csv",
        ]
    )

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find FER2013 split path. Pass --data-path explicitly. Tried: "
        + ", ".join(str(c) for c in candidates if c)
    )


def resolve_mask_root(config: Dict, override: Optional[str], split: str) -> Path:
    candidates: List[Path] = []
    if override and override != "auto":
        candidates.append(resolve_path(override))

    candidates.extend(
        [
            Path("/kaggle/working/outputs/mediapipe_failed_retry_masks/merged_mediapipe_region_masks"),
            PROJECT_ROOT / "outputs" / "mediapipe_failed_retry_masks" / "merged_mediapipe_region_masks",
        ]
    )

    config_mask = config.get("model", {}).get("mask_dir") or config.get("data", {}).get("mask_dir")
    if config_mask:
        candidates.append(resolve_path(config_mask))

    candidates.extend(
        [
            Path("/kaggle/input/datasets/lhongphuc3/mediapipe-mask-datasets/mediapipe_region_masks"),
            PROJECT_ROOT / "mediapipe_region_masks",
        ]
    )

    for candidate in candidates:
        if candidate and (candidate / split).exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find mask root with split '{split}'. Pass --mask-dir explicitly. Tried: "
        + ", ".join(str(c) for c in candidates if c)
    )


def make_output_dir(config: Dict, override: Optional[str]) -> Path:
    if override:
        output_dir = resolve_path(override)
    else:
        base = resolve_path(config.get("output_dir", "outputs"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = base / f"xai_convnext_mask_guided_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "correct").mkdir(exist_ok=True)
    (output_dir / "wrong").mkdir(exist_ok=True)
    return output_dir


def resolve_checkpoint(config: Dict, checkpoint_arg: str) -> Path:
    if checkpoint_arg != "auto-latest":
        checkpoint = resolve_path(checkpoint_arg)
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_name = config.get("model", {}).get("name", "convnext_tiny_mask_guided_region_attention")
    output_dir = resolve_path(config.get("output_dir", "outputs"))
    roots = [
        output_dir / "checkpoints" / model_name,
        PROJECT_ROOT / "outputs" / "checkpoints" / model_name,
        Path("/kaggle/working/outputs/checkpoints") / model_name,
        Path("/kaggle/working") / "checkpoints" / model_name,
    ]

    candidates: List[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/*_best.pth"))
            candidates.extend(root.glob("**/best*.pth"))

    if not candidates:
        raise FileNotFoundError(
            "auto-latest could not find a best checkpoint. Pass --checkpoint explicitly. Searched: "
            + ", ".join(str(root) for root in roots)
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    cleaned = {}
    for key, value in checkpoint.items():
        new_key = key
        stripped = True
        while stripped:
            stripped = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    stripped = True
        cleaned[new_key] = value
    return cleaned


def load_model(config: Dict, checkpoint_path: Path, device: torch.device, strict: bool) -> torch.nn.Module:
    model = get_model(config).to(device)
    state_dict = load_checkpoint_state(checkpoint_path)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            print(f"[load] Missing keys ({len(missing)}): {missing[:20]}")
        if unexpected:
            print(f"[load] Unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    if hasattr(model, "return_attn"):
        model.return_attn = True
    model.eval()
    return model


def build_dataset_and_loader(
    config: Dict,
    data_path: Path,
    mask_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
) -> Tuple[FER2013WithUNetMasks, DataLoader]:
    config.setdefault("data", {})
    config.setdefault("model", {})
    config["data"]["data_path"] = str(data_path)
    config["data"]["mask_dir"] = str(mask_root)
    config["model"]["mask_dir"] = str(mask_root)

    transform = build_landmark_transform(config, split)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    feature_layer = model_cfg.get("feature_layer", "layer4")
    image_size = data_cfg.get("image_size", 224)
    grid_sizes = {"layer2": image_size // 8, "layer3": image_size // 16, "layer4": image_size // 32}
    grid_size = grid_sizes.get(feature_layer, 7)

    dataset = FER2013WithUNetMasks(
        data_path=str(data_path),
        split=split,
        transforms=transform,
        mask_dir=str(mask_root),
        grid_size=grid_size,
        num_regions=model_cfg.get("num_regions", 6),
        mask_floor=model_cfg.get("mask_floor", 0.05),
        use_clean_filter=data_cfg.get("use_clean_filter", True),
    )

    if max_samples is not None:
        dataset.data = dataset.data.iloc[:max_samples].reset_index(drop=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def model_forward(model: torch.nn.Module, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    output = model(images, region_masks=masks)
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: FER2013WithUNetMasks,
    device: torch.device,
    num_correct: int,
    num_wrong: int,
    use_tta: bool,
) -> Tuple[List[Dict], List[Dict], Dict]:
    correct: List[Dict] = []
    wrong: List[Dict] = []
    total = 0
    hit = 0

    for batch_idx, batch in enumerate(loader):
        images, labels, masks = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits, attn = model_forward(model, images, masks)
        if use_tta:
            flipped_logits, _ = model_forward(model, torch.flip(images, dims=[3]), torch.flip(masks, dims=[3]))
            logits = (logits + flipped_logits) * 0.5

        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        matches = preds.eq(labels)

        for item_idx in range(images.size(0)):
            dataset_index = batch_idx * loader.batch_size + item_idx
            row = dataset.data.iloc[dataset_index]
            record = {
                "dataset_index": int(dataset_index),
                "original_idx": int(row["original_idx"]),
                "true_label": int(labels[item_idx].item()),
                "pred_label": int(preds[item_idx].item()),
                "confidence": float(confs[item_idx].item()),
                "correct": bool(matches[item_idx].item()),
                "attention": attn[item_idx].detach().cpu().numpy() if attn is not None else None,
            }

            if record["correct"]:
                hit += 1
                if len(correct) < num_correct:
                    correct.append(record)
            else:
                if len(wrong) < num_wrong:
                    wrong.append(record)

            total += 1

    summary = {
        "split_size": total,
        "accuracy": hit / max(total, 1),
        "correct_exported": len(correct),
        "wrong_exported": len(wrong),
    }
    return correct, wrong, summary


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook = target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def close(self) -> None:
        self.hook.remove()

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, target_class: int) -> np.ndarray:
        self.activations = None
        self.gradients = None
        image = image.clone().detach().requires_grad_(True)
        mask = mask.clone().detach()

        self.model.zero_grad(set_to_none=True)
        logits, _ = model_forward(self.model, image, mask)
        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook did not capture activations/gradients.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_np = normalize_01(cam_np)
        return cam_np


def find_convnext_final_feature_module(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "convnext_backbone") and hasattr(model.convnext_backbone, "backbone"):
        backbone = model.convnext_backbone.backbone
        if hasattr(backbone, "features"):
            return backbone.features
    raise AttributeError("Could not find ConvNeXt final feature module at model.convnext_backbone.backbone.features")


def normalize_01(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if max_value - min_value < eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value + eps)


def row_to_rgb224(row: pd.Series) -> np.ndarray:
    pixels = np.asarray(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
    image = Image.fromarray(pixels, mode="L").resize((224, 224), Image.BILINEAR).convert("RGB")
    return np.asarray(image)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42, cmap: str = "jet") -> np.ndarray:
    heatmap = normalize_01(heatmap)
    color = plt.get_cmap(cmap)(heatmap)[..., :3]
    mixed = (1.0 - alpha) * (image.astype(np.float32) / 255.0) + alpha * color
    return np.clip(mixed * 255.0, 0, 255).astype(np.uint8)


def upsample_small_map(small_map: np.ndarray, size: int = 224) -> np.ndarray:
    tensor = torch.tensor(small_map, dtype=torch.float32).view(1, 1, *small_map.shape)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor[0, 0].numpy()


def masks_to_combined_overlay(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim != 3:
        return image

    image_float = image.astype(np.float32)
    overlay = image_float.copy()
    for region_idx in range(min(masks.shape[0], len(REGION_COLORS))):
        mask = upsample_small_map(normalize_01(masks[region_idx]))
        color = REGION_COLORS[region_idx]
        overlay = overlay * (1.0 - 0.18 * mask[..., None]) + color * (0.18 * mask[..., None])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def attention_to_maps(attention: Optional[np.ndarray]) -> np.ndarray:
    if attention is None:
        return np.zeros((len(REGION_ORDER), 7, 7), dtype=np.float32)
    attention = np.asarray(attention, dtype=np.float32)
    if attention.ndim == 3:
        attention = attention.mean(axis=0)
    if attention.shape[-1] != 49:
        return np.zeros((len(REGION_ORDER), 7, 7), dtype=np.float32)
    return attention.reshape(attention.shape[0], 7, 7)


def init_face_mesh(skip: bool):
    if skip:
        return None
    try:
        face_mesh_solution = import_mediapipe_face_mesh()
        return face_mesh_solution.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        )
    except Exception as exc:
        print(f"[landmarks] MediaPipe unavailable, skipping landmarks: {exc}")
        return None


def draw_landmarks(image: np.ndarray, face_mesh) -> np.ndarray:
    if face_mesh is None:
        return image

    try:
        result = face_mesh.process(image)
    except Exception as exc:
        print(f"[landmarks] Failed to process one image: {exc}")
        return image

    if not getattr(result, "multi_face_landmarks", None):
        return image

    points = result.multi_face_landmarks[0].landmark
    output = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(output)
    for point in points:
        x = float(point.x) * image.shape[1]
        y = float(point.y) * image.shape[0]
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(0, 229, 255))
    return np.asarray(output)


def save_sample_figure(
    record: Dict,
    dataset: FER2013WithUNetMasks,
    model: torch.nn.Module,
    gradcam: GradCAM,
    face_mesh,
    device: torch.device,
    output_path: Path,
) -> None:
    row = dataset.data.iloc[record["dataset_index"]]
    base_image = row_to_rgb224(row)

    image_tensor, _, mask_tensor = dataset[record["dataset_index"]]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)
    masks_np = mask_tensor[0].detach().cpu().numpy()

    cam = gradcam(image_tensor, mask_tensor, record["pred_label"])
    gradcam_overlay = overlay_heatmap(base_image, cam, alpha=0.46, cmap="jet")
    landmarks_overlay = draw_landmarks(base_image, face_mesh)
    combined_mask_overlay = masks_to_combined_overlay(base_image, masks_np)

    attention_maps = attention_to_maps(record["attention"])
    mean_attention = normalize_01(attention_maps.mean(axis=0))
    mean_attention_overlay = overlay_heatmap(base_image, upsample_small_map(mean_attention), alpha=0.45, cmap="magma")

    fig, axes = plt.subplots(3, 6, figsize=(18, 9), dpi=140)
    for ax in axes.flat:
        ax.axis("off")

    axes[0, 0].imshow(base_image)
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 1].imshow(gradcam_overlay)
    axes[0, 1].set_title("Grad-CAM final ConvNeXt", fontsize=10)
    axes[0, 2].imshow(landmarks_overlay)
    axes[0, 2].set_title("MediaPipe landmarks", fontsize=10)
    axes[0, 3].imshow(combined_mask_overlay)
    axes[0, 3].set_title("6 region masks", fontsize=10)
    axes[0, 4].imshow(mean_attention_overlay)
    axes[0, 4].set_title("Mean region attention", fontsize=10)

    status = "correct" if record["correct"] else "wrong"
    true_name = EMOTION_NAMES[record["true_label"]]
    pred_name = EMOTION_NAMES[record["pred_label"]]
    info = (
        f"{status}\n"
        f"idx: {record['original_idx']}\n"
        f"true: {true_name}\n"
        f"pred: {pred_name}\n"
        f"conf: {record['confidence']:.3f}"
    )
    axes[0, 5].text(0.02, 0.98, info, va="top", ha="left", fontsize=11, family="monospace")

    for region_idx, region_name in enumerate(REGION_ORDER):
        mask_map = normalize_01(masks_np[region_idx])
        axes[1, region_idx].imshow(overlay_heatmap(base_image, upsample_small_map(mask_map), alpha=0.42, cmap="viridis"))
        axes[1, region_idx].set_title(f"Mask: {region_name}", fontsize=9)

        attn_map = normalize_01(attention_maps[region_idx])
        axes[2, region_idx].imshow(overlay_heatmap(base_image, upsample_small_map(attn_map), alpha=0.48, cmap="magma"))
        axes[2, region_idx].set_title(f"Attn: {region_name}", fontsize=9)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_visualizations(
    records: Sequence[Dict],
    kind: str,
    dataset: FER2013WithUNetMasks,
    model: torch.nn.Module,
    gradcam: GradCAM,
    face_mesh,
    device: torch.device,
    output_dir: Path,
) -> List[Dict]:
    rows: List[Dict] = []
    target_dir = output_dir / kind
    for order, record in enumerate(records, start=1):
        true_name = EMOTION_NAMES[record["true_label"]]
        pred_name = EMOTION_NAMES[record["pred_label"]]
        filename = (
            f"{order:03d}_idx{record['original_idx']:06d}_"
            f"true-{true_name}_pred-{pred_name}_conf-{record['confidence']:.3f}.png"
        )
        output_path = target_dir / filename
        save_sample_figure(record, dataset, model, gradcam, face_mesh, device, output_path)

        manifest_row = dict(record)
        manifest_row.pop("attention", None)
        manifest_row["kind"] = kind
        manifest_row["file"] = str(output_path)
        rows.append(manifest_row)
        print(f"[export] {kind} {order}/{len(records)} -> {output_path}")
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config, env=args.env)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_path = resolve_data_path(config, args.data_path)
    mask_root = resolve_mask_root(config, args.mask_dir, args.split)
    output_dir = make_output_dir(config, args.output_dir)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)

    print(f"[config] {args.config} env={args.env}")
    print(f"[data] {data_path}")
    print(f"[masks] {mask_root}")
    print(f"[checkpoint] {checkpoint_path}")
    print(f"[output] {output_dir}")
    print(f"[device] {device}")

    dataset, loader = build_dataset_and_loader(
        config=config,
        data_path=data_path,
        mask_root=mask_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    model = load_model(config, checkpoint_path, device=device, strict=args.strict_load)
    correct, wrong, summary = collect_predictions(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
        num_correct=args.num_correct,
        num_wrong=args.num_wrong,
        use_tta=not args.no_tta,
    )

    target_module = find_convnext_final_feature_module(model)
    gradcam = GradCAM(model, target_module)
    face_mesh = init_face_mesh(args.skip_landmarks)

    manifest_rows: List[Dict] = []
    try:
        manifest_rows.extend(
            export_visualizations(correct, "correct", dataset, model, gradcam, face_mesh, device, output_dir)
        )
        manifest_rows.extend(export_visualizations(wrong, "wrong", dataset, model, gradcam, face_mesh, device, output_dir))
    finally:
        gradcam.close()
        if face_mesh is not None and hasattr(face_mesh, "close"):
            face_mesh.close()

    summary.update(
        {
            "config": args.config,
            "env": args.env,
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "data_path": str(data_path),
            "mask_root": str(mask_root),
            "output_dir": str(output_dir),
            "tta": not args.no_tta,
        }
    )

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
