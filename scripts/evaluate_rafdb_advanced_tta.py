from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_rafdb_mask_guided import (  # noqa: E402
    extract_state_dict,
    safe_torch_load,
    strip_known_prefixes,
)
from src.data.rafdb_mask_dataset import CLASS_NAMES, RAFDBWithMasks  # noqa: E402
from src.data.transforms import build_landmark_transform  # noqa: E402
from src.models import get_model  # noqa: E402


DEFAULT_CHECKPOINT = "checkpoints/RAFDBVer1/rafdb_mgr_full_a035.pth"
DEFAULT_DATA_ROOT = "dataset/DATASET"
DEFAULT_MASK_ROOT = "dataset/rafdb_mediapipe_region_masks/outputs/rafdb_mediapipe_region_masks"
DEFAULT_OUTPUT_DIR = "outputs/rafdb_eval/rafdb_mgr_full_a035_advanced_tta_reproduce"


def log_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def prepare_config(checkpoint: dict, data_root: Path, mask_root: Path, batch_size: int, num_workers: int) -> dict:
    config = json.loads(json.dumps(checkpoint["config"]))
    config.setdefault("data", {})["root"] = str(data_root)
    config["data"]["mask_dir"] = str(mask_root)
    config["data"]["eval_batch_size"] = int(batch_size)
    config["data"]["num_workers"] = int(num_workers)
    config.setdefault("model", {})["mask_dir"] = str(mask_root)
    config["model"]["pretrained"] = False
    config["model"]["weights"] = None
    config["model"]["checkpoint_path"] = None
    config.setdefault("logging", {})["use_wandb"] = False
    return config


def mask_grid_size(config: dict) -> int:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    feature_layer = model_cfg.get("feature_layer", "layer4")
    grid_sizes = {"layer2": image_size // 8, "layer3": image_size // 16, "layer4": image_size // 32}
    return int(model_cfg.get("grid_size", grid_sizes.get(feature_layer, 7)))


def build_test_loader(config: dict, data_root: Path, mask_root: Path) -> DataLoader:
    dataset = RAFDBWithMasks(
        root=data_root,
        split="test",
        transform=build_landmark_transform(config, "test"),
        mask_root=mask_root,
        grid_size=mask_grid_size(config),
        num_regions=int(config["model"].get("num_regions", 6)),
        mask_floor=float(config["model"].get("mask_floor", 0.05)),
        mask_ablation=config["data"].get("mask_ablation", config["model"].get("mask_ablation", "none")),
        mask_region_permutation=config["data"].get(
            "mask_region_permutation",
            config["model"].get("mask_region_permutation"),
        ),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["data"].get("eval_batch_size", 48)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def load_model(checkpoint: dict, config: dict, device: torch.device) -> torch.nn.Module:
    model = get_model(name=config["model"]["name"], config=config)
    state_dict = strip_known_prefixes(extract_state_dict(checkpoint))
    load_result = model.load_state_dict(state_dict, strict=True)
    print(f"--> Load checkpoint: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def forward_components(model: torch.nn.Module, images: torch.Tensor, region_masks: torch.Tensor):
    batch_size = images.shape[0]
    backbone_outputs = model.convnext_backbone(images)
    if len(backbone_outputs) == 3:
        visual_features, global_feat, _ = backbone_outputs
        global_max_feat = None
    else:
        visual_features, global_feat, _, global_max_feat = backbone_outputs

    visual_features = visual_features + model.visual_pos_embed
    flat_masks = model._flatten_region_masks(region_masks, visual_features)
    region_tokens = model._region_tokens(batch_size)
    if model.mask_guided_attention:
        phi_sem, _ = model.alignment(region_tokens, visual_features, region_masks=flat_masks)
    else:
        phi_sem, _ = model.alignment(region_tokens, visual_features)

    hyper_visual = model._append_eye_fusion_token(phi_sem) if model.eye_fusion_mode == "post" else phi_sem
    hyper_visual = model._append_region_relation_tokens(hyper_visual)
    hyper_visual = hyper_visual + model.pos_embed
    global_context = (
        model.visual_proj(global_feat)
        if (model.use_global_visual_bias or model.use_global_feature_concat)
        else None
    )
    if model.use_global_visual_bias:
        hyper_visual = hyper_visual + global_context.unsqueeze(1)

    encoded = model.transformer_encoder(hyper_visual)
    encoded, region_weights = model._apply_dynamic_region_weighting(encoded, global_feat)
    pooled = model._pool_region_features(encoded, region_weights=region_weights)
    if model.use_global_feature_concat:
        pooled = torch.cat((pooled, global_context), dim=-1)

    attention_logits = model.classifier(pooled)
    cnn_aux_feat = model._cnn_aux_features(global_feat, global_max_feat)
    cnn_aux_logits = model.cnn_aux_classifier(cnn_aux_feat)
    return attention_logits, cnn_aux_logits


def resize_image(images: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def resize_masks(masks: torch.Tensor, size: int, mask_floor: float) -> torch.Tensor:
    return F.interpolate(masks, size=(size, size), mode="bilinear", align_corners=False).clamp(mask_floor, 1.0)


def crop_resize_pair(
    images: torch.Tensor,
    masks: torch.Tensor,
    top: int,
    left: int,
    crop_size: int,
    out_size: int,
    mask_floor: float,
):
    img_crop = images[:, :, top : top + crop_size, left : left + crop_size]
    img_aug = resize_image(img_crop, out_size)

    masks_224 = resize_masks(masks, out_size, mask_floor)
    mask_crop = masks_224[:, :, top : top + crop_size, left : left + crop_size]
    mask_aug = resize_masks(mask_crop, 7, mask_floor)
    return img_aug, mask_aug


def scale_center_pair(images: torch.Tensor, masks: torch.Tensor, scale: int, mask_floor: float, out_size: int = 224):
    if scale == out_size:
        return images, masks
    img_scaled = resize_image(images, scale)
    masks_scaled = resize_masks(masks, scale, mask_floor)
    top = (scale - out_size) // 2
    left = (scale - out_size) // 2
    img_aug = img_scaled[:, :, top : top + out_size, left : left + out_size]
    mask_crop = masks_scaled[:, :, top : top + out_size, left : left + out_size]
    mask_aug = resize_masks(mask_crop, 7, mask_floor)
    return img_aug, mask_aug


def hflip_pair(images: torch.Tensor, masks: torch.Tensor):
    return torch.flip(images, dims=[3]), torch.flip(masks, dims=[3])


def five_crop_pairs(images: torch.Tensor, masks: torch.Tensor, ratio: float, mask_floor: float, out_size: int = 224):
    crop_size = int(round(out_size * ratio))
    max_start = out_size - crop_size
    starts = [
        (0, 0),
        (0, max_start),
        (max_start, 0),
        (max_start, max_start),
        (max_start // 2, max_start // 2),
    ]
    return [crop_resize_pair(images, masks, top, left, crop_size, out_size, mask_floor) for top, left in starts]


def tta_variants(suite: str, images: torch.Tensor, masks: torch.Tensor, mask_floor: float):
    variants = []
    if suite == "single_center224":
        variants.append((images, masks))
    elif suite == "hflip2":
        variants.append((images, masks))
        variants.append(hflip_pair(images, masks))
    elif suite == "five_crop95":
        variants.extend(five_crop_pairs(images, masks, ratio=0.95, mask_floor=mask_floor))
    elif suite == "ten_crop95":
        crops = five_crop_pairs(images, masks, ratio=0.95, mask_floor=mask_floor)
        variants.extend(crops)
        variants.extend([hflip_pair(img, mask) for img, mask in crops])
    elif suite == "scale224_240_256_center":
        for scale in (224, 240, 256):
            variants.append(scale_center_pair(images, masks, scale=scale, mask_floor=mask_floor))
    elif suite == "scale224_240_256_hflip":
        for scale in (224, 240, 256):
            img, mask = scale_center_pair(images, masks, scale=scale, mask_floor=mask_floor)
            variants.append((img, mask))
            variants.append(hflip_pair(img, mask))
    elif suite == "combined_light":
        variants.append((images, masks))
        variants.append(hflip_pair(images, masks))
        crops = five_crop_pairs(images, masks, ratio=0.95, mask_floor=mask_floor)
        variants.extend(crops)
        variants.extend([hflip_pair(img, mask) for img, mask in crops])
        for scale in (240, 256):
            img, mask = scale_center_pair(images, masks, scale=scale, mask_floor=mask_floor)
            variants.append((img, mask))
            variants.append(hflip_pair(img, mask))
    else:
        raise ValueError(f"Unknown TTA suite: {suite}")
    return variants


def collect_tta_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    suite: str,
    mask_floor: float,
):
    y_true_batches = []
    attn_batches = []
    cnn_batches = []
    num_augments = None
    started = time.time()

    for batch_idx, (images, labels, region_masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        region_masks = region_masks.to(device, non_blocking=True)
        variants = tta_variants(suite, images, region_masks, mask_floor)
        num_augments = len(variants)

        attn_sum = None
        cnn_sum = None
        for img_aug, mask_aug in variants:
            attn_logits, cnn_logits = forward_components(model, img_aug, mask_aug)
            attn_sum = attn_logits if attn_sum is None else attn_sum + attn_logits
            cnn_sum = cnn_logits if cnn_sum is None else cnn_sum + cnn_logits

        attn_batches.append((attn_sum / num_augments).cpu().numpy())
        cnn_batches.append((cnn_sum / num_augments).cpu().numpy())
        y_true_batches.append(labels.numpy())
        if batch_idx % 8 == 0 or batch_idx == len(loader):
            print(f"--> {suite}: processed {batch_idx}/{len(loader)} batches")

    return {
        "suite": suite,
        "num_augments": int(num_augments or 0),
        "y_true": np.concatenate(y_true_batches),
        "attention_logits": np.concatenate(attn_batches),
        "cnn_logits": np.concatenate(cnn_batches),
        "elapsed_seconds": time.time() - started,
    }


def fused_probs(attention_logits: np.ndarray, cnn_logits: np.ndarray, cnn_weight: float, fusion_mode: str) -> np.ndarray:
    region_weight = 1.0 - float(cnn_weight)
    if fusion_mode == "logit_sum":
        return softmax_np(region_weight * attention_logits + float(cnn_weight) * cnn_logits)
    if fusion_mode == "prob_avg":
        return region_weight * softmax_np(attention_logits) + float(cnn_weight) * softmax_np(cnn_logits)
    raise ValueError(f"Unknown fusion mode: {fusion_mode}")


def metric_row(y_true: np.ndarray, probs: np.ndarray, labels_order: list[int]) -> dict:
    y_pred = probs.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100.0),
        "macro_f1": float(macro_f1),
        "macro_f1_percent": float(macro_f1 * 100.0),
        "negative_log_loss": float(log_loss(y_true, probs, labels=labels_order)),
    }


def save_eval_outputs(out_dir: Path, y_true: np.ndarray, probs: np.ndarray, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = probs.argmax(axis=1)
    labels_order = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels_order, average=None, zero_division=0)

    metrics = {
        **summary,
        "per_class_f1": dict(zip(CLASS_NAMES, [float(x) for x in per_class_f1])),
        "confusion_matrix": cm.tolist(),
    }
    log_json(metrics, out_dir / "test_metrics_summary.json")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    (out_dir / "test_classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    with (out_dir / "test_classification_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_or_average", "precision", "recall", "f1-score", "support"])
        for key, values in report_dict.items():
            if isinstance(values, dict):
                writer.writerow(
                    [
                        key,
                        values.get("precision"),
                        values.get("recall"),
                        values.get("f1-score"),
                        values.get("support"),
                    ]
                )

    with (out_dir / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, cm):
            writer.writerow([class_name, *row.tolist()])

    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "true_label", "true_name", "pred_label", "pred_name", "correct", "confidence"])
        confidence = probs.max(axis=1)
        for idx, (true_label, pred_label, conf) in enumerate(zip(y_true, y_pred, confidence)):
            writer.writerow(
                [
                    idx,
                    int(true_label),
                    CLASS_NAMES[int(true_label)],
                    int(pred_label),
                    CLASS_NAMES[int(pred_label)],
                    bool(true_label == pred_label),
                    float(conf),
                ]
            )


def run_fixed(args, model, loader, device, mask_floor: float, output_dir: Path) -> dict:
    collected = collect_tta_logits(model, loader, device, args.tta_suite, mask_floor)
    probs = fused_probs(
        collected["attention_logits"],
        collected["cnn_logits"],
        cnn_weight=args.cnn_weight,
        fusion_mode=args.fusion_mode,
    )
    labels_order = list(range(len(CLASS_NAMES)))
    summary = metric_row(collected["y_true"], probs, labels_order)
    summary.update(
        {
            "suite": args.tta_suite,
            "num_augments": collected["num_augments"],
            "alpha": float(args.alpha),
            "fusion_mode": args.fusion_mode,
            "cnn_weight": float(args.cnn_weight),
            "region_weight": float(1.0 - args.cnn_weight),
            "num_samples": int(len(collected["y_true"])),
            "elapsed_seconds": float(collected["elapsed_seconds"]),
        }
    )
    save_eval_outputs(output_dir, collected["y_true"], probs, summary)
    return summary


def run_fusion_sweep(args, model, loader, device, mask_floor: float, output_dir: Path) -> dict:
    collected_by_suite = {
        suite: collect_tta_logits(model, loader, device, suite, mask_floor)
        for suite in args.sweep_suites
    }
    labels_order = list(range(len(CLASS_NAMES)))
    weights = [round(float(args.min_cnn_weight) + i * float(args.step), 6) for i in range(int(round((args.max_cnn_weight - args.min_cnn_weight) / args.step)) + 1)]

    rows = []
    for suite, collected in collected_by_suite.items():
        for cnn_weight in weights:
            if cnn_weight < -1e-9 or cnn_weight > 1.0 + 1e-9:
                continue
            for fusion_mode in args.sweep_fusion_modes:
                probs = fused_probs(
                    collected["attention_logits"],
                    collected["cnn_logits"],
                    cnn_weight=cnn_weight,
                    fusion_mode=fusion_mode,
                )
                row = metric_row(collected["y_true"], probs, labels_order)
                row.update(
                    {
                        "suite": suite,
                        "num_augments": collected["num_augments"],
                        "alpha": float(args.alpha),
                        "fusion_mode": fusion_mode,
                        "cnn_weight": float(cnn_weight),
                        "region_weight": float(1.0 - cnn_weight),
                        "num_samples": int(len(collected["y_true"])),
                    }
                )
                rows.append(row)

    rows = sorted(rows, key=lambda row: (row["accuracy"], row["macro_f1"], -row["negative_log_loss"]), reverse=True)
    csv_path = output_dir / "advanced_tta_fusion_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "suite",
                "num_augments",
                "alpha",
                "fusion_mode",
                "cnn_weight",
                "region_weight",
                "accuracy",
                "accuracy_percent",
                "macro_f1",
                "macro_f1_percent",
                "negative_log_loss",
                "num_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_by_suite = []
    for suite in args.sweep_suites:
        best_by_suite.append(next(row for row in rows if row["suite"] == suite))
    with (output_dir / "best_by_tta_suite.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_by_suite[0].keys()))
        writer.writeheader()
        writer.writerows(best_by_suite)

    summary = {
        "best": rows[0],
        "best_by_suite": best_by_suite,
        "top20": rows[:20],
        "csv_path": str(csv_path),
        "best_by_suite_csv_path": str(output_dir / "best_by_tta_suite.csv"),
    }
    log_json(summary, output_dir / "advanced_tta_fusion_sweep_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce RAF-DB advanced TTA evaluation for MGR checkpoints.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--tta-suite", default="combined_light")
    parser.add_argument("--fusion-mode", default="prob_avg", choices=("logit_sum", "prob_avg"))
    parser.add_argument("--cnn-weight", type=float, default=0.86)
    parser.add_argument("--sweep-fusion", action="store_true")
    parser.add_argument("--min-cnn-weight", type=float, default=0.0)
    parser.add_argument("--max-cnn-weight", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--sweep-suites",
        nargs="+",
        default=[
            "single_center224",
            "hflip2",
            "five_crop95",
            "ten_crop95",
            "scale224_240_256_center",
            "scale224_240_256_hflip",
            "combined_light",
        ],
    )
    parser.add_argument("--sweep-fusion-modes", nargs="+", default=["logit_sum", "prob_avg"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    data_root = resolve_path(args.data_root)
    mask_root = resolve_path(args.mask_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError("Checkpoint must contain a saved config.")

    config = prepare_config(checkpoint, data_root, mask_root, args.batch_size, args.num_workers)
    requested_device = args.device.lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    print(f"--> Checkpoint: {checkpoint_path}")
    print(f"--> Checkpoint epoch: {checkpoint.get('epoch')}, monitor={checkpoint.get('monitor')}, best_score={checkpoint.get('best_score')}")
    print(f"--> Data root: {data_root}")
    print(f"--> Mask root: {mask_root}")
    print(f"--> Output dir: {output_dir}")
    print(f"--> Device: {device}")

    loader = build_test_loader(config, data_root, mask_root)
    model = load_model(checkpoint, config, device)
    model.alignment.mask_attention_alpha = float(args.alpha)
    mask_floor = float(config["model"].get("mask_floor", 0.05))

    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_monitor": checkpoint.get("monitor"),
        "checkpoint_best_score": checkpoint.get("best_score"),
        "trained_alpha": checkpoint["config"].get("model", {}).get("mask_attention_alpha"),
        "eval_alpha": float(args.alpha),
        "data_root": str(data_root),
        "mask_root": str(mask_root),
        "output_dir": str(output_dir),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "fixed_eval": {
            "tta_suite": args.tta_suite,
            "fusion_mode": args.fusion_mode,
            "cnn_weight": float(args.cnn_weight),
            "region_weight": float(1.0 - args.cnn_weight),
        },
        "sweep_fusion": bool(args.sweep_fusion),
    }
    log_json(manifest, output_dir / "run_manifest.json")

    fixed_summary = run_fixed(args, model, loader, device, mask_floor, output_dir)
    print(
        "RESULT "
        f"acc={fixed_summary['accuracy_percent']:.4f}% "
        f"macro_f1={fixed_summary['macro_f1_percent']:.4f}% "
        f"suite={fixed_summary['suite']} "
        f"fusion={fixed_summary['fusion_mode']} "
        f"cnn={fixed_summary['cnn_weight']:.2f} "
        f"region={fixed_summary['region_weight']:.2f}"
    )

    if args.sweep_fusion:
        sweep_summary = run_fusion_sweep(args, model, loader, device, mask_floor, output_dir)
        best = sweep_summary["best"]
        print(
            "SWEEP_BEST "
            f"acc={best['accuracy_percent']:.4f}% "
            f"macro_f1={best['macro_f1_percent']:.4f}% "
            f"suite={best['suite']} "
            f"fusion={best['fusion_mode']} "
            f"cnn={best['cnn_weight']:.2f} "
            f"region={best['region_weight']:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
