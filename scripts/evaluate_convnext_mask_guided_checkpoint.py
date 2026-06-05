from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_unet_mask import FER2013WithUNetMasks
from src.data.emotions_dict import EMOTION_DICT, EMOTION_NAMES
from src.data.transforms import build_landmark_transform
from src.models import get_model
from src.utils.config import load_config


DEFAULT_MASK_DIR_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "mediapipe_failed_retry_masks" / "merged_mediapipe_region_masks",
    PROJECT_ROOT / "outputs" / "mediapipe_region_masks",
    PROJECT_ROOT / "outputs" / "unet_region_masks",
)


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value or "run"


def log_json(obj, path: Path) -> None:
    def default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint does not contain a valid state dict.")


def strip_known_prefixes(state_dict):
    prefixes = ("module.", "_orig_mod.")
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        cleaned[name] = value
    return cleaned


def path_candidates(path_like) -> list[Path]:
    if not path_like:
        return []
    path = Path(path_like)
    if path.is_absolute():
        return [path]
    return [PROJECT_ROOT / path, path]


def resolve_split_dir(data_path, splits: Sequence[str]) -> Path:
    required = {f"{split}.csv" for split in splits}
    candidates = path_candidates(data_path)
    if not candidates:
        candidates = [PROJECT_ROOT / "dataset" / "fer13-split"]
    if Path("/kaggle/input").exists():
        candidates.append(Path("/kaggle/input"))

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        files = {p.name for p in candidate.iterdir() if p.is_file()}
        if required.issubset(files):
            return candidate.resolve()
        for current, _, files in os.walk(candidate):
            if required.issubset(set(files)):
                return Path(current).resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find split CSV files {sorted(required)}. Searched:\n{searched}")


def discover_mask_dir_by_name(folder_name: str, split: str) -> Path | None:
    roots = [PROJECT_ROOT]
    if Path("/kaggle/input").exists():
        roots.insert(0, Path("/kaggle/input"))

    for root in roots:
        for current, dirs, _ in os.walk(root):
            current_path = Path(current)
            if current_path.name == folder_name and split in dirs:
                return current_path.resolve()
    return None


def resolve_eval_mask_dir(config: dict, split: str, explicit_mask_dir=None) -> Path:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    configured = explicit_mask_dir or model_cfg.get("mask_dir") or data_cfg.get("mask_dir")

    candidates = []
    candidates.extend(path_candidates(configured))
    candidates.extend(DEFAULT_MASK_DIR_CANDIDATES)

    for candidate in candidates:
        if (candidate / split).exists():
            return candidate.resolve()

    if configured:
        discovered = discover_mask_dir_by_name(Path(configured).name, split)
        if discovered is not None:
            return discovered

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find mask_dir with split folder '{split}'. Searched:\n{searched}")


class FER2013MaskWithIndex(FER2013WithUNetMasks):
    def __getitem__(self, index):
        image, label, region_masks = super().__getitem__(index)
        original_idx = int(self.data.iloc[index]["original_idx"])
        return image, label, region_masks, int(index), original_idx


def mask_grid_size_for_config(config: dict) -> int:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    feature_layer = model_cfg.get("feature_layer", "layer4")
    image_size = int(data_cfg.get("image_size", 224))
    grid_sizes = {"layer2": image_size // 8, "layer3": image_size // 16, "layer4": image_size // 32}
    return int(grid_sizes.get(feature_layer, 7))


def prepare_config_for_eval(
    config: dict,
    batch_size: int,
    num_workers: int,
    mask_dir: Path,
) -> dict:
    config = copy.deepcopy(config)
    data_cfg = config.setdefault("data", {})
    data_cfg["image_size"] = 224
    data_cfg["channels"] = 3
    data_cfg["batch_size"] = int(batch_size)
    data_cfg["num_workers"] = int(num_workers)
    data_cfg["mask_dir"] = str(mask_dir)
    config.setdefault("logging", {})["use_wandb"] = False

    model_cfg = config.setdefault("model", {})
    model_cfg["checkpoint_path"] = None
    model_cfg["checkpoint_strict"] = False
    model_cfg["pretrained"] = False
    model_cfg["weights"] = None
    model_cfg["mask_dir"] = str(mask_dir)
    return config


def prepare_state_dict_for_eval(state_dict, config: dict, fuse_clip_tokens: bool, log=print):
    state_dict = strip_known_prefixes(state_dict)
    model_cfg = config.setdefault("model", {})

    if fuse_clip_tokens:
        has_mixed_tokens = (
            "learned_region_dict.token_embed.weight" in state_dict
            or any(k.startswith("clip_region_dict.") for k in state_dict)
            or "clip_region_gamma" in state_dict
        )
        if has_mixed_tokens:
            state_dict = dict(state_dict)
            learned_weight = state_dict.get("learned_region_dict.token_embed.weight")
            if learned_weight is None:
                raise ValueError(
                    "Checkpoint has learned/CLIP region-token keys but no "
                    "learned_region_dict.token_embed.weight."
                )

            fused_weight = learned_weight.clone()
            clip_token = state_dict.get("clip_region_dict.token_embed")
            if clip_token is not None:
                clip_weight = clip_token
                proj_weight = state_dict.get("clip_region_dict.proj.weight")
                proj_bias = state_dict.get("clip_region_dict.proj.bias")
                if proj_weight is not None:
                    clip_weight = F.linear(clip_weight, proj_weight, proj_bias)
                gamma = state_dict.get("clip_region_gamma", torch.tensor(0.0, dtype=fused_weight.dtype))
                fused_weight = fused_weight + gamma.to(dtype=fused_weight.dtype) * clip_weight.to(
                    dtype=fused_weight.dtype
                )
                log("Fused learned + gamma*CLIP region tokens into local region_dict tokens.")
            else:
                log("Remapped learned region tokens into local region_dict tokens.")

            for key in list(state_dict.keys()):
                if (
                    key.startswith("learned_region_dict.")
                    or key.startswith("clip_region_dict.")
                    or key == "clip_region_gamma"
                ):
                    state_dict.pop(key, None)
            state_dict["region_dict.token_embed.weight"] = fused_weight
            state_dict["region_dict.region_ids"] = torch.arange(fused_weight.shape[0], dtype=torch.long)
            model_cfg["use_learnable_clip_region_tokens"] = False
            model_cfg["use_clip_dictionary"] = False

    has_clip_token_param = "region_dict.token_embed" in state_dict
    has_local_embedding = "region_dict.token_embed.weight" in state_dict
    clip_proj_keys = [k for k in state_dict if k.startswith("region_dict.proj.")]

    if has_local_embedding:
        model_cfg["use_clip_dictionary"] = False
        model_cfg["use_learnable_clip_region_tokens"] = False

    if has_clip_token_param and not has_local_embedding:
        state_dict = dict(state_dict)
        token_weight = state_dict.pop("region_dict.token_embed")
        proj_weight = state_dict.pop("region_dict.proj.weight", None)
        proj_bias = state_dict.pop("region_dict.proj.bias", None)
        if proj_weight is not None:
            token_weight = F.linear(token_weight, proj_weight, proj_bias)
        for key in clip_proj_keys:
            state_dict.pop(key, None)
        state_dict["region_dict.token_embed.weight"] = token_weight
        model_cfg["use_clip_dictionary"] = False
        model_cfg["use_learnable_clip_region_tokens"] = False
        log("Remapped CLIP region_dict token parameter into local embedding.")

    if model_cfg.get("use_clip_dictionary") is False and "region_dict.region_ids" not in state_dict:
        state_dict = dict(state_dict)
        token_weight = state_dict.get("region_dict.token_embed.weight")
        num_regions = int(token_weight.shape[0]) if token_weight is not None else int(model_cfg.get("num_regions", 6))
        state_dict["region_dict.region_ids"] = torch.arange(num_regions, dtype=torch.long)

    return state_dict, config


def build_loader(config: dict, data_path: Path, split: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = build_landmark_transform(config, split=split)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    dataset = FER2013MaskWithIndex(
        data_path,
        split=split,
        transforms=transform,
        mask_dir=model_cfg.get("mask_dir") or data_cfg.get("mask_dir"),
        grid_size=mask_grid_size_for_config(config),
        num_regions=model_cfg.get("num_regions", 6),
        mask_floor=model_cfg.get("mask_floor", 0.05),
        use_clean_filter=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def checkpoint_metadata(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        return {}
    keys = ("epoch", "val_loss", "val_accuracy", "accuracy", "monitor", "best_score")
    return {key: checkpoint[key] for key in keys if key in checkpoint}


def load_model_from_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    env: str,
    batch_size: int,
    num_workers: int,
    mask_dir: Path,
    device: torch.device,
    load_strict: bool,
    fuse_clip_tokens: bool,
    log=print,
):
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    config = load_config(str(config_path), env=env)
    config = prepare_config_for_eval(config, batch_size=batch_size, num_workers=num_workers, mask_dir=mask_dir)
    state_dict, config = prepare_state_dict_for_eval(
        state_dict,
        config,
        fuse_clip_tokens=fuse_clip_tokens,
        log=log,
    )

    model = get_model(name=config["model"]["name"], config=config)
    load_result = model.load_state_dict(state_dict, strict=load_strict)
    model.to(device)
    model.eval()
    if hasattr(model, "return_attn"):
        model.return_attn = False

    diagnostics = {
        "load_strict": bool(load_strict),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        "checkpoint_meta": checkpoint_metadata(checkpoint),
        "model_name": config["model"]["name"],
        "logit_fusion": config.get("model", {}).get("logit_fusion"),
        "use_multiscale_se_fusion": config.get("model", {}).get("use_multiscale_se_fusion"),
        "use_cnn_aux_logits": config.get("model", {}).get("use_cnn_aux_logits"),
        "use_clip_dictionary_after_prepare": config.get("model", {}).get("use_clip_dictionary"),
        "use_learnable_clip_region_tokens_after_prepare": config.get("model", {}).get(
            "use_learnable_clip_region_tokens"
        ),
    }
    if getattr(model, "learnable_logit_fusion", False):
        diagnostics["current_cnn_logit_weight"] = model.current_cnn_logit_weight()
        diagnostics["current_region_logit_weight"] = model.current_region_logit_weight()
        if "logit_fusion_alpha" in diagnostics["missing_keys"]:
            diagnostics["fusion_warning"] = (
                "Config enables learnable_logit_fusion, but checkpoint has no "
                "logit_fusion_alpha. Evaluation will use the config init weight, "
                "not a trained fusion scalar."
            )
    return model, config, diagnostics


def unpack_logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("logits", "pred", "output"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
    raise TypeError(f"Cannot extract logits from output type {type(output)}")


@torch.inference_mode()
def forward_with_masks(model, images, region_masks):
    try:
        return unpack_logits(model(images, region_masks=region_masks))
    except TypeError:
        return unpack_logits(model(images))


@torch.inference_mode()
def evaluate_loader(model, loader: DataLoader, device: torch.device, use_tta: bool) -> dict:
    model.eval()
    all_row_pos, all_original_idx, all_true, all_pred, all_prob, all_logit = [], [], [], [], [], []
    start = time.time()

    for images, labels, region_masks, row_pos, original_idx in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        region_masks = region_masks.to(device, non_blocking=True)

        logits = forward_with_masks(model, images, region_masks)
        if use_tta:
            logits_flip = forward_with_masks(
                model,
                torch.flip(images, dims=[3]),
                torch.flip(region_masks, dims=[3]),
            )
            logits = (logits + logits_flip) / 2.0

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_row_pos.append(row_pos.cpu().numpy())
        all_original_idx.append(original_idx.cpu().numpy())
        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_prob.append(probs.cpu().numpy())
        all_logit.append(logits.cpu().numpy())

    return {
        "row_position": np.concatenate(all_row_pos, axis=0),
        "original_idx": np.concatenate(all_original_idx, axis=0),
        "y_true": np.concatenate(all_true, axis=0),
        "y_pred": np.concatenate(all_pred, axis=0),
        "y_prob": np.concatenate(all_prob, axis=0),
        "y_logit": np.concatenate(all_logit, axis=0),
        "elapsed_seconds": time.time() - start,
    }


@torch.inference_mode()
def forward_logit_components_with_masks(model, images, region_masks):
    """Return separate attention and CNN-aux logits for ConvNeXtRegionAttentionFER."""
    required_attrs = (
        "convnext_backbone",
        "visual_pos_embed",
        "_flatten_region_masks",
        "_region_tokens",
        "alignment",
        "transformer_encoder",
        "classifier",
        "cnn_aux_classifier",
    )
    missing_attrs = [name for name in required_attrs if not hasattr(model, name)]
    if missing_attrs:
        raise TypeError(f"Model does not expose ConvNeXt region-attention internals: {missing_attrs}")
    if model.cnn_aux_classifier is None:
        raise RuntimeError("This model has no cnn_aux_classifier, so CNN/region logit sweep is not available.")

    batch_size = images.shape[0]
    backbone_outputs = model.convnext_backbone(images)
    if len(backbone_outputs) == 3:
        visual_features, global_feat, pooled_map = backbone_outputs
        global_max_feat = None
    else:
        visual_features, global_feat, pooled_map, global_max_feat = backbone_outputs

    visual_features = visual_features + model.visual_pos_embed
    flat_masks = model._flatten_region_masks(region_masks, visual_features)
    region_tokens = model._region_tokens(batch_size)
    if model.mask_guided_attention:
        phi_sem, _ = model.alignment(region_tokens, visual_features, region_masks=flat_masks)
    else:
        phi_sem, _ = model.alignment(region_tokens, visual_features)

    hyper_visual = (
        model._append_eye_fusion_token(phi_sem)
        if model.eye_fusion_mode == "post"
        else phi_sem
    )
    hyper_visual = hyper_visual + model.pos_embed
    global_context = (
        model.visual_proj(global_feat)
        if (model.use_global_visual_bias or model.use_global_feature_concat)
        else None
    )
    if model.use_global_visual_bias:
        hyper_visual = hyper_visual + global_context.unsqueeze(1)

    encoded = model.transformer_encoder(hyper_visual)
    pooled = model._pool_region_features(encoded)
    if model.use_global_feature_concat:
        pooled = torch.cat((pooled, global_context), dim=-1)
    attention_logits = model.classifier(pooled)

    cnn_aux_feat = model._cnn_aux_features(global_feat, global_max_feat)
    cnn_aux_logits = model.cnn_aux_classifier(cnn_aux_feat)

    source_logits = None
    if model.logit_fusion in ("source", "sum"):
        source_logits = model.convnext_backbone.source_logits(pooled_map)
    configured_logits = model._combine_logits(attention_logits, source_logits, cnn_aux_logits)
    return {
        "attention_logits": attention_logits,
        "cnn_aux_logits": cnn_aux_logits,
        "configured_logits": configured_logits,
    }


@torch.inference_mode()
def evaluate_logit_components(model, loader: DataLoader, device: torch.device, use_tta: bool) -> dict:
    model.eval()
    all_row_pos, all_original_idx, all_true = [], [], []
    all_attention, all_cnn_aux, all_configured = [], [], []
    start = time.time()

    for images, labels, region_masks, row_pos, original_idx in tqdm(
        loader,
        desc="Evaluating logit components",
        leave=False,
    ):
        images = images.to(device, non_blocking=True)
        region_masks = region_masks.to(device, non_blocking=True)

        components = forward_logit_components_with_masks(model, images, region_masks)
        if use_tta:
            flipped = forward_logit_components_with_masks(
                model,
                torch.flip(images, dims=[3]),
                torch.flip(region_masks, dims=[3]),
            )
            for key in components:
                components[key] = (components[key] + flipped[key]) / 2.0

        all_row_pos.append(row_pos.cpu().numpy())
        all_original_idx.append(original_idx.cpu().numpy())
        all_true.append(labels.cpu().numpy())
        all_attention.append(components["attention_logits"].cpu().numpy())
        all_cnn_aux.append(components["cnn_aux_logits"].cpu().numpy())
        all_configured.append(components["configured_logits"].cpu().numpy())

    return {
        "row_position": np.concatenate(all_row_pos, axis=0),
        "original_idx": np.concatenate(all_original_idx, axis=0),
        "y_true": np.concatenate(all_true, axis=0),
        "attention_logits": np.concatenate(all_attention, axis=0),
        "cnn_aux_logits": np.concatenate(all_cnn_aux, axis=0),
        "configured_logits": np.concatenate(all_configured, axis=0),
        "elapsed_seconds": time.time() - start,
    }


def result_from_logit_components(
    component_result: dict,
    cnn_weight: float,
    fusion_mode: str = "logit",
) -> dict:
    cnn_weight = float(cnn_weight)
    region_weight = 1.0 - cnn_weight
    fusion_mode = fusion_mode.lower()

    cnn_logits = component_result["cnn_aux_logits"]
    region_logits = component_result["attention_logits"]
    if fusion_mode == "logit":
        logits = cnn_weight * cnn_logits + region_weight * region_logits
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    elif fusion_mode in ("prob", "softmax"):
        cnn_probs = torch.softmax(torch.from_numpy(cnn_logits), dim=1).numpy()
        region_probs = torch.softmax(torch.from_numpy(region_logits), dim=1).numpy()
        probs = cnn_weight * cnn_probs + region_weight * region_probs
        logits = np.log(np.clip(probs, 1e-12, 1.0))
    else:
        raise ValueError("fusion_mode must be 'logit' or 'prob'.")

    preds = probs.argmax(axis=1)
    return {
        "row_position": component_result["row_position"],
        "original_idx": component_result["original_idx"],
        "y_true": component_result["y_true"],
        "y_pred": preds,
        "y_prob": probs,
        "y_logit": logits,
        "elapsed_seconds": component_result.get("elapsed_seconds", 0.0),
        "fusion_mode": fusion_mode,
    }


def sweep_logit_weights(
    component_result: dict,
    cnn_weights: Sequence[float],
    fusion_mode: str = "logit",
) -> pd.DataFrame:
    rows = []
    for cnn_weight in cnn_weights:
        result = result_from_logit_components(
            component_result,
            cnn_weight,
            fusion_mode=fusion_mode,
        )
        summary = metrics_dict(result)
        summary["cnn_weight"] = float(cnn_weight)
        summary["region_weight"] = float(1.0 - float(cnn_weight))
        summary["fusion_mode"] = fusion_mode
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1", "weighted_f1"],
        ascending=False,
    )


def save_weight_sweep_plot(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ordered = metrics_df.sort_values("cnn_weight")
    ax.plot(ordered["cnn_weight"], ordered["accuracy_percent"], marker="o", label="accuracy")
    ax.plot(ordered["cnn_weight"], ordered["macro_f1"] * 100.0, marker="s", label="macro F1")
    ax.set_xlabel("CNN aux logits weight")
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def topk_accuracy(y_true, y_prob, k: int) -> float:
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    return float(np.mean([label in topk_i for label, topk_i in zip(y_true, topk)]))


def multiclass_brier(y_true, y_prob) -> float:
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def calibration_table(y_true, y_pred, y_prob, bins: int = 10) -> pd.DataFrame:
    confidence = y_prob.max(axis=1)
    correct = (y_true == y_pred).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for idx in range(bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        avg_conf = float(confidence[mask].mean()) if count else 0.0
        acc = float(correct[mask].mean()) if count else 0.0
        rows.append(
            {
                "bin_left": float(lo),
                "bin_right": float(hi),
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "abs_gap": abs(acc - avg_conf),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(calib_df: pd.DataFrame, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(((calib_df["count"] / total) * calib_df["abs_gap"]).sum())


def metrics_dict(result: dict) -> dict:
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_prob = result["y_prob"]
    labels_order = list(range(len(EMOTION_NAMES)))
    calib_df = calibration_table(y_true, y_pred, y_prob)

    output = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels_order, average="weighted", zero_division=0)),
        "top2_accuracy": topk_accuracy(y_true, y_prob, k=2),
        "top3_accuracy": topk_accuracy(y_true, y_prob, k=3),
        "negative_log_loss": float(log_loss(y_true, y_prob, labels=labels_order)),
        "multiclass_brier": multiclass_brier(y_true, y_prob),
        "ece_10_bins": expected_calibration_error(calib_df, len(y_true)),
        "num_samples": int(len(y_true)),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
        "samples_per_second": float(len(y_true) / max(result.get("elapsed_seconds", 0.0), 1e-9)),
    }
    output["accuracy_percent"] = output["accuracy"] * 100.0
    return output


def prediction_frame(result: dict) -> pd.DataFrame:
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_prob = result["y_prob"]
    order = np.argsort(-y_prob, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]

    df = pd.DataFrame(
        {
            "row_position": result["row_position"],
            "original_idx": result["original_idx"],
            "true_label": y_true,
            "true_name": [EMOTION_DICT[int(x)] for x in y_true],
            "pred_label": y_pred,
            "pred_name": [EMOTION_DICT[int(x)] for x in y_pred],
            "correct": y_true == y_pred,
            "top1_label": top1,
            "top1_name": [EMOTION_DICT[int(x)] for x in top1],
            "top1_prob": y_prob[np.arange(len(y_prob)), top1],
            "top2_label": top2,
            "top2_name": [EMOTION_DICT[int(x)] for x in top2],
            "top2_prob": y_prob[np.arange(len(y_prob)), top2],
            "top1_top2_margin": y_prob[np.arange(len(y_prob)), top1] - y_prob[np.arange(len(y_prob)), top2],
        }
    )
    for idx, name in enumerate(EMOTION_NAMES):
        df[f"prob_{name}"] = y_prob[:, idx]
    return df


def save_confusion_outputs(y_true, y_pred, out_dir: Path, title: str) -> None:
    labels_order = list(range(len(EMOTION_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=EMOTION_NAMES, columns=EMOTION_NAMES)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0) * 100.0
    pd.DataFrame(row_pct, index=EMOTION_NAMES, columns=EMOTION_NAMES).to_csv(
        out_dir / "confusion_matrix_row_percent.csv"
    )

    acc = accuracy_score(y_true, y_pred) * 100.0
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{title}, acc: {acc:.2f}%", fontsize=13)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(EMOTION_NAMES)))
    ax.set_yticks(range(len(EMOTION_NAMES)))
    ax.set_xticklabels(EMOTION_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_NAMES)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "#222222"
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.1f}%", ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def save_per_class_outputs(y_true, y_pred, out_dir: Path) -> pd.DataFrame:
    labels_order = list(range(len(EMOTION_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    per_class = pd.DataFrame(
        {
            "class_label": labels_order,
            "class_name": EMOTION_NAMES,
            "support": support,
            "correct": correct,
            "class_accuracy": np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0),
        }
    )
    per_class.to_csv(out_dir / "per_class_accuracy.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(per_class["class_name"], per_class["class_accuracy"] * 100.0, color="#357ABD")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class accuracy")
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(per_class["class_accuracy"] * 100.0):
        ax.text(idx, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    return per_class


def save_confidence_outputs(result: dict, out_dir: Path) -> pd.DataFrame:
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_prob = result["y_prob"]
    confidence = y_prob.max(axis=1)
    correct = y_true == y_pred
    calib_df = calibration_table(y_true, y_pred, y_prob)
    calib_df.to_csv(out_dir / "calibration_bins.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(confidence[correct], bins=20, alpha=0.65, label="correct", color="#2E7D32")
    ax.hist(confidence[~correct], bins=20, alpha=0.65, label="wrong", color="#C62828")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Samples")
    ax.set_title("Confidence distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_histogram.png", bbox_inches="tight")
    plt.close(fig)

    centers = (calib_df["bin_left"] + calib_df["bin_right"]) / 2.0
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#666666", label="perfect")
    ax.bar(centers, calib_df["accuracy"], width=0.08, alpha=0.7, label="accuracy")
    ax.plot(centers, calib_df["avg_confidence"], marker="o", color="#D2691E", label="confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Accuracy / confidence")
    ax.set_title("Calibration by confidence bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", bbox_inches="tight")
    plt.close(fig)
    return calib_df


def save_confusion_pairs(y_true, y_pred, out_dir: Path) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_NAMES))))
    rows = []
    for true_idx, true_name in enumerate(EMOTION_NAMES):
        for pred_idx, pred_name in enumerate(EMOTION_NAMES):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count:
                rows.append(
                    {
                        "true_label": true_idx,
                        "true_name": true_name,
                        "pred_label": pred_idx,
                        "pred_name": pred_name,
                        "count": count,
                    }
                )
    pairs_df = pd.DataFrame(rows).sort_values("count", ascending=False) if rows else pd.DataFrame(rows)
    pairs_df.to_csv(out_dir / "top_confusion_pairs.csv", index=False)
    return pairs_df


def save_sample_grid(csv_df: pd.DataFrame, pred_df: pd.DataFrame, out_path: Path, title: str, correct: bool, max_images=25):
    sample = pred_df[pred_df["correct"] == correct].head(max_images)
    if sample.empty:
        return

    ncols = 5
    nrows = int(np.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.1, nrows * 2.35))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, sample.iterrows()):
        row_pos = int(row["row_position"])
        pixels = np.fromstring(csv_df.iloc[row_pos, 1], sep=" ", dtype=np.uint8).reshape(48, 48)
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"T:{row['true_name']}\nP:{row['pred_name']} ({row['top1_prob']:.2f})",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_outputs(split: str, eval_name: str, result: dict, out_dir: Path, split_csv_path: Path, extra=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_prob = result["y_prob"]

    pred_df = prediction_frame(result)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    pred_df.loc[~pred_df["correct"]].to_csv(out_dir / "wrong_predictions.csv", index=False)

    labels_order = list(range(len(EMOTION_NAMES)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(out_dir / "classification_report.csv")
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=EMOTION_NAMES,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    save_confusion_outputs(y_true, y_pred, out_dir, title=f"{split} {eval_name}")
    save_per_class_outputs(y_true, y_pred, out_dir)
    save_confidence_outputs(result, out_dir)
    save_confusion_pairs(y_true, y_pred, out_dir)

    csv_df = pd.read_csv(split_csv_path, usecols=[0, 1])
    save_sample_grid(csv_df, pred_df, out_dir / "correct_samples.png", f"{split} {eval_name}: correct samples", True)
    save_sample_grid(csv_df, pred_df, out_dir / "wrong_samples.png", f"{split} {eval_name}: wrong samples", False)

    summary = metrics_dict(result)
    summary.update(extra or {})
    summary.update(
        {
            "split": split,
            "eval_name": eval_name,
            "output_dir": str(out_dir),
            "confusion_matrix_png": str(out_dir / "confusion_matrix.png"),
        }
    )
    log_json(summary, out_dir / "metrics_summary.json")
    return summary


def run_evaluation(
    config_path,
    checkpoint_path,
    env: str = "local",
    data_path=None,
    splits: Sequence[str] = ("val", "test"),
    batch_size: int = 16,
    num_workers: int = 0,
    output_dir=None,
    run_no_tta: bool = True,
    run_tta: bool = True,
    load_strict: bool = False,
    explicit_mask_dir=None,
    device=None,
    fuse_clip_tokens: bool = True,
):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    splits = tuple(splits)
    base_config = load_config(str(config_path), env=env)
    data_root = resolve_split_dir(data_path or base_config.get("data_path") or PROJECT_ROOT / "dataset" / "fer13-split", splits)
    mask_dir = resolve_eval_mask_dir(base_config, split=splits[0], explicit_mask_dir=explicit_mask_dir)

    if output_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "evaluation" / f"eval_{checkpoint_path.stem}_{run_tag}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "eval.log"
    log_file.write_text("", encoding="utf-8")

    def log(message=""):
        text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}" if message else ""
        print(text)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Config     : {config_path}")
    log(f"Checkpoint : {checkpoint_path}")
    log(f"Data root  : {data_root}")
    log(f"Mask dir   : {mask_dir}")
    log(f"Output dir : {output_dir}")
    log(f"Device     : {device}")

    model, config, diagnostics = load_model_from_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        env=env,
        batch_size=batch_size,
        num_workers=num_workers,
        mask_dir=mask_dir,
        device=device,
        load_strict=load_strict,
        fuse_clip_tokens=fuse_clip_tokens,
        log=log,
    )
    log_json(diagnostics, output_dir / "load_diagnostics.json")
    log(f"Missing keys: {len(diagnostics['missing_keys'])}; unexpected keys: {len(diagnostics['unexpected_keys'])}")
    if diagnostics.get("fusion_warning"):
        log(f"WARNING: {diagnostics['fusion_warning']}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "data_root": str(data_root),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
        "splits": list(splits),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "run_no_tta": run_no_tta,
        "run_tta": run_tta,
        "device": str(device),
        "load_diagnostics": diagnostics,
    }
    log_json(manifest, output_dir / "run_manifest.json")

    all_summaries = []
    for split in splits:
        split_csv_path = data_root / f"{split}.csv"
        loader = build_loader(config, data_root, split=split, batch_size=batch_size, num_workers=num_workers)
        log(f"Split {split}: {len(loader.dataset)} samples, {len(loader)} batches")

        runs = []
        if run_no_tta:
            runs.append(("no_tta", False))
        if run_tta:
            runs.append(("tta_hflip", True))

        for eval_name, use_tta in runs:
            log(f"Evaluating split={split}, eval={eval_name}")
            result = evaluate_loader(model, loader, device=device, use_tta=use_tta)
            out_dir = output_dir / split / eval_name
            summary = save_evaluation_outputs(
                split=split,
                eval_name=eval_name,
                result=result,
                out_dir=out_dir,
                split_csv_path=split_csv_path,
                extra={
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "use_tta": use_tta,
                    "batch_size": batch_size,
                    "mask_dir": str(mask_dir),
                },
            )
            all_summaries.append(summary)
            log(
                f"{split}/{eval_name}: acc={summary['accuracy_percent']:.4f}%, "
                f"macro_f1={summary['macro_f1']:.6f}, top2={summary['top2_accuracy'] * 100:.2f}%"
            )

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / "summary_all_runs.csv", index=False)
    log("Done.")
    log(f"Summary: {output_dir / 'summary_all_runs.csv'}")
    return summary_df, output_dir


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate a ConvNeXt mask-guided checkpoint.")
    parser.add_argument("--config", required=True, help="Config stem, YAML path, or absolute config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--env", default="local", choices=("local", "kaggle"))
    parser.add_argument("--data-path", default=None, help="Folder containing train/val/test CSV files.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to evaluate.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--skip-no-tta", action="store_true")
    parser.add_argument("--skip-tta", action="store_true")
    parser.add_argument("--load-strict", action="store_true")
    parser.add_argument("--no-fuse-clip-tokens", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        env=args.env,
        data_path=args.data_path,
        splits=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        run_no_tta=not args.skip_no_tta,
        run_tta=not args.skip_tta,
        load_strict=args.load_strict,
        explicit_mask_dir=args.mask_dir,
        device=args.device,
        fuse_clip_tokens=not args.no_fuse_clip_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
