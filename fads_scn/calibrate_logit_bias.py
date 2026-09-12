import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
import yaml

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import PureImageFER2013, build_transforms, seed_worker
from fads_scn.models.attentive_scn_model import AttentiveSCNFER


DEFAULT_CALIBRATION = {
    "enable_logit_bias": True,
    "metric": "hybrid",
    "search_on": "val",
    "use_tta": "flip",
    "class_names": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "bias_grid": {
        "fear": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "sad": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "neutral": [-0.8, -0.6, -0.4, -0.2, 0.0],
        "angry": [0.0, 0.2, 0.4],
        "disgust": [0.0, 0.2, 0.4],
        "surprise": [0.0],
        "happy": [0.0],
    },
    "tune_classes": ["fear", "sad", "neutral", "angry"],
    "save_path": "outputs/fads_scn_convnext/calibration_logit_bias.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate per-class logit bias for FADS-SCN")
    parser.add_argument("--config", type=str, default="fads_scn/configs/scn_convnext.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--data_path", type=str, default=None, help="Override FER split CSV directory")
    parser.add_argument("--search_split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_tta", type=str, default=None, choices=["none", "flip", "multiscale"])
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no_eval", action="store_true", help="Only search and save validation bias")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: str):
    config_path = Path(path)
    if not config_path.exists():
        config_path = repo_root / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f), config_path


def resolve_data_path(cfg: dict, env: str, override_path: str = None) -> str:
    if override_path is not None:
        return override_path

    data_path = cfg.get("data", {}).get("data_path", "dataset/fer13-split")
    if env != "kaggle":
        return data_path

    candidates = [
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
        "/kaggle/input/datasets/doduyquynii/fer13-split",
        "/kaggle/input/fer13-split/fer13-split",
        "/kaggle/input/fer13-split",
        "/kaggle/input/sgu-2026-facial-expression-recognition/dataset/fer13-split",
        "/kaggle/input/sgu-2026-facial-expression-recognition/fer13-split",
        "/kaggle/input/fer2013/dataset/fer13-split",
        "/kaggle/input/fer2013",
    ]
    for candidate in candidates:
        if (Path(candidate) / "val.csv").exists():
            return candidate
    return data_path


def tta_mode(value: str):
    if value in (None, "flip"):
        return True
    if value == "none":
        return False
    if value == "multiscale":
        return "multiscale"
    raise ValueError(f"Unsupported TTA mode: {value}")


def build_loader(cfg: dict, data_path: str, split: str, batch_size: int):
    transform = build_transforms(
        split,
        input_size=cfg.get("data", {}).get("input_size", 48),
        in_channels=cfg.get("model", {}).get("in_channels", 1),
        normalization=cfg.get("data", {}).get("normalization", "symmetric"),
    )
    dataset = PureImageFER2013(data_path=data_path, split=split, transform=transform)
    seed = cfg.get("seed", {}).get("random_seed", None)
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = seed_worker

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("data", {}).get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def build_model(cfg: dict, weights_path: str, device: torch.device):
    m_cfg = cfg["model"]
    model = AttentiveSCNFER(
        backbone_name=m_cfg.get("backbone", "resnet50"),
        num_classes=m_cfg.get("num_classes", 7),
        in_channels=m_cfg.get("in_channels", 1),
        embed_dim=m_cfg.get("embed_dim", 256),
        num_attn_heads=m_cfg.get("num_attn_heads", 8),
        use_latent_graph=m_cfg.get("use_latent_graph", True),
        use_spatial_attention=m_cfg.get("use_spatial_attention", True),
        graph_mode=m_cfg.get("graph_mode", "dense"),
        graph_topk=m_cfg.get("graph_topk", 3),
        graph_self_loop_bias=m_cfg.get("graph_self_loop_bias", 1.0),
        graph_fusion_mode=m_cfg.get("graph_fusion_mode", "legacy_add"),
        graph_gate_init=m_cfg.get("graph_gate_init", 0.01),
        dropout=0.0,
        classifier_type=m_cfg.get("classifier_type", "linear"),
        cosface_scale=m_cfg.get("cosface_scale", 30.0),
        cosface_margin=m_cfg.get("cosface_margin", 0.20),
        use_pretrained=False,
        pretrained_weights_path="",
        stem_init=m_cfg.get("stem_init", "mean"),
    )

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_logits_and_labels(model, loader, device, use_tta):
    all_logits = []
    all_labels = []
    model.eval()

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images, use_tta=use_tta)
        all_logits.append(outputs["logits"].detach().cpu())
        all_labels.append(targets.detach().cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def compute_metrics_from_logits(logits, labels, class_bias=None):
    logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

    if class_bias is not None:
        logits_np = logits_np + np.asarray(class_bias, dtype=np.float32).reshape(1, -1)

    preds = logits_np.argmax(axis=1)
    acc = accuracy_score(labels_np, preds)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(labels_np, preds)
    per_class_recall = recall_score(
        labels_np,
        preds,
        average=None,
        labels=list(range(logits_np.shape[1])),
        zero_division=0,
    )
    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(balanced_acc),
        "hybrid": float(acc * macro_f1),
        "per_class_recall": per_class_recall.tolist(),
    }


def build_bias_candidates(calib_cfg: dict):
    class_names = calib_cfg["class_names"]
    num_classes = len(class_names)
    bias_grid = calib_cfg.get("bias_grid", {})
    fixed_bias = calib_cfg.get("fixed_bias", {})
    tune_classes = set(calib_cfg.get("tune_classes", []))

    search_names = [name for name in bias_grid.keys() if not tune_classes or name in tune_classes]
    if not search_names:
        return [np.zeros(num_classes, dtype=np.float32)]

    candidates = []
    for values in itertools.product(*[bias_grid[name] for name in search_names]):
        bias = np.zeros(num_classes, dtype=np.float32)
        for name, value in fixed_bias.items():
            if name in class_names:
                bias[class_names.index(name)] = float(value)
        for name, value in zip(search_names, values):
            if name in class_names:
                bias[class_names.index(name)] = float(value)
        candidates.append(bias)
    return candidates


def search_best_logit_bias(logits, labels, calib_cfg: dict):
    metric_name = calib_cfg.get("metric", "hybrid")
    base_metrics = compute_metrics_from_logits(logits, labels)
    if metric_name not in base_metrics:
        raise ValueError(f"Unsupported calibration metric: {metric_name}")

    num_classes = logits.shape[1]
    best_bias = np.zeros(num_classes, dtype=np.float32)
    best_metrics = base_metrics
    best_score = base_metrics[metric_name]
    results = []

    logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    scale_factor = max(1.0, float(np.std(logits_np)) * 0.35)

    for bias in build_bias_candidates(calib_cfg):
        metrics = compute_metrics_from_logits(logits, labels, class_bias=bias)
        score = metrics[metric_name]
        results.append({"bias": bias.tolist(), **metrics})
        if score > best_score:
            best_score = score
            best_bias = bias.copy()
            best_metrics = metrics

    for step_range in (np.linspace(-2.0, 2.0, 17), np.linspace(-0.8, 0.8, 17)):
        steps = step_range * scale_factor
        current_bias = best_bias.copy()
        for _ in range(3):
            improved = False
            for class_idx in range(num_classes):
                for step in steps:
                    trial_bias = current_bias.copy()
                    trial_bias[class_idx] = step
                    metrics = compute_metrics_from_logits(logits, labels, class_bias=trial_bias)
                    score = metrics[metric_name]
                    results.append({"bias": trial_bias.tolist(), **metrics})
                    if score > best_score:
                        best_score = score
                        best_bias = trial_bias.copy()
                        current_bias = trial_bias.copy()
                        best_metrics = metrics
                        improved = True
            if not improved:
                break

    return {
        "metric": metric_name,
        "base_metrics": base_metrics,
        "best_bias": best_bias.tolist(),
        "best_metrics": best_metrics,
        "best_score": float(best_score),
        "num_candidates": len(results),
        "top_results": sorted(results, key=lambda item: item[metric_name], reverse=True)[:20],
    }


def print_metrics(prefix: str, metrics: dict):
    print(
        f"{prefix}: Acc={metrics['acc'] * 100:.2f}% | "
        f"MacroF1={metrics['macro_f1'] * 100:.2f}% | "
        f"BalancedAcc={metrics['balanced_acc'] * 100:.2f}% | "
        f"Hybrid={metrics['hybrid']:.4f}"
    )


def save_result(result: dict, save_path: str):
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVE] Calibration result -> {save_file}")


def main():
    args = parse_args()
    cfg, config_path = load_config(args.config)
    calib_cfg = DEFAULT_CALIBRATION.copy()
    calib_cfg.update(cfg.get("calibration", {}))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_path = resolve_data_path(cfg, args.env, args.data_path)
    batch_size = args.batch_size or cfg.get("data", {}).get("batch_size", 64)
    search_split = args.search_split or calib_cfg.get("search_on", "val")
    use_tta_name = args.use_tta or calib_cfg.get("use_tta", "flip")
    use_tta = tta_mode(use_tta_name)
    save_path = args.save_path or calib_cfg.get("save_path", DEFAULT_CALIBRATION["save_path"])

    print("\n=======================================================")
    print("[CALIBRATION] FADS-SCN Logit Bias")
    print(f"Config:       {config_path}")
    print(f"Weights:      {args.weights}")
    print(f"Data path:    {data_path}")
    print(f"Search split: {search_split} | Eval split: {args.eval_split} | TTA: {use_tta_name}")
    print(f"Device:       {device}")
    print("=======================================================\n")

    model = build_model(cfg, args.weights, device)
    search_loader = build_loader(cfg, data_path, search_split, batch_size)
    logits, labels = collect_logits_and_labels(model, search_loader, device, use_tta)

    result = search_best_logit_bias(logits, labels, calib_cfg)
    print_metrics(f"Raw {search_split}", result["base_metrics"])
    print_metrics(f"Calibrated {search_split}", result["best_metrics"])
    print(f"Best bias: {np.round(np.asarray(result['best_bias']), 3).tolist()}\n")

    if not args.no_eval:
        eval_loader = build_loader(cfg, data_path, args.eval_split, batch_size)
        eval_logits, eval_labels = collect_logits_and_labels(model, eval_loader, device, use_tta)
        raw_eval = compute_metrics_from_logits(eval_logits, eval_labels)
        calibrated_eval = compute_metrics_from_logits(eval_logits, eval_labels, class_bias=result["best_bias"])
        result["eval_split"] = args.eval_split
        result["eval_raw_metrics"] = raw_eval
        result["eval_calibrated_metrics"] = calibrated_eval
        print_metrics(f"Raw {args.eval_split}", raw_eval)
        print_metrics(f"Calibrated {args.eval_split}", calibrated_eval)

    save_result(result, save_path)


if __name__ == "__main__":
    main()
