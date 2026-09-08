import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import PureImageFER2013, seed_worker
from fads_scn import runtime
from fads_scn.data.dataset import transforms_from_config, EMOTION_NAMES


DEFAULT_CALIBRATION = {
    "enable_logit_bias": False,
    "metric": "acc",
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
    "save_path": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate per-class logit bias for FADS-SCN")
    parser.add_argument("--config", help="Optional calibration settings and fallback for legacy weights")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--data_path", type=str, default=None, help="Override FER split CSV directory")
    parser.add_argument("--search_split", type=str, default="val", choices=["val"])
    parser.add_argument("--eval_split", type=str, default=None, choices=["val", "test"])
    parser.add_argument("--enable", action="store_true", help="Explicitly enable bias calibration")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_tta", type=str, default=None, choices=["none", "flip", "multiscale"])
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no_eval", action="store_true", help="Only search and save validation bias")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_loader(cfg: dict, data_path: str, split: str, batch_size: int):
    transform = transforms_from_config(cfg, "val")
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
    macro_f1 = f1_score(labels_np, preds, labels=list(range(logits_np.shape[1])), average="macro", zero_division=0)
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


def _bias_spec(calib_cfg):
    names = calib_cfg["class_names"]
    if len(names) != len(set(names)):
        raise ValueError("class_names must be unique")
    fixed = calib_cfg.get("fixed_bias", {})
    grid = calib_cfg.get("bias_grid", {})
    tune = calib_cfg.get("tune_classes", list(grid))
    unknown = (set(fixed) | set(grid) | set(tune)) - set(names)
    if unknown:
        raise ValueError(f"Unknown calibration classes: {unknown}")
    search = [name for name in tune if name not in fixed]
    for name in search:
        values = np.asarray(grid.get(name, []), dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Missing or invalid bias_grid for {name}")
    if not all(np.isfinite(float(value)) for value in fixed.values()):
        raise ValueError("fixed_bias must be finite")
    return names, fixed, grid, search


def build_bias_candidates(calib_cfg):
    names, fixed, grid, search = _bias_spec(calib_cfg)
    candidates = []
    for values in itertools.product(*[grid[name] for name in search]):
        bias = np.zeros(len(names), dtype=np.float32)
        for name, value in fixed.items():
            bias[names.index(name)] = float(value)
        for name, value in zip(search, values):
            bias[names.index(name)] = float(value)
        candidates.append(bias)
    return candidates


def search_best_logit_bias(logits, labels, calib_cfg):
    names, fixed, grid, search = _bias_spec(calib_cfg)
    if len(names) != logits.shape[1]:
        raise ValueError("class_names does not match logits")
    metric = calib_cfg.get("metric", "acc")
    if metric not in ("acc", "macro_f1", "balanced_acc", "hybrid"):
        raise ValueError(f"Unsupported calibration metric: {metric}")
    base_metrics = compute_metrics_from_logits(logits, labels)
    best_bias, best_metrics, best_score = None, None, -float("inf")
    results, seen = [], set()

    def consider(bias):
        nonlocal best_bias, best_metrics, best_score
        key = tuple(np.round(bias, 7))
        if key in seen:
            return
        seen.add(key)
        metrics = compute_metrics_from_logits(logits, labels, class_bias=bias)
        score = metrics[metric]
        results.append({"bias": bias.tolist(), **metrics})
        if score > best_score:
            best_bias, best_metrics, best_score = bias.copy(), metrics, score

    # Always honor fixed biases, including the initial candidate.
    initial = np.zeros(len(names), dtype=np.float32)
    for name, value in fixed.items():
        initial[names.index(name)] = value
    for name in search:
        initial[names.index(name)] = np.clip(0.0, min(grid[name]), max(grid[name]))
    consider(initial)
    for bias in build_bias_candidates(calib_cfg):
        consider(bias)
    # Coordinate refinement only on allowed classes, within configured bounds.
    for refinement in range(2):
        for name in search:
            index = names.index(name)
            lower, upper = min(grid[name]), max(grid[name])
            radius = (upper - lower) / (2 ** (refinement + 1))
            center = float(best_bias[index])
            for value in np.linspace(max(lower, center - radius), min(upper, center + radius), 17):
                candidate = best_bias.copy()
                candidate[index] = value
                consider(candidate)
    return {"metric": metric, "base_metrics": base_metrics, "best_bias": best_bias.tolist(),
            "best_metrics": best_metrics, "best_score": float(best_score), "num_candidates": len(results),
            "top_results": sorted(results, key=lambda item: item[metric], reverse=True)[:20]}


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
    with open(save_file, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"[SAVE] Calibration result -> {save_file}")


def main():
    args = parse_args()
    supplied_cfg = runtime.load_config(args.config)[0] if args.config else None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, checkpoint = runtime.load_checkpoint_model(args.weights, device, supplied_cfg)
    # Architecture and preprocessing always come from the checkpoint, if embedded.
    settings = supplied_cfg if supplied_cfg is not None else cfg
    calib_cfg = runtime.merge_config(DEFAULT_CALIBRATION, settings.get("calibration", {}))
    if not (args.enable or calib_cfg.get("enable_logit_bias", False)):
        raise ValueError("Calibration is disabled; pass --enable to run validation bias search")
    if calib_cfg.get("class_names") != EMOTION_NAMES:
        raise ValueError("Calibration class_names must match the FER2013 class order")
    if calib_cfg.get("search_on", "val") != "val":
        raise ValueError("Bias search must use validation, never the test split")
    splits = ("val", args.eval_split) if args.eval_split and not args.no_eval else ("val",)
    data_path = runtime.resolve_data_path(cfg, args.env, args.data_path, splits)
    batch_size = args.batch_size or cfg.get("data", {}).get("batch_size", 64)
    search_split = "val"
    use_tta_name = args.use_tta or calib_cfg.get("use_tta", "flip")
    use_tta = runtime.tta_mode(use_tta_name)
    save_path = (args.save_path or calib_cfg.get("save_path")
                 or Path(args.weights).resolve().parent / "calibration_logit_bias.json")
    if Path(save_path).exists():
        raise FileExistsError(f"Choose a new --save_path: {save_path}")

    print("\n=======================================================")
    print("[CALIBRATION] FADS-SCN Logit Bias")
    print(f"Config:       {args.config or 'embedded checkpoint config'}")
    print(f"Weights:      {args.weights}")
    print(f"Data path:    {data_path}")
    print(f"Search split: {search_split} | Eval split: {args.eval_split} | TTA: {use_tta_name}")
    print(f"Device:       {device}")
    print("=======================================================\n")

    search_loader = build_loader(cfg, data_path, search_split, batch_size)
    logits, labels = collect_logits_and_labels(model, search_loader, device, use_tta)

    result = search_best_logit_bias(logits, labels, calib_cfg)
    result.update(search_split="val", class_names=EMOTION_NAMES, tta=runtime.tta_mode(use_tta),
                  checkpoint_sha256=runtime.checkpoint_hash(args.weights), calibration_config=calib_cfg)
    print_metrics(f"Raw {search_split}", result["base_metrics"])
    print_metrics(f"Calibrated {search_split}", result["best_metrics"])
    print(f"Best bias: {np.round(np.asarray(result['best_bias']), 3).tolist()}\n")

    if args.eval_split is not None and not args.no_eval:
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
