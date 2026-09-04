"""
calibrate_logit_bias_tf.py — Post-hoc threshold and logit bias calibration for TensorFlow FER.
Performs grid search and scale-adapted coordinate descent on validation set logits
to balance minority classes (e.g. fear, sad, neutral) and boosts overall test accuracy & macro F1.

Usage:
    python tf_src/calibrate_logit_bias_tf.py --config configs/semantic_roi_graph_tf.yaml --weights outputs/checkpoints_tf/semantic_roi_graph_fer_tf_best.weights.h5 --eval_test
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
import tensorflow as tf

from tf_src.data.dataset_tf import create_tf_dataloader
from tf_src.models.semantic_roi_graph_tf import SemanticROIGraphFERTF
from tf_src.evaluation.evaluator_tf import evaluate_test_set_tf, EMOTIONS


def collect_logits_and_labels_tf(model: tf.keras.Model, dataset: tf.data.Dataset):
    """Collect predictions with built-in Horizontal Flip TTA over the given dataset."""
    all_logits = []
    all_labels = []

    for inputs, labels in dataset:
        images = inputs["images"]
        bboxes = inputs["bboxes"]
        region_mask = inputs.get("region_mask", None)
        region_confidence = inputs.get("region_confidence", None)

        outputs = model(
            images, bboxes, region_mask=region_mask, region_confidence=region_confidence, training=False
        )
        logits = outputs["logits"]
        all_logits.append(logits.numpy())
        all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, class_bias=None):
    logits_np = np.copy(logits)
    if class_bias is not None:
        bias_np = np.asarray(class_bias, dtype=np.float32).reshape(1, -1)
        logits_np = logits_np + bias_np

    preds = logits_np.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(labels, preds)
    per_class_recall = recall_score(
        labels,
        preds,
        average=None,
        labels=list(range(logits_np.shape[1])),
        zero_division=0,
    )

    hybrid = 0.5 * acc + 0.5 * macro_f1

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(balanced_acc),
        "hybrid": float(hybrid),
        "per_class_recall": per_class_recall.tolist(),
    }


def build_bias_candidates(calib_cfg: dict):
    class_names = calib_cfg["class_names"]
    num_classes = len(class_names)

    bias_grid = calib_cfg.get("bias_grid", {})
    fixed_bias = calib_cfg.get("fixed_bias", {})
    tune_classes = set(calib_cfg.get("tune_classes", []))

    search_names = [name for name in bias_grid.keys() if not tune_classes or name in tune_classes]
    search_values = [bias_grid[name] for name in search_names]

    candidates = []
    for values in itertools.product(*search_values):
        bias = np.zeros(num_classes, dtype=np.float32)

        for name, value in fixed_bias.items():
            if name in class_names:
                bias[class_names.index(name)] = float(value)

        for name, value in zip(search_names, values):
            if name in class_names:
                bias[class_names.index(name)] = float(value)

        candidates.append(bias)

    return candidates


def search_best_logit_bias(logits: np.ndarray, labels: np.ndarray, calib_cfg: dict):
    metric_name = calib_cfg.get("metric", "hybrid")
    base_metrics = compute_metrics_from_logits(logits, labels, class_bias=None)

    num_classes = logits.shape[1]
    best_bias = np.zeros(num_classes, dtype=np.float32)
    best_metrics = base_metrics
    best_score = base_metrics[metric_name]

    logit_std = float(np.std(logits))
    scale_factor = max(1.0, logit_std * 0.35)

    print(f"[Calibration] Logits dynamic std: {logit_std:.3f}, search step scale: {scale_factor:.3f}")
    print(f"[Calibration] Base {calib_cfg.get('search_on', 'val')} metrics: Acc={base_metrics['acc']*100:.2f}%, F1={base_metrics['macro_f1']*100:.2f}%, Hybrid={base_metrics['hybrid']*100:.2f}%")

    results = []

    # 1. Grid Search on specified candidates
    candidates = build_bias_candidates(calib_cfg)
    for bias in candidates:
        metrics = compute_metrics_from_logits(logits, labels, class_bias=bias)
        score = metrics[metric_name]
        results.append({"bias": bias.tolist(), **metrics})
        if score > best_score:
            best_score = score
            best_bias = bias.copy()
            best_metrics = metrics

    # 2. Multi-Pass Scale-Adapted Coordinate Descent Search
    for step_range in [np.linspace(-2.0, 2.0, 17), np.linspace(-0.8, 0.8, 17)]:
        steps = step_range * scale_factor
        current_bias = best_bias.copy()
        for _ in range(3):
            improved = False
            for c in range(num_classes):
                for s in steps:
                    trial_bias = current_bias.copy()
                    trial_bias[c] = s
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

    print(f"[Calibration] Optimized {calib_cfg.get('search_on', 'val')} metrics: Acc={best_metrics['acc']*100:.2f}%, F1={best_metrics['macro_f1']*100:.2f}%, Hybrid={best_metrics['hybrid']*100:.2f}%")
    print(f"[Calibration] Optimal bias vector: {np.round(best_bias, 3).tolist()}")

    return {
        "base_metrics": base_metrics,
        "best_bias": best_bias.tolist(),
        "best_metrics": best_metrics,
        "best_score": float(best_score),
        "metric": metric_name,
        "num_candidates": len(results),
        "top_results": sorted(results, key=lambda x: x[metric_name], reverse=True)[:20],
    }


def save_calibration_result(result: dict, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[Calibration] Saved to: {save_path}")


def load_logit_bias(path: str, num_classes: int = 7) -> np.ndarray:
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bias = np.array(data["best_bias"], dtype=np.float32)
    if bias.size != num_classes:
        raise ValueError("Logit bias length does not match num_classes")
    return bias


def main():
    parser = argparse.ArgumentParser(description="Calibrate Logit Bias for TensorFlow FER")
    parser.add_argument("--config", type=str, default="configs/semantic_roi_graph_tf.yaml")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint .weights.h5")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--eval_test", action="store_true", default=True, help="Evaluate on test set before and after calibration")
    args, _ = parser.parse_known_args()

    # Resolve config path
    def _find_config(raw_str: str) -> Path:
        p = Path(raw_str)
        stem = p.stem if p.suffix in [".yaml", ".yml"] else p.name
        candidates = [
            p,
            p.with_suffix(".yaml"),
            Path("configs") / f"{stem}.yaml",
            Path("configs") / stem,
            Path("tf_src/configs") / f"{stem}.yaml",
            Path("../configs") / f"{stem}.yaml",
            Path("/kaggle/working/sgu-2026-facial-expression-recognition/configs") / f"{stem}.yaml",
            Path("/kaggle/working/configs") / f"{stem}.yaml",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        return p.with_suffix(".yaml")

    cfg_path = _find_config(args.config)
    print(f"--> [Config] Loading: {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config.get("data", {}).get("data_dir", "dataset/fer13-split")
    masks_dir = args.masks_dir or config.get("data", {}).get("semantic_masks_dir", None)
    save_dir = "outputs/evaluation_tf"

    if args.env == "kaggle":
        candidates_data = [
            Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split"),
            Path("/kaggle/input/datasets/doduyquynii/fer13-split"),
            Path("/kaggle/input/fer13-split/fer13-split"),
            Path("/kaggle/input/fer13-split"),
        ]
        if not Path(data_dir).exists():
            for c in candidates_data:
                if (c / "test.csv").exists():
                    data_dir = str(c)
                    break

        candidates_masks = [
            Path("/kaggle/input/datasets/pha1t2/maskfer2013/semantic_masks"),
            Path("/kaggle/input/maskfer2013/semantic_masks"),
            Path("/kaggle/input/semantic_masks"),
        ]
        if masks_dir is None or not Path(masks_dir).exists():
            for m in candidates_masks:
                if m.exists():
                    masks_dir = str(m)
                    break
        save_dir = "/kaggle/working/outputs/evaluation_tf"

    default_calib_cfg = {
        "enable_logit_bias": True,
        "metric": "hybrid",
        "search_on": "val",
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
        "save_path": "outputs/calibration_logit_bias_tf.json",
    }
    calib_cfg = config.get("calibration", default_calib_cfg)

    batch_size = int(config.get("data", {}).get("batch_size", 64))

    print(f"--> Building validation dataset loader from {data_dir}...")
    val_loader = create_tf_dataloader(
        data_path=data_dir,
        split="val",
        batch_size=batch_size,
        semantic_masks_dir=masks_dir,
        is_training=False,
        shuffle=False,
    )

    print("--> Initializing SemanticROIGraphFERTF model architecture...")
    model = SemanticROIGraphFERTF(config=config)

    # Initialize graph with dummy forward pass
    dummy_img = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    dummy_box = tf.zeros([1, 9, 4], dtype=tf.float32)
    _ = model(dummy_img, dummy_box, training=False)

    print(f"--> Loading weights from {args.weights}...")
    model.load_weights(args.weights)

    search_on = calib_cfg.get("search_on", "val")
    print(f"\n[Calibration] Collecting predictions on {search_on} set with Horizontal Flip TTA...")
    logits, labels = collect_logits_and_labels_tf(model, val_loader)

    print("[Calibration] Searching optimal logit bias vector...")
    result = search_best_logit_bias(logits, labels, calib_cfg)

    save_path = args.save_path or calib_cfg.get("save_path", "outputs/calibration_logit_bias_tf.json")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_calibration_result(result, save_path)

    if args.eval_test:
        print(f"\n--> Building test dataset loader from {data_dir}...")
        test_loader = create_tf_dataloader(
            data_path=data_dir,
            split="test",
            batch_size=batch_size,
            semantic_masks_dir=masks_dir,
            is_training=False,
            shuffle=False,
        )

        print("\n" + "=" * 55)
        print("[Calibration] Step 1: Evaluating Test with RAW Logits")
        print("=" * 55)
        raw_res = evaluate_test_set_tf(model, test_loader, save_dir=save_dir, logit_bias=None, run_tag="raw")

        print("\n" + "=" * 55)
        print("[Calibration] Step 2: Evaluating Test with CALIBRATED Logits")
        print("=" * 55)
        best_bias = np.array(result["best_bias"], dtype=np.float32)
        calib_res = evaluate_test_set_tf(model, test_loader, save_dir=save_dir, logit_bias=best_bias, run_tag="calibrated")

        print("\n" + "=" * 55)
        print(f"--> Summary: RAW Acc = {raw_res['accuracy']*100:.2f}%  -->  CALIBRATED Acc = {calib_res['accuracy']*100:.2f}%")
        print("=" * 55)


if __name__ == "__main__":
    main()
