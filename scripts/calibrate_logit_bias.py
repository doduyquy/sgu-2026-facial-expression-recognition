import argparse
import itertools
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from src.data.dataloader import build_dataloader
from src.evaluation.evaluator import evaluate_and_show
from src.models import get_model
from src.utils.config import load_config


@torch.no_grad()
def collect_logits_and_labels(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in dataloader:
        semantic_meta = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                images, labels, bboxes, semantic_meta = batch
            elif len(batch) == 3:
                images, labels, bboxes = batch
            else:
                images, labels = batch[:2]
                bboxes = None
        elif isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
            bboxes = batch.get("bboxes", None)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device)
        labels = labels.to(device)
        if bboxes is not None:
            bboxes = bboxes.to(device)

        if bboxes is not None:
            if isinstance(semantic_meta, dict) and "region_mask" in semantic_meta:
                region_mask = semantic_meta["region_mask"].to(device)
                region_confidence = semantic_meta.get("region_confidence", None)
                if region_confidence is not None:
                    region_confidence = region_confidence.to(device)
                outputs = model(
                    images,
                    bboxes,
                    region_mask=region_mask,
                    region_confidence=region_confidence,
                )
            else:
                outputs = model(images, bboxes)
        else:
            outputs = model(images)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels


def compute_metrics_from_logits(logits, labels, class_bias=None):
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = logits

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    if class_bias is not None:
        bias_np = np.asarray(class_bias, dtype=np.float32).reshape(1, -1)
        logits_np = logits_np + bias_np

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

    hybrid = 0.5 * acc + 0.5 * macro_f1

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(balanced_acc),
        "hybrid": float(hybrid),
        "per_class_recall": per_class_recall.tolist(),
    }


def build_bias_candidates(calib_cfg):
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


def search_best_logit_bias(logits, labels, calib_cfg):
    metric_name = calib_cfg.get("metric", "hybrid")
    candidates = build_bias_candidates(calib_cfg)

    base_metrics = compute_metrics_from_logits(logits, labels, class_bias=None)

    best_bias = np.zeros(logits.shape[1], dtype=np.float32)
    best_metrics = base_metrics
    best_score = base_metrics[metric_name]

    results = []

    for bias in candidates:
        metrics = compute_metrics_from_logits(logits, labels, class_bias=bias)
        score = metrics[metric_name]

        results.append({
            "bias": bias.tolist(),
            **metrics,
        })

        if score > best_score:
            best_score = score
            best_bias = bias.copy()
            best_metrics = metrics

    return {
        "base_metrics": base_metrics,
        "best_bias": best_bias.tolist(),
        "best_metrics": best_metrics,
        "best_score": float(best_score),
        "metric": metric_name,
        "num_candidates": len(candidates),
        "top_results": sorted(results, key=lambda x: x[metric_name], reverse=True)[:20],
    }


def save_calibration_result(result, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[Calibration] saved to:", save_path)
    print("[Calibration] base:", result["base_metrics"])
    print("[Calibration] best:", result["best_metrics"])
    print("[Calibration] best_bias:", result["best_bias"])


def load_logit_bias(path, num_classes=7):
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bias = torch.tensor(data["best_bias"], dtype=torch.float32)
    if bias.numel() != num_classes:
        raise ValueError("Logit bias length does not match num_classes")
    return bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--eval_test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config, args.env)
    calib_cfg = config.get("calibration", {})

    if not calib_cfg.get("enable_logit_bias", False):
        raise ValueError("Calibration is disabled in config. Set calibration.enable_logit_bias=true")

    # data path and root path for each platform
    if config["env"]["platform"] == "kaggle":
        data_path = config["kaggle"].get("data_path", "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
        root_path = config["kaggle"].get("root_path", "/kaggle/working/sgu-2026-facial-expression-recognition/")
    else:
        data_path = config["local"].get("data_path", "../dataset")
        root_path = config["local"].get("root_path", "../")

    train_loader, val_loader, test_loader = build_dataloader(config=config, data_path=data_path)

    model = get_model(name=config["model"]["name"], config=config)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    search_on = calib_cfg.get("search_on", "val")
    search_loader = val_loader if search_on == "val" else test_loader

    logits, labels = collect_logits_and_labels(model, search_loader, device)
    result = search_best_logit_bias(logits, labels, calib_cfg)

    save_path = args.save_path or calib_cfg.get("save_path", "calibration_logit_bias.json")
    if not os.path.isabs(save_path):
        save_path = os.path.join(root_path, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_calibration_result(result, save_path)

    if args.eval_test:
        eval_dir_path = os.path.join(root_path, "outputs/figures")
        os.makedirs(eval_dir_path, exist_ok=True)
        testset_path = os.path.join(data_path, "test.csv")

        print("\n[Calibration] Evaluate test raw logits...")
        evaluate_and_show(model, test_loader, testset_path, device, eval_dir_path, logit_bias=None, run_tag="raw")

        print("\n[Calibration] Evaluate test with calibrated bias...")
        logit_bias = load_logit_bias(save_path, num_classes=logits.shape[1])
        evaluate_and_show(model, test_loader, testset_path, device, eval_dir_path, logit_bias=logit_bias, run_tag="calibrated")


if __name__ == "__main__":
    main()
