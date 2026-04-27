"""
Evaluator cho GNN FER-2013.
Chạy inference trên test_loader, tính metric, plot confusion matrix.
"""
import os
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix
from src.utils.visualization import plot_prediction_grid

EMOTION_NAMES = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]


def _forward_batch(model, batch: dict, device) -> torch.Tensor:
    """
    Dispatch forward giống Trainer:
      - GNN batch : model(x, edge_index, edge_valid, mask)
      - MLP batch : model(x, mask=mask)
      - Plain     : model(x)
    """
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    x = batch.get("x")

    if "candidate_x" in batch:
        out = model(batch)
        return out["logits"] if isinstance(out, dict) else out

    if {"motif_score_vector", "match_scores", "matched_class"}.issubset(batch.keys()):
        out = model(batch)
        return out["logits"] if isinstance(out, dict) else out

    if "edge_index" in batch and "edge_valid" in batch:
        edge_index = batch["edge_index"]
        edge_valid = batch["edge_valid"]
        mask = batch.get("mask")
        out = model(x, edge_index=edge_index, edge_valid=edge_valid, mask=mask)
        return out["logits"] if isinstance(out, dict) else out

    mask = batch.get("mask")
    if mask is not None:
        out = model(x, mask=mask)
        return out["logits"] if isinstance(out, dict) else out

    out = model(x)
    return out["logits"] if isinstance(out, dict) else out


def _load_raw_test_images(config: dict):
    """Load raw FER test split để vẽ prediction grids nếu có data_path."""
    if config is None:
        return None

    data_path = config.get("data_path")
    if not data_path:
        return None

    test_csv = Path(data_path) / "test.csv"
    if not test_csv.exists():
        return None

    try:
        from data.raw_fer_dataset import RawFERDataset
        return RawFERDataset(test_csv, split="test")
    except Exception as exc:
        print(f"[WARN] Cannot load raw test images for visualization: {exc}")
        return None


def _save_prediction_grids(
    raw_test_ds,
    graph_ids: list[int],
    y_true: list[int],
    y_pred: list[int],
    save_dir: str,
) -> list[str]:
    """Lưu grid ảnh đoán đúng/sai nếu raw FER test split khả dụng."""
    if raw_test_ds is None or not graph_ids:
        return []

    correct_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p][:10]
    wrong_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p][:10]

    saved_paths = []
    for indices, filename, title in [
        (correct_idx, "correct_preds.png", "Correct Predictions"),
        (wrong_idx, "wrong_preds.png", "Wrong Predictions"),
    ]:
        if not indices:
            continue

        images = []
        labels = []
        preds = []
        for i in indices:
            sample = raw_test_ds[int(graph_ids[i])]
            images.append(sample.image)
            labels.append(int(y_true[i]))
            preds.append(int(y_pred[i]))

        save_path = os.path.join(save_dir, filename)
        fig = plot_prediction_grid(images, labels, preds, title=title, save_path=save_path)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        saved_paths.append(save_path)

    return saved_paths


def _log_eval_artifacts_to_wandb(metrics: dict, cm_path: str, extra_paths: list[str]) -> None:
    """Log scalar metrics và figure files lên WandB nếu run đang active."""
    try:
        import wandb
        if wandb.run is None:
            return
    except Exception:
        return

    from src.utils.logger_wandb import log_metrics

    log_metrics({
        "Test/Accuracy": metrics["accuracy"],
        "Test/MacroF1": metrics["macro_f1"],
        "Test/WeightedF1": metrics["weighted_f1"],
    })

    artifact_payload = {}
    for tag, path in [
        ("Test/ConfusionMatrix", cm_path),
        ("Test/CorrectPredictions", next((p for p in extra_paths if p.endswith("correct_preds.png")), None)),
        ("Test/WrongPredictions", next((p for p in extra_paths if p.endswith("wrong_preds.png")), None)),
    ]:
        if path and os.path.exists(path):
            artifact_payload[tag] = wandb.Image(path)

    if artifact_payload:
        wandb.log(artifact_payload)


def evaluate_and_show(model, test_loader, device, save_dir: str, config: dict | None = None) -> dict:
    """
    Chạy evaluation trên test set, plot confusion matrix và in kết quả.

    Args:
        model:       PyTorch model
        test_loader: DataLoader trả batch {"x": tensor, "y": tensor}
        device:      torch.device
        save_dir:    thư mục lưu ảnh
        config:      config runtime, dùng để resolve raw FER test.csv nếu cần

    Returns:
        metrics dict
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_preds = []
    all_trues = []
    all_graph_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            y = batch.get("y", batch.get("label")).to(device)
            logits = _forward_batch(model, batch, device)
            preds = torch.argmax(logits, dim=1)

            all_trues.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            if "graph_id" in batch:
                graph_ids = batch["graph_id"]
                if torch.is_tensor(graph_ids):
                    all_graph_ids.extend(graph_ids.cpu().numpy().tolist())
                else:
                    all_graph_ids.extend(list(graph_ids))

    metrics = compute_classification_metrics(all_trues, all_preds)

    # Print results
    print("\n" + "=" * 55)
    print("TEST SET EVALUATION")
    print("=" * 55)
    print(f"--> Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"--> Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"--> Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\n--> Classification Report:")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    print(report_df.to_string())

    # Plot Confusion Matrix
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        all_trues, all_preds,
        class_names=EMOTION_NAMES,
        acc=metrics["accuracy"],
        save_path=cm_path,
    )

    raw_test_ds = _load_raw_test_images(config)
    grid_paths = _save_prediction_grids(
        raw_test_ds=raw_test_ds,
        graph_ids=all_graph_ids,
        y_true=all_trues,
        y_pred=all_preds,
        save_dir=save_dir,
    )
    _log_eval_artifacts_to_wandb(metrics, cm_path, grid_paths)

    print(f"\n--> Figures saved to: {save_dir}")
    return metrics
