"""
Evaluator cho GNN FER-2013.
Chạy inference trên test_loader, tính metric, plot confusion matrix.
"""
import os
import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix

EMOTION_NAMES = [
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
]


def _forward_batch(model, batch: dict, device) -> torch.Tensor:
    """
    Dispatch forward giống Trainer:
      - GNN batch : model(x, edge_index, edge_valid, mask)
      - MLP batch : model(x, mask=mask)
      - Plain     : model(x)
    """
    x = batch["x"].to(device)

    if "edge_index" in batch and "edge_valid" in batch:
        edge_index = batch["edge_index"].to(device)
        edge_valid = batch["edge_valid"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        return model(x, edge_index=edge_index, edge_valid=edge_valid, mask=mask)

    mask = batch.get("mask")
    if mask is not None:
        mask = mask.to(device)
        return model(x, mask=mask)

    return model(x)


def evaluate_and_show(model, test_loader, device, save_dir: str) -> dict:
    """
    Chạy evaluation trên test set, plot confusion matrix và in kết quả.

    Args:
        model:       PyTorch model
        test_loader: DataLoader trả batch {"x": tensor, "y": tensor}
        device:      torch.device
        save_dir:    thư mục lưu ảnh

    Returns:
        metrics dict
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            y = batch["y"].to(device)
            logits = _forward_batch(model, batch, device)
            preds = torch.argmax(logits, dim=1)

            all_trues.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

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

    print(f"\n--> Figures saved to: {save_dir}")
    return metrics
