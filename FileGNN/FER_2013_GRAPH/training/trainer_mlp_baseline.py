from typing import Dict
import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import compute_classification_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "loss": epoch_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict:
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "loss": epoch_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }