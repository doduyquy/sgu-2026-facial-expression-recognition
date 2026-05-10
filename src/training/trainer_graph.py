import copy
import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm
from src.utils.metrics import compute_classification_metrics

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train_one_epoch_graph(model, loader, optimizer, criterion, device, ema_model=None, ema_decay=0.999, global_step=0):
    model.train()

    running_loss = 0.0
    y_true = []
    y_pred = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        logits = model(x, targets=y)
        
        # criterion requires model for aux losses in EMO-GNP
        if hasattr(criterion, 'forward') and 'model' in criterion.forward.__code__.co_varnames:
            loss = criterion(logits, y, model=model)
        else:
            loss = criterion(logits, y)
            
        loss.backward()
        
        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        if ema_model is not None:
            update_ema_variables(model, ema_model, ema_decay, global_step)
            global_step += 1

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "loss": epoch_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "global_step": global_step
    }

@torch.no_grad()
def evaluate_graph(model, loader, criterion, device) -> Dict:
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        
        # For evaluation, we mainly care about cross entropy
        if hasattr(criterion, 'ce'):
            loss = criterion.ce(logits, y)
        else:
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