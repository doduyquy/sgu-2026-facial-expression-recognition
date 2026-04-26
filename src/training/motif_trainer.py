import torch
from tqdm import tqdm
from src.utils.metrics import compute_classification_metrics

def train_one_epoch_motif(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    pbar = tqdm(loader, desc="Train (Motif)", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Model returns logits and motif-related data for consistency loss
        logits, top_k_idx, centers, scores = model(images, return_selection=True)
        
        # Loss criterion should be CombinedMotifLoss
        loss = criterion(logits, labels, scores, top_k_idx, model=model)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "loss": epoch_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }

@torch.no_grad()
def evaluate_motif(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    pbar = tqdm(loader, desc="Eval (Motif)", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits, top_k_idx, centers, scores = model(images, return_selection=True)
        loss = criterion(logits, labels, scores, top_k_idx, model=model)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "loss": epoch_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
