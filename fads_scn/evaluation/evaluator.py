import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from ..data.dataset import EMOTION_NAMES


@torch.no_grad()
def evaluate_model(model, dataloader, device, use_tta: bool = True):
    """
    Evaluate model on a dataloader.
    Returns:
        metrics: dict with 'accuracy', 'macro_f1', 'hybrid_score', 'per_class_acc', 'report', 'confusion_matrix'
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_alphas = []

    for batch in dataloader:
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images, use_tta=use_tta)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        if "alpha" in outputs:
            all_alphas.extend(outputs["alpha"].cpu().view(-1).numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    hybrid_score = float(acc * macro_f1)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(EMOTION_NAMES))))
    # Per-class accuracy
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)

    per_class_dict = {
        name: round(float(acc_val) * 100, 2)
        for name, acc_val in zip(EMOTION_NAMES, per_class_acc)
    }

    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(EMOTION_NAMES))),
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "hybrid_score": hybrid_score,
        "per_class_acc": per_class_dict,
        "confusion_matrix": cm,
        "report": report,
        "mean_alpha": float(np.mean(all_alphas)) if len(all_alphas) > 0 else 1.0,
    }
