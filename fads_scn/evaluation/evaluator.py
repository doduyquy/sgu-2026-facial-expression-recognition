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


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """
    Renders and saves a publication-quality confusion matrix heatmap.
    Displays both count and row-normalized percentage in each cell.
    """
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Calculate row-normalized matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    display_mat = cm_norm if normalize else cm
    im = ax.imshow(display_mat, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Rate" if normalize else "Count", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = (display_mat.max() + display_mat.min()) / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            text_str = f"{count}\n({pct:.1f}%)" if normalize else f"{count}"
            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="medium",
                color="white" if display_mat[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

