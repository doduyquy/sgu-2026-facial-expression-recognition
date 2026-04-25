import numpy as np


def compute_classification_metrics(y_true, y_pred) -> dict:
    """
    Tính các metric phân loại.

    Returns:
        {
            "accuracy":         float,
            "macro_f1":         float,
            "weighted_f1":      float,
            "confusion_matrix": np.ndarray,
            "report":           dict
        }
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "report": report,
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None, acc=None, save_path=None):
    """Plot confusion matrix với matplotlib."""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(9, 7))
    try:
        import seaborn as sns
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
    except Exception:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    title = "Confusion Matrix"
    if acc is not None:
        title += f"  (Acc: {acc * 100:.2f}%)"
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved: {save_path}")

    return fig
