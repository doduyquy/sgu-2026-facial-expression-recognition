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
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
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