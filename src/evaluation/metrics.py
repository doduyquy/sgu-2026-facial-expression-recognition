import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from src.data.emotions_dict import EMOTION_NAMES


def compute_metrics(true_labels, pred_labels):
    """Compute metrics: acc, f1, precision, recall"""
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=EMOTION_NAMES, output_dict=True)

    return acc, report

def plot_confusion_matrix(true_labels, pred_labels, class_distribution, accuracy, save_path=None):
    labels = list(range(len(EMOTION_NAMES)))
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=labels)

    # Row-wise denominator = number of actual samples per class.
    # Prefer provided class_distribution to keep consistency with dataset stats.
    if class_distribution is not None:
        actual_counts = np.array([class_distribution.get(i, 0) for i in labels], dtype=float)
    else:
        actual_counts = conf_matrix.sum(axis=1).astype(float)

    row_percent = np.divide(
        conf_matrix,
        actual_counts[:, None],
        out=np.zeros_like(conf_matrix, dtype=float),
        where=actual_counts[:, None] != 0,
    ) * 100.0

    annot = np.empty_like(conf_matrix, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            annot[i, j] = f"{conf_matrix[i, j]}\n{row_percent[i, j]:.1f}%"

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(
        conf_matrix,
        annot=annot,
        fmt='',
        cmap='Blues',
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES,
        ax=ax,
    )
    plt.ylabel("True label")
    plt.xlabel('Pred label')
    plt.title(f'Confusion matrix on test set, acc: {accuracy*100:.2f}%')

    # save fig
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig




if __name__ == "__main__":

    # Synthetic sample for quick local check of confusion matrix annotations.
    y_true = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6]
    y_pred = [0, 1, 0, 1, 2, 2, 2, 3, 3, 0, 4, 6, 5, 4, 6, 6, 3]

    class_distribution = pd.Series(
        [y_true.count(i) for i in range(len(EMOTION_NAMES))],
        index=list(range(len(EMOTION_NAMES))),
    )

    accuracy, _ = compute_metrics(y_true, y_pred)
    fig = plot_confusion_matrix(
        true_labels=y_true,
        pred_labels=y_pred,
        class_distribution=class_distribution,
        accuracy=accuracy,
    )
    plt.show()