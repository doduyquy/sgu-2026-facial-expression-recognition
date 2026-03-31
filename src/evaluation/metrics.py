import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.emotions_dict import EMOTION_NAMES


def compute_metrics(true_labels, pred_labels):
    """Compute metrics: acc, f1, precision, recall"""
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=EMOTION_NAMES, output_dict=True)

    return acc, report

def plot_confusion_matrix(true_labels, pred_labels, accuracy, save_path=None):
    
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax)
    plt.ylabel("True label")
    plt.xlabel('Pred label')
    plt.title(f'Confusion matrix on test set, acc: {accuracy*100:.2f}%')

    # save fig
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig