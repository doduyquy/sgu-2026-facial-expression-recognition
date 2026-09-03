"""
evaluator_tf.py — Evaluation pipeline on the FER2013 test set in TensorFlow.
Computes overall accuracy, macro F1, and detailed per-class classification report.
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def evaluate_test_set_tf(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    weights_path: Optional[str] = None,
    save_dir: str = "outputs/evaluation_tf",
) -> Dict[str, float]:
    """Run evaluation on the test set and print full classification metrics."""
    if weights_path is not None:
        print(f"--> Loading trained weights from: {weights_path}")
        model.load_weights(weights_path)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_preds = []
    all_targets = []

    print("--> Evaluating test set with Horizontal Flip TTA...")
    for inputs, labels in test_dataset:
        images = inputs["images"]
        bboxes = inputs["bboxes"]
        region_mask = inputs.get("region_mask", None)
        region_confidence = inputs.get("region_confidence", None)

        # Call with training=False executes the built-in 72.92% Horizontal Flip TTA
        outputs = model(
            images, bboxes, region_mask=region_mask, region_confidence=region_confidence, training=False
        )
        logits = outputs["logits"]
        preds = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
        targets = labels.numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    print("\n" + "=" * 50)
    print(f"--> Test Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    print(f"--> Classification Report:\n{report_df.to_string()}\n")

    # Save metrics report
    report_df.to_csv(save_path / "test_classification_report.csv")
    cm = confusion_matrix(y_true, y_pred)
    np.save(save_path / "test_confusion_matrix.npy", cm)

    return {"accuracy": acc, "report": report_dict}
