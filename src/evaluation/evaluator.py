import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from src.utils.visualization import plot_prediction_grid
from src.utils.logger_wandb import log_image_to_wandb
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix
from src.utils.data_stats import get_class_distribution
from src.models.utils import apply_multi_scale_tta


def evaluate_and_show(model, test_loader, testset_path, save_dir, use_tta=False) -> None:
    """Test set, 10 ảnh đoán đúng, 10 ảnh đoán sai và Visualize and log to wandb.
    
    Args:
        model: tf.keras.Model
        test_loader: tf.data.Dataset (batched)
        testset_path: str, path to test CSV
        save_dir: str, directory to save figures
        use_tta: bool, whether to use test-time augmentation
    """
    correct_images, correct_trues, correct_preds = [], [], []
    wrong_images, wrong_trues, wrong_preds = [], [], []
    
    all_preds = []
    all_trues = []

    os.makedirs(save_dir, exist_ok=True)
    
    for batch in tqdm(test_loader, desc="Evaluate test set..."):
        # Support different batch formats
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
            bboxes = batch.get('bboxes', None)
            region_mask = batch.get('region_mask', None)
            region_confidence = batch.get('region_confidence', None)
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                images, labels, bboxes, semantic_meta = batch
                if isinstance(semantic_meta, dict):
                    region_mask = semantic_meta.get('region_mask', None)
                    region_confidence = semantic_meta.get('region_confidence', None)
                else:
                    region_mask = None
                    region_confidence = None
            elif len(batch) == 3:
                images, labels, bboxes = batch
                region_mask = None
                region_confidence = None
            else:
                images, labels = batch[:2]
                bboxes = None
                region_mask = None
                region_confidence = None
        else:
            images, labels = batch
            bboxes = None
            region_mask = None
            region_confidence = None

        # Forward pass (no gradient tape needed for inference)
        if bboxes is not None:
            if region_mask is not None:
                if use_tta:
                    outputs = apply_multi_scale_tta(model, images, bboxes, region_mask, region_confidence)
                else:
                    outputs = model(images, bboxes, region_mask=region_mask,
                                    region_confidence=region_confidence, training=False)
            else:
                if use_tta:
                    outputs = apply_multi_scale_tta(model, images, bboxes)
                else:
                    outputs = model(images, bboxes, training=False)
        else:
            if use_tta:
                outputs = apply_multi_scale_tta(model, images)
            else:
                outputs = model(images, training=False)

        logits = outputs["logits"] if isinstance(outputs, dict) else (
            outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
        preds = tf.argmax(logits, axis=-1)
        
        # Convert to numpy
        imgs_np = images.numpy()
        labels_np = labels.numpy() if isinstance(labels, tf.Tensor) else np.array(labels)
        preds_np = preds.numpy()
        
        all_trues.extend(labels_np)
        all_preds.extend(preds_np)
        
        for i in range(len(preds_np)):
            img = imgs_np[i]
            true_label = int(labels_np[i])
            pred_label = int(preds_np[i])
            if true_label == pred_label:
                if len(correct_images) < 10:
                    correct_images.append(img)
                    correct_trues.append(true_label)
                    correct_preds.append(pred_label)
            else:
                if len(wrong_images) < 10:
                    wrong_images.append(img)
                    wrong_trues.append(true_label)
                    wrong_preds.append(pred_label)
                    
    # Plot and push W&B
    print("\nPushing to WandB & Dashboard...")

    # metrics and confusion matrix
    print("Compute metrics and confusion matrix...")
    acc, report = compute_metrics(all_trues, all_preds)
    print(f"--> Accuracy: {acc*100:.2f}%")
    print(f"--> Report:\n {pd.DataFrame(report).transpose().to_string()}")

    # Plot Confusion Matrix
    class_distribution = get_class_distribution(testset_path)
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig_cm = plot_confusion_matrix(all_trues, all_preds, class_distribution, acc, save_path=cm_path)
    log_image_to_wandb("Evaluation/Confusion_Matrix", fig_cm)

    if len(correct_images) > 0:
        fig_corr = plot_prediction_grid(
            correct_images, correct_trues, correct_preds, 
            title="Correct Predictions", 
            save_path=os.path.join(save_dir, "correct_preds.png")
        )
        log_image_to_wandb("Evaluation/Correct_Samples", fig_corr)
        
    if len(wrong_images) > 0:
        fig_wrong = plot_prediction_grid(
            wrong_images, wrong_trues, wrong_preds, 
            title="Incorrect Predictions", 
            save_path=os.path.join(save_dir, "wrong_preds.png")
        )
        log_image_to_wandb("Evaluation/Wrong_Samples", fig_wrong)

    print(f"Done! Save file at: {save_dir}")