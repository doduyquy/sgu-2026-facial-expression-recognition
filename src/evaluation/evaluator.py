import os
import pandas as pd
from tqdm import tqdm
import torch
from src.utils.visualization import plot_prediction_grid
from src.utils.logger_wandb import log_image_to_wandb
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix
from src.utils.data_stats import get_class_distribution

def evaluate_and_show(model, test_loader, testset_path, device, save_dir) -> None:
    """Test set, 10 ảnh đoán đúng, 10 ảnh đoán sai và Visualize and log to wandb"""
    model.eval()
    
    correct_images, correct_trues, correct_preds = [], [], []
    wrong_images, wrong_trues, wrong_preds = [], [], []
    
    all_preds = []
    all_trues = []

    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluate test set..."):
            # Support DataLoader returning (images, labels) or (images, labels, bboxes)
            # or (images, labels, bboxes, semantic_meta)
            semantic_meta = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    images, labels, bboxes, semantic_meta = batch
                elif len(batch) == 3:
                    images, labels, bboxes = batch
                else:
                    images, labels = batch[:2]
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            # If bounding boxes are present, forward with them. If semantic_meta
            # contains region-level masks/confidences, pass them through to the model
            if 'bboxes' in locals() and bboxes is not None:
                bboxes = bboxes.to(device)
                if isinstance(semantic_meta, dict) and "region_mask" in semantic_meta:
                    region_mask = semantic_meta["region_mask"].to(device)
                    region_confidence = semantic_meta.get("region_confidence", None)
                    if region_confidence is not None:
                        region_confidence = region_confidence.to(device)
                    outputs = model(
                        images,
                        bboxes,
                        region_mask=region_mask,
                        region_confidence=region_confidence,
                    )
                else:
                    outputs = model(images, bboxes)
            else:
                outputs = model(images)

            logits = outputs["logits"] if isinstance(outputs, dict) else (outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
            _, preds = torch.max(logits, 1)
            
            imgs_cpu = images.cpu()
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            
            all_trues.extend(labels_cpu)
            all_preds.extend(preds_cpu)
            
            for i in range(len(preds_cpu)):
                img, true_label, pred_label = imgs_cpu[i], labels_cpu[i], preds_cpu[i]
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

    # metrics and confusoin matrix
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