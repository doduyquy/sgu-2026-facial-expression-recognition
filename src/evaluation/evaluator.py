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
    wrong_masks = []
    
    all_preds = []
    all_trues = []

    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluate test set..."):
            if len(batch) == 4:
                images, labels, landmarks, landmark_mask = batch
                landmarks = landmarks.to(device)
                landmark_mask = landmark_mask.to(device)
            elif len(batch) == 3:
                images, labels, landmarks = batch
                landmarks = landmarks.to(device)
                landmark_mask = None
            else:
                images, labels = batch
                landmarks = None
                landmark_mask = None

            images, labels = images.to(device), labels.to(device)

            if landmarks is None:
                outputs = model(images)
            else:
                if landmark_mask is not None:
                    try:
                        outputs = model(images, landmarks, landmark_mask)
                    except TypeError:
                        outputs = model(images, landmarks)
                else:
                    try:
                        outputs = model(images, landmarks)
                    except TypeError:
                        outputs = model(images)

            _, preds = torch.max(outputs, 1)
            
            imgs_cpu = images.cpu()
            landmarks_cpu = landmarks.cpu() if landmarks is not None else None
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
                        if landmarks_cpu is not None and landmarks_cpu.dim() == 4:
                            wrong_masks.append(landmarks_cpu[i])
                        else:
                            wrong_masks.append(None)
                        
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
            title="Incorrect Predictions (Mask Overlay)",
            save_path=os.path.join(save_dir, "wrong_preds_with_mask.png"),
            masks=wrong_masks,
        )
        log_image_to_wandb("Evaluation/Wrong_Samples", fig_wrong)

    print(f"Done! Save file at: {save_dir}")
