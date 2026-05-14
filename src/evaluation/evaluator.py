import os
import pandas as pd
from tqdm import tqdm
import torch
from src.utils.visualization import plot_prediction_grid, plot_attention_heatmap_grid
from src.utils.logger_wandb import log_image_to_wandb
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix
from src.utils.data_stats import get_class_distribution

def evaluate_and_show(model, test_loader, testset_path, device, save_dir) -> None:
    """Evaluate, log metrics, and visualize 30 correct / 30 wrong samples."""
    model.eval()
    
    sample_limit = 30
    correct_images, correct_trues, correct_preds, correct_attns = [], [], [], []
    wrong_images, wrong_trues, wrong_preds, wrong_attns = [], [], [], []
    
    all_preds = []
    all_trues = []

    # Tell model to return attention weights if supported
    if hasattr(model, 'return_attn'):
        model.return_attn = True
        
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluate test set..."):
            if len(batch) == 3:
                images, labels, region_masks = batch
                region_masks = region_masks.to(device)
            else:
                images, labels = batch
                region_masks = None
            images, labels = images.to(device), labels.to(device)
            
            # --- Test Time Augmentation (TTA) - Horizontal Flip ---
            if region_masks is not None:
                out_orig = model(images, region_masks=region_masks)
            else:
                out_orig = model(images)
            images_flipped = torch.flip(images, dims=[3])
            if region_masks is not None:
                region_masks_flipped = torch.flip(region_masks, dims=[3])
                out_flipped = model(images_flipped, region_masks=region_masks_flipped)
            else:
                out_flipped = model(images_flipped)
            
            attn_weights_batch = None
            if isinstance(out_orig, tuple) and len(out_orig) == 2:
                logits_orig, attn_weights_batch = out_orig
                logits_flipped, attn_weights_flipped = out_flipped
                # Average predictions for TTA Boost
                logits = (logits_orig + logits_flipped) / 2.0
                # Keep heatmaps tied to the original image. TTA still improves
                # logits, but averaging original+flipped attention tends to blur
                # the spatial explanation.
            else:
                logits_orig = out_orig
                logits_flipped = out_flipped
                logits = (logits_orig + logits_flipped) / 2.0
            # ------------------------------------------------------
                
            _, preds = torch.max(logits, 1)
            
            imgs_cpu = images.cpu()
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            
            all_trues.extend(labels_cpu)
            all_preds.extend(preds_cpu)
            
            for i in range(len(preds_cpu)):
                img, true_label, pred_label = imgs_cpu[i], labels_cpu[i], preds_cpu[i]
                attn_w = attn_weights_batch[i].cpu().numpy() if attn_weights_batch is not None else None
                
                if true_label == pred_label:
                    if len(correct_images) < sample_limit:
                        correct_images.append(img)
                        correct_trues.append(true_label)
                        correct_preds.append(pred_label)
                        if attn_w is not None: correct_attns.append(attn_w)
                else:
                    if len(wrong_images) < sample_limit:
                        wrong_images.append(img)
                        wrong_trues.append(true_label)
                        wrong_preds.append(pred_label)
                        if attn_w is not None: wrong_attns.append(attn_w)
                        
    # Plot and push W&B
    print("\nPushing to WandB & Dashboard...")

    # metrics and confusoin matrix
    print("Compute metrics and confusion matrix...")
    acc, report = compute_metrics(all_trues, all_preds)
    print(f"--> Accuracy: {acc*100:.2f}%")
    print(f"--> Report:\n {pd.DataFrame(report).transpose().to_string()}")

    # Plot Confusion Matrix
    csv_path = testset_path
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, "test.csv")
        
    class_distribution = get_class_distribution(csv_path)
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig_cm = plot_confusion_matrix(all_trues, all_preds, class_distribution, acc, save_path=cm_path)
    log_image_to_wandb("Evaluation/Confusion_Matrix", fig_cm)


    if len(correct_images) > 0:
        if len(correct_attns) == len(correct_images):
            fig_corr = plot_attention_heatmap_grid(
                correct_images, correct_trues, correct_preds, correct_attns,
                title="Correct Predictions with Attention Heatmap", 
                save_path=os.path.join(save_dir, "correct_preds_attn.png")
            )
            log_image_to_wandb("Evaluation/Correct_Samples_Attention", fig_corr)
        else:
            fig_corr = plot_prediction_grid(
                correct_images, correct_trues, correct_preds, 
                title="Correct Predictions", 
                save_path=os.path.join(save_dir, "correct_preds.png")
            )
            log_image_to_wandb("Evaluation/Correct_Samples", fig_corr)
        
    if len(wrong_images) > 0:
        if len(wrong_attns) == len(wrong_images):
            fig_wrong = plot_attention_heatmap_grid(
                wrong_images, wrong_trues, wrong_preds, wrong_attns,
                title="Incorrect Predictions with Attention Heatmap", 
                save_path=os.path.join(save_dir, "wrong_preds_attn.png")
            )
            log_image_to_wandb("Evaluation/Wrong_Samples_Attention", fig_wrong)
        else:
            fig_wrong = plot_prediction_grid(
                wrong_images, wrong_trues, wrong_preds, 
                title="Incorrect Predictions", 
                save_path=os.path.join(save_dir, "wrong_preds.png")
            )
            log_image_to_wandb("Evaluation/Wrong_Samples", fig_wrong)

    print(f"Done! Save file at: {save_dir}")
