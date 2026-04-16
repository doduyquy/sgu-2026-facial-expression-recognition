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
        for images, labels in tqdm(test_loader, desc="Evaluate test set..."):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            imgs_cpu = images.cpu()
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            
            all_trues.extend(labels_cpu)
            all_preds.extend(preds_cpu)
            
            for i in range(len(preds_cpu)):
                img, true_label, pred_label = imgs_cpu[i], labels_cpu[i], preds_cpu[i]
                if true_label == pred_label:
                    if len(correct_images) < 25:
                        correct_images.append(img)
                        correct_trues.append(true_label)
                        correct_preds.append(pred_label)
                else:
                    if len(wrong_images) < 25:
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
    csv_path = testset_path
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, "test.csv")
        
    class_distribution = get_class_distribution(csv_path)
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig_cm = plot_confusion_matrix(all_trues, all_preds, class_distribution, acc, save_path=cm_path)
    log_image_to_wandb("Evaluation/Confusion_Matrix", fig_cm)


    # ------ PROCESS GRAD-CAM HEATMAP OVERLAY ------ #
    cam_generator = None
    if hasattr(model, 'layer4'):
        try:
            from src.utils.gradcam import SimpleGradCAM, apply_heatmap
            cam_generator = SimpleGradCAM(model, model.layer4[-1])
            print("--> Grad-CAM Heatmap Active: Layer-4 Hooked!")
        except Exception as e:
            print(f"-!- Could not hook Grad-CAM: {e}")

    def inject_heatmap(image_list, true_label_list):
        if cam_generator is None: return image_list
        result = []
        for img, lbl in zip(image_list, true_label_list):
            input_tensor = img.unsqueeze(0).to(device)
            # Tạo map theo đúng label thực tế để xem model chú ý đúng khu vực chứa emotion thật chưa
            hm_mask = cam_generator.generate(input_tensor, target_class=lbl)
            rgb_numpy = apply_heatmap(img, hm_mask)
            # Chuyển về Tensor RGB [3, H, W] phục vụ thư viện plot_prediction_grid
            rgb_tensor = torch.from_numpy(rgb_numpy).permute(2, 0, 1)
            result.append(rgb_tensor)
        return result


    if len(correct_images) > 0:
        final_corrs = inject_heatmap(correct_images, correct_trues)
        fig_corr = plot_prediction_grid(
            final_corrs, correct_trues, correct_preds, 
            title="Correct Predictions with Grad-CAM Heatmap", 
            save_path=os.path.join(save_dir, "correct_preds.png")
        )
        log_image_to_wandb("Evaluation/Correct_Samples", fig_corr)
        
    if len(wrong_images) > 0:
        final_wrongs = inject_heatmap(wrong_images, wrong_trues)
        fig_wrong = plot_prediction_grid(
            final_wrongs, wrong_trues, wrong_preds, 
            title="Incorrect Predictions with Grad-CAM Heatmap", 
            save_path=os.path.join(save_dir, "wrong_preds.png")
        )
        log_image_to_wandb("Evaluation/Wrong_Samples", fig_wrong)

    # ------ CROSS-ATTENTION HEATMAP (CNNDictionary) ------ #
    # Tự động phát hiện model có cross-attention weights → vẽ heatmap
    if hasattr(model, 'attn_weights') and hasattr(model, 'dic_region'):
        try:
            from src.utils.attention_heatmap import generate_attention_heatmaps
            print("\n--> [Attention Heatmap] Generating region attention visualization...")
            fig_attn_corr, fig_attn_wrong = generate_attention_heatmaps(
                model, test_loader, device, save_dir,
                num_correct=4, num_wrong=4
            )
            if fig_attn_corr:
                log_image_to_wandb("Evaluation/Attention_Heatmap_Correct", fig_attn_corr)
            if fig_attn_wrong:
                log_image_to_wandb("Evaluation/Attention_Heatmap_Wrong", fig_attn_wrong)
            print("--> [Attention Heatmap] Done!")
        except Exception as e:
            print(f"-!- Attention Heatmap failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"Done! Save file at: {save_dir}")

