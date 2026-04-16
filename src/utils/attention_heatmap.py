"""
Attention Heatmap Visualization cho CNNDictionary V2.

Hiển thị cross-attention weights của mỗi region token lên ảnh gốc,
cho phép kiểm tra model có attend đúng vùng khuôn mặt không.

Usage:
    Được gọi tự động trong evaluator.py sau khi evaluate xong.

Output:
    - 1 figure lớn: mỗi hàng = 1 sample, mỗi cột = 1 region (forehead, eyebrows, ...)
    - Cột cuối = tổng hợp tất cả regions
    - Save ở outputs/figures/attention_heatmap.png + log lên WandB
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.data.emotions_dict import EMOTION_DICT


REGION_NAMES = ["forehead", "eyebrows", "eyes", "nose", "mouth", "chin"]


def get_attention_and_prediction(model, images, device):
    """
    Forward pass và lấy attention weights từ CNNDictionary.
    
    Args:
        model: CNNDictionary model (đã load checkpoint)
        images: tensor [B, C, H, W]
        device: torch.device
    
    Returns:
        preds: [B] predicted class indices
        attn_weights: [B, K, 36] cross-attention weights
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Lấy attention weights đã lưu trong forward pass
        attn_weights = model.attn_weights  # [B, K, 36]
    
    return preds.cpu().numpy(), attn_weights.cpu().numpy()


def attention_to_heatmap(attn_weights_single, grid_size=6):
    """
    Chuyển attention weights [K, 36] thành heatmaps [K, grid_size, grid_size].
    
    Args:
        attn_weights_single: [K, 36] attention weights cho 1 sample
        grid_size: kích thước spatial grid (6×6)
    
    Returns:
        heatmaps: [K, grid_size, grid_size] normalized heatmaps
    """
    K = attn_weights_single.shape[0]
    heatmaps = attn_weights_single.reshape(K, grid_size, grid_size)
    
    # Normalize mỗi region map về [0, 1]
    for i in range(K):
        hm = heatmaps[i]
        hm_min, hm_max = hm.min(), hm.max()
        if hm_max - hm_min > 1e-8:
            heatmaps[i] = (hm - hm_min) / (hm_max - hm_min)
        else:
            heatmaps[i] = np.zeros_like(hm)
    
    return heatmaps


def overlay_heatmap(img_tensor, heatmap, alpha=0.5, image_size=48):
    """
    Overlay heatmap lên ảnh gốc.
    
    Args:
        img_tensor: [C, H, W] image tensor
        heatmap: [grid, grid] attention heatmap 
        alpha: blend ratio
        image_size: output size
    
    Returns:
        overlay: [H, W, 3] RGB numpy array
    """
    # Ảnh gốc → numpy
    img_np = img_tensor.cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)  # [H, W]
        img_np = np.stack([img_np] * 3, axis=-1)  # [H, W, 3]
    elif img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Normalize ảnh về [0, 1]
    img_np = img_np.astype(np.float32)
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    # Undo normalization (mean=0.5, std=0.5) cho grayscale
    img_np = img_np * 0.5 + 0.5
    img_np = np.clip(img_np, 0, 1)
    
    # Resize heatmap lên kích thước ảnh
    heatmap_resized = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    
    # Tạo colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = heatmap_color.astype(np.float32) / 255.0
    
    # Blend
    overlay = heatmap_color * alpha + img_np * (1 - alpha)
    overlay = overlay / overlay.max()
    
    return overlay


def plot_attention_heatmaps(model, images, labels, preds, device, 
                             save_path=None, num_samples=8, image_size=48):
    """
    Vẽ figure lớn: mỗi hàng = 1 sample, mỗi cột = 1 region + cột tổng hợp.
    
    Args:
        model: CNNDictionary model
        images: list of image tensors [C, H, W]
        labels: list/array of true labels
        preds: list/array of predicted labels  
        device: torch.device
        save_path: path to save figure
        num_samples: số samples hiển thị
        image_size: kích thước ảnh
    
    Returns:
        fig: matplotlib figure object
    """
    num_samples = min(num_samples, len(images))
    
    # Stack images thành batch
    batch = torch.stack(images[:num_samples]).to(device)
    
    # Forward pass để lấy attention weights
    _, attn_weights = get_attention_and_prediction(model, batch, device)
    # attn_weights: [B, K, 36]
    
    num_regions = attn_weights.shape[1]
    num_cols = num_regions + 2  # original + 6 regions + combined
    
    fig, axes = plt.subplots(num_samples, num_cols, 
                              figsize=(num_cols * 2.2, num_samples * 2.5))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]  # Ensure 2D
    
    # Column headers
    col_titles = ["Original"] + REGION_NAMES[:num_regions] + ["Combined"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight='bold')
    
    for i in range(num_samples):
        img_tensor = images[i]
        true_label = int(labels[i])
        pred_label = int(preds[i])
        
        # Heatmaps cho sample i
        heatmaps = attention_to_heatmap(attn_weights[i])  # [K, 6, 6]
        
        # Cột 0: ảnh gốc
        img_np = img_tensor.cpu().numpy()
        if img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)
            img_np = img_np * 0.5 + 0.5  # undo normalize
            axes[i, 0].imshow(img_np, cmap='gray')
        elif img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = img_np * 0.5 + 0.5
            axes[i, 0].imshow(np.clip(img_np, 0, 1))
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i, 0].set_ylabel(
            f"T:{EMOTION_DICT[true_label]}\nP:{EMOTION_DICT[pred_label]}", 
            fontsize=9, color=color, fontweight='bold', rotation=0, labelpad=60
        )
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Cột 1→K: từng region heatmap
        for r in range(num_regions):
            overlay = overlay_heatmap(img_tensor, heatmaps[r], alpha=0.5, image_size=image_size)
            axes[i, r + 1].imshow(overlay)
            axes[i, r + 1].axis('off')
        
        # Cột cuối: combined — trung bình tất cả regions
        combined_heatmap = heatmaps.mean(axis=0)  # [6, 6]
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
        overlay_combined = overlay_heatmap(img_tensor, combined_heatmap, alpha=0.5, image_size=image_size)
        axes[i, num_cols - 1].imshow(overlay_combined)
        axes[i, num_cols - 1].axis('off')
    
    plt.suptitle("Cross-Attention Heatmap: Region Dictionary → Visual Features", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"--> Saved attention heatmap to {save_path}")
    
    return fig


def generate_attention_heatmaps(model, test_loader, device, save_dir, 
                                 num_correct=4, num_wrong=4):
    """
    Entry point: lấy samples đúng/sai từ test set rồi vẽ heatmap.
    Được gọi từ evaluator.py.
    
    Args:
        model: CNNDictionary model (đã eval)
        test_loader: DataLoader cho test set
        device: torch.device
        save_dir: thư mục save figure
        num_correct: số sample đoán đúng cần hiển thị  
        num_wrong: số sample đoán sai cần hiển thị
    
    Returns:
        fig_correct, fig_wrong: matplotlib figures (hoặc None nếu không đủ samples)
    """
    import os
    
    model.eval()
    
    correct_images, correct_labels, correct_preds = [], [], []
    wrong_images, wrong_labels, wrong_preds = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(preds)):
                img = images[i].cpu()
                true_l = labels[i].item()
                pred_l = preds[i].item()
                
                if true_l == pred_l and len(correct_images) < num_correct:
                    correct_images.append(img)
                    correct_labels.append(true_l)
                    correct_preds.append(pred_l)
                elif true_l != pred_l and len(wrong_images) < num_wrong:
                    wrong_images.append(img)
                    wrong_labels.append(true_l)
                    wrong_preds.append(pred_l)
                
                if len(correct_images) >= num_correct and len(wrong_images) >= num_wrong:
                    break
            
            if len(correct_images) >= num_correct and len(wrong_images) >= num_wrong:
                break
    
    fig_correct, fig_wrong = None, None
    
    if correct_images:
        print(f"\n--> Generating attention heatmap for {len(correct_images)} CORRECT samples...")
        fig_correct = plot_attention_heatmaps(
            model, correct_images, correct_labels, correct_preds, device,
            save_path=os.path.join(save_dir, "attention_heatmap_correct.png"),
            num_samples=len(correct_images)
        )
    
    if wrong_images:
        print(f"--> Generating attention heatmap for {len(wrong_images)} WRONG samples...")
        fig_wrong = plot_attention_heatmaps(
            model, wrong_images, wrong_labels, wrong_preds, device,
            save_path=os.path.join(save_dir, "attention_heatmap_wrong.png"),
            num_samples=len(wrong_images)
        )
    
    return fig_correct, fig_wrong
