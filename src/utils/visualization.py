import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.data.emotions_dict import EMOTION_DICT


def plot_loss_curves(train_losses, val_losses, save_path=None):
    
    epoch_axis = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, train_losses, marker='o', label='Train loss')
    plt.plot(epoch_axis, val_losses, marker='x', label='Val loss')
    plt.title("Train and val loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("Saved plot at:", save_path)

    plt.show()


def plot_prediction_grid(images, true_labels, pred_labels, title, save_path=None):
    """Plot 10 true pred and 10 wrong pred images
    Args: 
        images: 10 image (numpy array)
        true_labels, pred_labels: list 10 number (category)
    Return: (show)
        figure (object)
    """
    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    fig.suptitle(title, fontsize=16)

    # Dùng zip để lặp qua từng ô (ax) và dữ liệu tương ứng
    for ax, img, true, pred in zip(axes, images, true_labels, pred_labels):
        
        # 1. Chuyển ảnh về Numpy và xử lý shape
        # Nếu img là Tensor (C, H, W), ta cần chuyển về (H, W) để vẽ ảnh xám
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()
        
        # Nếu ảnh có dạng (1, H, W) thì bóp về (H, W)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        # add-on for RGB
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # 2. Vẽ ảnh
        ax.imshow(img, cmap='gray')
        
        # 3. Đặt tiêu đề cho từng ô nhỏ
        # Đổi màu tiêu đề: xanh nếu đúng, đỏ nếu sai để dễ nhìn
        color = 'green' if true == pred else 'red'
        ax.set_title(f"T: {EMOTION_DICT[int(true)]}\nP: {EMOTION_DICT[int(pred)]}", 
                     fontsize=12, color=color)
        
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"--> Saved prediction grid to {save_path}")

    # Trả về fig object để log lên wandb
    return fig


def plot_attention_heatmap_grid(images, true_labels, pred_labels, attns, title, save_path=None):
    """
    Plot images with attention heatmap overlay
    Args: 
        images: list of images (tensor or numpy)
        true_labels, pred_labels: list of labels
        attns: list of attention weights [6, 18] or [6, 9] (numpy)
    """
    fig, axes = plt.subplots(1, len(images), figsize=(20, 3))
    if len(images) == 1:
        axes = [axes]
        
    fig.suptitle(title, fontsize=16)

    for ax, img, true, pred, attn in zip(axes, images, true_labels, pred_labels, attns):
        
        # 1. Image
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
 
 
        # 2. Attention Heatmap
        if attn.shape == (6, 18):
            vgg = attn[:, :9]
            res = attn[:, 9:]
            attn = (vgg + res) / 2.0  # Average 2 backbone [6, 9]
            
        if attn.ndim == 2 and attn.shape[0] == 6:
            attn = attn.mean(axis=0)  # Average 6 regions -> [num_visual_tokens]

        if attn.ndim == 1:
            side = int(np.sqrt(attn.shape[0]))
            if side * side == attn.shape[0]:
                attn = attn.reshape(side, side)
            else:
                attn = attn.reshape(1, -1)
            
        # Normalize to 0-1
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # Resize to original image size
        attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Convert grayscale image to RGB for overlay
        if img.ndim == 2:
            img_color = np.stack((img,)*3, axis=-1)
        else:
            img_color = img
            
        # Scale to 0-255
        img_color_uint8 = np.uint8(255 * img_color) if img_color.max() <= 1.0 else np.uint8(img_color)
        
        # Apply colormap JET
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img_color_uint8, 0.6, heatmap, 0.4, 0)
        
        # Plot
        ax.imshow(overlay)
        color = 'green' if true == pred else 'red'
        ax.set_title(f"T: {EMOTION_DICT[int(true)]}\nP: {EMOTION_DICT[int(pred)]}", 
                     fontsize=12, color=color)
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"--> Saved attention heatmap grid to {save_path}")

    return fig
