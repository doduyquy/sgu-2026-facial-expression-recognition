import torch
import matplotlib.pyplot as plt
import numpy as np
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


def plot_prediction_grid(images, true_labels, pred_labels, title, save_path=None, masks=None, mask_alpha=0.35):
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
    for idx, (ax, img, true, pred) in enumerate(zip(axes, images, true_labels, pred_labels)):
        
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

        # 2.1 Optional mask overlay (heatmap-style).
        if masks is not None and idx < len(masks) and masks[idx] is not None:
            m = masks[idx]
            if torch.is_tensor(m):
                m = m.cpu().detach().numpy()
            if m.ndim == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            if m.ndim == 3:
                m = m.sum(axis=0)
            if m.ndim == 2:
                m = m.astype(np.float32)
                m_min, m_max = float(m.min()), float(m.max())
                if m_max > m_min:
                    m = (m - m_min) / (m_max - m_min)
                ax.imshow(m, cmap='Reds', alpha=mask_alpha, vmin=0.0, vmax=1.0)
        
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
