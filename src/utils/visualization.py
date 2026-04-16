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


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                          freeze_epochs=None, save_path=None):
    """
    Vẽ biểu đồ Loss + Accuracy song song, đánh dấu phase transition + best epoch.
    
    Args:
        train_losses, val_losses: list of floats
        train_accs, val_accs: list of floats (0-1 range)
        freeze_epochs: epoch mà backbone unfreeze (vẽ đường chia phase)
        save_path: path to save figure
    
    Returns:
        fig: matplotlib figure object (để log WandB)
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Tìm best epoch (val_loss thấp nhất)
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = min(val_losses)
    best_val_acc = val_accs[best_epoch - 1] if val_accs else None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training Curves", fontsize=16, fontweight='bold', y=1.02)
    
    # ── Loss Plot ──
    ax1.plot(epochs, train_losses, 'b-', linewidth=1.5, label='Train Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=1.5, label='Val Loss', alpha=0.8)
    
    # Đánh dấu best epoch
    ax1.scatter([best_epoch], [best_val_loss], color='gold', s=120, zorder=5, 
                edgecolors='black', linewidth=1.5, label=f'Best (ep {best_epoch})')
    ax1.annotate(f'Best: {best_val_loss:.4f}\nEpoch {best_epoch}', 
                 xy=(best_epoch, best_val_loss),
                 xytext=(best_epoch + 2, best_val_loss + 0.02),
                 fontsize=9, fontweight='bold', color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
    
    # Phase transition line
    if freeze_epochs and freeze_epochs < len(train_losses):
        ax1.axvline(x=freeze_epochs, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(freeze_epochs + 0.5, ax1.get_ylim()[1] * 0.95, 'Unfreeze\nBackbone', 
                 fontsize=9, color='green', fontweight='bold', va='top')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ── Accuracy Plot ──
    train_accs_pct = [a * 100 for a in train_accs]
    val_accs_pct = [a * 100 for a in val_accs]
    
    ax2.plot(epochs, train_accs_pct, 'b-', linewidth=1.5, label='Train Acc', alpha=0.8)
    ax2.plot(epochs, val_accs_pct, 'r-', linewidth=1.5, label='Val Acc', alpha=0.8)
    
    # Đánh dấu best epoch trên accuracy
    if best_val_acc is not None:
        ax2.scatter([best_epoch], [best_val_acc * 100], color='gold', s=120, zorder=5,
                    edgecolors='black', linewidth=1.5, label=f'Best (ep {best_epoch})')
        ax2.annotate(f'{best_val_acc*100:.2f}%', 
                     xy=(best_epoch, best_val_acc * 100),
                     xytext=(best_epoch + 2, best_val_acc * 100 + 1.5),
                     fontsize=9, fontweight='bold', color='darkblue',
                     arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
    
    # Max val accuracy
    max_val_acc_epoch = int(np.argmax(val_accs_pct)) + 1
    max_val_acc = max(val_accs_pct)
    if max_val_acc_epoch != best_epoch:
        ax2.scatter([max_val_acc_epoch], [max_val_acc], color='lime', s=80, zorder=5,
                    edgecolors='black', linewidth=1, label=f'Max Acc (ep {max_val_acc_epoch})')
    
    # Phase transition line
    if freeze_epochs and freeze_epochs < len(train_accs):
        ax2.axvline(x=freeze_epochs, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.text(freeze_epochs + 0.5, ax2.get_ylim()[0] + 1, 'Unfreeze\nBackbone',
                 fontsize=9, color='green', fontweight='bold', va='bottom')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ── Summary text box ──
    gap = train_accs_pct[-1] - val_accs_pct[-1]
    summary = (f"Final Train Acc: {train_accs_pct[-1]:.2f}%\n"
               f"Final Val Acc: {val_accs_pct[-1]:.2f}%\n"
               f"Overfit Gap: {gap:.2f}%\n"
               f"Best Val Loss: {best_val_loss:.4f} (ep {best_epoch})")
    
    fig.text(0.5, -0.06, summary, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"--> Saved training curves to {save_path}")
    
    return fig



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
