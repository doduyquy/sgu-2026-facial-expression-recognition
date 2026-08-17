import numpy as np
import matplotlib.pyplot as plt
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
    """Plot 10 true pred and 10 wrong pred images.
    
    Args: 
        images: list of numpy arrays (NHWC format for TF)
        true_labels, pred_labels: list of integer labels
    Return: 
        figure object
    """
    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    fig.suptitle(title, fontsize=16)

    for ax, img, true, pred in zip(axes, images, true_labels, pred_labels):
        
        # Convert to numpy if needed
        if hasattr(img, 'numpy'):
            img = img.numpy()
        
        # Handle TenCrop input: (10, H, W, C) -> select center crop
        if img.ndim == 4 and img.shape[0] == 10:
            img = img[4]  # Index 4 is the center crop
            
        # TF uses NHWC: (H, W, C)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)  # (H, W) for grayscale
        elif img.ndim == 3 and img.shape[-1] == 3:
            pass  # Already in (H, W, 3) format for RGB
        # Handle legacy NCHW format just in case
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # Denormalize if needed (assuming mean=0.5, std=0.5)
        if img.min() < 0:
            img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        ax.imshow(img, cmap='gray')
        
        color = 'green' if true == pred else 'red'
        ax.set_title(f"T: {EMOTION_DICT[int(true)]}\nP: {EMOTION_DICT[int(pred)]}", 
                     fontsize=12, color=color)
        
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"--> Saved prediction grid to {save_path}")

    return fig

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') 
    
    # Test with standard 3D image (H, W, C) NHWC
    dummy_imgs_3d = [np.random.randn(48, 48, 1).astype(np.float32) for _ in range(10)]
    labels = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    preds = [0, 1, 0, 3, 4, 6, 6, 0, 1, 3]
    
    print("Testing 3D visualization...")
    plot_prediction_grid(dummy_imgs_3d, labels, preds, "Test 3D")
    
    print("Visualization tests completed.")