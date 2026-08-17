import wandb 
import os
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image

def init_wandb(config, run_name=None):
    """Init wandb with config"""
    
    # get wandb api key from environment
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key: 
        wandb.login(key=wandb_api_key)

    wandb.init(
        project=config.get('logging', {}).get('project_name', "FER2013"), 
        entity=config.get('logging', {}).get('wandb_entity', 'phucga15062005'),
        name=run_name,
        config=config,
        resume="allow" 
    )

def log_metrics(metrics_dict, epoch=None):
    if wandb.run is not None:
        wandb.log(metrics_dict, step=epoch)

def log_image_to_wandb(tag, fig):
    """Log figure to wandb"""
    if wandb.run is not None:
        wandb.log({tag: wandb.Image(fig)})

def log_heatmap_samples(images, labels, preds, node_attn_all, sampling_grid_all, epoch, n_samples=10):
    """
    Log 10 correct and 10 incorrect heatmap visualizations to WandB.
    Handles both normal (4D) and TenCrop (5D) inputs.
    
    Note: images should be in NHWC format for TF.
    """
    if wandb.run is None or node_attn_all is None or sampling_grid_all is None:
        return

    # Convert to numpy if needed
    if hasattr(images, 'numpy'):
        images = images.numpy()
    if hasattr(node_attn_all, 'numpy'):
        node_attn_all = node_attn_all.numpy()
    if hasattr(sampling_grid_all, 'numpy'):
        sampling_grid_all = sampling_grid_all.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    if hasattr(preds, 'numpy'):
        preds = preds.numpy()

    # Handle TenCrop: (B, 10, H, W, C) -> Visualize only the first crop
    if images.ndim == 5:
        B, T = images.shape[:2]
        images = images[:, 0]  # (B, H, W, C)
        
        num_cands_per_crop = node_attn_all.shape[0] // (B * T)
        node_attn_all = node_attn_all.reshape(B, T, num_cands_per_crop, *node_attn_all.shape[1:])[:, 0]
        node_attn_all = node_attn_all.reshape(B * num_cands_per_crop, *node_attn_all.shape[2:])
        
        sampling_grid_all = sampling_grid_all.reshape(B, T, *sampling_grid_all.shape[1:])[:, 0]

    correct_imgs, incorrect_imgs = [], []
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    num_cands = node_attn_all.shape[0] // images.shape[0]

    for i in range(images.shape[0]):
        if len(correct_imgs) >= n_samples and len(incorrect_imgs) >= n_samples:
            break
            
        is_correct = (preds[i] == labels[i])
        if is_correct and len(correct_imgs) >= n_samples: continue
        if not is_correct and len(incorrect_imgs) >= n_samples: continue
        
        # --- Create Heatmap Image ---
        img = images[i]  # (H, W, C) NHWC format
        pred_label = int(preds[i])
        true_label = int(labels[i])
        
        # Denormalize (assuming mean=0.5, std=0.5)
        img = (img * 0.5 + 0.5).clip(0, 1)
        
        # Attention processing
        sample_attn = node_attn_all[i*num_cands : (i+1)*num_cands]
        attn = sample_attn[:, pred_label].mean(axis=1).flatten()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # Grid coords
        sample_grid = sampling_grid_all[i]
        coords = ((sample_grid.squeeze() + 1) / 2 * 224)
        
        # Plot
        fig, ax = plt.subplots(figsize=(4, 4))
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(-1), cmap='gray')
        else:
            ax.imshow(img)
        ax.scatter(coords[:, 0], coords[:, 1], s=attn * 150, c='red', alpha=0.6, edgecolors='white', linewidth=0.5)
        
        title = f"P: {emotions[pred_label]} | T: {emotions[true_label]}"
        color = 'green' if is_correct else 'red'
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        viz_img = Image.open(buf)
        
        if is_correct:
            correct_imgs.append(wandb.Image(viz_img, caption=f"Correct_{len(correct_imgs)}"))
        else:
            incorrect_imgs.append(wandb.Image(viz_img, caption=f"Wrong_{len(incorrect_imgs)}"))

    if correct_imgs:
        wandb.log({"viz/correct_samples": correct_imgs}, step=epoch)
    if incorrect_imgs:
        wandb.log({"viz/incorrect_samples": incorrect_imgs}, step=epoch)

def save_model_to_wandb(model_path, model_name="cnn"):
    """Upload checkpoint files to Artifacts"""
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(name=f"{model_name}_{wandb.run.id}", type="model")
            # TF checkpoints have multiple files (.index, .data-00000-of-00001)
            checkpoint_dir = os.path.dirname(model_path)
            checkpoint_prefix = os.path.basename(model_path)
            for f in os.listdir(checkpoint_dir):
                if f.startswith(checkpoint_prefix):
                    artifact.add_file(os.path.join(checkpoint_dir, f))
            wandb.log_artifact(artifact)
            print(f"\t--> [WandB] Send checkpoint `{checkpoint_prefix}` to cloud successfully!")
        except Exception as e:
            print(f"\t-!- [WandB] Error when upload Model: {e}")
