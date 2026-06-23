import io
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image


def init_wandb(config, run_name=None):
    """Init wandb with config"""

    # get wandb api key from environment
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    wandb.init(
        project=config.get("logging", {}).get("project_name", "FER2013"),
        entity=config.get("logging", {}).get("wandb_entity", "phucga15062005"),
        name=run_name,
        config=config,
        resume="allow",
    )


def log_metrics(metrics_dict, epoch=None):
    if wandb.run is not None:
        wandb.log(metrics_dict, step=epoch)


def log_image_to_wandb(tag, fig):
    """Log 10 true image and 10 wrong image"""
    if wandb.run is not None:
        wandb.log({tag: wandb.Image(fig)})


def log_heatmap_samples(
    images, labels, preds, node_attn_all, sampling_grid_all, epoch, n_samples=10
):
    """
    Log 10 correct and 10 incorrect heatmap visualizations to WandB.
    Handles both normal (4D) and TenCrop (5D) inputs.
    """
    if wandb.run is None or node_attn_all is None or sampling_grid_all is None:
        return

    # Handle TenCrop: (B, 10, C, H, W) -> Visualize only the first crop
    if images.dim() == 5:
        B, T, C, H, W = images.shape
        images = images[:, 0]  # (B, C, H, W)

        # node_attn_all is (B*T*Cands, ...)
        # sampling_grid_all is (B*T, ...)
        num_cands_per_crop = node_attn_all.shape[0] // (B * T)

        # Reshape and take the first crop's data
        node_attn_all = node_attn_all.view(
            B, T, num_cands_per_crop, *node_attn_all.shape[1:]
        )[:, 0]
        node_attn_all = node_attn_all.reshape(
            B * num_cands_per_crop, *node_attn_all.shape[1:]
        )

        sampling_grid_all = sampling_grid_all.view(B, T, *sampling_grid_all.shape[1:])[
            :, 0
        ]

    correct_imgs, incorrect_imgs = [], []
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    num_cands = node_attn_all.shape[0] // images.shape[0]

    for i in range(images.shape[0]):
        if len(correct_imgs) >= n_samples and len(incorrect_imgs) >= n_samples:
            break

        is_correct = preds[i] == labels[i]
        if is_correct and len(correct_imgs) >= n_samples:
            continue
        if not is_correct and len(incorrect_imgs) >= n_samples:
            continue

        # --- Create Heatmap Image ---
        img_tensor = images[i]
        pred_label = preds[i].item()
        true_label = labels[i].item()

        # Denormalize
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = (
            img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        # Attention processing: Mean over motifs to get importance per node
        # sample_attn: (num_cands, num_classes, motifs_per_class, 16)
        sample_attn = node_attn_all[i * num_cands : (i + 1) * num_cands]
        # (num_cands, motifs_per_class, 16) -> (num_cands, 16)
        attn = sample_attn[:, pred_label].mean(dim=1).flatten().cpu().numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        # Grid coords
        sample_grid = sampling_grid_all[i]
        coords = ((sample_grid.squeeze() + 1) / 2 * 224).cpu().numpy()

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=attn * 150,
            c="red",
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
        )

        title = f"P: {emotions[pred_label]} | T: {emotions[true_label]}"
        color = "green" if is_correct else "red"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        viz_img = Image.open(buf)

        if is_correct:
            correct_imgs.append(
                wandb.Image(viz_img, caption=f"Correct_{len(correct_imgs)}")
            )
        else:
            incorrect_imgs.append(
                wandb.Image(viz_img, caption=f"Wrong_{len(incorrect_imgs)}")
            )

    if correct_imgs:
        wandb.log({"viz/correct_samples": correct_imgs}, step=epoch)
    if incorrect_imgs:
        wandb.log({"viz/incorrect_samples": incorrect_imgs}, step=epoch)


def save_model_to_wandb(model_path, model_name="cnn"):
    """Lưu file pth trực tiếp vào Artifacts"""
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(name=f"{model_name}_{wandb.run.id}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            print(
                f"\t--> [WandB] Send File `{os.path.basename(model_path)}` to cloud successfully!"
            )
        except Exception as e:
            print(f"\t-!- [WandB] Error when upload Model: {e}")
