"""Motif visualization script — TensorFlow version.

Note: This script references the older MotifGraphModel which is no longer 
the primary model. The current architecture uses SemanticROIGraphFER.
This script is kept for reference and can be adapted for the new model.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml


def visualize_inference(model_path, config_path, image_path=None):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Init model
    from src.models import get_model
    model = get_model(config["model"]["name"], config)
    
    if os.path.exists(model_path + ".index"):
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(model_path).expect_partial()
        print(f"[OK] Loaded checkpoint from {model_path}")

    # Get sample image
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        # NHWC format: (1, 48, 48, 1)
        img_tensor = tf.constant(img.astype(np.float32)[None, :, :, None] / 255.0)
    else:
        # Dummy
        img_tensor = tf.random.normal([1, 48, 48, 1])
        img = ((img_tensor[0, :, :, 0].numpy() + 1) * 127.5).astype(np.uint8)

    # Forward
    bboxes = tf.random.uniform([1, 9, 4], minval=5, maxval=40)
    outputs = model(img_tensor, bboxes, training=False)
    logits = outputs["logits"]
    probs = tf.nn.softmax(logits, axis=1)
    pred_class = tf.argmax(probs, axis=1).numpy()[0]

    # Class mapping (FER2013 standard)
    classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    plt.figure(figsize=(10, 5))

    # 1. Original Image
    plt.subplot(1, 2, 1)
    plt.title(f"Prediction: {classes[pred_class]} ({probs[0, pred_class]:.2f})")
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    # 2. Class-wise Similarity (Logits breakdown)
    plt.subplot(1, 2, 2)
    class_logits = logits[0].numpy()
    plt.bar(classes, class_logits, color="orange")
    plt.title("Logits per Class")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("motif_visualization.png")
    print("Visualization saved to motif_visualization.png")
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = "outputs/checkpoints/best_model"
    CONFIG_PATH = "configs/semantic_roi_graph.yaml"
    visualize_inference(MODEL_PATH, CONFIG_PATH)
