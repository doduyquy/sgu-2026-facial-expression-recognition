"""Tiny demo runner for Semantic ROI Graph FER — TensorFlow version."""

import tensorflow as tf
import numpy as np

from src.models import SemanticRoiGraphConfig, SemanticROIGraphFER


def random_bboxes(batch_size, num_regions):
    """Generate random bounding boxes for demo."""
    bboxes = np.zeros((batch_size, num_regions, 4), dtype=np.float32)
    for b in range(batch_size):
        for r in range(num_regions):
            x1 = np.random.randint(0, 30)
            y1 = np.random.randint(0, 30)
            x2 = x1 + np.random.randint(8, 18)
            y2 = y1 + np.random.randint(8, 18)
            bboxes[b, r] = [x1, y1, min(x2, 47), min(y2, 47)]
    return tf.constant(bboxes)


def main():
    config = SemanticRoiGraphConfig(
        num_classes=7, num_regions=9, roi_grid=4, feature_dim=256
    )
    model = SemanticROIGraphFER(config)

    # TF uses NHWC format: (B, H, W, C)
    images = tf.random.normal((2, 48, 48, 1))
    bboxes = random_bboxes(2, config.num_regions)

    outputs = model(images, bboxes, training=False)

    for key, value in outputs.items():
        if isinstance(value, tf.Tensor):
            print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
