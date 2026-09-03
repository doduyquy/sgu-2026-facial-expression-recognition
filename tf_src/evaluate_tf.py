"""
evaluate_tf.py — CLI tool to evaluate a trained TensorFlow model checkpoint on FER2013 test set.
Usage:
    python tf_src/evaluate_tf.py --config tf_src/configs/semantic_roi_graph_tf.yaml --weights outputs/checkpoints_tf/semantic_roi_graph_fer_tf_best.weights.h5
"""

import argparse
import yaml
from pathlib import Path

from tf_src.data.dataset_tf import create_tf_dataloader
from tf_src.models.semantic_roi_graph_tf import SemanticROIGraphFERTF
from tf_src.evaluation.evaluator_tf import evaluate_test_set_tf


def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic ROI Graph FER in TensorFlow")
    parser.add_argument("--config", type=str, default="tf_src/configs/semantic_roi_graph_tf.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to .weights.h5 checkpoint")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/evaluation_tf")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config.get("data", {}).get("data_dir", "dataset/fer13-split")
    masks_dir = args.masks_dir or config.get("data", {}).get("semantic_masks_dir", None)
    batch_size = int(config.get("data", {}).get("batch_size", 64))

    print(f"--> Initializing TensorFlow Test Data Loader from {data_dir}...")
    test_loader = create_tf_dataloader(
        data_path=data_dir,
        split="test",
        batch_size=batch_size,
        semantic_masks_dir=masks_dir,
        is_training=False,
        shuffle=False,
    )

    print("--> Initializing model architecture...")
    model = SemanticROIGraphFERTF(config=config)

    # Build model variables with a dummy forward pass before loading weights
    import tensorflow as tf
    dummy_img = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    dummy_box = tf.zeros([1, 9, 4], dtype=tf.float32)
    _ = model(dummy_img, dummy_box, training=False)

    print(f"--> Loading weights from {args.weights}...")
    model.load_weights(args.weights)

    print("--> Running test evaluation with Horizontal Flip TTA...")
    evaluate_test_set_tf(model, test_loader, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
