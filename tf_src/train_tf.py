"""
train_tf.py — Main TensorFlow training entry point.
Usage:
    python tf_src/train_tf.py --config tf_src/configs/semantic_roi_graph_tf.yaml
"""

import argparse
from pathlib import Path
import yaml
import tensorflow as tf

from tf_src.data.dataset_tf import create_tf_dataloader
from tf_src.models.semantic_roi_graph_tf import SemanticROIGraphFERTF
from tf_src.training.optimizer_tf import build_optimizer_tf
from tf_src.training.trainer_tf import TrainerTF
from tf_src.evaluation.evaluator_tf import evaluate_test_set_tf


def main():
    parser = argparse.ArgumentParser(description="Train Semantic ROI Graph FER in TensorFlow")
    parser.add_argument("--config", type=str, default="tf_src/configs/semantic_roi_graph_tf.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default=None, help="Override FER2013 data split folder")
    parser.add_argument("--masks_dir", type=str, default=None, help="Override semantic masks folder")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config.get("data", {}).get("data_dir", "dataset/fer13-split")
    masks_dir = args.masks_dir or config.get("data", {}).get("semantic_masks_dir", None)
    batch_size = int(config.get("data", {}).get("batch_size", 64))

    print(f"--> Initializing TensorFlow Data Loaders (batch_size={batch_size})...")
    train_loader = create_tf_dataloader(
        data_path=data_dir,
        split="train",
        batch_size=batch_size,
        semantic_masks_dir=masks_dir,
        is_training=True,
        shuffle=True,
    )
    val_loader = create_tf_dataloader(
        data_path=data_dir,
        split="val",
        batch_size=batch_size,
        semantic_masks_dir=masks_dir,
        is_training=False,
        shuffle=False,
    )
    test_loader = create_tf_dataloader(
        data_path=data_dir,
        split="test",
        batch_size=batch_size,
        semantic_masks_dir=masks_dir,
        is_training=False,
        shuffle=False,
    )

    # 28709 training samples // 64 ≈ 448 steps per epoch
    steps_per_epoch = 28709 // batch_size

    print("--> Initializing SemanticROIGraphFERTF model...")
    model = SemanticROIGraphFERTF(config=config)

    print("--> Building AdamW optimizer & CosineDecayRestarts scheduler...")
    optimizer = build_optimizer_tf(config, steps_per_epoch=steps_per_epoch)

    trainer = TrainerTF(
        model=model,
        train_dataset=train_loader,
        val_dataset=val_loader,
        optimizer=optimizer,
        config=config,
    )

    # Run training
    best_weights = trainer.fit()

    # Final test evaluation with 2-View Horizontal Flip TTA
    print("\n===================================================")
    print("Evaluate on FER2013 Test Set with Horizontal Flip TTA")
    print("===================================================")
    evaluate_test_set_tf(model, test_loader, weights_path=best_weights)


if __name__ == "__main__":
    main()
