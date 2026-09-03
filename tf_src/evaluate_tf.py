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
    parser.add_argument("--config", type=str, default="configs/semantic_roi_graph_tf.yaml")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"], help="Environment: local or kaggle")
    parser.add_argument("--weights", type=str, required=True, help="Path to .weights.h5 checkpoint")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        for candidate in [
            Path("configs") / cfg_path.name,
            Path("tf_src/configs") / cfg_path.name,
            Path("../configs") / cfg_path.name,
        ]:
            if candidate.exists():
                cfg_path = candidate
                break

    print(f"--> [Config] Loading: {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config.get("data", {}).get("data_dir", "dataset/fer13-split")
    masks_dir = args.masks_dir or config.get("data", {}).get("semantic_masks_dir", None)
    save_dir = args.save_dir or "outputs/evaluation_tf"

    if args.env == "kaggle":
        candidates_data = [
            Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split"),
            Path("/kaggle/input/datasets/doduyquynii/fer13-split"),
            Path("/kaggle/input/fer13-split/fer13-split"),
            Path("/kaggle/input/fer13-split"),
        ]
        if not Path(data_dir).exists():
            for c in candidates_data:
                if (c / "test.csv").exists():
                    data_dir = str(c)
                    break

        candidates_masks = [
            Path("/kaggle/input/datasets/pha1t2/maskfer2013/semantic_masks"),
            Path("/kaggle/input/maskfer2013/semantic_masks"),
            Path("/kaggle/input/semantic_masks"),
        ]
        if masks_dir is None or not Path(masks_dir).exists():
            for m in candidates_masks:
                if m.exists():
                    masks_dir = str(m)
                    break

        if args.save_dir is None:
            save_dir = "/kaggle/working/outputs/evaluation_tf"

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
    evaluate_test_set_tf(model, test_loader, save_dir=save_dir)


if __name__ == "__main__":
    main()
