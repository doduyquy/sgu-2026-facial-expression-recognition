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
    parser.add_argument("--config", type=str, default="configs/semantic_roi_graph_tf.yaml", help="Path to config YAML")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"], help="Environment: local or kaggle")
    parser.add_argument("--data_dir", type=str, default=None, help="Override FER2013 data split folder")
    parser.add_argument("--masks_dir", type=str, default=None, help="Override semantic masks folder")
    parser.add_argument("--save_dir", type=str, default=None, help="Override checkpoint save directory")
    args, _ = parser.parse_known_args()

    # Enable GPU memory growth if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"--> [GPU] Found {len(gpus)} GPU(s), memory growth enabled.")
        except RuntimeError as e:
            print(f"--> [GPU] Memory growth notice: {e}")

    # Smart config path resolution (handles with or without .yaml, relative and absolute paths)
    def _find_config(raw_str: str) -> Path:
        p = Path(raw_str)
        stem = p.stem if p.suffix in [".yaml", ".yml"] else p.name
        candidates = [
            p,
            p.with_suffix(".yaml"),
            Path("configs") / f"{stem}.yaml",
            Path("configs") / stem,
            Path("tf_src/configs") / f"{stem}.yaml",
            Path("../configs") / f"{stem}.yaml",
            Path("/kaggle/working/sgu-2026-facial-expression-recognition/configs") / f"{stem}.yaml",
            Path("/kaggle/working/configs") / f"{stem}.yaml",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        return p.with_suffix(".yaml")

    cfg_path = _find_config(args.config)
    print(f"--> [Config] Loading: {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config.get("data", {}).get("data_dir", "dataset/fer13-split")
    masks_dir = args.masks_dir or config.get("data", {}).get("semantic_masks_dir", None)
    save_dir = args.save_dir or "outputs/checkpoints_tf"

    if args.env == "kaggle":
        # Resolve Kaggle dataset paths automatically if default path does not exist
        candidates_data = [
            Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split"),
            Path("/kaggle/input/datasets/doduyquynii/fer13-split"),
            Path("/kaggle/input/fer13-split/fer13-split"),
            Path("/kaggle/input/fer13-split"),
        ]
        if not Path(data_dir).exists():
            for c in candidates_data:
                if (c / "train.csv").exists():
                    data_dir = str(c)
                    print(f"--> [Kaggle] Auto-detected FER2013 data path: {data_dir}")
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
                    print(f"--> [Kaggle] Auto-detected semantic masks path: {masks_dir}")
                    break

        if args.save_dir is None:
            save_dir = "/kaggle/working/outputs/checkpoints_tf"

    batch_size = int(config.get("data", {}).get("batch_size", 64))

    print(f"--> Initializing TensorFlow Data Loaders (batch_size={batch_size})...")
    print(f"    Data Dir : {data_dir}")
    print(f"    Masks Dir: {masks_dir}")
    print(f"    Save Dir : {save_dir}")
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
        save_dir=save_dir,
    )

    # Run training
    best_weights = trainer.fit()

    # Final test evaluation with 2-View Horizontal Flip TTA
    print("\n===================================================")
    print("Evaluate on FER2013 Test Set with Horizontal Flip TTA")
    print("===================================================")
    evaluate_test_set_tf(model, test_loader, weights_path=best_weights, save_dir=save_dir)


if __name__ == "__main__":
    main()
