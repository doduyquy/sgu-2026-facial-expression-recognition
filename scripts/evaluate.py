import os
import tensorflow as tf
import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_config
from src.data.dataloader import build_dataloader
from src.models import get_model
from src.evaluation.evaluator import evaluate_and_show

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    args = parser.parse_args()

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"--- Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("--- No GPU found, using CPU")

    config = load_config(args.config, args.env)

    if config['env']['platform'] == 'kaggle':
        data_path = config['kaggle'].get('data_path', "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
        root_path = config['kaggle'].get('root_path', "/kaggle/working/sgu-2026-facial-expression-recognition/")
    else: 
        data_path = config['local'].get('data_path', "../dataset")
        root_path = config['local'].get('root_path', "../")

    _, _, test_loader = build_dataloader(config=config, data_path=data_path)

    model = get_model(name=config['model']['name'], config=config)

    print(f"--> Loading ckpt from {args.ckpt}")
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.ckpt).expect_partial()
    print("[OK] Checkpoint loaded")

    eval_dir_path = os.path.join(root_path, "outputs/figures")
    os.makedirs(eval_dir_path, exist_ok=True)
    
    testset_path = os.path.join(data_path, "test.csv")
    evaluate_and_show(model, test_loader, testset_path, eval_dir_path, use_tta=True)
    print("Evaluation completed!")

if __name__ == "__main__":
    main()