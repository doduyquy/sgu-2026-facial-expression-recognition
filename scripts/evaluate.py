import os
import torch
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Use device:", device)

    config = load_config(args.config, args.env)

    if config['env']['platform'] == 'kaggle':
        data_path = config['kaggle'].get('data_path', "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
        root_path = config['kaggle'].get('root_path', "/kaggle/working/sgu-2026-facial-expression-recognition/")
    else: 
        data_path = config['local'].get('data_path', "../dataset")
        root_path = config['local'].get('root_path', "../")

    _, _, test_loader = build_dataloader(config=config, data_path=data_path)

    model = get_model(name=config['model']['name'], config=config)
    model.to(device)

    print(f"--> Loading ckpt from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    # Checkpoint structure might have 'model_state_dict' or just be the state dict directly
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    eval_dir_path = os.path.join(root_path, "outputs/figures")
    os.makedirs(eval_dir_path, exist_ok=True)
    
    testset_path = os.path.join(data_path, "test.csv")
    evaluate_and_show(model, test_loader, testset_path, device, eval_dir_path, use_tta=False)
    print("Evaluation completed!")

if __name__ == "__main__":
    main()