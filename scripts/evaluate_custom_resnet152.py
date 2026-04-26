import os
import sys
# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from src.utils.config import load_config
from src.data.dataloader import build_dataloader
from src.models import get_model
from src.evaluation.evaluator import evaluate_and_show

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="resnet152_eval")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    args = parser.parse_args()

    # Load config and override model/data settings
    config = load_config(args.config, args.env)
    config['model']['name'] = 'resnet152'
    config['data']['channels'] = 3 # The checkpoint is 3 channels
    config['data']['image_size'] = 224
    config['data']['normalize'] = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # Get data path from merged config
    data_path = config.get('data_path', "./dataset/fer13-split")
    
    # Nếu đang chạy local, đảm bảo đường dẫn tuyệt đối để tránh lỗi
    if args.env == 'local':
        data_path = os.path.abspath(data_path)
    
    print(f"--> Data path: {data_path}")
    
    _, _, test_loader = build_dataloader(config, data_path)

    # Initialize model
    model = get_model(name='resnet152', config=config)
    model.to(device)

    # Load checkpoint
    if hasattr(model, 'load_from_checkpoint'):
        model.load_from_checkpoint(args.ckpt, device)
    else:
        # Fallback to standard loading if method doesn't exist
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt['net'] if 'net' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    # Evaluate
    print("\n" + "="*50)
    print("Evaluating ResNet152 on Test Set...")
    print("="*50)

    eval_dir_path = "outputs/evaluation_resnet152"
    os.makedirs(eval_dir_path, exist_ok=True)
    
    testset_path = os.path.join(data_path, "test.csv")
    evaluate_and_show(model, test_loader, testset_path, device, eval_dir_path)

    print(f"\nDone! Evaluation results saved to: {eval_dir_path}")

if __name__ == "__main__":
    main()
