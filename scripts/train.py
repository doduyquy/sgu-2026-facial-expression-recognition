import os
import wandb
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path so `src` imports work when running scripts
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger_wandb import init_wandb

from src.data.dataloader import build_dataloader
from src.models import get_model
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer, ReduceLROnPlateau
from src.utils.checkpoint import load_checkpoints
from src.evaluation.evaluator import evaluate_and_show
from src.utils.logger_wandb import save_model_to_wandb
from src.utils.data_stats import get_class_distribution

from datetime import datetime
#-------------------------------------------------------------

def main():
    print("\t\t--> In main <--\t\t")

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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
    
    # load config
    config = load_config(args.config, args.env)
    set_seed(config['seed'].get('random_seed', 21))

    # data path and root path for each platform
    if config['env']['platform'] == 'kaggle':
        data_path = config['kaggle'].get('data_path', "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
        root_path = config['kaggle'].get('root_path', "/kaggle/working/sgu-2026-facial-expression-recognition/")
    else: 
        data_path = config['local'].get('data_path', "../dataset")
        root_path = config['local'].get('root_path', "../")
       

    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    run_name = f"{config['model'].get('name', 'cnn')}_{timestamp}"

    # load data, loss, optim, model
    train_loader, val_loader, test_loader = build_dataloader(config=config, data_path=data_path)
    
    model = get_model(
        name=config['model']['name'],
        config=config)

    # get class_distribution for class_weights
    trainset_path = os.path.join(data_path, "train.csv")
    train_class_distribution = get_class_distribution(trainset_path)
    train_class_distribution_np = train_class_distribution.values
    class_counts = tf.constant(train_class_distribution_np, dtype=tf.float32)

    class_weight_mode = config['training'].get('class_weight_mode', 'inverse')
    use_class_weights = config['training'].get('use_class_weights', True)

    class_weights = None
    if use_class_weights:
        if class_weight_mode == 'manual':
            manual_weights = config['training'].get('manual_class_weights', [1.2, 2.0, 1.5, 0.8, 0.8, 1.0, 1.0])
            class_weights = tf.constant(manual_weights, dtype=tf.float32)
        elif class_weight_mode == 'sqrt_inverse':
            class_weights = 1.0 / tf.sqrt(class_counts)
        elif class_weight_mode == 'inverse':
            class_weights = 1.0 / class_counts
        else:
            raise ValueError(f"Unsupported class_weight_mode: {class_weight_mode}")

        class_weights = class_weights / tf.reduce_sum(class_weights)
        print(f"--- Class weight mode: {class_weight_mode}")
        print(f"--- Class weights: {class_weights.numpy()}")
    else:
        print("--- Class weights disabled")

    optimizer = build_optimizer(model=model, config=config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    
    # set path to save ckpt
    path_save_ckpt = os.path.join(root_path, f"outputs/checkpoints/{config['model'].get('name', 'cnn')}/{run_name}_best")
    os.makedirs(os.path.dirname(path_save_ckpt), exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataset=train_loader,
        val_dataset=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        run_name=run_name,
        save_path=path_save_ckpt,
        class_weights=class_weights
    )
    train_losses, val_losses = trainer.fit()

    # evaluate
    print("\n" + "="*51)
    print("Evaluate in test set")
    print("="*51)
    
    # Load best checkpoint
    load_checkpoints(model, optimizer, path_save_ckpt)
    
    eval_dir_path = os.path.join(root_path, "outputs/figures")
    os.makedirs(eval_dir_path, exist_ok=True)
    print(f"Evaluation save path: {eval_dir_path}")

    # test data path
    testset_path = os.path.join(data_path, "test.csv")
    evaluate_and_show(model, test_loader, testset_path, eval_dir_path, use_tta=True)
    
    # upload best ckpt to wandb
    if config['logging'].get('use_wandb', True):
        print("\n\t--> Uploading best ckpt to WandB, please wait...")
        save_model_to_wandb(path_save_ckpt)
        wandb.finish()

    print("\n\t\tDONE!\n")

    

if __name__ == "__main__":
    main()