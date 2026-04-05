import os
import copy
import wandb
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger_wandb import init_wandb

from src.data.dataloader import build_dataloader
from src.data.dataset import FER2013
from src.data.transforms import build_transform
from src.models import get_model # in __init__ gfile
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.optimizer import build_scheduler
from src.utils.checkpoint import load_checkpoints
from src.evaluation.evaluator import evaluate_and_show
from src.utils.logger_wandb import save_model_to_wandb
from src.utils.data_stats import get_class_distribution # testing: class weight

from datetime import datetime
#-------------------------------------------------------------


def _collect_misclassified_indices(model, data_path, config, device):
    eval_trans = build_transform(config, split="val")
    eval_dataset = FER2013(data_path=data_path, split="train", transforms=eval_trans)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 2),
        shuffle=False,
        pin_memory=True,
    )

    model.eval()
    wrong_indices = []
    base_idx = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            wrong_mask = preds != labels

            local_wrong = torch.nonzero(wrong_mask, as_tuple=False).squeeze(1).tolist()
            if isinstance(local_wrong, int):
                local_wrong = [local_wrong]

            wrong_indices.extend([base_idx + i for i in local_wrong])
            base_idx += labels.size(0)

    return wrong_indices


def _build_hard_train_loader(data_path, config, hard_indices):
    train_trans = build_transform(config, split="train")
    full_train_dataset = FER2013(data_path=data_path, split="train", transforms=train_trans)
    hard_dataset = Subset(full_train_dataset, hard_indices)
    hard_batch_size = config['training'].get('hard_mining_batch_size', config['data']['batch_size'])

    hard_loader = DataLoader(
        hard_dataset,
        batch_size=hard_batch_size,
        num_workers=config['data'].get('num_workers', 2),
        shuffle=True,
        pin_memory=True,
    )
    return hard_loader

def main():
    print("\t\t--> In main <--\t\t")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print("--- Use device:", device)

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    args = parser.parse_args()
    
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
    

    # get class_distribution for class_weights (for testing)
    trainset_path = os.path.join(data_path, "train.csv")
    train_class_distribution = get_class_distribution(trainset_path)
    train_class_distribution_np = train_class_distribution.values
    class_counts = torch.tensor(train_class_distribution_np, dtype=torch.float)

    class_weight_mode = config['training'].get('class_weight_mode', 'inverse')
    use_class_weights = config['training'].get('use_class_weights', True)

    class_weights = None
    if use_class_weights:
        if class_weight_mode == 'sqrt_inverse':
            class_weights = 1.0 / torch.sqrt(class_counts)
        elif class_weight_mode == 'inverse':
            class_weights = 1.0 / class_counts
        else:
            raise ValueError(f"Unsupported class_weight_mode: {class_weight_mode}")

        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
        print(f"--- Class weight mode: {class_weight_mode}")
    else:
        print("--- Class weights disabled")


    loss = build_loss(config=config, class_weights=class_weights)
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)
    
    # set path to save ckpt
    path_save_ckpt = os.path.join(root_path, f"outputs/checkpoints/{config['model'].get('name', 'cnn')}/{run_name}_best.pth")
    os.makedirs(os.path.dirname(path_save_ckpt), exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        run_name=run_name,
        save_dir=path_save_ckpt
    )
    train_losses, val_losses = trainer.fit()
    final_optimizer = optimizer

    # optional phase-2 hard example mining (misclassification replay)
    hard_cfg_enabled = config['training'].get('hard_mining_enabled', False)
    if hard_cfg_enabled:
        print("\n" + "="*51)
        print("Hard Mining Phase (phase-2)")
        print("="*51)

        load_checkpoints(model, optimizer, path_save_ckpt, device)
        wrong_indices = _collect_misclassified_indices(model, data_path, config, device)

        max_samples = config['training'].get('hard_mining_max_samples', 0)
        if max_samples and len(wrong_indices) > max_samples:
            wrong_indices = wrong_indices[:max_samples]

        print(f"--- Hard samples found: {len(wrong_indices)}")

        if len(wrong_indices) > 0:
            hard_loader = _build_hard_train_loader(data_path, config, wrong_indices)

            phase2_config = copy.deepcopy(config)
            phase2_config['training']['epochs'] = config['training'].get('hard_mining_epochs', 8)
            phase2_config['training']['patience'] = config['training'].get('hard_mining_patience', 3)
            phase2_config['training']['lr'] = config['training']['lr'] * config['training'].get('hard_mining_lr_scale', 0.2)
            phase2_config['logging']['use_wandb'] = False

            phase2_optimizer = build_optimizer(model=model, config=phase2_config)
            phase2_scheduler = build_scheduler(optimizer=phase2_optimizer, config=phase2_config)

            phase2_trainer = Trainer(
                model=model,
                train_loader=hard_loader,
                val_loader=val_loader,
                criterion=loss,
                optimizer=phase2_optimizer,
                scheduler=phase2_scheduler,
                config=phase2_config,
                device=device,
                run_name=f"{run_name}_hardmining",
                save_dir=path_save_ckpt,
            )
            phase2_trainer.fit()
            load_checkpoints(model, phase2_optimizer, path_save_ckpt, device)
            final_optimizer = phase2_optimizer

    # evaluate
    print("\n" + "="*51)
    print("Evaluate in test set")
    print("="*51)
    
    # Get path of file best  
    load_checkpoints(model, final_optimizer, path_save_ckpt, device)
    
    eval_dir_path = os.path.join(root_path ,"outputs/figures")
    os.makedirs(eval_dir_path, exist_ok=True)
    print(f"Evaluatoin save path: {eval_dir_path}")


    # test data path
    testset_path = os.path.join(data_path, "test.csv")
    evaluate_and_show(model, test_loader, testset_path, device, eval_dir_path)
    
    # upload best ckpt to wandb
    if config['logging'].get('use_wandb', True):
        print("\n\t--> Uploading best ckpt to WandB, please wait...")
        save_model_to_wandb(path_save_ckpt)
        
        # Đóng cửa sổ WandB, tránh bị kẹt quá trình upload trên hệ thống ngầm của Kaggle
        wandb.finish()

    print("\n\t\tDONE!\n")

    

if __name__ == "__main__":
    main()
