import os
import wandb
import torch
import argparse
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger_wandb import init_wandb

from src.data.dataloader import build_dataloader
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

def resolve_data_path(data_path):
    required_files = {"train.csv", "val.csv", "test.csv"}
    if os.path.isdir(data_path):
        if required_files.issubset(set(os.listdir(data_path))):
            return data_path

        for current_dir, _, files in os.walk(data_path):
            if required_files.issubset(set(files)):
                print(f"--> Data path not exact; using discovered split folder: {current_dir}")
                return current_dir

    raise FileNotFoundError(
        f"Could not find train.csv, val.csv, test.csv under data_path: {data_path}"
    )

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

    # load_config() merges env.yaml into top-level keys: data_path, output_dir, root_path.
    path_cfg = config.get('paths', {})
    data_path = path_cfg.get('data_path', config.get('data_path', "dataset/fer13-split"))
    output_dir = path_cfg.get('output_dir', config.get('output_dir', "outputs"))
    root_path = config.get('root_path', ".")
    data_path = resolve_data_path(data_path)
       

    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    run_name = f"{config['model'].get('name', 'cnn')}_{timestamp}"

    # load data, loss, optim, model
    train_loader, val_loader, test_loader = build_dataloader(config=config, data_path=data_path)
    
    model = get_model(
        name=config['model']['name'],
        config=config)
    

    # ── Transfer Learning: load pretrained backbone weights ──
    pretrained_vgg = config['model'].get('pretrained_vgg_path', None)
    pretrained_resnet = config['model'].get('pretrained_resnet_path', None)
    
    if hasattr(model, 'load_pretrained_backbones'):
        if pretrained_vgg and pretrained_resnet:
            print("\n" + "="*50 + "\n[Transfer Learning] Loading dual pretrained backbones...\n" + "="*50)
            model.load_pretrained_backbones(pretrained_vgg, pretrained_resnet, device=device)
            model.freeze_backbones()
            print("="*50 + "\n")
        elif pretrained_resnet:
            print("\n" + "="*50 + "\n[Transfer Learning] Loading ResNet pretrained backbone...\n" + "="*50)
            model.load_pretrained_backbones(resnet_ckpt_path=pretrained_resnet, device=device)
            model.freeze_backbones()
            print("="*50 + "\n")
        elif pretrained_vgg:
            print("\n" + "="*50 + "\n[Transfer Learning] Loading VGG pretrained backbone...\n" + "="*50)
            model.load_pretrained_backbones(vgg_ckpt_path=pretrained_vgg, device=device)
            model.freeze_backbones()
            print("="*50 + "\n")


    # get class_distribution for class_weights (optional)
    use_class_weights = config['training'].get('use_class_weights', False)
    class_weights = None
    
    if use_class_weights:
        print("--> Using Class Weights to handle imbalance...")
        trainset_path = os.path.join(data_path, "train.csv")
        train_class_distribution = get_class_distribution(trainset_path)
        train_class_distribution_np = train_class_distribution.values
        class_weights = 1.0 / torch.tensor(train_class_distribution_np, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)

    loss = build_loss(config=config, class_weights=class_weights)
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)
    
    # set path to save ckpt
    path_save_ckpt = os.path.join(output_dir, f"checkpoints/{config['model'].get('name', 'cnn')}/{run_name}_best.pth")
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

    # evaluate
    print("\n" + "="*51)
    print("Evaluate in test set")
    print("="*51)
    
    # Get path of file best  
    load_checkpoints(model, optimizer, path_save_ckpt, device)
    
    eval_dir_path = os.path.join(output_dir, "figures")
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
