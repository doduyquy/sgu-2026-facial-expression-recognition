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
    

    # ── Transfer Learning: load pretrained backbone weights ──
    pretrained_vgg = config['model'].get('pretrained_vgg_path', None)
    pretrained_resnet = config['model'].get('pretrained_resnet_path', None)
    pretrained_backbone = config['model'].get('pretrained_backbone_path', None)
    
    # Case 1: Dual backbone (VGG + ResNet) — RegionAlignedFER, DualFusion
    if pretrained_vgg and pretrained_resnet and hasattr(model, 'load_pretrained_backbones'):
        print("\n" + "="*50)
        print("[Transfer Learning] Loading pretrained backbones...")
        print("="*50)
        model.load_pretrained_backbones(pretrained_vgg, pretrained_resnet, device=device)
        model.freeze_backbones()
        print("="*50 + "\n")

    # Case 2: Single backbone (ResNet35) — CNNDictionary
    elif pretrained_backbone and hasattr(model, 'load_pretrained_backbone'):
        print("\n" + "="*50)
        print("[Transfer Learning] Loading pretrained backbone...")
        print("="*50)
        model.load_pretrained_backbone(pretrained_backbone, device=device)
        model.freeze_backbone()
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

    # evaluate
    print("\n" + "="*51)
    print("Evaluate in test set")
    print("="*51)
    
    # Get path of file best  
    load_checkpoints(model, optimizer, path_save_ckpt, device)
    
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
