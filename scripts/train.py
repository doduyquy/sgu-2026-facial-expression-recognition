import os
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
        data_path = config['kaggle'].get('data_path', "/kaggle/input/datasets/doduyquynii/fer13-split")
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
    
    loss = build_loss(config=config)
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


    evaluate_and_show(model, test_loader, device, eval_dir_path)
    
    print("\n\t\tDONE!\n")

    

if __name__ == "__main__":
    main()
