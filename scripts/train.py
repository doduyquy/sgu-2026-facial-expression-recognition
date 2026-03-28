import argparse
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_wandb
from src.data.dataloader import build_dataloaders
from src.models import get_model
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    args = parser.parse_args()
    
    config = load_config(args.config, args.env)
    set_seed(config.seed)
    setup_wandb(config)
    
    train_loader, val_loader, test_loader = build_dataloaders(config)
    model = get_model(config.model.name, num_classes=config.data.num_classes, 
                      pretrained=config.model.pretrained)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()

if __name__ == "__main__":
    main()
