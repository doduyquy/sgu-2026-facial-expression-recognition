import wandb 
import os

def init_wandb(config, run_name=None):
    """Init wandb with config"""
    
    # get wandb api key from environment
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key: 
        wandb.login(key=wandb_api_key)

    wandb.init(
        project=config['logging'].get('project_name', "FER2013"), 
        entity=config['logging'].get('wandb_entity', 'phucga15062005'),
        name=run_name,
        config=config,
        resume="allow" 
    )

def log_metrics(metrics_dict, epoch=None):
    if wandb.run is not None:
        wandb.log(metrics_dict, step=epoch)

def log_image_to_wandb(tag, fig):
    """Log 10 true image and 10 wrong image"""
    if wandb.run is not None:
        wandb.log({tag: wandb.Image(fig)})

