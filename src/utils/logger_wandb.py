import wandb 
import os

def init_wandb(config, run_name=None):
    """Init wandb with config"""
    
    # get wandb api key from environment
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key: 
        wandb.login(key=wandb_api_key)

    logging_cfg = config.get('logging', {})
    project_name = logging_cfg.get(
        'project_name',
        logging_cfg.get('wandb_project', "FER2013"),
    )

    wandb.init(
        project=project_name,
        entity=logging_cfg.get('wandb_entity', 'phucga15062005'),
        name=run_name,
        config=config,
        resume="allow" 
    )
    wandb.define_metric("Epoch")
    wandb.define_metric("Train/*", step_metric="Epoch")
    wandb.define_metric("Val/*", step_metric="Epoch")
    wandb.define_metric("Best/*", step_metric="Epoch")
    wandb.define_metric("Learning_Rate*", step_metric="Epoch")
    wandb.define_metric("Scheduler/*", step_metric="Epoch")
    wandb.define_metric("EarlyStopping/*", step_metric="Epoch")
    wandb.define_metric("Checkpoint/*", step_metric="Epoch")
    wandb.define_metric("Training/*", step_metric="Epoch")

def log_metrics(metrics_dict, epoch=None):
    if wandb.run is not None:
        wandb.log(metrics_dict, step=epoch)

def log_image_to_wandb(tag, fig):
    """Log 10 true image and 10 wrong image"""
    if wandb.run is not None:
        wandb.log({tag: wandb.Image(fig)})

def save_model_to_wandb(model_path, model_name="cnn"):
    """Lưu file pth trực tiếp vào Artifacts"""
    if wandb.run is not None:
        try:
            # Gói file -> Artifacts (Đánh dấu ID để phân biệt các lần chạy khác nhau)
            artifact = wandb.Artifact(name=f"{model_name}_{wandb.run.id}", type="model")
            artifact.add_file(model_path) 
            
            # Push Server WandB
            wandb.log_artifact(artifact)
            print(f"\t--> [WandB] Send File `{os.path.basename(model_path)}` to cloud successfully!")
        except Exception as e:
            print(f"\t-!- [WandB] Error when upload Model: {e}")


