import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wandb
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger_wandb import init_wandb

from src.data.dataloader import build_dataloader, build_landmark_dataloader
from src.models import get_model # in __init__ gfile
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.optimizer import build_scheduler
from src.utils.checkpoint import load_checkpoints
from src.evaluation.evaluator import evaluate_and_show
from src.utils.logger_wandb import save_model_to_wandb
from src.utils.data_stats import get_class_distribution # testing: class weight

from datetime import datetime, timedelta
#-------------------------------------------------------------

def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("DDP needs CUDA GPUs. Run without torchrun for CPU/single process.")

    cuda_count = torch.cuda.device_count()
    if local_rank >= cuda_count:
        raise RuntimeError(
            f"DDP launched local_rank={local_rank}, but torch sees only {cuda_count} CUDA device(s). "
            "On Kaggle, switch Accelerator to GPU T4 x2 or set --nproc_per_node to the number of visible GPUs."
        )

    torch.cuda.set_device(local_rank)
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    if rank == 0:
        print(f"--- DDP backend: {backend}")
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

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

@record
def main():
    distributed, rank, world_size, local_rank = setup_distributed()
    try:
        if is_main_process():
            print("\t\t--> In main <--\t\t")

        # device
        device = torch.device(f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu"))
        if is_main_process():
            print("--- Use device:", device)
            if distributed:
                print(f"--- DDP enabled: world_size={world_size}")

        # get args 
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
        args = parser.parse_args()
        
        # load config
        config = load_config(args.config, args.env)
        set_seed(config['seed'].get('random_seed', 21) + rank)
        ddp_cfg = config.get('ddp', {})

        # load_config() merges env.yaml into top-level keys: data_path, output_dir, root_path.
        path_cfg = config.get('paths', {})
        data_path = path_cfg.get('data_path', config.get('data_path', "dataset/fer13-split"))
        output_dir = path_cfg.get('output_dir', config.get('output_dir', "outputs"))
        root_path = config.get('root_path', ".")
        data_path = resolve_data_path(data_path)
           

        timestamp = datetime.now().strftime("%d%m%Y_%H%M")
        run_name = f"{config['model'].get('name', 'cnn')}_{timestamp}"

        # load data, loss, optim, model
        dataloader_builder = (
            build_landmark_dataloader
            if config['model'].get('name') == 'resnet152_landmark_attention'
            else build_dataloader
        )
        if is_main_process() and dataloader_builder is build_landmark_dataloader:
            print("--> Using landmark dataloader for landmark-guided attention.")

        train_loader, val_loader, test_loader = dataloader_builder(
            config=config,
            data_path=data_path,
            distributed=distributed,
            world_size=world_size,
        )
        
        model = get_model(
            name=config['model']['name'],
            config=config)
        

        # ── Transfer Learning: load pretrained backbone weights ──
        pretrained_vgg = config['model'].get('pretrained_vgg_path', None)
        pretrained_resnet = config['model'].get('pretrained_resnet_path', None)
        
        if hasattr(model, 'load_pretrained_backbones'):
            if pretrained_vgg and pretrained_resnet:
                if is_main_process():
                    print("\n" + "="*50 + "\n[Transfer Learning] Loading dual pretrained backbones...\n" + "="*50)
                model.load_pretrained_backbones(pretrained_vgg, pretrained_resnet, device=device)
                model.freeze_backbones()
                if is_main_process():
                    print("="*50 + "\n")
            elif pretrained_resnet:
                if is_main_process():
                    print("\n" + "="*50 + "\n[Transfer Learning] Loading ResNet pretrained backbone...\n" + "="*50)
                model.load_pretrained_backbones(resnet_ckpt_path=pretrained_resnet, device=device)
                model.freeze_backbones()
                if is_main_process():
                    print("="*50 + "\n")
            elif pretrained_vgg:
                if is_main_process():
                    print("\n" + "="*50 + "\n[Transfer Learning] Loading VGG pretrained backbone...\n" + "="*50)
                model.load_pretrained_backbones(vgg_ckpt_path=pretrained_vgg, device=device)
                model.freeze_backbones()
                if is_main_process():
                    print("="*50 + "\n")


        # get class_distribution for class_weights (optional)
        use_class_weights = config['training'].get('use_class_weights', False)
        class_weights = None
        
        if use_class_weights:
            if is_main_process():
                print("--> Using Class Weights to handle imbalance...")
            trainset_path = os.path.join(data_path, "train.csv")
            train_class_distribution = get_class_distribution(trainset_path)
            train_class_distribution_np = train_class_distribution.values
            class_weights = 1.0 / torch.tensor(train_class_distribution_np, dtype=torch.float)
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(device)

        model = model.to(device)
        if distributed:
            if ddp_cfg.get('sync_batchnorm', False):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=ddp_cfg.get('find_unused_parameters', False),
                broadcast_buffers=ddp_cfg.get('broadcast_buffers', True),
            )

        loss = build_loss(config=config, class_weights=class_weights)
        optimizer = build_optimizer(model=model, config=config)
        scheduler = build_scheduler(optimizer=optimizer, config=config)
        
        # set path to save ckpt
        path_save_ckpt = os.path.join(output_dir, f"checkpoints/{config['model'].get('name', 'cnn')}/{run_name}_best.pth")
        if is_main_process():
            os.makedirs(os.path.dirname(path_save_ckpt), exist_ok=True)
        if distributed:
            dist.barrier()

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
        if distributed:
            dist.barrier()

        if not is_main_process():
            return

        print("\n" + "="*51)
        print("Evaluate in test set")
        print("="*51)
        
        eval_model = model.module if hasattr(model, "module") else model

        # Get path of file best  
        load_checkpoints(eval_model, optimizer, path_save_ckpt, device)
        
        eval_dir_path = os.path.join(output_dir, "figures")
        os.makedirs(eval_dir_path, exist_ok=True)
        print(f"Evaluatoin save path: {eval_dir_path}")


        # test data path
        testset_path = os.path.join(data_path, "test.csv")
        evaluate_and_show(eval_model, test_loader, testset_path, device, eval_dir_path)
        
        # upload best ckpt to wandb
        if config['logging'].get('use_wandb', True):
            print("\n\t--> Uploading best ckpt to WandB, please wait...")
            save_model_to_wandb(path_save_ckpt)
            
            # Đóng cửa sổ WandB, tránh bị kẹt quá trình upload trên hệ thống ngầm của Kaggle
            wandb.finish()

        print("\n\t\tDONE!\n")
    finally:
        cleanup_distributed()

    

if __name__ == "__main__":
    main()
