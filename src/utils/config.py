# Load and merge YAML (file configs)
import yaml
import os
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).parent.parent.parent
CONFIG_DIR = os.path.join(ROOT_DIRECTORY, "configs")
# print(CONFIG_DIR)

def _deep_update(base_dict, update_dict):
    """Merge các nhánh trùng nhau, ghi đè F1 nếu có"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_config(model='simple_cnn', env='kaggle') -> dict:
    """Load file config và cả file base mà nó kế thừa
    Trong config, các path mà nó trả về sẽ là path tương đối,
    tùy vào môi trường chạy (kaggle/kaggle) mà path sẽ khác nhau

    Args:
        model (str): Tên file config (không có .yaml)
        env (str): Môi trường chạy (local/kaggle), tốt nhất: kaggle

    Returns:
        dict: config (gồm base) đã ghi đè (nếu có) và các config env tương ứng
    """
    model_config_path = os.path.join(CONFIG_DIR, f"{model}.yaml")
    env_config_path = os.path.join(CONFIG_DIR, "env.yaml")
    base_config_path = os.path.join(CONFIG_DIR, "base.yaml")

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Merge base config and model config
    config = _deep_update(base_config, model_config)

    if env == "local":
        env_config = env_config["local"]
    elif env == "kaggle":
        env_config = env_config["kaggle"]
    
    # Merge configs
    config = {**config, **env_config}
    return config


if __name__ == "__main__":
    config = load_config("vgg19", "kaggle")

    # print("Base:", config['_base_']['data'])

    # print config
    print(type(config))
    print("Batch size after merger:", config['data']['batch_size'])
    print("="*50)
    print(config)

    """
    <class 'dict'>
    Batch size after merger: 32
    ==================================================
    {'data': {'name': 'fer13-split', 'num_classes': 7, 'image_size': 224, 'batch_size': 32, 'num_workers': 2}, 'seed': {'random_seed': 42}, 'model': {'name': 'vgg19', 'pretrained': True}, 'training': {'epochs': 30, 'lr': 0.0001, 'optimizer': 'adam', 'scheduler': 'reduce_lr_on_plateau', 'weight_decay': 0.0001}, 'logging': {'use_wandb': True, 'project_name': 'fer2013-sgu2026'}, 'env': {'platform': 'kaggle'}, '_base_': 'base.yaml', 'data_path': '/kaggle/input/datasets/doduyquynii/', 'output_dir': '/kaggle/working/outputs', 'num_workers': 2}

    """
