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


def _config_path(config_name):
    path = Path(config_name)
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    if not path.is_absolute():
        path = Path(CONFIG_DIR) / path
    return path


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_config_with_base(config_name, seen=None):
    """Load YAML recursively, honoring `_base_` inheritance."""
    seen = seen or set()
    config_path = _config_path(config_name)
    resolved_path = config_path.resolve()

    if resolved_path in seen:
        raise ValueError(f"Circular config inheritance detected at {config_path}")

    config = _load_yaml(config_path)
    base_name = config.get("_base_")
    if base_name is None and config_path.name != "base.yaml":
        base_name = "base.yaml"

    if base_name:
        base_config = _load_config_with_base(base_name, seen | {resolved_path})
        return _deep_update(base_config, config)

    return config


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
    env_config_path = os.path.join(CONFIG_DIR, "env.yaml")

    config = _load_config_with_base(model)
    env_config = _load_yaml(env_config_path)

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
