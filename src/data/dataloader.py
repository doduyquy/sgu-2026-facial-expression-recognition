import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import FER2013
from .transforms import build_transform


def _build_weighted_sampler(data_train, num_classes, mode="inverse"):
    labels = data_train.data.iloc[:, 0].astype(int).values
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    class_counts[class_counts == 0] = 1.0
    class_counts_tensor = torch.tensor(class_counts, dtype=torch.float)

    if mode == "sqrt_inverse":
        class_weights = 1.0 / torch.sqrt(class_counts_tensor)
    else:
        class_weights = 1.0 / class_counts_tensor

    sample_weights = class_weights[torch.tensor(labels, dtype=torch.long)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

def build_dataloader(config, data_path):
    """ Dataloader: Group dataset into batch (mini-batch) 
    Args: 
        config: config for data, dataloader (Q cho ca config goc)
        data_path: path to fer13-split dir
    Return: 
        train_loader, val_loader, test_loader
    """
    # transform
    trans_train = build_transform(config, "train")
    trans_val = build_transform(config, "val")
    trans_test = build_transform(config, "test")

    # build dataset
    data_train = FER2013(data_path=data_path, split="train", transforms=trans_train)
    data_val = FER2013(data_path=data_path, split="val", transforms=trans_val)
    data_test = FER2013(data_path=data_path, split="test", transforms=trans_test)

    use_weighted_sampler = config['training'].get('use_weighted_sampler', False)
    sampler_weight_mode = config['training'].get('sampler_weight_mode', 'inverse')
    train_sampler = None

    if use_weighted_sampler:
        train_sampler = _build_weighted_sampler(
            data_train=data_train,
            num_classes=config['data'].get('num_classes', 7),
            mode=sampler_weight_mode,
        )
        print(f"--- Train sampler: weighted ({sampler_weight_mode})")

    # batch the dataset
    train_loader = DataLoader(
        data_train, 
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True, # push data to cache (-> send to GPU)
        shuffle=(train_sampler is None),
        sampler=train_sampler)
    val_loader = DataLoader(
        data_val, 
        batch_size=config['data']['batch_size'], 
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True, # push data to cache (-> send to GPU)
        shuffle=False)
    test_loader = DataLoader(
        data_test, 
        batch_size=config['data']['batch_size'], 
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True, # push data to cache (-> send to GPU)
        shuffle=False)
    
    return train_loader, val_loader, test_loader




if __name__ == "__main__":

    import os, sys
    # go back to root directory
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 
    from src.utils.config import load_config

    config = load_config(model='vgg19', env='kaggle')
    

    data_path = "./dataset/fer13-split"

    print("Create dataloader for train | val | test ...")
    train_loader, val_loader, test_loader = build_dataloader(config, data_path)

    # test 1: get batch from train_loader
    images, labels = next(iter(train_loader))

    print("--> Check one batch from train loader <--")
    print("     - Batch tensor image, expect: (32, 1, 48, 48) ||", images.shape) # becase batch_size in VGG19 override batch_size in base
    print("     - Batch tensor label, expect: (64) ||", labels.shape)     # torch.Size([64])
    print("     - Image dtype, expect: float32 ||", images.dtype)     # float32
    print("     - Label dtype, REQUIRED: int64 ||", labels.dtype)     # int64 (Nếu là int8 thì model ko chạy đc)
    print("     - Max pixel, expect:  ~1.0 ||", images.max().item())  # Quanh quẩn ~1.0
    print("     - Min pixel, expect: ~-1.0 ||", images.min().item())  # Quanh quẩn ~ -1.0

    print(labels)
