import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import FER2013
from .transforms import build_transform


def _build_weighted_sampler(data_train, train_csv_path: str) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that over-samples rare classes.

    Strategy:
        sample_weight[i] = 1 / count(label[i])

    This ensures each class is drawn with equal probability per epoch,
    which stabilises Supervised Contrastive Loss by guaranteeing enough
    positive pairs for every emotion class (Fear/Sad/Disgust are rare in FER2013).

    Args:
        data_train: FER2013 dataset instance (already constructed).
        train_csv_path: path to train.csv — used to read labels fast via pandas
                        instead of iterating __getitem__ over 28k samples.

    Returns:
        WeightedRandomSampler with replacement=True, num_samples=len(data_train)
    """
    df = pd.read_csv(train_csv_path, usecols=[0])      # only 'emotion' column
    labels_all = df.iloc[:, 0].values                  # shape (N,)

    # class_counts[c] = number of samples for class c
    num_classes = 7
    class_counts = torch.zeros(num_classes, dtype=torch.float)
    for lbl in labels_all:
        class_counts[int(lbl)] += 1.0

    # class_weight[c] = inverse frequency — rare classes get higher weight
    class_weight = 1.0 / class_counts.clamp_min(1.0)

    # per-sample weight
    sample_weights = torch.tensor(
        [class_weight[int(lbl)] for lbl in labels_all],
        dtype=torch.float,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(data_train),   # same epoch length as without sampler
        replacement=True,              # must be True for weighted over-sampling
    )

    print("[WeightedRandomSampler] Class counts:", class_counts.long().tolist())
    print("[WeightedRandomSampler] Class weights (relative):",
          [f"{w:.4f}" for w in (class_weight / class_weight.sum()).tolist()])
    return sampler


def build_dataloader(config, data_path):
    """Build DataLoaders for train / val / test.

    Task 1: When training.use_weighted_sampler=true, replaces shuffle=True
    with WeightedRandomSampler so that each batch sees roughly equal counts
    of all 7 emotion classes. Critical for Supervised Contrastive Loss which
    needs positive pairs from every class in each batch.

    Args:
        config: full config dict
        data_path: path to fer13-split dir (contains train.csv / val.csv / test.csv)

    Returns:
        train_loader, val_loader, test_loader
    """
    # transforms
    trans_train = build_transform(config, "train")
    trans_val   = build_transform(config, "val")
    trans_test  = build_transform(config, "test")

    data_cfg   = config.get('data', {})
    train_cfg  = config.get('training', {})

    use_semantic_masks = bool(data_cfg.get('use_semantic_masks', False))
    semantic_masks_dir = data_cfg.get('semantic_masks_dir')
    num_regions        = int(config.get('model', {}).get('num_regions', 9))
    batch_size         = data_cfg['batch_size']
    num_workers        = data_cfg.get('num_workers', 2)

    if semantic_masks_dir and not os.path.isabs(semantic_masks_dir):
        semantic_masks_dir = os.path.join(os.path.dirname(data_path), semantic_masks_dir)

    masks_dir_arg = semantic_masks_dir if use_semantic_masks else None

    # build datasets
    data_train = FER2013(data_path=data_path, split="train",
                         transforms=trans_train,
                         semantic_masks_dir=masks_dir_arg,
                         num_regions=num_regions)
    data_val   = FER2013(data_path=data_path, split="val",
                         transforms=trans_val,
                         semantic_masks_dir=masks_dir_arg,
                         num_regions=num_regions)
    data_test  = FER2013(data_path=data_path, split="test",
                         transforms=trans_test,
                         semantic_masks_dir=masks_dir_arg,
                         num_regions=num_regions)

    # ── Task 1: WeightedRandomSampler ────────────────────────────────────────
    use_weighted_sampler = bool(train_cfg.get('use_weighted_sampler', False))
    train_sampler = None
    train_shuffle = True          # default: plain random shuffle

    if use_weighted_sampler:
        train_csv_path = os.path.join(data_path, "train.csv")
        train_sampler  = _build_weighted_sampler(data_train, train_csv_path)
        train_shuffle  = False    # IMPORTANT: shuffle=True is mutually exclusive with sampler
        print("[DataLoader] WeightedRandomSampler enabled — shuffle disabled")
    else:
        print("[DataLoader] WeightedRandomSampler disabled — using plain shuffle")

    # ── Build loaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=train_shuffle,
        sampler=train_sampler,      # None when not using weighted sampler
    )
    val_loader = DataLoader(
        data_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
    test_loader = DataLoader(
        data_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

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