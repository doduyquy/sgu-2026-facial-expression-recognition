import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .dataset import FER2013
from .transforms import build_landmark_transform, build_transform

def build_dataloader(config, data_path, distributed=False, world_size=1):
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

    train_sampler = DistributedSampler(data_train, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(data_val, shuffle=False) if distributed else None
    test_sampler = None

    batch_size = config['data']['batch_size']
    ddp_cfg = config.get('ddp', {})
    if distributed and ddp_cfg.get('batch_size_is_global', True):
        batch_size = max(batch_size // max(world_size, 1), 1)
        print(f"--> DDP per-process batch_size: {batch_size}")

    # batch the dataset
    train_loader = DataLoader(
        data_train, 
        batch_size=batch_size,
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None))
    val_loader = DataLoader(
        data_val, 
        batch_size=batch_size, 
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True,
        sampler=val_sampler,
        shuffle=False)
    test_loader = DataLoader(
        data_test, 
        batch_size=batch_size, 
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True,
        sampler=test_sampler,
        shuffle=False)
    
    return train_loader, val_loader, test_loader


def build_landmark_dataloader(config, data_path, distributed=False, world_size=1):
    """
    Same as build_dataloader but uses FER2013WithLandmarks.
    Each batch returns (images, labels, region_masks).
    """
    from .dataset_landmark import FER2013WithLandmarks

    trans_train = build_landmark_transform(config, "train")
    trans_val = build_landmark_transform(config, "val")
    trans_test = build_landmark_transform(config, "test")

    model_cfg = config.get("model", {})
    feature_layer = model_cfg.get("feature_layer", "layer3")
    image_size = config["data"].get("image_size", 224)
    grid_sizes = {"layer2": image_size // 8, "layer3": image_size // 16, "layer4": image_size // 32}
    grid_size = grid_sizes.get(feature_layer, 14)
    sigma = model_cfg.get("landmark_sigma", 1.5)
    num_regions = model_cfg.get("num_regions", 6)
    predictor_path = model_cfg.get("landmark_predictor_path", None)
    cache_masks = config.get("data", {}).get("cache_landmark_masks", True)

    ds_kwargs = dict(
        grid_size=grid_size,
        sigma=sigma,
        num_regions=num_regions,
        predictor_path=predictor_path,
        cache_masks=cache_masks,
    )

    data_train = FER2013WithLandmarks(data_path, split="train", transforms=trans_train, **ds_kwargs)
    data_val = FER2013WithLandmarks(data_path, split="val", transforms=trans_val, **ds_kwargs)
    data_test = FER2013WithLandmarks(data_path, split="test", transforms=trans_test, **ds_kwargs)

    train_sampler = DistributedSampler(data_train, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(data_val, shuffle=False) if distributed else None
    test_sampler = None

    batch_size = config["data"]["batch_size"]
    ddp_cfg = config.get("ddp", {})
    if distributed and ddp_cfg.get("batch_size_is_global", True):
        batch_size = max(batch_size // max(world_size, 1), 1)
        print(f"--> DDP per-process batch_size: {batch_size}")

    num_workers = config["data"].get("num_workers", 2)
    train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader = DataLoader(data_val, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, sampler=val_sampler, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, sampler=test_sampler, shuffle=False)

    return train_loader, val_loader, test_loader
