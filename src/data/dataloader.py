import os
import tensorflow as tf
from .dataset import create_fer2013_dataset
from .transforms import build_transform

def build_dataloader(config, data_path):
    """ Dataloader: Group dataset into batch (mini-batch) 
    Args: 
        config: config for data, dataloader
        data_path: path to fer13-split dir
    Return: 
        train_loader, val_loader, test_loader
    """
    trans_train = build_transform(config, "train")
    trans_val = build_transform(config, "val")
    trans_test = build_transform(config, "test")

    use_semantic_masks = bool(config.get('data', {}).get('use_semantic_masks', False))
    semantic_masks_dir = config.get('data', {}).get('semantic_masks_dir')
    num_regions = int(config.get('model', {}).get('num_regions', 9))
    batch_size = config.get('data', {}).get('batch_size', 32)

    if semantic_masks_dir and not os.path.isabs(semantic_masks_dir):
        semantic_masks_dir = os.path.join(os.path.dirname(data_path), semantic_masks_dir)

    train_ds = create_fer2013_dataset(
        data_path=data_path,
        split="train",
        transforms=trans_train,
        semantic_masks_dir=semantic_masks_dir if use_semantic_masks else None,
        num_regions=num_regions,
    )
    val_ds = create_fer2013_dataset(
        data_path=data_path,
        split="val",
        transforms=trans_val,
        semantic_masks_dir=semantic_masks_dir if use_semantic_masks else None,
        num_regions=num_regions,
    )
    test_ds = create_fer2013_dataset(
        data_path=data_path,
        split="test",
        transforms=trans_test,
        semantic_masks_dir=semantic_masks_dir if use_semantic_masks else None,
        num_regions=num_regions,
    )

    train_loader = train_ds.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_loader = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_loader = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_loader, val_loader, test_loader