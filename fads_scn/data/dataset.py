import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


EMOTION_NAMES = [
    "angry",      # 0
    "disgust",    # 1
    "fear",       # 2
    "happy",      # 3
    "sad",        # 4
    "surprise",   # 5
    "neutral",    # 6
]


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(split: str = "train", input_size: int = 48, use_random_erasing: bool = True, erasing_prob: float = 0.3):
    """
    Build data transformation pipeline for pure image FER.
    Train: Flip + Affine + ToTensor + Normalize + RandomErasing
    Val/Test: ToTensor + Normalize
    """
    if split == "train":
        transform_list = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
        if use_random_erasing:
            transform_list.append(
                T.RandomErasing(p=erasing_prob, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0.0)
            )
        return T.Compose(transform_list)
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])


class PureImageFER2013(Dataset):
    """
    Pure Image-Based FER2013 Dataset.
    Only takes raw 48x48 pixel values from CSV.
    Zero dependency on bounding boxes, landmarks, or .npz files.
    """

    def __init__(self, data_path: str, split: str = "train", transform=None):
        super().__init__()
        self.split = split
        self.transform = transform

        # Resolve CSV path
        csv_candidates = [
            Path(data_path) / f"{split}.csv",
            Path(data_path) / f"fer13-split/{split}.csv",
            Path("dataset/fer13-split") / f"{split}.csv",
        ]
        csv_file = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_file = candidate
                break

        if csv_file is None:
            # Fallback to direct path
            csv_file = Path(data_path) / f"{split}.csv"

        df = pd.read_csv(csv_file, usecols=[0, 1])
        # Vectorized parsing into numpy array of uint8 images
        self.labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
        raw_pixels = df.iloc[:, 1].tolist()
        
        # Pre-parse pixels to (N, 48, 48) uint8 array for ultra-fast loading
        parsed_imgs = []
        for p_str in raw_pixels:
            arr = np.fromstring(p_str, sep=' ', dtype=np.uint8).reshape(48, 48)
            parsed_imgs.append(arr)
        self.images = np.stack(parsed_imgs, axis=0)  # [N, 48, 48]

        # Mutable labels for SCN dynamic relabeling
        self.relabelled_count = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        img_arr = self.images[index]
        label = int(self.labels[index])

        img = Image.fromarray(img_arr)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index

    def update_label(self, index: int, new_label: int):
        """Update label dynamically for SCN relabeling."""
        if 0 <= index < len(self.labels) and self.labels[index] != new_label:
            self.labels[index] = new_label
            self.relabelled_count += 1

    def get_class_counts(self):
        """Return counts per class for computing class weights."""
        counts = np.bincount(self.labels, minlength=7)
        return counts


def build_dataloaders(cfg: dict):
    """Factory to build train, val, and test dataloaders."""
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("data_path", "dataset/fer13-split")
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 2)
    use_random_erasing = data_cfg.get("use_random_erasing", True)
    erasing_prob = data_cfg.get("erasing_prob", 0.3)
    seed = cfg.get("seed", {}).get("random_seed", None)
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = seed_worker

    train_tf = build_transforms("train", use_random_erasing=use_random_erasing, erasing_prob=erasing_prob)
    val_tf = build_transforms("val")
    test_tf = build_transforms("test")

    train_ds = PureImageFER2013(data_path, split="train", transform=train_tf)
    val_ds = PureImageFER2013(data_path, split="val", transform=val_tf)
    test_ds = PureImageFER2013(data_path, split="test", transform=test_tf)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return train_loader, val_loader, test_loader
