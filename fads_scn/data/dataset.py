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


def build_transforms(
    split: str = "train",
    input_size: int = 48,
    use_random_erasing: bool = True,
    erasing_prob: float = 0.3,
    in_channels: int = 1,
    normalization: str = "symmetric",
):
    """
    Build data transformation pipeline for pure image FER.
    Train: Flip + Affine + ToTensor + Normalize + RandomErasing
    Val/Test: ToTensor + Normalize
    """
    if input_size < 16:
        raise ValueError("input_size must be at least 16")
    if in_channels not in (1, 3):
        raise ValueError("FER2013 supports in_channels=1 or 3")
    if normalization not in ("symmetric", "imagenet"):
        raise ValueError("normalization must be symmetric or imagenet")
    if normalization == "imagenet" and in_channels != 3:
        raise ValueError("imagenet normalization requires 3 channels")

    # Defaults intentionally preserve the prior 48x48 grayscale recipe exactly.
    transform_list = []
    if input_size != 48:
        transform_list.append(T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR))
    if in_channels == 3:
        # FER2013 is grayscale: replicate its one channel; this does not invent colour.
        transform_list.append(T.Grayscale(num_output_channels=3))
    if split == "train":
        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            T.ToTensor(),
        ])
    else:
        transform_list.append(T.ToTensor())

    if normalization == "imagenet":
        transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        transform_list.append(T.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels))

    if split == "train":
        if use_random_erasing:
            transform_list.append(
                T.RandomErasing(p=erasing_prob, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0.0)
            )
        return T.Compose(transform_list)
    return T.Compose(transform_list)


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

    input_size = data_cfg.get("input_size", 48)
    in_channels = cfg.get("model", {}).get("in_channels", 1)
    normalization = data_cfg.get("normalization", "symmetric")
    train_tf = build_transforms("train", input_size, use_random_erasing, erasing_prob, in_channels, normalization)
    val_tf = build_transforms("val", input_size, in_channels=in_channels, normalization=normalization)
    test_tf = build_transforms("test", input_size, in_channels=in_channels, normalization=normalization)

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
