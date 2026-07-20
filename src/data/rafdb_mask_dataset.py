import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from .transforms import build_landmark_transform


RAFDB_FOLDER_TO_NAME = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}
CLASS_FOLDERS = [str(i) for i in range(1, 8)]
CLASS_NAMES = [RAFDB_FOLDER_TO_NAME[str(i)] for i in range(1, 8)]


def resolve_rafdb_root(raw_root):
    root_value = str(raw_root)
    if root_value.lower() != "auto":
        root = Path(root_value)
        if not root.exists():
            raise FileNotFoundError(f"Configured RAF-DB root does not exist: {root}")
        return root

    candidates = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.glob("*/DATASET"))
        candidates.extend(input_root.glob("*/*/DATASET"))

    candidates.extend(Path.cwd().glob("*/DATASET"))
    candidates.extend(Path.cwd().glob("DATASET"))

    valid = [
        path
        for path in candidates
        if (path / "train").is_dir() and (path / "test").is_dir()
    ]
    if not valid:
        searched = "/kaggle/input/*/DATASET, /kaggle/input/*/*/DATASET, ./DATASET"
        raise FileNotFoundError(f"Could not auto-find RAF-DB DATASET root. Searched: {searched}")

    valid = sorted(set(path.resolve() for path in valid), key=lambda p: str(p))
    print(f"--> Auto-found RAF-DB root: {valid[0]}")
    return valid[0]


def resolve_mask_root(mask_root, split):
    requested = Path(mask_root)
    if (requested / split).exists():
        return requested

    def _find_root_with_split(search_root):
        if not search_root.exists():
            return None
        for current_dir, dirs, _ in os.walk(search_root):
            current = Path(current_dir)
            if split in dirs:
                return current
        return None

    recursive_candidate = _find_root_with_split(requested)
    if recursive_candidate is not None:
        print(f"--> [RAFDBWithMasks] Using discovered mask_root: {recursive_candidate}")
        return recursive_candidate

    search_roots = [Path.cwd()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        search_roots.insert(0, kaggle_input)

    for root in search_roots:
        recursive_candidate = _find_root_with_split(root)
        if recursive_candidate is not None:
            print(f"--> [RAFDBWithMasks] Using discovered mask_root: {recursive_candidate}")
            return recursive_candidate

    return requested


def validate_imagefolder_classes(dataset, split_name):
    classes = list(dataset.classes)
    if classes != CLASS_FOLDERS:
        raise ValueError(
            f"{split_name} folders must be exactly {CLASS_FOLDERS}, got {classes}. "
            "RAF-DB folder ids are part of the label mapping."
        )


def count_by_target(targets):
    counts = Counter(int(target) for target in targets)
    return [counts.get(idx, 0) for idx in range(len(CLASS_NAMES))]


class RAFDBWithMasks(Dataset):
    def __init__(
        self,
        root,
        split,
        transform=None,
        mask_root="outputs/rafdb_mediapipe_region_masks",
        grid_size=7,
        num_regions=6,
        mask_floor=0.05,
        mask_ablation="none",
        mask_region_permutation=None,
    ):
        if split not in ("train", "test"):
            raise ValueError("RAFDBWithMasks split must be 'train' or 'test'.")

        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.dataset = ImageFolder(self.split_dir)
        validate_imagefolder_classes(self.dataset, split)
        self.transform = transform
        self.mask_root = resolve_mask_root(mask_root, split)
        self.grid_size = int(grid_size)
        self.num_regions = int(num_regions)
        self.mask_floor = float(mask_floor)
        self.mask_ablation = str(mask_ablation or "none").lower()
        if self.mask_ablation not in ("none", "uniform", "shuffle_regions"):
            raise ValueError("mask_ablation must be one of: none, uniform, shuffle_regions")
        if mask_region_permutation is None:
            mask_region_permutation = [4, 2, 0, 5, 1, 3]
        self.mask_region_permutation = [int(i) for i in mask_region_permutation]
        if sorted(self.mask_region_permutation) != list(range(self.num_regions)):
            raise ValueError("mask_region_permutation must be a permutation of region indices.")

        self.split_mask_dir = self.mask_root / split
        if not self.split_mask_dir.exists():
            raise FileNotFoundError(
                f"Mask split directory not found: {self.split_mask_dir}. "
                "Run scripts/precompute_rafdb_mediapipe_region_masks.py first."
            )

        print(
            f"--> [RAFDBWithMasks] split={split}, samples={len(self.dataset)}, "
            f"mask_dir={self.split_mask_dir}, grid={self.grid_size}x{self.grid_size}, "
            f"K={self.num_regions}, mask_ablation={self.mask_ablation}"
        )

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    @property
    def targets(self):
        return self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def _mask_path(self, image_path):
        relative = Path(image_path).relative_to(self.split_dir)
        return self.split_mask_dir / relative.parent / f"{relative.name}.npy"

    def _load_region_masks(self, image_path):
        mask_path = self._mask_path(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing precomputed mask: {mask_path}")

        masks = np.load(mask_path).astype(np.float32)
        if masks.ndim != 3:
            raise ValueError(f"Expected mask shape [K,H,W], got {masks.shape} at {mask_path}")
        if masks.shape[0] != self.num_regions:
            raise ValueError(
                f"Expected {self.num_regions} region masks, got {masks.shape[0]} at {mask_path}"
            )

        masks = torch.from_numpy(masks).float().clamp(0.0, 1.0)
        if masks.shape[-2:] != (self.grid_size, self.grid_size):
            masks = F.interpolate(
                masks.unsqueeze(0),
                size=(self.grid_size, self.grid_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.mask_ablation == "uniform":
            masks = torch.ones_like(masks)
        elif self.mask_ablation == "shuffle_regions":
            masks = masks[self.mask_region_permutation]

        return masks.clamp(min=self.mask_floor, max=1.0)

    def __getitem__(self, index):
        image_path, label = self.dataset.samples[index]
        image = Image.open(image_path).convert("RGB")
        region_masks = self._load_region_masks(image_path)

        if self.transform is not None:
            if getattr(self.transform, "accepts_label", False):
                image, region_masks = self.transform(image, region_masks, label=label)
            else:
                image, region_masks = self.transform(image, region_masks)

        return image, int(label), region_masks


def build_rafdb_mask_loaders(config, root, distributed=False, world_size=1):
    if distributed:
        raise NotImplementedError("RAF-DB mask trainer currently supports single-process training only.")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    seed = int(config.get("seed", {}).get("random_seed", config.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    mask_root = model_cfg.get("mask_dir") or data_cfg.get("mask_dir") or "outputs/rafdb_mediapipe_region_masks"
    image_size = int(data_cfg.get("image_size", 224))
    feature_layer = model_cfg.get("feature_layer", "layer4")
    grid_sizes = {"layer2": image_size // 8, "layer3": image_size // 16, "layer4": image_size // 32}
    grid_size = int(model_cfg.get("grid_size", grid_sizes.get(feature_layer, 7)))
    num_regions = int(model_cfg.get("num_regions", 6))
    mask_floor = float(model_cfg.get("mask_floor", 0.05))
    mask_ablation = data_cfg.get("mask_ablation", model_cfg.get("mask_ablation", "none"))
    mask_region_permutation = data_cfg.get(
        "mask_region_permutation",
        model_cfg.get("mask_region_permutation"),
    )

    train_transform = build_landmark_transform(config, "train")
    eval_transform = build_landmark_transform(config, "test")
    dataset_kwargs = {
        "mask_root": mask_root,
        "grid_size": grid_size,
        "num_regions": num_regions,
        "mask_floor": mask_floor,
        "mask_ablation": mask_ablation,
        "mask_region_permutation": mask_region_permutation,
    }
    train_full = RAFDBWithMasks(root, "train", transform=train_transform, **dataset_kwargs)
    val_full = RAFDBWithMasks(root, "train", transform=eval_transform, **dataset_kwargs)
    test_dataset = RAFDBWithMasks(root, "test", transform=eval_transform, **dataset_kwargs)

    targets = np.array(train_full.targets)
    indices = np.arange(len(targets))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )
    train_dataset = Subset(train_full, train_indices.tolist())
    val_dataset = Subset(val_full, val_indices.tolist())

    batch_size = int(data_cfg.get("batch_size", 32))
    eval_batch_size = int(data_cfg.get("eval_batch_size", batch_size))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(data_cfg.get("drop_last", False)),
    )
    eval_kwargs = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    counts = {
        "train_raw": count_by_target(targets),
        "internal_train": count_by_target(targets[train_indices]),
        "val": count_by_target(targets[val_indices]),
        "test": count_by_target(test_dataset.targets),
    }
    return (
        train_loader,
        DataLoader(val_dataset, **eval_kwargs),
        DataLoader(test_dataset, **eval_kwargs),
        train_full.class_to_idx,
        counts,
    )
