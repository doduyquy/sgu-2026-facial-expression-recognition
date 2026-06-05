"""
FER2013 dataset with precomputed U-Net / face-parsing region masks.

Each sample returns:
    image, label, region_masks

where region_masks has shape [K, Hf, Wf] and is used as a soft spatial guide
for region attention.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import FER2013


def resolve_mask_dir(mask_dir, split):
    requested = Path(mask_dir)
    if (requested / split).exists():
        return requested

    folder_name = requested.name
    search_roots = [Path.cwd()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        search_roots.insert(0, kaggle_input)

    for root in search_roots:
        for current_dir, dirs, _ in os.walk(root):
            current = Path(current_dir)
            if current.name == folder_name and split in dirs:
                print(
                    f"--> [FER2013WithUNetMasks] Using discovered mask_dir: {current}"
                )
                return current

    return requested


class FER2013WithUNetMasks(FER2013):
    """
    FER2013 dataset that reads precomputed region masks from disk.

    Expected mask layout:
        mask_dir/
          train/000000.npy
          val/000000.npy
          test/000000.npy

    The numeric file stem is the original row index from split CSV.
    """

    def __init__(
        self,
        data_path,
        split="train",
        transforms=None,
        mask_dir="outputs/unet_region_masks",
        grid_size=7,
        num_regions=6,
        mask_floor=0.05,
        use_clean_filter=True,
        bad_row_indices_path=None,
    ):
        super().__init__(
            data_path=data_path,
            split=split,
            transforms=transforms,
            use_clean_filter=use_clean_filter,
            bad_row_indices_path=bad_row_indices_path,
        )
        self.mask_dir = resolve_mask_dir(mask_dir, split)
        self.split_mask_dir = self.mask_dir / split
        self.grid_size = int(grid_size)
        self.num_regions = int(num_regions)
        self.mask_floor = float(mask_floor)

        if not self.split_mask_dir.exists():
            raise FileNotFoundError(
                f"Mask split directory not found: {self.split_mask_dir}. "
                "Run scripts/precompute_face_parsing_region_masks.py first."
            )

        print(
            f"--> [FER2013WithUNetMasks] split={split}, "
            f"mask_dir={self.split_mask_dir}, grid={self.grid_size}x{self.grid_size}, "
            f"K={self.num_regions}"
        )

    @staticmethod
    def _read_image_from_pixels(pixels):
        image_vec = np.fromstring(pixels, sep=" ", dtype=np.uint8)
        return image_vec.reshape((48, 48))

    def _load_region_masks(self, original_idx):
        mask_path = self.split_mask_dir / f"{int(original_idx):06d}.npy"
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

        return masks.clamp(min=self.mask_floor, max=1.0)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        label = int(row.iloc[0])
        pixels = row.iloc[1]
        original_idx = int(row["original_idx"])

        image_np = self._read_image_from_pixels(pixels)
        image = Image.fromarray(image_np)
        region_masks = self._load_region_masks(original_idx)

        if self.transform is not None:
            if getattr(self.transform, "accepts_masks", False):
                if getattr(self.transform, "accepts_label", False):
                    image, region_masks = self.transform(image, region_masks, label=label)
                else:
                    image, region_masks = self.transform(image, region_masks)
            elif getattr(self.transform, "accepts_label", False):
                image = self.transform(image, label=label)
            else:
                image = self.transform(image)

        return image, label, region_masks
