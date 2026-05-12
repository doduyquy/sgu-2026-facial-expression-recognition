"""
FER2013 Dataset with Landmark Masks.

Extends the base FER2013 dataset to also return per-sample region masks
[K, Hf, Wf] generated from Dlib 68-point facial landmarks.

DataLoader returns: (image_tensor, label, region_masks)
"""

import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from .dataset import FER2013
from .landmark_mask import DlibLandmarkMaskGenerator


class FER2013WithLandmarks(FER2013):
    """
    FER2013 dataset that also returns landmark-based region masks.

    __getitem__ returns:
        (image_tensor, label, region_masks)

    where region_masks is a [K, Hf, Wf] float tensor (0..1).
    """

    def __init__(self, data_path, split="train", transforms=None,
                 grid_size=14, sigma=1.5, num_regions=6, predictor_path=None):
        super().__init__(data_path, split=split, transforms=transforms)

        self.mask_generator = DlibLandmarkMaskGenerator(
            grid_size=grid_size,
            sigma=sigma,
            num_regions=num_regions,
            predictor_path=predictor_path,
        )
        print(
            f"--> [FER2013WithLandmarks] split={split}, "
            f"grid={grid_size}x{grid_size}, sigma={sigma}, K={num_regions}"
        )

    def __getitem__(self, index):
        """
        Returns:
            image:        Tensor [C, H, W] after transforms
            label:        int (0-6)
            region_masks: Tensor [K, Hf, Wf] float32 (0..1)
        """
        row = self.data.iloc[index]
        emotion = row.iloc[0]
        pixels = row.iloc[1]
        label = int(emotion)

        # Parse the raw 48x48 grayscale image
        image_vec = np.fromstring(pixels, sep=' ', dtype=np.uint8)
        image_np = image_vec.reshape((48, 48))

        # Generate landmark masks BEFORE any augmentation transforms.
        # Masks are based on the original face geometry, not the augmented version.
        region_masks = self.mask_generator(image_np)

        # Convert to PIL and apply image transforms (resize, augment, normalize)
        image = Image.fromarray(image_np)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, region_masks
