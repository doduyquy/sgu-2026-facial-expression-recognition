from PIL.Image import fromarray
import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT


class FER2013(Dataset):
    """Load one sample for dataloader"""

    def __init__(self, data_path, split="train", transforms=None, semantic_masks_dir=None, num_regions=9, use_semantic_manifest=True):
        # set relative path to train|val|test in dataset
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        # because Q splitted dataset, so we only need 2 column: emotion(for category) and pixels for images
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        self.transform = transforms
        self.split = split
        self.semantic_masks_dir = Path(semantic_masks_dir) if semantic_masks_dir else None
        self.num_regions = int(num_regions)
        self.use_semantic_manifest = bool(use_semantic_manifest)

        self.semantic_manifest = None
        if self.semantic_masks_dir is not None and self.use_semantic_manifest:
            manifest_path = self.semantic_masks_dir / f"semantic_manifest_{split}.csv"
            if manifest_path.exists():
                self.semantic_manifest = pd.read_csv(manifest_path)

    def __len__(self):
        # return len(rows) of dataframe which we have read 
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Arg: 
            index: index of row in dataframe in dataset 
        Return 
            (image, label) & apply transform for image (if have)"""
        # get row and convert to numpy array
        emotion, pixels = self.data.iloc[index].values
        label = int(emotion)

        # convert image vector to image 48x48
        image_vec = np.fromstring(pixels, sep=' ', dtype=np.uint8)
        image_np = image_vec.reshape((48, 48))
        image = Image.fromarray(image_np)

        # apply transform if it not None
        if self.transform is not None:
            image = self.transform(image)

        if self.semantic_masks_dir is not None:
            # Do not trust save_path inside manifest. Always resolve from split/index.
            mask_path = self.semantic_masks_dir / self.split / f"{int(index):06d}.npz"

            manifest_row = None
            detect_success = True
            fallback_used = False
            variant_used = "unknown"
            if self.semantic_manifest is not None and index < len(self.semantic_manifest):
                manifest_row = self.semantic_manifest.iloc[index]
                if "success" in manifest_row:
                    detect_success = bool(manifest_row["success"])
                if "fallback_used" in manifest_row:
                    fallback_used = bool(manifest_row["fallback_used"])
                if "variant_used" in manifest_row:
                    variant_used = str(manifest_row["variant_used"])

            if mask_path.exists():
                with np.load(mask_path, allow_pickle=False) as npz:
                    bboxes = npz["bboxes"].astype(np.float32)
            else:
                fallback_used = True
                detect_success = False
                bboxes = np.zeros((self.num_regions, 4), dtype=np.float32)
                bboxes[:, 0] = 0.0
                bboxes[:, 1] = 0.0
                bboxes[:, 2] = 47.0
                bboxes[:, 3] = 47.0

            # Per-region validity: one bad region should not invalidate the whole face.
            x1 = bboxes[:, 0]
            y1 = bboxes[:, 1]
            x2 = bboxes[:, 2]
            y2 = bboxes[:, 3]
            finite_mask = np.isfinite(bboxes).all(axis=1)
            order_mask = (x2 > x1) & (y2 > y1)
            size_mask = ((x2 - x1) >= 2.0) & ((y2 - y1) >= 2.0)
            region_mask = (finite_mask & order_mask & size_mask).astype(np.float32)

            # Confidence is region-level. When detect fails, keep masks per-region
            # but suppress trust in those regions via low confidence.
            if detect_success:
                width = np.clip(x2 - x1, 1.0, None)
                height = np.clip(y2 - y1, 1.0, None)
                area = (width * height) / float(48 * 48)
                region_confidence = np.clip(0.5 + 0.5 * area, 0.0, 1.0).astype(np.float32)
            else:
                region_confidence = (0.15 * region_mask).astype(np.float32)

            semantic_meta = {
                "detect_success": np.array(detect_success, dtype=np.bool_),
                "fallback_used": np.array(fallback_used, dtype=np.bool_),
                "variant_used": variant_used,
                "region_mask": region_mask,
                "region_confidence": region_confidence,
            }
            return (image, label, bboxes, semantic_meta)

        return (image, label)
    
    def label_to_emotion(self, label):
        return EMOTION_DICT[label]

    
if __name__ == "__main__":
    import os
    from pathlib import Path
    root_dir = Path.cwd().resolve().parent.parent
    print(root_dir)

    data_path = os.path.join(root_dir, "dataset/fer13-split")
    data_train = FER2013(data_path=data_path, split='train')
    

    print("Emotion for label 3:", data_train.label_to_emotion(3))