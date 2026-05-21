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

    def __init__(self, data_path, split="train", transforms=None, semantic_masks_dir=None, num_regions=9):
        # set relative path to train|val|test in dataset
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        # because Q splitted dataset, so we only need 2 column: emotion(for category) and pixels for images
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        self.transform = transforms
        self.split = split
        self.semantic_masks_dir = Path(semantic_masks_dir) if semantic_masks_dir else None
        self.num_regions = int(num_regions)

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
            mask_path = self.semantic_masks_dir / self.split / f"{int(index):06d}.npz"
            if mask_path.exists():
                with np.load(mask_path, allow_pickle=False) as npz:
                    bboxes = npz["bboxes"].astype(np.float32)
            else:
                bboxes = np.zeros((self.num_regions, 4), dtype=np.float32)
                bboxes[:, 0] = 0.0
                bboxes[:, 1] = 0.0
                bboxes[:, 2] = 47.0
                bboxes[:, 3] = 47.0
            return (image, label, bboxes)

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