from PIL.Image import fromarray
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT
from .landmarks import LandmarkExtractor


class FER2013(Dataset):
    """Load one sample for dataloader"""

    def __init__(self, data_path, split="train", transforms=None, use_landmarks=False, landmark_config=None):
        # set relative path to train|val|test in dataset
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        # because Q splitted dataset, so we only need 2 column: emotion(for category) and pixels for images
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        self.transform = transforms
        self.use_landmarks = use_landmarks
        self.landmark_config = landmark_config or {}
        self.landmark_representation = self.landmark_config.get("representation", "coords")
        self.landmark_heatmap_size = self.landmark_config.get("heatmap_size", 48)
        self.landmark_heatmap_sigma = self.landmark_config.get("heatmap_sigma", 1.5)
        self.landmark_normalize_mode = self.landmark_config.get("normalize_mode", "relative")
        self.landmark_semantic_weighting = self.landmark_config.get("semantic_weighting", True)
        self.landmark_extractor = LandmarkExtractor(
            enabled=use_landmarks,
            backend=self.landmark_config.get("backend", "mediapipe"),
            landmark_indexes=self.landmark_config.get("landmark_indexes", None),
        )

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

        landmarks = None
        landmark_mask = None
        if self.use_landmarks:
            points_abs, mask = self.landmark_extractor.extract_points_with_mask(image_np)
            landmark_mask = torch.from_numpy(mask.astype(np.float32))

            if self.landmark_representation == "heatmap":
                hm = LandmarkExtractor.points_to_heatmaps(
                    points_abs,
                    mask=mask,
                    heatmap_size=self.landmark_heatmap_size,
                    sigma=self.landmark_heatmap_sigma,
                    semantic_weighting=self.landmark_semantic_weighting,
                )
                landmarks = torch.from_numpy(hm)
            else:
                points_out = points_abs
                if self.landmark_normalize_mode == "relative":
                    points_out = LandmarkExtractor.normalize_points_relative(points_abs, mask)
                landmarks = torch.from_numpy(points_out.reshape(-1).astype(np.float32))

        # apply transform if it not None
        if self.transform is not None:
            image = self.transform(image)

        if self.use_landmarks:
            return (image, label, landmarks, landmark_mask)

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
