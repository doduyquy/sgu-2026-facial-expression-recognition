from PIL.Image import fromarray
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT


class FER2013(Dataset):
    """Load one sample for dataloader"""

    def __init__(self, data_path, split="train", transforms=None):
        # set relative path to train|val|test in dataset
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        # because Q splitted dataset, so we only need 2 column: emotion(for category) and pixels for images
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        self.transform = transforms
        self.split = split

        # Nạp Landmark tĩnh chỉ dành cho tập Train (Teacher KD)
        self.landmarks = None
        self.lm_status = None
        if split == "train":
            lm_path = os.path.join(os.path.dirname(__file__), "Data", f"landmarks_{split} (1).csv")
            if not os.path.exists(lm_path):
                lm_path = os.path.join(data_path, f"landmarks_{split}.csv")
            if os.path.exists(lm_path):
                print(f"--> [Landmark KD] Nạp Teacher Landmarks từ {lm_path}...")
                ldf = pd.read_csv(lm_path)
                self.landmarks = ldf.iloc[:, 2:].values # (N, 20)
                self.lm_status = (ldf['status'] == 'success').values.astype(np.float32) # (N,) 1.0 if success, 0.0 if failed
            else:
                print(f"[WARNING] Không tìm thấy file Landmark tại {lm_path}. Bỏ qua Soft Supervision.")

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

        # Xử lý Landmark KD cho tập Train
        landmarks = None
        valid_lm = 0.0
        if self.split == "train" and self.landmarks is not None:
            lm_arr = self.landmarks[index].reshape(10, 2).copy()
            valid_lm = float(self.lm_status[index])
            
            # Khôi phục lật ngang thủ công an toàn cho Landmark
            import random
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                lm_arr[:, 0] = 47.0 - lm_arr[:, 0]
                lm_arr[[2, 3, 4, 5, 7, 8], :] = lm_arr[[3, 2, 5, 4, 8, 7], :]
            
            landmarks = torch.tensor(lm_arr, dtype=torch.float32)

        # apply transform if it not None
        if self.transform is not None:
            image = self.transform(image)

        if self.split == "train":
            if landmarks is None:
                landmarks = torch.zeros((10, 2), dtype=torch.float32)
            return image, label, landmarks, torch.tensor(valid_lm, dtype=torch.float32)
        return (image, label)
    
    def label_to_emotion(self, label):
        return EMOTION_DICT[label]

    
if __name__ == "__main__":
    import os
    from pathlib import Path
    root_dir = Path.cwd().resolve().parent
    print(root_dir)

    data_path = os.path.join(root_dir, "dataset/fer13-split")
    data_train = FER2013(data_path=data_path, split='train')

    

    print("Emotion for label 3:", data_train.label_to_emotion(3))