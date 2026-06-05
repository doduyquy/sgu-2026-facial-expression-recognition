from PIL.Image import fromarray
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT


class FER2013(Dataset):
    """Load one sample for dataloader"""

    def __init__(
        self,
        data_path,
        split="train",
        transforms=None,
        use_clean_filter=True,
        bad_row_indices_path=None,
    ):
        # set relative path to train|val|test in dataset
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        # because Q splitted dataset, so we only need 2 column: emotion(for category) and pixels for images
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        
        # Keep original index just in case future features need it
        self.split = split
        self.data['original_idx'] = self.data.index

        # ── Data cleaning: drop bad rows for train split only ──
        # bad_row_indices.txt chứa các chỉ số dòng (0-based) trong train.csv
        # tương ứng với ảnh lỗi (đen, trắng, không phải mặt người, v.v.)
        if split == "train" and use_clean_filter:
            # Tìm file blacklist ở nhiều vị trí (local & Kaggle)
            if bad_row_indices_path and not os.path.exists(bad_row_indices_path):
                raise FileNotFoundError(
                    f"Configured bad_row_indices_path not found: {bad_row_indices_path}"
                )
            candidate_paths = [
                bad_row_indices_path,
                os.path.join(data_path, "bad_row_indices.txt"),                     # local: cùng thư mục train.csv
                "/kaggle/input/datasets/lphuccc/id-error/bad_row_indices.txt",      # kaggle dataset
            ]
            blacklist_path = None
            for p in candidate_paths:
                if p and os.path.exists(p):
                    blacklist_path = p
                    break

            if blacklist_path is not None:
                with open(blacklist_path, 'r') as f:
                    bad_indices = set(
                        int(line.strip()) for line in f if line.strip()
                    )
                before = len(self.data)
                self.data = self.data[
                    ~self.data['original_idx'].isin(bad_indices)
                ].reset_index(drop=True)
                after = len(self.data)
                print(f"[FER2013] Filtered {before - after} bad rows "
                      f"({before} -> {after}) using {blacklist_path}")
        elif split == "train":
            print("[FER2013] Clean bad-row filter disabled; using original train.csv rows.")

        self.transform = transforms

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
        row = self.data.iloc[index]
        emotion = row.iloc[0]
        pixels = row.iloc[1]
        original_idx = row['original_idx']
        
        label = int(emotion)

        # convert image vector to image 48x48
        image_vec = np.fromstring(pixels, sep=' ', dtype=np.uint8)
        image_np = image_vec.reshape((48, 48))
        image = Image.fromarray(image_np)

        # apply transform if it not None
        if self.transform is not None:
            if getattr(self.transform, "accepts_label", False):
                image = self.transform(image, label=label)
            else:
                image = self.transform(image)

        return (image, label)
    
    def label_to_emotion(self, label):
        return EMOTION_DICT[label]

    
if __name__ == "__main__":
    import os

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    print("Root dir:", root_dir)

    data_path = os.path.join(root_dir, "dataset", "fer13-split")
    print("Data path:", data_path)

    data_train = FER2013(data_path=data_path, split="train")
    print("Train samples:", len(data_train))
    

    print("Emotion for label 3:", data_train.label_to_emotion(3))
