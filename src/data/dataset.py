from PIL.Image import fromarray
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT


class FER2013(Dataset):
    """Load and Cache Dataset into RAM for Ultra-Fast Dataloading"""

    def __init__(self, data_path, split="train", transforms=None):
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        self.transform = transforms

        # VŨ KHÍ TỐI ƯU CPU: Cắt chuỗi và nạp toàn bộ ảnh lên RAM 1 lần duy nhất
        print(f"--> Pre-processing & Caching {split} dataset into RAM...")
        
        self.labels = self.data.iloc[:, 0].astype(int).tolist()
        pixels_list = self.data.iloc[:, 1].tolist()
        
        # Parse list of strings into a single optimized list of numpy arrays (Loại bỏ DeprecationWarning)
        self.images = [np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in pixels_list]
        
        print(f"--> Done caching {len(self.images)} images for {split}.")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """ Lấy ảnh trực tiếp từ RAM (O(1) Time Complexity) """
        label = self.labels[index]
        image_np = self.images[index]
        
        image = Image.fromarray(image_np)

        if self.transform is not None:
            image = self.transform(image)

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