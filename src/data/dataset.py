from PIL.Image import fromarray
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT


class FER2013(Dataset):
    """Load and Cache Dataset into RAM for Ultra-Fast Dataloading"""

    def __init__(self, data_path, split="train", transforms=None, landmark_dir=None):
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        self.transform = transforms

        # VŨ KHÍ TỐI ƯU CPU: Cắt chuỗi và nạp toàn bộ ảnh lên RAM 1 lần duy nhất
        print(f"--> Pre-processing & Caching {split} dataset into RAM...")
        
        self.labels = self.data.iloc[:, 0].astype(int).tolist()
        pixels_list = self.data.iloc[:, 1].tolist()
        
        # Parse list of strings into a single optimized list of numpy arrays (Loại bỏ DeprecationWarning)
        self.images = [np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in pixels_list]
        
        # CAY GHEP BO 10 DIEM VANG (The 10 Golden Landmarks)
        base_dir = os.path.dirname(data_path)
        possible_paths = [
            os.path.join(data_path, "landmarks", f"landmarks_{split}.csv"),
            os.path.join(base_dir, "landmarks", f"landmarks_{split}.csv"),
            os.path.join("dataset", "landmarks", f"landmarks_{split}.csv"),
            os.path.join("/kaggle/input/datasets/ltlttt/datalandmark", f"landmarks_{split} (1).csv")
        ]
        if landmark_dir and os.path.exists(landmark_dir):
            possible_paths.insert(0, os.path.join(landmark_dir, f"landmarks_{split}.csv"))
            possible_paths.insert(0, os.path.join(landmark_dir, f"landmarks_{split} (1).csv"))

        landmark_path = None
        for p in possible_paths:
            if os.path.exists(p):
                landmark_path = p
                break
                
        if landmark_path and os.path.exists(landmark_path):
            print(f"--> [Landmarks] Loading 10 Golden Landmarks from {landmark_path}...")
            ldf = pd.read_csv(landmark_path)
            self.landmarks = ldf.iloc[:, 2:].values # (N, 20)
            if 'status' in ldf.columns:
                status_col = ldf['status'].values
                self.statuses = np.array([1.0 if s == 'success' else 0.0 for s in status_col], dtype=np.float32)
            else:
                self.statuses = np.ones(len(self.labels), dtype=np.float32)
        else:
            print(f"[WARNING] Landmark CSV not found for {split}. Using default mean 10-landmark matrix (treated as failed/subgraph fallback).")
            mean_matrix = np.array([
                20.0, 6.0,   # forehead
                20.0, 12.0,  # glabella
                12.0, 10.0,  # left_eyebrow
                28.0, 10.0,  # right_eyebrow
                12.0, 16.0,  # left_eye
                28.0, 16.0,  # right_eye
                20.0, 24.0,  # nose
                12.0, 36.0,  # left_mouth
                28.0, 36.0,  # right_mouth
                20.0, 44.0   # chin
            ])
            self.landmarks = np.tile(mean_matrix, (len(self.labels), 1))
            self.statuses = np.zeros(len(self.labels), dtype=np.float32)
        
        print(f"--> Done caching {len(self.images)} images for {split}.")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """ Lấy ảnh trực tiếp từ RAM (O(1) Time Complexity) """
        label = self.labels[index]
        image_np = self.images[index]
        
        image = Image.fromarray(image_np)
        
        # Lấy Landmark gốc
        landmarks = self.landmarks[index].reshape(10, 2).copy()
        
        # 🟢 ANTI-OVERFITTING: Đưa Augmentation Lật ngang (Horizontal Flip) trở lại!
        # Vì Landmark là điểm hình học, ta không thể để transforms.RandomHorizontalFlip tự làm
        # Do đó, ta lật BẰNG TAY: Lật cả Ảnh, lật trục X của Landmark, và Hoán đổi Trái/Phải
        import random
        if hasattr(self, 'data_split_path') and 'train' in self.data_split_path and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Lật trục X (Ảnh 48x48 nên X_new = 48.0 - X_old)
            landmarks[:, 0] = 48.0 - landmarks[:, 0]
            
            # Hoán đổi (Swap) các cặp điểm đối xứng Trái <-> Phải
            # [2, 3]: Lông mày trái <-> phải
            # [4, 5]: Mắt trái <-> phải
            # [7, 8]: Mép miệng trái <-> phải
            # (Trán [0], Ấn đường [1], Mũi [6], Cằm [9] nằm trục giữa -> không đổi)
            landmarks[[2, 3, 4, 5, 7, 8], :] = landmarks[[3, 2, 5, 4, 8, 7], :]

        if self.transform is not None:
            image = self.transform(image)

        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
        status_tensor = torch.tensor(self.statuses[index], dtype=torch.float32)
        return image, label, landmarks_tensor, status_tensor
    
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