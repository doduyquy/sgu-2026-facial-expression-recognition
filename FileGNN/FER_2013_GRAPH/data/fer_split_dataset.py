import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class FERSplitDataset(Dataset):
    """
    Dataset đọc 1 file split riêng:
        - train.csv
        - val.csv
        - test.csv

    Yêu cầu cột:
        - emotion
        - pixels
        - Usage (nếu có)

    Mỗi sample trả:
        {
            "id": int,
            "image": np.ndarray shape (48, 48), dtype float32,
            "label": int,
            "usage": str,
            "split": str
        }
    """

    def __init__(self, csv_path: str, split_name: str, image_size: int = 48):
        self.csv_path = csv_path
        self.split_name = split_name
        self.image_size = image_size

        self.df = pd.read_csv(csv_path)
        self._validate_dataframe()

    def _validate_dataframe(self):
        required_cols = {"emotion", "pixels"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {missing}")

        self.df["emotion"] = self.df["emotion"].astype(int)

        if "Usage" not in self.df.columns:
            self.df["Usage"] = self.split_name

    def __len__(self):
        return len(self.df)

    def _parse_pixels(self, pixel_str: str) -> np.ndarray:
        arr = np.fromstring(str(pixel_str), sep=" ", dtype=np.float32)
        expected = self.image_size * self.image_size

        if arr.size != expected:
            raise ValueError(
                f"Sai số pixel: cần {expected}, nhưng nhận {arr.size}"
            )

        return arr.reshape(self.image_size, self.image_size)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = self._parse_pixels(row["pixels"])
        label = int(row["emotion"])
        usage = str(row["Usage"])

        return {
            "id": idx,
            "image": image,
            "label": label,
            "usage": usage,
            "split": self.split_name,
        }

    def get_class_distribution(self):
        return self.df["emotion"].value_counts().sort_index()

    def summary(self):
        return {
            "split": self.split_name,
            "num_samples": len(self.df),
            "class_distribution": self.get_class_distribution().to_dict(),
        }