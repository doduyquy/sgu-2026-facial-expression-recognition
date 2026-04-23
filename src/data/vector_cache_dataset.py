"""
src/data/vector_cache_dataset.py

Dataset nhỏ gọn cho MLP Baseline — đọc từ pre-computed vector cache.

File cache format: {'x': Tensor[N, D], 'y': Tensor[N], 'split': str}
Được build bởi scripts/build_vector_cache.py (~1MB thay vì ~24GB)
"""
import torch
from torch.utils.data import Dataset


class VectorCacheDataset(Dataset):
    """
    Load pre-computed graph vectors (9-dim) từ file .pt nhỏ gọn.
    Không load PixelGraph đầy đủ → RAM-safe cho Kaggle.

    Output mỗi sample:
        {
            "x": torch.FloatTensor [D],    # graph-level vector
            "y": torch.LongTensor  []      # label 0-6
        }
    """

    def __init__(self, vector_path: str):
        data = torch.load(vector_path, map_location="cpu", weights_only=True)

        if not isinstance(data, dict) or "x" not in data or "y" not in data:
            raise ValueError(
                f"File {vector_path} phải có format {{'x': Tensor, 'y': Tensor}}.\n"
                f"Hãy build bằng scripts/build_vector_cache.py"
            )

        self.x = data["x"]   # [N, D]
        self.y = data["y"]   # [N]
        self.split = data.get("split", "unknown")

        assert len(self.x) == len(self.y), "x và y phải cùng số lượng mẫu"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
        }

    def get_input_dim(self) -> int:
        return self.x.shape[1]

    def get_num_classes(self) -> int:
        return int(self.y.max().item()) + 1
