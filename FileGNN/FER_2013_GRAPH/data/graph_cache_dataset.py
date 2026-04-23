import torch
from torch.utils.data import Dataset


class GraphCacheDataset(Dataset):
    """
    Dataset đọc danh sách PixelGraph đã lưu trong file .pt.

    Mỗi sample trả về:
        {
            "graph": PixelGraph
        }
    """

    def __init__(self, graph_path: str):
        self.graphs = torch.load(graph_path)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return {"graph": self.graphs[idx]}