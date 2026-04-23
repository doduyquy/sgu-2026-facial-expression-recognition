import torch
from torch.utils.data import Dataset
import numpy as np

from data.graph_cache_dataset import GraphCacheDataset
from features.graph_vectorizer import GraphVectorizer


class GraphVectorDataset(Dataset):
    """
    Dataset chuyển PixelGraph -> graph vector + label.

    Output mỗi sample:
        {
            "x": torch.FloatTensor [D],
            "y": torch.LongTensor []
        }
    """

    def __init__(self, graph_path: str, vectorizer: GraphVectorizer):
        self.base_dataset = GraphCacheDataset(graph_path)
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        graph = self.base_dataset[idx]["graph"]

        x = self.vectorizer.transform(graph)   # np.ndarray [D]
        y = int(graph.label)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }

    def get_input_dim(self) -> int:
        graph = self.base_dataset[0]["graph"]
        node_feature_dim = graph.node_features.shape[1]
        return self.vectorizer.infer_output_dim(node_feature_dim)

    def get_num_classes(self) -> int:
        labels = set()
        for i in range(len(self.base_dataset)):
            labels.add(int(self.base_dataset[i]["graph"].label))
        return len(labels)