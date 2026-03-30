import torch
from torch.utils.data import Dataset

EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

class FER2013(Dataset):
    """Load one sample for dataloader"""

    def __init__(self, args):
        self.dataset_name =     