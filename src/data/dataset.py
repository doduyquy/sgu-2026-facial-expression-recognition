from PIL.Image import fromarray
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from src.data.emotions_dict import EMOTION_DICT
import torchvision.transforms.functional as TF


class FER2013(Dataset):
    """Load one sample for dataloader"""

    def __init__(self, data_path, split="train", transforms=None):
        # set relative path to train|val|test in dataset
        self.data_split_path = os.path.join(data_path, f"{split}.csv")
        # because Q splitted dataset, so we only need 2 column: emotion(for category) and pixels for images
        self.data = pd.read_csv(self.data_split_path, usecols=[0, 1])
        
        # Keep original index to apply specific transformations and filtering
        self.split = split
        self.data['original_idx'] = self.data.index
        
        if self.split == 'train':
            IGNORE_INDICES = set([59, 2059, 2809, 3262, 3931, 4275, 5274, 5439, 5881, 6458, 7172, 7496, 7629, 8737, 8856, 9026, 9679, 10423, 11286, 11846, 12352, 13148, 13402, 13988, 14279, 15144, 15838, 15894, 17081, 19238, 19632, 20222, 20712, 20817, 21817, 22198, 22927, 24891, 25219, 25909, 26383, 26897, 28601])
            self.GAMMA_DARK_INDICES = set([422, 821, 1526, 1645, 1839, 2181, 3233, 3508, 3822, 4102, 4615, 5201, 5527, 5534, 6051, 6185, 6660, 7117, 7671, 8145, 8424, 9213, 9404, 9868, 10431, 10560, 11046, 11792, 11800, 12365, 12579, 13156, 13314, 13367, 13377, 13464, 13923, 15655, 15702, 15795, 16483, 16691, 17077, 17284, 17701, 18140, 18737, 19461, 20248, 20288, 20417, 20544, 20561, 20927, 21093, 21194, 21387, 21672, 21684, 21786, 23424, 23731, 23738, 23757, 24493, 25082, 25205, 25436, 26431, 26541, 27299, 27605, 27735])
            self.GAMMA_BRIGHT_INDICES = set([163, 276, 1211, 2501, 2548, 3791, 4358, 4481, 5023, 5243, 5595, 5623, 5714, 5905, 6624, 6668, 6681, 6716, 7497, 8182, 8540, 8581, 8594, 9017, 9757, 10184, 10315, 10580, 12075, 12479, 12775, 12794, 12947, 13072, 14461, 14476, 15024, 15542, 16021, 16454, 17555, 17853, 18196, 18230, 18262, 19176, 20354, 20929, 21293, 22351, 23249, 23497, 23590, 24948, 25003, 25566, 26624, 26795, 27666, 28268, 28335])
            
            # Remove the ignored rows completely from self.data
            self.data.drop(index=list(IGNORE_INDICES), inplace=True, errors='ignore')
            
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

        # Apply specific gamma correction dynamically
        if self.split == 'train':
            if original_idx in self.GAMMA_DARK_INDICES:
                image = TF.adjust_gamma(image, gamma=0.5)
            elif original_idx in self.GAMMA_BRIGHT_INDICES:
                image = TF.adjust_gamma(image, gamma=2.0)

        # apply transform if it not None
        if self.transform is not None:
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
