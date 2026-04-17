from torchvision.transforms import Compose
from posixpath import split
import torch 
from torchvision.transforms import v2

def build_transform(config, split="train") -> Compose: # train | val | test
    """Buid transform (augmentatio) for our data
    Args: 
        config: for image size
        split: train | val | test (transform for train is diff from val and test)
    Return: 
        compose: a transform compose
    """
    image_size = config['data']['image_size']
    if split == "train":
        trans = v2.Compose([
            # v2.Grayscale(num_output_channels=1),
            
            # Augmentation
            v2.Resize(size=(image_size, image_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(21),
            # crop image with output shape: (image_size, image_size), small zoom 
            v2.RandomResizedCrop(size=(image_size), scale=(0.8, 1)),

            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), # scale=True: / 255
            v2.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        trans = v2.Compose([
            # v2.Grayscale(num_output_channels=1),
            v2.Resize(size=(image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5])
        ])

    return trans


# With transfer learning: VGG hay ResNet:
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    # create random array
    dummy_pixels = np.random.randint(0, 256, (48, 48), dtype=np.uint8)

    # convert uint8 array to PIL image grayscale ('L' mode)
    dummy_image = Image.fromarray(dummy_pixels)
    
    # expect: L (48, 48) --> ok
    print("Before transform: ", dummy_image.mode, dummy_image.size)

    # create a mock config
    mock_config = {
        'data':{
            'image_size': 48 # change to 224 if using VGG, Q use 48 for basic CNN
        }
    }

    train_trans = build_transform(mock_config, split="train")
    out_tensor = train_trans(dummy_image)

    # 6. Kiểm tra kết quả
    print("Tensor after Transform:")
    print("   - shape = ", out_tensor.shape)       # Kỳ vọng: [1, 48, 48]
    print("   - float32? .dtype = ", out_tensor.dtype)     # Kỳ vọng: torch.float32
    print(f"   - Max (scale & normalize) = {out_tensor.max().item():.3f}")  # Kỳ vọng xoay quanh ~ 1.0
    print(f"   - Min (scale & normalize) = {out_tensor.min().item():.3f}")  # Kỳ vọng xoay quanh ~ -1.0
