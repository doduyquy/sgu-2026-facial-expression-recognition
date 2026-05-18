from torchvision.transforms import Compose
from posixpath import split
import torch 
from torchvision import transforms

def build_transform(config, split="train") -> Compose: # train | val | test
    """Build transform matching SOTA pipeline
    
    Args: 
        config: for image size
        split: train | val | test (transform for train is diff from val and test)
    Return: 
        compose: a transform compose
    """
    image_size = config['data']['image_size']
    mu = 0.5
    st = 0.5
    
    if split == "train":
        # Standard augmentation matching SOTA pipeline
        # LƯU Ý QUAN TRỌNG: RandomHorizontalFlip đã được thực hiện thủ công bên trong dataset.py 
        # để đồng bộ hóa chính xác với việc lật tọa độ Landmark (tránh lệch pha ảnh và tọa độ).
        trans = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random'),
            transforms.Normalize(mean=(mu,), std=(st,))
        ])
    else:
        # Clean validation/testing pipeline (NO TenCrop, NO augment)
        trans = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,))
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