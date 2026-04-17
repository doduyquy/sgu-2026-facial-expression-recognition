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
    image_size = config['data'].get('image_size', 224)
    channels = config['data'].get('channels', 3)
    mean = config['data'].get('mean', None)
    std = config['data'].get('std', None)

    if mean is None or std is None:
        if channels == 3:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5]
            std = [0.5]

    if split == "train":
        transform_ops = [
            # Nếu channels=3, convert ảnh xám sang RGB (3 kênh giống nhau)
            v2.Lambda(lambda x: x.convert('RGB')) if channels == 3 else v2.Lambda(lambda x: x),
            
            v2.Resize(size=(image_size, image_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Thêm tịnh tiến và thu phóng
            v2.ColorJitter(brightness=0.15, contrast=0.15), # Thêm biến đổi ánh sáng

            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)) # Xóa vùng ảnh (chống overfit)
        ]
    else:
        transform_ops = [
            v2.Lambda(lambda x: x.convert('RGB')) if channels == 3 else v2.Lambda(lambda x: x),
            v2.Resize(size=(image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ]

    # Loại bỏ Lambda(identity) nếu channels=1 để tối ưu
    if channels == 1:
        transform_ops = [op for op in transform_ops if not isinstance(op, v2.Lambda)]

    return v2.Compose(transform_ops)


# With transfer learning: VGG hay ResNet:
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    # create random array
    dummy_pixels = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_pixels)
    
    print(f"Original image: {dummy_image.mode} {dummy_image.size}")

    for ch in [1, 3]:
        print(f"\n--- Testing Scenario: {ch} Channel(s) ---")
        mock_config = {
            'data': {
                'image_size': 224,
                'channels': ch
            }
        }

        trans = build_transform(mock_config, split="train")
        out_tensor = trans(dummy_image)

        print(f"   - Channels config: {ch}")
        print(f"   - Tensor shape: {out_tensor.shape}")
        print(f"   - Standard deviation: {out_tensor.std().item():.3f}")
        
        # Verify shape
        assert out_tensor.shape == (ch, 224, 224), f"Shape mismatch! Expected ({ch}, 224, 224), got {out_tensor.shape}"

    print("\n[SUCCESS] Transforms test passed for both scenarios!")

