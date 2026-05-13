import torch
from torchvision import transforms
from PIL import Image

def build_transform(config, split="train") -> transforms.Compose:
    # 1. Đồng bộ image size theo config của ResNet (224)
    image_size = config['data']['input_size'] 
    
    # 2. Thay đổi tham số Normalize theo ImageNet (dùng cho ResNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if split == "train":
        trans = transforms.Compose([
            # Chuyển ảnh xám ('L') thành ảnh 3 kênh giống tác giả repo thực hiện
            transforms.Grayscale(num_output_channels=3), 
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Scale 224 -> ~256 để cắt lấy 224 cho TenCrop
        resize_dim = int(image_size / 0.875) 
        
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((resize_dim, resize_dim)),
            transforms.TenCrop(image_size), # Cắt 10 ảnh size 224
            transforms.Lambda(lambda crops: torch.stack([
                transforms.ToTensor()(crop) for crop in crops
            ])),  
            transforms.Lambda(lambda tensors: torch.stack([
                transforms.Normalize(mean=mean, std=std)(t) for t in tensors
            ])),  
        ])

    return trans

if __name__ == "__main__":
    import numpy as np

    # Giả lập ảnh xám 1 kênh kích thước gốc là 48x48
    dummy_pixels = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_pixels)

    print("Before transform: ", dummy_image.mode, dummy_image.size)

    # Cập nhật theo file fer2013_config.json của repo
    mock_config = {
        'data':{
            'input_size': 224 
        }
    }

    # Test Train split
    train_trans = build_transform(mock_config, split="train")
    out_tensor_train = train_trans(dummy_image)
    
    # Test TTA split
    test_trans = build_transform(mock_config, split="test")
    out_tensor_test = test_trans(dummy_image)

    print("\n--- TRAIN TENSOR ---")
    print(" - shape = ", out_tensor_train.shape)      # Kỳ vọng: [3, 224, 224]
    
    print("\n--- TEST (TTA) TENSOR ---")
    print(" - shape = ", out_tensor_test.shape)       # Kỳ vọng: [10, 3, 224, 224]