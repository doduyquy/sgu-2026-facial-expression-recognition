from torchvision.transforms import Compose
import torch
from torchvision import transforms


def build_transform(config, split="train") -> Compose:
    """Build transforms for train / val / test.

    Fix 1: config key is 'input_size' (not 'image_size'). Falls back to
            'image_size' for backward compatibility with older configs.
    Fix 6: When semantic masks are used, TenCrop is incompatible because
            bounding-box coordinates are defined in the original image space
            and become invalid after any spatial crop. Simple resize is used
            instead so that bbox coordinates stay correct.

    Args:
        config: full config dict
        split: 'train' | 'val' | 'test'

    Returns:
        Compose transform pipeline
    """
    # Fix 1: accept both key names
    data_cfg = config.get('data', {})
    image_size = data_cfg.get('input_size', data_cfg.get('image_size', 48))
    mu = 0.5
    st = 0.5

    use_semantic_masks = bool(data_cfg.get('use_semantic_masks', False))

    if split == "train":
        trans = transforms.Compose([
            transforms.Resize((image_size, image_size)), # An toàn: Giữ nguyên tọa độ bboxes
            transforms.RandomHorizontalFlip(p=0.5),      # Chấp nhận được vì khuôn mặt có tính đối xứng
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Đổi ánh sáng thay vì tọa độ
            transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.5, 2.0), value=0) # Che khuất ngẫu nhiên (Cutout)
        ])
    else:
        if use_semantic_masks:
            # Fix 6: no TenCrop — semantic bbox coords are in original image space
            # and would be wrong after any spatial crop. Simple resize preserves coords.
            trans = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(mu,), std=(st,)),
            ])
        else:
            # TenCrop TTA for models that do not use bounding boxes.
            # Note: TenCrop order is [tl, tr, bl, br, center, ...flips].
            # Center crop is at index 4, NOT image.size(1)//2 = 5.
            larger = int(image_size * 56 / 48)
            trans = transforms.Compose([
                transforms.Resize((larger, larger)),
                transforms.TenCrop(image_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(c) for c in crops]
                )),
                transforms.Lambda(lambda tensors: torch.stack(
                    [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]
                )),
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