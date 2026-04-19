from torchvision.transforms import Compose
import torch
from torchvision.transforms import v2
from torchvision.transforms import functional as TF


class RandomGamma(torch.nn.Module):
    """
    Random gamma correction for grayscale/RGB facial images.
    Gamma < 1.0  -> brighten
    Gamma > 1.0  -> darken
    """
    def __init__(self, p=0.5, gamma_range=(0.7, 1.4)):
        super().__init__()
        self.p = p
        self.gamma_range = gamma_range

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            gamma = torch.empty(1).uniform_(self.gamma_range[0], self.gamma_range[1]).item()
            img = TF.adjust_gamma(img, gamma=gamma)
        return img


def build_transform(config, split="train") -> Compose:
    """
    Build transform for FER2013 / face expression recognition.

    Args:
        config: config dict
        split: train | val | test

    Return:
        torchvision.transforms.v2.Compose
    """
    image_size = config["data"].get("image_size", 48)
    channels = 1  # Chỉ có 48x48x1 theo yêu cầu

    mean = [0.5]
    std = [0.5]

    if split == "train":
        transform_ops = [
            v2.Resize((image_size, image_size)),

            # FER-safe augmentations
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),

            # Lighting augmentation: nhẹ thôi
            v2.ColorJitter(brightness=0.15, contrast=0.15),

            # Gamma augmentation mạnh hơn để xử lý ảnh quá sáng / quá tối
            RandomGamma(p=0.5, gamma_range=(0.5, 2.0)),

            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]

    else:
        transform_ops = [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]

    return v2.Compose(transform_ops)


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    dummy_pixels = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_pixels)

    print(f"Original image: mode={dummy_image.mode}, size={dummy_image.size}")

    for split in ["train", "val", "test"]:
        print(f"\n--- Testing split={split}, channels=1 ---")

        mock_config = {
            "data": {
                "image_size": 48,
                "channels": 1
            }
        }

        trans = build_transform(mock_config, split=split)
        out_tensor = trans(dummy_image)

        print(f"Tensor shape: {out_tensor.shape}")
        print(f"Tensor dtype : {out_tensor.dtype}")
        print(f"Tensor mean  : {out_tensor.mean().item():.4f}")
        print(f"Tensor std   : {out_tensor.std().item():.4f}")

        assert out_tensor.shape == (1, 48, 48), \
            f"Shape mismatch! Expected (1, 48, 48), got {out_tensor.shape}"

    print("\n[SUCCESS] Transform test passed.")
