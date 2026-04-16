from torchvision.transforms import Compose
from posixpath import split
import torch 
from torchvision.transforms import v2
import torch.nn.functional as F


class SobelConcat:
    """Create a second channel using Sobel edge magnitude and concat with grayscale image."""

    def __init__(self, edge_scale=0.5, blur_kernel_size=3):
        self.edge_scale = edge_scale
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, H, W) in [0, 1]
        if x.ndim != 3 or x.shape[0] != 1:
            return x

        img = x.unsqueeze(0)

        if self.blur_kernel_size >= 3:
            # Lightweight Gaussian-like smoothing to suppress FER noise before Sobel.
            blur_kernel = torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                dtype=img.dtype,
                device=img.device,
            )
            blur_kernel = (blur_kernel / blur_kernel.sum()).view(1, 1, 3, 3)
            img = F.conv2d(img, blur_kernel, padding=1)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=img.dtype,
            device=img.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=img.dtype,
            device=img.device,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        edge = edge / edge.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        edge = torch.clamp(edge * self.edge_scale, min=0.0, max=1.0)

        x_out = torch.cat([x, edge.squeeze(0)], dim=0)
        return x_out

def build_transform(config, split="train") -> Compose: # train | val | test
    """Buid transform (augmentatio) for our data
    Args: 
        config: for image size
        split: train | val | test (transform for train is diff from val and test)
    Return: 
        compose: a transform compose
    """
    image_size = config['data']['image_size']
    use_sobel_channel = config['data'].get('use_sobel_channel', False)
    sobel_edge_scale = config['data'].get('sobel_edge_scale', 0.5)
    sobel_blur_kernel_size = config['data'].get('sobel_blur_kernel_size', 3)

    normalize = v2.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]) if use_sobel_channel else v2.Normalize(mean=[0.5], std=[0.5])
    maybe_sobel = [SobelConcat(edge_scale=sobel_edge_scale, blur_kernel_size=sobel_blur_kernel_size)] if use_sobel_channel else []

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
            *maybe_sobel,
            normalize,
        ])
    else:
        trans = v2.Compose([
            # v2.Grayscale(num_output_channels=1),
            v2.Resize(size=(image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            *maybe_sobel,
            normalize,
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

