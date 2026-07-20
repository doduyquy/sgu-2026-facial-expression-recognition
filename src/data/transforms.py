from torchvision.transforms import Compose
import torch
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms import functional as TF


class ToChannels(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mode = 'RGB' if channels == 3 else 'L'

    def forward(self, img):
        return img.convert(self.mode)


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


def _pair_from_config(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError("Expected a two-value list/tuple.")


class ClassAwareAugment(torch.nn.Module):
    """
    Extra light augmentation for selected labels.

    The normal train augmentation still applies to every sample. This module
    only adds a small extra perturbation to hard classes when enabled by config.
    """

    def __init__(self, config):
        super().__init__()
        cfg = config.get("data", {}).get("class_aware_augmentation", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.target_labels = {int(label) for label in cfg.get("target_labels", [])}
        self.extra_prob = float(cfg.get("extra_prob", 0.35))
        self.rotation_degrees = float(cfg.get("rotation_degrees", 5.0))
        self.brightness = float(cfg.get("brightness", 0.10))
        self.contrast = float(cfg.get("contrast", 0.10))
        self.gamma_range = _pair_from_config(cfg.get("gamma_range", None), (0.7, 1.5))
        self.random_erasing_p = float(cfg.get("random_erasing_p", 0.15))
        self.erase_scale = _pair_from_config(cfg.get("erase_scale", None), (0.02, 0.08))
        self.erase_value = cfg.get("erase_value", "random")

        self.color_jitter = (
            v2.ColorJitter(brightness=self.brightness, contrast=self.contrast)
            if self.brightness > 0.0 or self.contrast > 0.0
            else None
        )
        self.random_gamma = (
            RandomGamma(p=1.0, gamma_range=self.gamma_range)
            if self.gamma_range is not None
            else None
        )
        self.random_erasing = (
            v2.RandomErasing(p=1.0, scale=self.erase_scale, value=self.erase_value)
            if self.random_erasing_p > 0.0
            else None
        )

    def sample(self, label):
        if not self.enabled:
            return None
        if label is None:
            return None
        if self.target_labels and int(label) not in self.target_labels:
            return None
        if torch.rand(1).item() >= self.extra_prob:
            return None

        angle = 0.0
        if self.rotation_degrees > 0.0:
            angle = torch.empty(1).uniform_(
                -self.rotation_degrees,
                self.rotation_degrees,
            ).item()

        return {
            "angle": angle,
            "erase": (
                self.random_erasing is not None
                and torch.rand(1).item() < self.random_erasing_p
            ),
        }

    def apply_to_image(self, img, params):
        if params is None:
            return img

        angle = params.get("angle", 0.0)
        if angle:
            img = TF.rotate(
                img,
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
            )
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        if self.random_gamma is not None:
            img = self.random_gamma(img)
        return img

    def apply_to_masks(self, masks, params):
        if params is None:
            return masks

        angle = params.get("angle", 0.0)
        if angle:
            masks = TF.rotate(
                masks,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
            )
            masks = masks.clamp_(0.0, 1.0)
        return masks

    def apply_to_tensor(self, img, params):
        if params is None or not params.get("erase", False):
            return img
        return self.random_erasing(img)


class LabelAwareTrainTransform(torch.nn.Module):
    """Train transform that can inspect the class label."""

    def __init__(self, config):
        super().__init__()
        self.accepts_label = True
        self.image_size = config["data"].get("image_size", 48)
        self.channels = config["data"].get("channels", 1)
        self.normalize = config["data"].get("normalize", True)

        if self.channels == 3:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5]
            self.std = [0.5]

        self.to_channels = ToChannels(self.channels)
        self.resize = v2.Resize((self.image_size, self.image_size))
        self.color_jitter = v2.ColorJitter(brightness=0.15, contrast=0.15)
        self.random_gamma = RandomGamma(p=0.5, gamma_range=(0.5, 2.0))
        self.random_erasing = v2.RandomErasing(p=0.4, scale=(0.02, 0.15), value="random")
        self.class_aware = ClassAwareAugment(config)

    def forward(self, img, label=None):
        img = self.to_channels(img)
        img = self.resize(img)

        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)

        img = TF.rotate(
            img,
            angle=torch.empty(1).uniform_(-15.0, 15.0).item(),
            interpolation=InterpolationMode.NEAREST,
        )
        img = self.color_jitter(img)
        img = self.random_gamma(img)

        class_aware_params = self.class_aware.sample(label)
        img = self.class_aware.apply_to_image(img, class_aware_params)

        img = v2.ToImage()(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        if self.normalize:
            img = v2.Normalize(mean=self.mean, std=self.std)(img)

        img = self.random_erasing(img)
        img = self.class_aware.apply_to_tensor(img, class_aware_params)
        return img


class LandmarkPairedTransform(torch.nn.Module):
    """
    Apply the same geometric train-time augmentation to an image and its masks.

    Color / intensity operations remain image-only, while horizontal flips and
    rotations are mirrored onto the soft landmark masks so the guidance stays
    aligned with the transformed face.
    """

    def __init__(self, config, split="train"):
        super().__init__()
        self.accepts_masks = True
        self.split = split
        self.image_size = config["data"].get("image_size", 48)
        self.channels = config["data"].get("channels", 1)
        self.normalize = config["data"].get("normalize", True)
        train_aug_cfg = config.get("augmentation", {}).get("train", {})
        paired_aug_cfg = config.get("data", {}).get("paired_augmentation", {})
        aug_cfg = {**train_aug_cfg, **paired_aug_cfg}
        self.train_aug_enabled = bool(aug_cfg.get("enabled", True))
        self.hflip_prob = float(aug_cfg.get("hflip_prob", 0.5))
        self.rotation_degrees = float(aug_cfg.get("rotation_degrees", 15.0))
        self.gamma_prob = float(aug_cfg.get("gamma_prob", 0.5))
        self.gamma_range = _pair_from_config(aug_cfg.get("gamma_range", None), (0.5, 2.0))
        self.random_erasing_prob = float(aug_cfg.get("random_erasing_prob", 0.4))
        self.random_erasing_scale = _pair_from_config(
            aug_cfg.get("random_erasing_scale", None),
            (0.02, 0.15),
        )
        self.random_erasing_value = aug_cfg.get("random_erasing_value", "random")
        jitter_cfg = aug_cfg.get("color_jitter", {})
        if jitter_cfg is None:
            jitter_cfg = {}
        self.color_jitter_brightness = float(jitter_cfg.get("brightness", 0.15))
        self.color_jitter_contrast = float(jitter_cfg.get("contrast", 0.15))
        self.color_jitter_saturation = float(jitter_cfg.get("saturation", 0.0))
        self.color_jitter_hue = float(jitter_cfg.get("hue", 0.0))

        if self.channels == 3:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5]
            self.std = [0.5]

        self.to_channels = ToChannels(self.channels)
        self.resize = v2.Resize((self.image_size, self.image_size))
        self.color_jitter = v2.ColorJitter(
            brightness=self.color_jitter_brightness,
            contrast=self.color_jitter_contrast,
            saturation=self.color_jitter_saturation,
            hue=self.color_jitter_hue,
        )
        self.random_gamma = RandomGamma(p=self.gamma_prob, gamma_range=self.gamma_range)
        self.random_erasing = v2.RandomErasing(
            p=self.random_erasing_prob,
            scale=self.random_erasing_scale,
            value=self.random_erasing_value,
        )
        self.accepts_label = True
        self.class_aware = ClassAwareAugment(config)

    def forward(self, img, masks, label=None):
        img = self.to_channels(img)
        img = self.resize(img)

        if self.split == "train" and self.train_aug_enabled:
            if self.hflip_prob > 0.0 and torch.rand(1).item() < self.hflip_prob:
                img = TF.hflip(img)
                masks = torch.flip(masks, dims=[-1])

            if self.rotation_degrees > 0.0:
                angle = torch.empty(1).uniform_(
                    -self.rotation_degrees,
                    self.rotation_degrees,
                ).item()
                img = TF.rotate(
                    img,
                    angle=angle,
                    interpolation=InterpolationMode.NEAREST,
                )
                masks = TF.rotate(
                    masks,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                )
                masks = masks.clamp_(0.0, 1.0)

            img = self.color_jitter(img)
            img = self.random_gamma(img)

            class_aware_params = self.class_aware.sample(label)
            img = self.class_aware.apply_to_image(img, class_aware_params)
            masks = self.class_aware.apply_to_masks(masks, class_aware_params)
        else:
            class_aware_params = None

        img = v2.ToImage()(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        if self.normalize:
            img = v2.Normalize(mean=self.mean, std=self.std)(img)

        if self.split == "train" and self.train_aug_enabled:
            img = self.random_erasing(img)
            img = self.class_aware.apply_to_tensor(img, class_aware_params)

        return img, masks


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
    channels = config["data"].get("channels", 1)

    normalize = config["data"].get("normalize", True)
    class_aware_cfg = config.get("data", {}).get("class_aware_augmentation", {})

    if split == "train" and class_aware_cfg.get("enabled", False):
        return LabelAwareTrainTransform(config)

    if channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5]
        std = [0.5]

    common_ops = [
        ToChannels(channels),
        v2.Resize((image_size, image_size)),
    ]

    if split == "train":
        transform_ops = common_ops + [

            # FER-safe augmentations
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),

            # Lighting augmentation: nhẹ thôi
            v2.ColorJitter(brightness=0.15, contrast=0.15),

            # Gamma augmentation mạnh hơn để xử lý ảnh quá sáng / quá tối
            RandomGamma(p=0.5, gamma_range=(0.5, 2.0)),

            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if normalize:
            transform_ops.append(v2.Normalize(mean=mean, std=std))

        transform_ops += [
            # Thêm Random Erasing (Cutout) chống overfitting & ép Region Attention
            v2.RandomErasing(p=0.4, scale=(0.02, 0.15), value='random'),
        ]

    else:
        transform_ops = common_ops + [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if normalize:
            transform_ops.append(v2.Normalize(mean=mean, std=std))

    return v2.Compose(transform_ops)


def build_landmark_transform(config, split="train") -> LandmarkPairedTransform:
    return LandmarkPairedTransform(config, split=split)


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
