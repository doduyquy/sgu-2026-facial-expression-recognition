from PIL import Image

from src.data.transforms import build_transform


def _config(enabled=True):
    return {
        "data": {
            "image_size": 48,
            "channels": 1,
            "normalize": False,
            "class_aware_augmentation": {
                "enabled": enabled,
                "target_labels": [0, 2, 4, 6],
                "extra_prob": 1.0,
                "rotation_degrees": 2.0,
                "brightness": 0.05,
                "contrast": 0.05,
                "gamma_range": [0.9, 1.1],
                "random_erasing_p": 0.0,
            },
        }
    }


def test_class_aware_train_transform_accepts_label():
    transform = build_transform(_config(enabled=True), split="train")
    image = Image.new("L", (48, 48), color=128)

    output = transform(image, label=0)

    assert getattr(transform, "accepts_label", False)
    assert tuple(output.shape) == (1, 48, 48)


def test_class_aware_transform_only_replaces_train_transform():
    val_transform = build_transform(_config(enabled=True), split="val")
    image = Image.new("L", (48, 48), color=128)

    output = val_transform(image)

    assert not getattr(val_transform, "accepts_label", False)
    assert tuple(output.shape) == (1, 48, 48)
