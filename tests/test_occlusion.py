import torch

from src.training.occlusion import RegionOcclusionGenerator


def test_region_occlusion_generator_masks_selected_samples():
    torch.manual_seed(7)
    images = torch.zeros(4, 3, 32, 32)
    generator = RegionOcclusionGenerator(
        {
            "apply_prob": 1.0,
            "min_area": 0.10,
            "max_area": 0.10,
            "fill_value": 0.5,
            "policy": "mixed_face_regions",
        }
    )

    masked, applied = generator(images)

    assert masked.shape == images.shape
    assert applied.tolist() == [True, True, True, True]
    assert torch.count_nonzero(masked).item() > 0
    assert torch.all(masked >= 0.0)
    assert torch.all(masked <= 0.5)


def test_region_occlusion_generator_can_be_disabled_by_probability():
    images = torch.ones(2, 3, 24, 24)
    generator = RegionOcclusionGenerator(
        {
            "apply_prob": 0.0,
            "fill_value": 0.5,
        }
    )

    masked, applied = generator(images)

    assert torch.equal(masked, images)
    assert applied.tolist() == [False, False]
