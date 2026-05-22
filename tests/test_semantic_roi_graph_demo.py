"""Smoke test for Semantic ROI Graph FER model."""

import torch
from torch.optim.swa_utils import AveragedModel

from src.models.semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig
from src.models.semantic_roi_graph_losses import compute_semantic_roi_graph_losses


def test_forward_shapes():
    config = SemanticRoiGraphConfig(num_classes=7, num_regions=9, roi_grid=4, feature_dim=256)
    model = SemanticROIGraphFER(config)
    images = torch.randn(2, 1, 48, 48)
    bboxes = torch.tensor(
        [
            [[5, 5, 20, 20]] * 9,
            [[10, 10, 30, 30]] * 9,
        ],
        dtype=torch.float32,
    )
    outputs = model(images, bboxes)

    assert outputs["logits"].shape == (2, 7)
    assert outputs["logits_motif"].shape == (2, 7)
    assert outputs["logits_global"].shape == (2, 7)
    assert outputs["micro_motif_attention"].shape[0] == 2
    assert outputs["region_embeddings"].shape[1] == 9
    assert outputs["macro_embeddings"].shape[1] == 9


def test_forward_handles_tencrop_batches():
    config = SemanticRoiGraphConfig(num_classes=7, num_regions=9, roi_grid=4, feature_dim=256, use_pretrained=False)
    model = SemanticROIGraphFER(config)
    images = torch.randn(2, 10, 1, 48, 48)
    bboxes = torch.tensor(
        [
            [[5, 5, 20, 20]] * 9,
            [[10, 10, 30, 30]] * 9,
        ],
        dtype=torch.float32,
    )

    outputs = model(images, bboxes)

    assert outputs["logits"].shape == (2, 7)
    assert outputs["region_embeddings"].shape == (2, 9, 256)


def test_semantic_losses_support_averaged_model_wrapper():
    config = SemanticRoiGraphConfig(num_classes=7, num_regions=9, roi_grid=4, feature_dim=256, use_pretrained=False)
    model = SemanticROIGraphFER(config)
    ema_model = AveragedModel(model)
    images = torch.randn(2, 1, 48, 48)
    bboxes = torch.tensor(
        [
            [[5, 5, 20, 20]] * 9,
            [[10, 10, 30, 30]] * 9,
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    outputs = ema_model(images, bboxes)
    loss_dict = compute_semantic_roi_graph_losses(ema_model, outputs, labels)

    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"])
