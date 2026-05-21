"""Tiny demo runner for Semantic ROI Graph FER."""

import torch

from src.models.semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig


def random_bboxes(batch_size, num_regions):
    bboxes = torch.zeros(batch_size, num_regions, 4)
    for b in range(batch_size):
        for r in range(num_regions):
            x1 = torch.randint(0, 30, (1,)).item()
            y1 = torch.randint(0, 30, (1,)).item()
            x2 = x1 + torch.randint(8, 18, (1,)).item()
            y2 = y1 + torch.randint(8, 18, (1,)).item()
            bboxes[b, r] = torch.tensor([x1, y1, min(x2, 47), min(y2, 47)])
    return bboxes


def main():
    config = SemanticRoiGraphConfig(num_classes=7, num_regions=9, roi_grid=4, feature_dim=256)
    model = SemanticROIGraphFER(config)
    model.eval()

    images = torch.randn(2, 1, 48, 48)
    bboxes = random_bboxes(2, config.num_regions)

    with torch.no_grad():
        outputs = model(images, bboxes)

    for key, value in outputs.items():
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
