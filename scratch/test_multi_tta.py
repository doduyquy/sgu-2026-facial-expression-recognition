import sys
sys.path.append('.')
import torch
from src.models.utils import apply_multi_scale_tta
import torch.nn as nn

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, images, bboxes=None, region_mask=None, region_confidence=None):
        return {"logits": torch.randn(images.shape[0], 7)}

model = MockModel()

images = torch.randn(4, 1, 48, 48)
bboxes = torch.tensor([
    [
        [10.0, 10.0, 20.0, 20.0],
        [25.0, 10.0, 35.0, 20.0], # right eye
        [25.0, 10.0, 35.0, 20.0],
        [25.0, 10.0, 35.0, 20.0],
        [25.0, 10.0, 35.0, 20.0],
        [25.0, 10.0, 35.0, 20.0],
        [25.0, 10.0, 35.0, 20.0],
        [25.0, 10.0, 35.0, 20.0],
        [25.0, 10.0, 35.0, 20.0],
    ]
] * 4)

region_mask = torch.ones(4, 9, 1, 12, 12)
region_confidence = torch.ones(4, 9, 1, 1)

try:
    print("Testing apply_multi_scale_tta...")
    outputs = apply_multi_scale_tta(model, images, bboxes, region_mask, region_confidence, scale=1.05)
    print("Logits shape:", outputs["logits"].shape)
    print("Done!")
except Exception as e:
    import traceback
    traceback.print_exc()
