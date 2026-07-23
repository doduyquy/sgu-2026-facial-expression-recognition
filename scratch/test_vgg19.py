import sys
sys.path.append('.')
import torch
import yaml
from src.models.semantic_roi_graph import SemanticROIGraphFER, SemanticRoiGraphConfig

print("Testing VGG19-BN Backbone integration...")
try:
    with open('configs/semantic_roi_graph.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = SemanticRoiGraphConfig(**config_dict['model'])
    assert config.backbone_type == 'vgg19_bn', "Config not updated correctly!"
    
    model = SemanticROIGraphFER(config)
    print("Model instantiated successfully.")
    
    # Test forward pass
    images = torch.randn(2, 1, 48, 48)
    bboxes = torch.tensor([
        [
            [10.0, 10.0, 20.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
            [25.0, 10.0, 35.0, 20.0],
        ]
    ] * 2)
    region_mask = torch.ones(2, 9)
    region_confidence = torch.ones(2, 9)
    
    outputs = model(images, bboxes, region_mask, region_confidence)
    print("Logits shape:", outputs["logits"].shape)
    
    print("All tests passed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
