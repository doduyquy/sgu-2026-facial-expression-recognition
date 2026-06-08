
import sys
import os
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(r'd:\DIP\Hydric_Graph_CNN')))

from src.models.semantic_roi_graph_fer import SemanticROIGraphFER, SemanticRoiGraphConfig

def test():
    with open(r'd:\DIP\Hydric_Graph_CNN\configs\semantic_roi_graph.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    config = SemanticRoiGraphConfig(**cfg_dict['model'])
    model = SemanticROIGraphFER(config)
    model.eval()
    
    # Mock data
    image = torch.randn(2, 1, 48, 48) # Batch=2, 1 channel, 48x48
    
    bboxes = torch.zeros(2, 9, 4)
    bboxes[:, :, 2] = 10.0
    bboxes[:, :, 3] = 10.0
    
    region_mask = torch.ones(2, 9)
    region_confidence = torch.ones(2, 9)
    
    with torch.no_grad():
        out = model(image, bboxes, region_mask, region_confidence)
    print('SUCCESS! Output keys:', list(out.keys()))
    print('logits shape:', out['logits'].shape)

if __name__ == '__main__':
    test()

