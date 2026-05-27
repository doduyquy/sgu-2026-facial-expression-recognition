import sys, os
sys.path.insert(0, os.path.abspath('.'))
from src.models.semantic_roi_graph_fer import SemanticRoiGraphConfig, SemanticROIGraphFER
import torch

def main():
    cfg = SemanticRoiGraphConfig()
    cfg.use_pretrained = False
    cfg.use_layer4 = False
    cfg.feature_dim = 256
    m = SemanticROIGraphFER(cfg)
    m.eval()
    img = torch.randn(2,1,48,48)
    with torch.no_grad():
        out = m(img)
    print('logits', out['logits'].shape)
    print('logits_motif', out['logits_motif'].shape)
    print('logits_fused', out['logits_fused'].shape)
    print('structure_gate', out['structure_gate'].shape)

if __name__ == '__main__':
    main()
