from .simple_cnn import SimpleCNN
# from .vgg import VGG
# from .resnet import ResNet
# from .resmaskingnet import ResMaskingNet
from .resnet import ResNetDualBranch

"""Hi, guy, tạo model mới thì:
    1. Tạo file src/models/model_name.py
    2. Thêm vào MODEL_REGISTRY ở dưới
    3. Tạo file configs/model_name.yaml để set config cho nó.    
"""


MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "resnet": lambda config, **kw: ResNet50(
        num_classes=config['data'].get('num_classes', 7),
        in_channels=config['data'].get('in_channels', 1),
        use_learned_landmark_branch=config['model'].get('use_learned_landmark_branch', True),
        landmark_num_points=config['model'].get('landmark_num_points', 6),
        landmark_tau=config['model'].get('landmark_tau', 0.07),
        landmark_feature_dropout_p=config['model'].get('landmark_feature_dropout_p', 0.3),
        landmark_head_dropout_p=config['model'].get('landmark_head_dropout_p', 0.2),
        landmark_edge_guidance_beta=config['model'].get('landmark_edge_guidance_beta', 1.0),
        landmark_edge_alpha=config['model'].get('landmark_edge_alpha', 6.0),
        landmark_edge_feat_guidance_beta=config['model'].get('landmark_edge_feat_guidance_beta', 0.3),
        landmark_edge_dropout_prob=config['model'].get('landmark_edge_dropout_prob', 0.3),
        landmark_edge_head_scale_std=config['model'].get('landmark_edge_head_scale_std', 0.1),
        landmark_edge_mask_threshold=config['model'].get('landmark_edge_mask_threshold', 0.3),
        landmark_edge_gamma=config['model'].get('landmark_edge_gamma', 1.7),
        landmark_from_stage=config['model'].get('landmark_from_stage', 3),
    ),
    "resnet_dual": lambda config, **kw: ResNetDualBranch(
        num_classes=config['data'].get('num_classes', 7),
        use_cbam_stage34=config['model'].get('use_cbam_stage34', True),
        cbam_reduction=config['model'].get('cbam_reduction', 16),
        cbam_kernel_size=config['model'].get('cbam_kernel_size', 7),
    ),
}

def get_model(name: str, **kwargs):
    """Factory function: tạo model theo tên trong config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
