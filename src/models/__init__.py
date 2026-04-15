from .simple_cnn import SimpleCNN
# from .vgg import VGG
# from .resnet import ResNet
# from .resmaskingnet import ResMaskingNet
from .resnet import ResNet50

"""Hi, guy, tạo model mới thì:
    1. Tạo file src/models/model_name.py
    2. Thêm vào MODEL_REGISTRY ở dưới
    3. Tạo file configs/model_name.yaml để set config cho nó.    
"""

MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    # "vgg11": lambda **kw: VGG(variant="vgg11", **kw),
    # "vgg19": lambda **kw: VGG(variant="vgg19", **kw),
    # "resnet18": lambda **kw: ResNet(variant="resnet18", **kw),
    # "resnet34": lambda **kw: ResNet(variant="resnet34", **kw),
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
    # "resmaskingnet": ResMaskingNet,
}

def get_model(name: str, **kwargs):
    """Factory function: tạo model theo tên trong config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
