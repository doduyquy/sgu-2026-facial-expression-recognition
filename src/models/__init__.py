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
        landmark_num_points=config['model'].get('landmark_num_points', 12),
        landmark_tau=config['model'].get('landmark_tau', 0.03),
        landmark_feature_dropout_p=config['model'].get('landmark_feature_dropout_p', 0.3),
        landmark_mask_prob=config['model'].get('landmark_mask_prob', 0.2),
        landmark_prior_strength=config['model'].get('landmark_prior_strength', 0.05),
        landmark_prior_sigma=config['model'].get('landmark_prior_sigma', 0.22),
        landmark_keypoint_dropout_p=config['model'].get('landmark_keypoint_dropout_p', 0.1),
        landmark_prior_min_strength=config['model'].get('landmark_prior_min_strength', 0.0),
        landmark_prior_anneal_power=config['model'].get('landmark_prior_anneal_power', 1.5),
        landmark_part_mask_expand=config['model'].get('landmark_part_mask_expand', 0.08),
        landmark_part_target_inside=config['model'].get('landmark_part_target_inside', 0.35),
        landmark_prior_disable_after_progress=config['model'].get('landmark_prior_disable_after_progress', 0.3),
        landmark_use_cross_keypoint_competition=config['model'].get('landmark_use_cross_keypoint_competition', False),
        landmark_post_softmax_sharpness=config['model'].get('landmark_post_softmax_sharpness', 1.3),
        landmark_use_soft_face_mask=config['model'].get('landmark_use_soft_face_mask', True),
        landmark_face_mask_strength=config['model'].get('landmark_face_mask_strength', 0.15),
        landmark_use_dynamic_patch_localization=config['model'].get('landmark_use_dynamic_patch_localization', True),
        landmark_patch_window_sigma=config['model'].get('landmark_patch_window_sigma', 0.22),
        landmark_patch_gate_strength=config['model'].get('landmark_patch_gate_strength', 0.7),
        landmark_patch_center_detach_for_gate=config['model'].get('landmark_patch_center_detach_for_gate', False),
        landmark_from_stage=config['model'].get('landmark_from_stage', 3),
    ),
    # "resmaskingnet": ResMaskingNet,
}

def get_model(name: str, **kwargs):
    """Factory function: tạo model theo tên trong config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
