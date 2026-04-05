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
        use_eca_stage34=config['model'].get('use_eca_stage34', config['model'].get('use_cbam_stage34', True)),
        eca_kernel_size=config['model'].get('eca_kernel_size', config['model'].get('cbam_kernel_size', 3)),
        use_arcface=config['model'].get('use_arcface', False),
        arcface_s=config['model'].get('arcface_s', 30.0),
        arcface_m=config['model'].get('arcface_m', 0.5),
        arcface_easy_margin=config['model'].get('arcface_easy_margin', False),
    ),
    # "resmaskingnet": ResMaskingNet,
}

def get_model(name: str, **kwargs):
    """Factory function: tạo model theo tên trong config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
