from .inception import Inception
from .simple_cnn import SimpleCNN
from .vgg import VGG19, VGGFusion
from .efficientnet import EfficientNetForFER2013
# from .resnet import ResNet
# from .resmaskingnet import ResMaskingNet


"""Hi, guy, tạo model mới thì:
    1. Tạo file src/models/model_name.py
    2. Thêm vào MODEL_REGISTRY ở dưới
    3. Tạo file configs/model_name.yaml để set config cho nó.    
"""

MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "inception": Inception,
    # "vgg11": lambda **kw: VGG(variant="vgg11", **kw),
    "vgg19": lambda **kw: VGG19(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_fusion": lambda **kw: VGGFusion(config=kw['config'], channels=kw['config']['data']['channels']),
    # "resnet18": lambda **kw: ResNet(variant="resnet18", **kw),
    # "resnet34": lambda **kw: ResNet(variant="resnet34", **kw),
    # "resnet50": lambda **kw: ResNet(variant="resnet50", **kw),
    # "resmaskingnet": ResMaskingNet,
    "efficientnet_fer2013": EfficientNetForFER2013,
}

def get_model(name: str, **kwargs):
    """Factory function: tạo model theo tên trong config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
