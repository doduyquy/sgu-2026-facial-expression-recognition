from .inception import Inception
from .simple_cnn import SimpleCNN
from .vgg import VGG19, VGGFusionSpatial, VGGFusionCBAM, VGGFusionSpatialCNN
from .transformer_encoder import VGGFusionTransformer, VGGFusionTransformerV2, VGGFusionTransformerEA
from .efficientnet import EfficientNetForFER2013
from .vgg_ea import VGGEA_CNN
from .resnet import ResNet50
from .dual_fusion import VGGResNetAttentionFusion
from .region_attention import RegionAlignedFER
from .resnetBASE import Resnet35
# from .vit_ea_6x6 import VisionTransformerEA_6x6


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
    "vgg_spatial": lambda **kw: VGGFusionSpatial(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_cbam": lambda **kw: VGGFusionCBAM(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_spatial_cnn": lambda **kw: VGGFusionSpatialCNN(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_transformer": lambda **kw: VGGFusionTransformer(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_transformer_v2": lambda **kw: VGGFusionTransformerV2(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_transformer_ea": lambda **kw: VGGFusionTransformerEA(config=kw['config'], channels=kw['config']['data']['channels']),
    # "resnet18": lambda **kw: ResNet(variant="resnet18", **kw),
    # "resnet34": lambda **kw: ResNet(variant="resnet34", **kw),
    "resnet50": lambda **kw: ResNet50(config=kw['config'], channels=kw['config']['data']['channels']),
    "resnet35": lambda **kw: Resnet35(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_resnet_attention": lambda **kw: VGGResNetAttentionFusion(config=kw['config'], channels=kw['config']['data']['channels']),
    "region_aligned_fer": lambda **kw: RegionAlignedFER(config=kw['config'], channels=kw['config']['data']['channels']),
    # "resmaskingnet": ResMaskingNet,
    "efficientnet_fer2013": EfficientNetForFER2013,
    "vgg_ea_cnn": lambda **kw: VGGEA_CNN(config=kw['config'], channels=kw['config']['data']['channels']),
    # "vit_ea_6x6": lambda **kw: VisionTransformerEA_6x6(config=kw['config'], channels=kw['config']['data']['channels']),
}

def get_model(name: str, **kwargs):
    """Factory function: tạo model theo tên trong config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
