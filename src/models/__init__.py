from .inception import Inception
from .simple_cnn import SimpleCNN
from .vgg import VGG19, VGGFusionSpatial, VGGFusionCBAM, VGGFusionSpatialCNN
from .transformer_encoder import VGGFusionTransformer, VGGFusionTransformerV2, VGGFusionTransformerEA
from .efficientnet import EfficientNetForFER2013
from .vgg_ea import VGGEA_CNN
from .resnet import ResNet50, ResNet152
from .dual_fusion import VGGResNetAttentionFusion
from .region_attention import RegionAlignedFER
from .resnet_region_attention import ResNetRegionAlignedFER
from .resnet152_region_attention import ResNet152RegionAttentionFER
from .convnext_region_attention import ConvNeXtRegionAttentionFER
from .resnet152_landmark_attention import ResNet152LandmarkAttentionFER
from .resnet152_unet_mask_attention import ResNet152UNetMaskAttentionFER
from .swin_region_attention import SwinRegionAlignedFER
from .torchvision_cnn import TorchvisionCNNFER
from .GraphNN import MotifGNN
from .PixelGNN import PixelGNN
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
    "resnet152": lambda **kw: ResNet152(config=kw['config'], channels=kw['config']['data']['channels']),
    "vgg_resnet_attention": lambda **kw: VGGResNetAttentionFusion(config=kw['config'], channels=kw['config']['data']['channels']),
    "region_aligned_fer": lambda **kw: RegionAlignedFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "resnet_region_aligned_fer": lambda **kw: ResNetRegionAlignedFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "resnet152_region_attention": lambda **kw: ResNet152RegionAttentionFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "convnext_tiny_region_attention": lambda **kw: ConvNeXtRegionAttentionFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "convnext_tiny_mask_guided_region_attention": lambda **kw: ConvNeXtRegionAttentionFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "resnet152_landmark_attention": lambda **kw: ResNet152LandmarkAttentionFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "resnet152_unet_mask_attention": lambda **kw: ResNet152UNetMaskAttentionFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "swin_region_aligned_fer": lambda **kw: SwinRegionAlignedFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "torchvision_cnn": lambda **kw: TorchvisionCNNFER(config=kw['config'], channels=kw['config']['data']['channels']),
    "convnext_tiny_fer2013": lambda **kw: TorchvisionCNNFER(config=kw['config'], channels=kw['config']['data']['channels'], arch="convnext_tiny"),
    "efficientnetv2_s_fer2013": lambda **kw: TorchvisionCNNFER(config=kw['config'], channels=kw['config']['data']['channels'], arch="efficientnet_v2_s"),
    "regnet_y_8gf_fer2013": lambda **kw: TorchvisionCNNFER(config=kw['config'], channels=kw['config']['data']['channels'], arch="regnet_y_8gf"),
    "motif_gnn": lambda **kw: MotifGNN(config=kw['config'], channels=kw['config']['data']['channels']),
    "pixel_gnn": lambda **kw: PixelGNN(config=kw['config'], channels=kw['config']['data']['channels']),
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
