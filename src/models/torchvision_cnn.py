import torch.nn as nn
import torchvision.models as models


class TorchvisionCNNFER(nn.Module):
    """Generic torchvision CNN classifier for FER2013 ensemble members."""

    _WEIGHT_ENUMS = {
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
        "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
        "efficientnet_v2_m": "EfficientNet_V2_M_Weights",
        "regnet_y_8gf": "RegNet_Y_8GF_Weights",
        "regnet_y_16gf": "RegNet_Y_16GF_Weights",
    }

    def __init__(self, config, channels=3, arch=None):
        super().__init__()
        self.config = config
        self.num_classes = config["data"]["num_classes"]
        model_cfg = config.get("model", {})
        self.arch = arch or model_cfg.get("arch")
        if not self.arch:
            raise ValueError("TorchvisionCNNFER needs model.arch in config.")

        self.freeze_backbone_on_start = bool(model_cfg.get("freeze_backbone", False))
        self.unfreeze_epoch = model_cfg.get("unfreeze_epoch", None)
        self.unfreeze_backbone_scope = model_cfg.get("unfreeze_backbone_scope", "all")
        self.trainable_backbone_layers = model_cfg.get("trainable_backbone_layers", [])
        if isinstance(self.trainable_backbone_layers, str):
            self.trainable_backbone_layers = [self.trainable_backbone_layers]

        weights = self._resolve_weights(model_cfg)
        builder = getattr(models, self.arch, None)
        if builder is None:
            raise ValueError(f"torchvision.models has no builder named '{self.arch}'.")

        self.backbone = builder(weights=weights)
        if channels != 3:
            self._adapt_first_conv(channels)

        self.feature_dim = self._replace_classifier(model_cfg)
        self.is_frozen = False

        if self.freeze_backbone_on_start:
            self.freeze_backbone()

        weight_name = "none" if weights is None else "DEFAULT"
        print(
            f"--> [TorchvisionCNNFER] arch={self.arch}, weights={weight_name}, "
            f"feature_dim={self.feature_dim}"
        )

    def _resolve_weights(self, model_cfg):
        if not bool(model_cfg.get("pretrained", True)):
            return None

        weights_name = model_cfg.get("weights", "DEFAULT")
        if weights_name in (None, "none", "None", False):
            return None

        enum_name = self._WEIGHT_ENUMS.get(self.arch)
        if enum_name is None or not hasattr(models, enum_name):
            return weights_name

        weights_enum = getattr(models, enum_name)
        if weights_name == "DEFAULT":
            return weights_enum.DEFAULT
        return getattr(weights_enum, weights_name)

    def _adapt_first_conv(self, channels):
        first_conv_parent, first_conv_name, first_conv = self._find_first_conv(self.backbone)
        if first_conv is None:
            raise ValueError(f"Could not find first Conv2d in {self.arch}.")

        new_conv = nn.Conv2d(
            channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=(first_conv.bias is not None),
            padding_mode=first_conv.padding_mode,
        )
        if channels == 1 and first_conv.in_channels == 3:
            new_conv.weight.data.copy_(first_conv.weight.data.mean(dim=1, keepdim=True))
            if first_conv.bias is not None:
                new_conv.bias.data.copy_(first_conv.bias.data)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        self._set_child(first_conv_parent, first_conv_name, new_conv)

    def _replace_classifier(self, model_cfg):
        parent, child_name, last_linear, path = self._find_last_linear(self.backbone)
        if last_linear is None:
            raise ValueError(f"Could not find final Linear classifier in {self.arch}.")

        feature_dim = last_linear.in_features
        head = self._build_head(feature_dim, model_cfg)
        self._set_child(parent, child_name, head)
        object.__setattr__(self, "_head_module", head)

        head_param_ids = {id(param) for param in head.parameters()}
        self._head_param_names = {
            name
            for name, param in self.backbone.named_parameters()
            if id(param) in head_param_ids
        }
        self._head_module_path = path
        return feature_dim

    def _build_head(self, feature_dim, model_cfg):
        head_type = model_cfg.get("head_type", "mlp").lower()
        dropout = float(model_cfg.get("head_dropout", 0.35))

        if head_type == "linear":
            return nn.Linear(feature_dim, self.num_classes)

        hidden_dim = int(model_cfg.get("head_hidden_dim", 512))
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    @classmethod
    def _find_last_linear(cls, module, prefix=()):
        found = None
        for name, child in module.named_children():
            path = prefix + (name,)
            if isinstance(child, nn.Linear):
                found = (module, name, child, path)
            deeper = cls._find_last_linear(child, path)
            if deeper is not None:
                found = deeper
        return found

    @classmethod
    def _find_first_conv(cls, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                return module, name, child
            found = cls._find_first_conv(child)
            if found[2] is not None:
                return found
        return None, None, None

    @staticmethod
    def _set_child(parent, child_name, child):
        if isinstance(parent, nn.Sequential) and str(child_name).isdigit():
            parent[int(child_name)] = child
        else:
            setattr(parent, child_name, child)

    def parameter_role(self, clean_name):
        local_name = clean_name[len("backbone."):] if clean_name.startswith("backbone.") else clean_name
        return "head" if local_name in self._head_param_names else "visual"

    def freeze_backbone(self):
        trainable_prefixes = tuple(self.trainable_backbone_layers)
        for name, param in self.backbone.named_parameters():
            is_head_param = name in self._head_param_names
            is_trainable_backbone = bool(trainable_prefixes) and name.startswith(trainable_prefixes)
            param.requires_grad = is_head_param or is_trainable_backbone

        self.is_frozen = True
        trainable = ["head", *self.trainable_backbone_layers]
        print(f"--> [TorchvisionCNNFER] Frozen {self.arch} except: {', '.join(trainable)}.")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        self.is_frozen = False
        print(f"--> [TorchvisionCNNFER] Unfrozen full {self.arch} backbone.")

    def check_unfreeze(self, epoch):
        if self.unfreeze_epoch is None:
            return False
        if self.is_frozen and epoch >= int(self.unfreeze_epoch):
            self.unfreeze_backbone()
            return True
        return False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.is_frozen:
            self.backbone.eval()
            self._head_module.train(mode)
        return self

    def forward(self, x, labels=None):
        return self.backbone(x)
