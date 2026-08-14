import os

import torch
import torch.nn as nn
import torchvision.models as models


class TorchvisionCNNFER(nn.Module):
    """Generic torchvision CNN classifier for FER2013 ensemble members."""

    _WEIGHT_ENUMS = {
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
        "efficientnet_b3": "EfficientNet_B3_Weights",
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
        self.checkpoint_path = model_cfg.get("checkpoint_path")
        self.checkpoint_strict = bool(model_cfg.get("checkpoint_strict", True))
        self.checkpoint_skip_mismatch = bool(
            model_cfg.get("checkpoint_skip_mismatch", not self.checkpoint_strict)
        )

        weights = self._resolve_weights(model_cfg)
        builder = getattr(models, self.arch, None)
        if builder is None:
            raise ValueError(f"torchvision.models has no builder named '{self.arch}'.")

        self.backbone = builder(weights=weights)
        if channels != 3:
            self._adapt_first_conv(channels)

        self.feature_dim = self._replace_classifier(model_cfg)
        self.is_frozen = False

        if self.checkpoint_path:
            self.load_from_checkpoint(
                self.checkpoint_path,
                device="cpu",
                strict=self.checkpoint_strict,
                skip_mismatch=self.checkpoint_skip_mismatch,
            )

        if self.freeze_backbone_on_start:
            self.freeze_backbone()

        weight_name = "none" if weights is None else "DEFAULT"
        print(
            f"--> [TorchvisionCNNFER] arch={self.arch}, weights={weight_name}, "
            f"feature_dim={self.feature_dim}"
        )

    @staticmethod
    def _safe_torch_load(path, map_location="cpu"):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model", "net"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value

        if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

        raise ValueError("Checkpoint does not contain a valid state dict.")

    @staticmethod
    def _strip_known_prefixes(state_dict):
        prefixes = ("module.", "_orig_mod.")
        cleaned = {}
        for key, value in state_dict.items():
            name = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        changed = True
            cleaned[name] = value
        return cleaned

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required.")

        if os.path.exists(checkpoint_path):
            return checkpoint_path

        basename = os.path.basename(checkpoint_path)
        search_roots = [os.getcwd()]
        if os.path.exists("/kaggle/input"):
            search_roots.insert(0, "/kaggle/input")

        for root in search_roots:
            for current_dir, _, files in os.walk(root):
                if basename in files:
                    found = os.path.join(current_dir, basename)
                    print(f"--> [TorchvisionCNNFER] Using discovered checkpoint: {found}")
                    return found

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    @staticmethod
    def _align_external_state_dict(state_dict, target_state):
        """Accept bare torchvision checkpoints by mapping conv1.* -> backbone.conv1.*."""
        if any(key.startswith("backbone.") for key in state_dict):
            return state_dict

        aligned = {}
        for key, value in state_dict.items():
            prefixed_key = f"backbone.{key}"
            aligned[prefixed_key if prefixed_key in target_state else key] = value
        return aligned

    @staticmethod
    def _filter_mismatched_keys(state_dict, target_state):
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            target_value = target_state.get(key)
            if target_value is None:
                skipped.append((key, "unexpected"))
                continue
            if tuple(value.shape) != tuple(target_value.shape):
                skipped.append((key, f"shape {tuple(value.shape)} != {tuple(target_value.shape)}"))
                continue
            filtered[key] = value
        return filtered, skipped

    def load_from_checkpoint(self, checkpoint_path, device="cpu", strict=True, skip_mismatch=False):
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = self._safe_torch_load(checkpoint_path, map_location=device)
        state_dict = self._strip_known_prefixes(self._extract_state_dict(checkpoint))
        target_state = self.state_dict()
        state_dict = self._align_external_state_dict(state_dict, target_state)

        skipped = []
        if skip_mismatch:
            state_dict, skipped = self._filter_mismatched_keys(state_dict, target_state)
            strict = False

        load_result = self.load_state_dict(state_dict, strict=strict)
        print(
            f"--> [TorchvisionCNNFER] Loaded checkpoint: {checkpoint_path} "
            f"(strict={strict}, skip_mismatch={skip_mismatch})"
        )
        if skipped:
            print(f"--> [TorchvisionCNNFER] Skipped checkpoint keys: {len(skipped)}")
        if getattr(load_result, "missing_keys", None):
            print(f"--> [TorchvisionCNNFER] Missing keys: {len(load_result.missing_keys)}")
        if getattr(load_result, "unexpected_keys", None):
            print(f"--> [TorchvisionCNNFER] Unexpected keys: {len(load_result.unexpected_keys)}")
        return load_result

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
