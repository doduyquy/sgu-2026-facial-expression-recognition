import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .region_attention import (
    CLIPFacialRegionDictionary,
    DropPath,
    FacialRegionDictionary,
    SubGraphFusion,
)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint does not contain a valid state dict.")


def strip_known_prefixes(state_dict):
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


class CrossDimSemanticVisualAlignment(nn.Module):
    """
    Region tokens query ConvNeXt visual tokens.

    Query lives in embed_dim, while ConvNeXt keys/values stay in visual_dim.
    """

    def __init__(self, embed_dim=512, visual_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=visual_dim,
            vdim=visual_dim,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(dropout if dropout > 0.0 else 0.0)

    def forward(self, region_tokens, visual_features):
        attn_out, attn_weights = self.cross_attn(
            query=region_tokens,
            key=visual_features,
            value=visual_features,
        )
        region_enriched = self.norm1(region_tokens + self.drop_path(attn_out))
        ffn_out = self.ffn(region_enriched)
        region_enriched = self.norm2(region_enriched + self.drop_path(ffn_out))
        return region_enriched, attn_weights


class MaskGuidedCrossDimSemanticVisualAlignment(CrossDimSemanticVisualAlignment):
    """
    Region-token cross-attention with an additive soft mask prior.

    The mask is injected before softmax as:
        attention_score += alpha * log(clamp(mask, floor, 1.0))
    """

    def __init__(
        self,
        embed_dim=512,
        visual_dim=768,
        num_heads=4,
        dropout=0.1,
        mask_attention_alpha=0.3,
        mask_floor=0.05,
    ):
        super().__init__(
            embed_dim=embed_dim,
            visual_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mask_attention_alpha = float(mask_attention_alpha)
        self.mask_floor = float(mask_floor)
        if self.mask_attention_alpha < 0.0:
            raise ValueError("model.mask_attention_alpha must be >= 0.")
        if not 0.0 < self.mask_floor <= 1.0:
            raise ValueError("model.mask_floor must be in (0, 1].")

    def _build_log_mask(self, region_masks):
        masks = region_masks.clamp(min=self.mask_floor, max=1.0)
        log_mask = self.mask_attention_alpha * torch.log(masks + 1e-6)
        return log_mask.repeat_interleave(self.cross_attn.num_heads, dim=0)

    def forward(self, region_tokens, visual_features, region_masks=None):
        attn_mask = None
        if region_masks is not None and self.mask_attention_alpha > 0.0:
            attn_mask = self._build_log_mask(region_masks)

        attn_out, attn_weights = self.cross_attn(
            query=region_tokens,
            key=visual_features,
            value=visual_features,
            attn_mask=attn_mask,
        )
        region_enriched = self.norm1(region_tokens + self.drop_path(attn_out))
        ffn_out = self.ffn(region_enriched)
        region_enriched = self.norm2(region_enriched + self.drop_path(ffn_out))
        return region_enriched, attn_weights


class SwinLocalRefinerBlock(nn.Module):
    """Shifted-window self-attention over a ConvNeXt feature map."""

    def __init__(
        self,
        dim,
        num_heads=4,
        window_size=4,
        shift_size=0,
        mlp_ratio=2.0,
        dropout=0.1,
        layer_scale_init=1e-4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = max(int(window_size), 1)
        self.shift_size = max(int(shift_size), 0)

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(self.dim * float(mlp_ratio))
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(dropout if dropout > 0.0 else 0.0)
        self.gamma_attn = nn.Parameter(torch.full((self.dim,), float(layer_scale_init)))
        self.gamma_ffn = nn.Parameter(torch.full((self.dim,), float(layer_scale_init)))

    @staticmethod
    def _window_partition(x, window_size):
        b, h, w, c = x.shape
        x = x.view(
            b,
            h // window_size,
            window_size,
            w // window_size,
            window_size,
            c,
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, window_size * window_size, c)

    @staticmethod
    def _window_reverse(windows, batch_size, height, width, window_size):
        x = windows.view(
            batch_size,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, height, width, -1)

    def _pad_to_window(self, x):
        b, h, w, c = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h == 0 and pad_w == 0:
            valid_mask = torch.ones((b, h, w), device=x.device, dtype=torch.bool)
            return x, valid_mask, h, w

        padded = x.new_zeros((b, h + pad_h, w + pad_w, c))
        padded[:, :h, :w, :] = x
        valid_mask = torch.zeros(
            (b, h + pad_h, w + pad_w),
            device=x.device,
            dtype=torch.bool,
        )
        valid_mask[:, :h, :w] = True
        return padded, valid_mask, h, w

    def _build_shift_mask(self, height, width, shift_size, device):
        if shift_size <= 0:
            return None

        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = self._window_partition(img_mask, self.window_size).squeeze(-1)
        return mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)

    def forward(self, feat_map):
        b, c, h, w = feat_map.shape
        x = feat_map.permute(0, 2, 3, 1).contiguous()
        shortcut = x

        attn_input = self.norm1(x)
        attn_input, valid_mask, orig_h, orig_w = self._pad_to_window(attn_input)
        padded_h, padded_w = attn_input.shape[1:3]
        shift_size = (
            min(self.shift_size, self.window_size - 1)
            if min(padded_h, padded_w) > self.window_size
            else 0
        )

        if shift_size > 0:
            attn_input = torch.roll(
                attn_input,
                shifts=(-shift_size, -shift_size),
                dims=(1, 2),
            )
            valid_mask = torch.roll(
                valid_mask,
                shifts=(-shift_size, -shift_size),
                dims=(1, 2),
            )

        windows = self._window_partition(attn_input, self.window_size)
        valid_windows = self._window_partition(
            valid_mask.unsqueeze(-1).float(),
            self.window_size,
        ).squeeze(-1)
        key_padding_mask = valid_windows < 0.5

        attn_mask = self._build_shift_mask(
            padded_h,
            padded_w,
            shift_size,
            device=attn_input.device,
        )
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(b, 1, 1)
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        attn_out, _ = self.attn(
            windows,
            windows,
            windows,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self._window_reverse(
            attn_out,
            b,
            padded_h,
            padded_w,
            self.window_size,
        )

        if shift_size > 0:
            attn_out = torch.roll(
                attn_out,
                shifts=(shift_size, shift_size),
                dims=(1, 2),
            )

        attn_out = attn_out[:, :orig_h, :orig_w, :]
        x = shortcut + self.drop_path(self.gamma_attn * attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop_path(self.gamma_ffn * ffn_out)
        return x.permute(0, 3, 1, 2).contiguous()


class SwinLocalRefiner(nn.Module):
    """Small W-MSA/SW-MSA stack placed before local-token flattening."""

    def __init__(
        self,
        dim,
        depth=2,
        num_heads=4,
        window_size=4,
        mlp_ratio=2.0,
        dropout=0.1,
        layer_scale_init=1e-4,
    ):
        super().__init__()
        blocks = []
        for index in range(int(depth)):
            shift_size = 0 if index % 2 == 0 else max(int(window_size) // 2, 1)
            blocks.append(
                SwinLocalRefinerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, feat_map):
        return self.blocks(feat_map)


class ChannelGate(nn.Module):
    """SE/ECA channel gate with an optional residual multiplier."""

    def __init__(
        self,
        channels,
        attention_type="se",
        reduction=16,
        eca_kernel_size=3,
        gate_mode="residual",
        gamma_init=0.1,
        gamma_learnable=True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.attention_type = attention_type.lower()
        self.gate_mode = gate_mode.lower()
        if self.gate_mode not in ("residual", "sigmoid"):
            raise ValueError("multiscale_se_gate_mode must be 'residual' or 'sigmoid'.")

        if self.attention_type == "se":
            hidden_dim = max(self.channels // max(int(reduction), 1), 4)
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channels, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, self.channels, kernel_size=1),
            )
        elif self.attention_type == "eca":
            kernel_size = max(int(eca_kernel_size), 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.attn = nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
        else:
            raise ValueError("multiscale_se_type must be 'se' or 'eca'.")

        gamma = torch.tensor(float(gamma_init), dtype=torch.float32)
        if gamma_learnable:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

    def _weights(self, x):
        if self.attention_type == "se":
            return torch.sigmoid(self.attn(x))

        pooled = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        weights = self.attn(pooled).transpose(1, 2).unsqueeze(-1)
        return torch.sigmoid(weights)

    def forward(self, x):
        weights = self._weights(x)
        if self.gate_mode == "sigmoid":
            return x * weights

        gamma = self.gamma.to(device=x.device, dtype=x.dtype)
        gate = 2.0 * weights - 1.0
        return x * (1.0 + gamma * gate)


class DynamicRegionWeighter(nn.Module):
    """Predict per-image region gates from the global ConvNeXt feature."""

    def __init__(
        self,
        global_dim,
        num_regions,
        hidden_dim,
        dropout=0.1,
        temperature=1.0,
        output_scale=1.0,
        zero_init=True,
    ):
        super().__init__()
        self.num_regions = int(num_regions)
        self.temperature = float(temperature)
        self.output_scale = float(output_scale)
        if self.temperature <= 0.0:
            raise ValueError("model.region_weight_temperature must be > 0.")

        self.mlp = nn.Sequential(
            nn.LayerNorm(int(global_dim)),
            nn.Linear(int(global_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_regions),
        )
        if bool(zero_init):
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, global_feature):
        logits = self.mlp(global_feature)
        return F.softmax(logits / self.temperature, dim=-1) * self.output_scale


class RegionRelationTokenBuilder(nn.Module):
    """Build explicit relation tokens from configured facial-region groups."""

    def __init__(self, embed_dim, relation_pairs, dropout=0.1):
        super().__init__()
        self.relation_pairs = relation_pairs
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(int(embed_dim) * 2),
                nn.Linear(int(embed_dim) * 2, int(embed_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(embed_dim), int(embed_dim)),
            )
            for _ in relation_pairs
        ])

    @staticmethod
    def _group_feature(region_features, indices):
        selected = region_features[:, indices, :]
        return selected.mean(dim=1)

    def forward(self, region_features):
        relation_tokens = []
        for pair, fusion in zip(self.relation_pairs, self.fusions):
            left_feat = self._group_feature(region_features, pair["left"])
            right_feat = self._group_feature(region_features, pair["right"])
            relation_input = torch.cat((left_feat, right_feat), dim=-1)
            relation_tokens.append(fusion(relation_input).unsqueeze(1))

        if not relation_tokens:
            return region_features
        return torch.cat((region_features, *relation_tokens), dim=1)


class ConvNeXtMultiScaleSEFusion(nn.Module):
    """Fuse ConvNeXt stage3 and stage4 maps before tokenization."""

    def __init__(self, model_cfg, out_channels):
        super().__init__()
        self.stage3_channels = int(model_cfg.get("multiscale_stage3_channels", 384))
        self.stage4_channels = int(model_cfg.get("multiscale_stage4_channels", out_channels))
        self.out_channels = int(out_channels)
        attention_type = model_cfg.get("multiscale_se_type", "se")
        gate_mode = model_cfg.get("multiscale_se_gate_mode", "residual")
        gamma_init = float(model_cfg.get("multiscale_se_gamma_init", 0.1))
        gamma_learnable = bool(model_cfg.get("multiscale_se_gamma_learnable", True))
        reduction = int(model_cfg.get("multiscale_se_reduction", 16))
        eca_kernel_size = int(model_cfg.get("multiscale_eca_kernel_size", 3))

        self.stage3_gate = ChannelGate(
            self.stage3_channels,
            attention_type=attention_type,
            reduction=reduction,
            eca_kernel_size=eca_kernel_size,
            gate_mode=gate_mode,
            gamma_init=gamma_init,
            gamma_learnable=gamma_learnable,
        )
        self.stage4_gate = ChannelGate(
            self.stage4_channels,
            attention_type=attention_type,
            reduction=reduction,
            eca_kernel_size=eca_kernel_size,
            gate_mode=gate_mode,
            gamma_init=gamma_init,
            gamma_learnable=gamma_learnable,
        )

        dropout = float(model_cfg.get("multiscale_fusion_dropout", 0.0))
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.stage3_channels + self.stage4_channels,
                self.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(1, self.out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.fusion_residual = bool(model_cfg.get("multiscale_fusion_residual", True))
        fusion_gamma = torch.tensor(
            float(model_cfg.get("multiscale_fusion_gamma_init", 0.1)),
            dtype=torch.float32,
        )
        if bool(model_cfg.get("multiscale_fusion_gamma_learnable", True)):
            self.fusion_gamma = nn.Parameter(fusion_gamma)
        else:
            self.register_buffer("fusion_gamma", fusion_gamma)

    def forward(self, stage3, stage4):
        if stage3.size(1) != self.stage3_channels:
            raise ValueError(
                f"Expected stage3 channels={self.stage3_channels}, got {stage3.size(1)}."
            )
        if stage4.size(1) != self.stage4_channels:
            raise ValueError(
                f"Expected stage4 channels={self.stage4_channels}, got {stage4.size(1)}."
            )

        stage3_enhanced = self.stage3_gate(stage3)
        stage4_enhanced = self.stage4_gate(stage4)
        stage3_down = F.adaptive_avg_pool2d(stage3_enhanced, stage4_enhanced.shape[-2:])
        fused = self.fusion(torch.cat((stage3_down, stage4_enhanced), dim=1))
        if not self.fusion_residual:
            return fused

        gamma = self.fusion_gamma.to(device=stage4_enhanced.device, dtype=stage4_enhanced.dtype)
        return stage4_enhanced + gamma * fused


class ConvNeXtSpatialTokenizer(nn.Module):
    """ConvNeXt ImageNet features -> local visual tokens for region attention."""

    _WEIGHT_ENUMS = {
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
        "efficientnet_b3": "EfficientNet_B3_Weights",
        "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
        "efficientnet_v2_m": "EfficientNet_V2_M_Weights",
    }

    def __init__(self, config, channels=3):
        super().__init__()
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})

        self.num_classes = data_cfg.get("num_classes", 7)
        self.arch = model_cfg.get("arch", "convnext_tiny")
        self.pool_visual_tokens = bool(model_cfg.get("pool_visual_tokens", False))
        self.token_grid_size = int(model_cfg.get("token_grid_size", 7))

        weights = self._resolve_weights(model_cfg)
        builder = getattr(models, self.arch, None)
        if builder is None:
            raise ValueError(f"torchvision.models has no builder named '{self.arch}'.")
        self.backbone = builder(weights=weights)
        if channels != 3:
            self._adapt_first_conv(channels)

        self.feature_dim = self._infer_feature_dim()
        self.use_source_classifier = bool(model_cfg.get("use_source_classifier", False))
        self.num_visual_tokens = (
            self.token_grid_size ** 2
            if self.pool_visual_tokens
            else (data_cfg.get("image_size", 224) // 32) ** 2
        )
        self.token_pool = (
            nn.AdaptiveAvgPool2d((self.token_grid_size, self.token_grid_size))
            if self.pool_visual_tokens
            else nn.Identity()
        )
        self.use_swin_local_refiner = bool(
            model_cfg.get("use_swin_local_refiner", False)
        )
        self.use_multiscale_se_fusion = bool(
            model_cfg.get("use_multiscale_se_fusion", False)
        )
        self.use_layer4_se = bool(model_cfg.get("use_layer4_se", False))
        if self.use_multiscale_se_fusion and self.use_layer4_se:
            raise ValueError(
                "use_layer4_se and use_multiscale_se_fusion are separate ablations; "
                "enable only one of them."
            )
        self.multiscale_stage3_index = int(model_cfg.get("multiscale_stage3_index", 5))
        self.multiscale_stage4_index = int(
            model_cfg.get(
                "multiscale_stage4_index",
                len(self.backbone.features) - 1,
            )
        )
        if self.use_multiscale_se_fusion:
            self.multiscale_fusion = ConvNeXtMultiScaleSEFusion(
                model_cfg,
                out_channels=self.feature_dim,
            )
        else:
            self.multiscale_fusion = None
        if self.use_layer4_se:
            self.layer4_se_gate = ChannelGate(
                int(model_cfg.get("layer4_se_channels", self.feature_dim)),
                attention_type=model_cfg.get("layer4_se_type", "se"),
                reduction=int(model_cfg.get("layer4_se_reduction", 16)),
                eca_kernel_size=int(model_cfg.get("layer4_eca_kernel_size", 3)),
                gate_mode=model_cfg.get("layer4_se_gate_mode", "residual"),
                gamma_init=float(model_cfg.get("layer4_se_gamma_init", 0.1)),
                gamma_learnable=bool(model_cfg.get("layer4_se_gamma_learnable", True)),
            )
        else:
            self.layer4_se_gate = None
        if self.use_swin_local_refiner:
            self.swin_refiner = SwinLocalRefiner(
                dim=self.feature_dim,
                depth=int(model_cfg.get("swin_refiner_depth", 2)),
                num_heads=int(model_cfg.get("swin_refiner_heads", model_cfg.get("num_heads", 4))),
                window_size=int(model_cfg.get("swin_window_size", 4)),
                mlp_ratio=float(model_cfg.get("swin_refiner_mlp_ratio", 2.0)),
                dropout=float(
                    model_cfg.get(
                        "swin_refiner_dropout",
                        model_cfg.get("transformer_dropout", 0.1),
                    )
                ),
                layer_scale_init=float(
                    model_cfg.get("swin_refiner_layer_scale_init", 1e-4)
                ),
            )
        else:
            self.swin_refiner = nn.Identity()
        if self.use_source_classifier:
            self.source_classifier = self._build_source_classifier(model_cfg)
        else:
            # The region-attention branch uses ConvNeXt as a pure feature extractor:
            # keep features, drop the final avgpool/classifier path.
            self.backbone.avgpool = nn.Identity()
            self.backbone.classifier = nn.Identity()
            self.source_classifier = None

        weight_name = "none" if weights is None else "DEFAULT"
        print(
            f"--> [ConvNeXtTokenizer] arch={self.arch}, weights={weight_name}, "
            f"feature_dim={self.feature_dim}, tokens={self.num_visual_tokens}, "
            f"source_classifier={self.use_source_classifier}"
        )
        if self.use_swin_local_refiner:
            print(
                "--> [ConvNeXtTokenizer] Swin local refiner enabled before "
                "local-token flattening."
            )
        if self.use_multiscale_se_fusion:
            print(
                "--> [ConvNeXtTokenizer] Multi-scale SE fusion enabled: "
                f"stage3_index={self.multiscale_stage3_index}, "
                f"stage4_index={self.multiscale_stage4_index}"
            )
        if self.use_layer4_se:
            print(
                "--> [ConvNeXtTokenizer] Layer4 SE gate enabled: "
                f"channels={self.feature_dim}"
            )

    def _resolve_weights(self, model_cfg):
        if not bool(model_cfg.get("pretrained", False)):
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

    def _infer_feature_dim(self):
        _, _, last_linear, _ = self._find_last_linear(self.backbone)
        if last_linear is None:
            raise ValueError(f"Could not infer feature dim for {self.arch}.")
        return last_linear.in_features

    def _build_source_classifier(self, model_cfg):
        head_type = model_cfg.get("head_type", "mlp").lower()
        dropout = float(model_cfg.get("head_dropout", 0.35))

        if head_type == "linear":
            head = nn.Linear(self.feature_dim, self.num_classes)
        else:
            hidden_dim = int(model_cfg.get("head_hidden_dim", 512))
            head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )

        parent, child_name, last_linear, _ = self._find_last_linear(self.backbone.classifier)
        if last_linear is None:
            raise ValueError(f"Could not find classifier Linear in {self.arch}.")
        self._set_child(parent, child_name, head)
        return self.backbone.classifier

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

    def _forward_multiscale_features(self, x):
        stage3 = None
        stage4 = None
        feat = x
        for index, layer in enumerate(self.backbone.features):
            feat = layer(feat)
            if index == self.multiscale_stage3_index:
                stage3 = feat
            if index == self.multiscale_stage4_index:
                stage4 = feat

        if stage3 is None:
            raise RuntimeError(
                "Could not capture ConvNeXt stage3 feature. "
                "Check model.multiscale_stage3_index."
            )
        if stage4 is None:
            raise RuntimeError(
                "Could not capture ConvNeXt stage4 feature. "
                "Check model.multiscale_stage4_index."
            )
        return self.multiscale_fusion(stage3, stage4)

    def forward(self, x):
        if self.use_multiscale_se_fusion:
            feat_map = self._forward_multiscale_features(x)
        else:
            feat_map = self.backbone.features(x)
            if self.layer4_se_gate is not None:
                feat_map = self.layer4_se_gate(feat_map)
        token_map = self.token_pool(feat_map)
        token_map = self.swin_refiner(token_map)
        visual_tokens = token_map.flatten(2).transpose(1, 2)
        pooled_map = F.adaptive_avg_pool2d(feat_map, 1)
        max_pooled_map = F.adaptive_max_pool2d(feat_map, 1)
        global_feat = torch.flatten(pooled_map, 1)
        global_max_feat = torch.flatten(max_pooled_map, 1)
        return visual_tokens, global_feat, pooled_map, global_max_feat

    def source_logits(self, pooled_map):
        if self.source_classifier is None:
            return None
        return self.source_classifier(pooled_map)

    def load_from_checkpoint(self, checkpoint_path, device="cpu"):
        checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
        print(f"--> [ConvNeXtTokenizer] Loading source checkpoint: {checkpoint_path}")

        checkpoint = safe_torch_load(checkpoint_path, map_location=device)
        state_dict = strip_known_prefixes(extract_state_dict(checkpoint))

        features_state = {}
        classifier_state = {}
        skipped = []

        features_ref = self.backbone.features.state_dict()
        classifier_ref = (
            self.source_classifier.state_dict()
            if self.source_classifier is not None
            else {}
        )

        for key, value in state_dict.items():
            name = key
            if name.startswith("backbone."):
                name = name[len("backbone."):]

            if name.startswith("features."):
                feature_name = name[len("features."):]
                if feature_name in features_ref and features_ref[feature_name].shape == value.shape:
                    features_state[feature_name] = value
                else:
                    skipped.append(key)
                continue

            if name.startswith("classifier."):
                if self.source_classifier is None:
                    skipped.append(key)
                    continue
                classifier_name = name[len("classifier."):]
                if (
                    classifier_name in classifier_ref
                    and classifier_ref[classifier_name].shape == value.shape
                ):
                    classifier_state[classifier_name] = value
                else:
                    skipped.append(key)
                continue

            skipped.append(key)

        missing, unexpected = self.backbone.features.load_state_dict(features_state, strict=False)
        if self.source_classifier is not None and classifier_state:
            self.source_classifier.load_state_dict(classifier_state, strict=False)

        print(f"--> [ConvNeXtTokenizer] Features loaded: {len(features_state)} tensors")
        if self.source_classifier is not None and classifier_state:
            print("--> [ConvNeXtTokenizer] Source classifier loaded for residual logits.")
        if missing:
            print(f"--> [ConvNeXtTokenizer] Missing feature keys: {len(missing)}")
        if unexpected:
            print(f"--> [ConvNeXtTokenizer] Unexpected feature keys: {len(unexpected)}")
        if skipped:
            print(f"--> [ConvNeXtTokenizer] Skipped checkpoint keys: {len(skipped)}")

    @staticmethod
    def resolve_checkpoint_path(checkpoint_path):
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
                    print(f"--> [ConvNeXtTokenizer] Using discovered checkpoint: {found}")
                    return found

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


class ConvNeXtRegionAttentionFER(nn.Module):
    """
    Region-token attention head on top of an ImageNet-pretrained ConvNeXt.

    ConvNeXt contributes only the feature extractor; its final avgpool/classifier
    path is discarded by default. Region tokens, cross-attention, region
    transformer, and classifier are trained as the new branch in the diagram.
    """

    def __init__(self, config, channels=3):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        self.embed_dim = int(model_cfg.get("embed_dim", 512))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.num_regions = int(model_cfg.get("num_regions", 6))
        self.num_layers = int(model_cfg.get("num_encoder_layers", 2))
        self.dropout_rate = float(model_cfg.get("transformer_dropout", 0.1))
        self.use_visual_pos_embed = bool(model_cfg.get("use_visual_pos_embed", True))
        self.use_region_slot_embed = bool(model_cfg.get("use_region_slot_embed", True))
        self.use_global_visual_bias = bool(model_cfg.get("use_global_visual_bias", True))
        self.use_global_feature_concat = bool(model_cfg.get("use_global_feature_concat", False))
        self.fusion_type = model_cfg.get("fusion_type", "transformer")
        self.region_pooling = model_cfg.get("region_pooling", "concat").lower()
        self.classifier_hidden_dim = int(model_cfg.get("classifier_hidden_dim", 1024))
        self.ortho_loss_type = model_cfg.get("ortho_loss_type", "squared_offdiag").lower()
        self.logit_fusion = model_cfg.get("logit_fusion", "attention")
        self.finetune_logit_fusion = model_cfg.get("finetune_logit_fusion", self.logit_fusion)
        self.attention_logit_weight = float(model_cfg.get("attention_logit_weight", 1.0))
        self.source_logit_weight = float(model_cfg.get("source_logit_weight", 1.0))
        self.cnn_aux_logit_weight = float(model_cfg.get("cnn_aux_logit_weight", 0.2))
        self.learnable_logit_fusion = bool(model_cfg.get("learnable_logit_fusion", False))
        self.learnable_logit_fusion_min = float(model_cfg.get("learnable_logit_fusion_min", 0.0))
        self.learnable_logit_fusion_max = float(model_cfg.get("learnable_logit_fusion_max", 1.0))
        self.learnable_logit_fusion_init = float(
            model_cfg.get("learnable_logit_fusion_init", self.cnn_aux_logit_weight)
        )
        self.cnn_aux_pooling = model_cfg.get("cnn_aux_pooling", "avg").lower()
        self.use_cnn_aux_loss = bool(model_cfg.get("use_cnn_aux_loss", False))
        self.use_cnn_aux_logits = bool(
            model_cfg.get(
                "use_cnn_aux_logits",
                self.logit_fusion in ("cnn_aux", "cnn_aux_sum")
                or self.finetune_logit_fusion in ("cnn_aux", "cnn_aux_sum"),
            )
        )
        self.use_cnn_aux_classifier = self.use_cnn_aux_loss or self.use_cnn_aux_logits
        self.freeze_epochs = int(model_cfg.get("freeze_backbone_epochs", 0))
        self.unfreeze_backbone = bool(model_cfg.get("unfreeze_backbone", False))
        self.unfreeze_backbone_scope = model_cfg.get("unfreeze_backbone_scope", "all").lower()
        self.freeze_unfrozen_batchnorm = bool(model_cfg.get("freeze_unfrozen_batchnorm", False))
        self.current_epoch_index = 0
        self.is_frozen = False
        self.return_attn = False
        self.return_region_weights = False
        self.checkpoint_strict = bool(model_cfg.get("checkpoint_strict", False))
        self.mask_guided_attention = bool(model_cfg.get("mask_guided_attention", False))
        self.use_learnable_clip_region_tokens = bool(
            model_cfg.get("use_learnable_clip_region_tokens", False)
        )
        self.use_eye_fusion_token = bool(model_cfg.get("use_eye_fusion_token", False))
        self.eye_fusion_mode = model_cfg.get("eye_fusion_mode", "post").lower()
        self.eye_fusion_left_index = int(model_cfg.get("eye_fusion_left_index", 0))
        self.eye_fusion_right_index = int(model_cfg.get("eye_fusion_right_index", 1))
        self.use_region_relation_tokens = bool(
            model_cfg.get("use_region_relation_tokens", False)
        )
        self.region_relation_pairs = self._parse_region_relation_pairs(
            model_cfg.get("region_relation_pairs")
        )
        self.num_relation_tokens = (
            len(self.region_relation_pairs) if self.use_region_relation_tokens else 0
        )
        self.num_output_regions = (
            self.num_regions
            + (1 if self.use_eye_fusion_token else 0)
            + self.num_relation_tokens
        )
        self.num_region_tokens = (
            self.num_regions + (1 if self.use_eye_fusion_token else 0)
            if self.use_eye_fusion_token and self.eye_fusion_mode == "mask_union"
            else self.num_regions
        )

        num_classes = int(data_cfg.get("num_classes", 7))
        if self.region_pooling not in ("mean", "concat"):
            raise ValueError("model.region_pooling must be one of: mean, concat")
        if self.ortho_loss_type not in ("mean_offdiag", "squared_offdiag"):
            raise ValueError("model.ortho_loss_type must be one of: mean_offdiag, squared_offdiag")
        if self.cnn_aux_pooling not in ("avg", "avgmax"):
            raise ValueError("model.cnn_aux_pooling must be one of: avg, avgmax")
        self.use_dynamic_region_weighting = bool(
            model_cfg.get("use_dynamic_region_weighting", False)
        )
        self.region_weight_hidden_dim = int(
            model_cfg.get("region_weight_hidden_dim", self.embed_dim)
        )
        self.region_weight_dropout = float(
            model_cfg.get("region_weight_dropout", self.dropout_rate)
        )
        self.region_weight_temperature = float(
            model_cfg.get("region_weight_temperature", 1.0)
        )
        self.region_weight_scale = float(model_cfg.get("region_weight_scale", 1.0))
        self.region_weight_zero_init = bool(
            model_cfg.get("region_weight_zero_init", True)
        )
        valid_logit_fusions = ("attention", "source", "sum", "cnn_aux", "cnn_aux_sum")
        if self.logit_fusion not in valid_logit_fusions:
            raise ValueError(
                "model.logit_fusion must be one of: "
                + ", ".join(valid_logit_fusions)
            )
        if self.finetune_logit_fusion not in valid_logit_fusions:
            raise ValueError(
                "model.finetune_logit_fusion must be one of: "
                + ", ".join(valid_logit_fusions)
            )
        if not 0.0 <= self.learnable_logit_fusion_min < self.learnable_logit_fusion_max <= 1.0:
            raise ValueError(
                "model.learnable_logit_fusion_min/max must satisfy "
                "0 <= min < max <= 1."
            )
        if self.learnable_logit_fusion and "cnn_aux_sum" not in {
            self.logit_fusion,
            self.finetune_logit_fusion,
        }:
            raise ValueError(
                "model.learnable_logit_fusion only supports "
                "logit_fusion='cnn_aux_sum'."
            )
        if self.eye_fusion_mode not in ("post", "mask_union"):
            raise ValueError("model.eye_fusion_mode must be one of: post, mask_union")
        if self.use_eye_fusion_token:
            if not (0 <= self.eye_fusion_left_index < self.num_regions):
                raise ValueError("model.eye_fusion_left_index is out of range.")
            if not (0 <= self.eye_fusion_right_index < self.num_regions):
                raise ValueError("model.eye_fusion_right_index is out of range.")
            if self.eye_fusion_left_index == self.eye_fusion_right_index:
                raise ValueError("eye-fusion needs two different region indices.")
        if self.use_region_relation_tokens:
            self._validate_region_relation_pairs(self.region_relation_pairs)

        self.convnext_backbone = ConvNeXtSpatialTokenizer(config, channels=channels)
        self.visual_dim = self.convnext_backbone.feature_dim
        if self.logit_fusion == "attention" and self.convnext_backbone.source_classifier is not None:
            for param in self.convnext_backbone.source_classifier.parameters():
                param.requires_grad = False

        num_visual_tokens = self.convnext_backbone.num_visual_tokens
        if self.use_visual_pos_embed:
            self.visual_pos_embed = nn.Parameter(
                torch.randn(1, num_visual_tokens, self.visual_dim) * 0.02
            )
        else:
            self.register_buffer(
                "visual_pos_embed",
                torch.zeros(1, num_visual_tokens, self.visual_dim),
            )

        self.use_clip_dictionary = bool(model_cfg.get("use_clip_dictionary", True))
        if (
            self.use_eye_fusion_token
            and self.eye_fusion_mode == "mask_union"
            and self.use_clip_dictionary
        ):
            raise ValueError(
                "model.eye_fusion_mode='mask_union' currently supports learned-only "
                "region tokens. Set model.use_clip_dictionary: false."
            )
        self.learned_region_dict = None
        self.clip_region_dict = None
        self.clip_region_gamma = None
        if self.use_learnable_clip_region_tokens:
            self.learned_region_dict = FacialRegionDictionary(
                num_regions=self.num_region_tokens,
                embed_dim=self.embed_dim,
            )
            self.region_dict = self.learned_region_dict
            gamma_init = float(model_cfg.get("clip_region_gamma_init", 0.1))
            gamma_value = torch.tensor(gamma_init, dtype=torch.float32)
            if bool(model_cfg.get("clip_region_gamma_learnable", True)):
                self.clip_region_gamma = nn.Parameter(gamma_value)
            else:
                self.register_buffer("clip_region_gamma", gamma_value)

            if self.use_clip_dictionary:
                clip_model_name = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
                try:
                    self.clip_region_dict = CLIPFacialRegionDictionary(
                        num_regions=self.num_region_tokens,
                        embed_dim=self.embed_dim,
                        clip_model_name=clip_model_name,
                    )
                    print(
                        "--> [ConvNeXtRegionAttention] Mixed region tokens: "
                        f"learned + gamma*CLIP, gamma_init={gamma_init}"
                    )
                except Exception as exc:
                    if not bool(model_cfg.get("clip_fallback_to_learned", True)):
                        raise
                    print(
                        "--> [ConvNeXtRegionAttention] CLIP region tokens unavailable; "
                        "using learned region tokens only. "
                        f"Reason: {exc}"
                    )
                    self.clip_region_dict = None
            else:
                print("--> [ConvNeXtRegionAttention] Learned region tokens only.")
        elif self.use_clip_dictionary:
            clip_model_name = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
            try:
                self.region_dict = CLIPFacialRegionDictionary(
                    num_regions=self.num_region_tokens,
                    embed_dim=self.embed_dim,
                    clip_model_name=clip_model_name,
                )
                print(f"--> [ConvNeXtRegionAttention] CLIP text region tokens: K={self.num_region_tokens}")
            except Exception as exc:
                if not bool(model_cfg.get("clip_fallback_to_learned", True)):
                    raise
                print(
                    "--> [ConvNeXtRegionAttention] CLIP region tokens unavailable; "
                    f"using learned region tokens instead. Reason: {exc}"
                )
                self.region_dict = FacialRegionDictionary(
                    num_regions=self.num_region_tokens,
                    embed_dim=self.embed_dim,
                )
        else:
            self.region_dict = FacialRegionDictionary(
                num_regions=self.num_region_tokens,
                embed_dim=self.embed_dim,
            )
            print(f"--> [ConvNeXtRegionAttention] Learned region tokens: K={self.num_region_tokens}")

        if self.mask_guided_attention:
            self.alignment = MaskGuidedCrossDimSemanticVisualAlignment(
                embed_dim=self.embed_dim,
                visual_dim=self.visual_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                mask_attention_alpha=float(model_cfg.get("mask_attention_alpha", 0.3)),
                mask_floor=float(model_cfg.get("mask_floor", 0.05)),
            )
            print(
                "--> [ConvNeXtRegionAttention] Mask-guided cross-attention enabled: "
                f"alpha={self.alignment.mask_attention_alpha}, floor={self.alignment.mask_floor}"
            )
        else:
            self.alignment = CrossDimSemanticVisualAlignment(
                embed_dim=self.embed_dim,
                visual_dim=self.visual_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
            )

        if self.use_eye_fusion_token and self.eye_fusion_mode == "post":
            self.eye_fusion = nn.Sequential(
                nn.LayerNorm(self.embed_dim * 2),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            print(
                "--> [ConvNeXtRegionAttention] Eye-fusion token enabled: "
                f"indices=({self.eye_fusion_left_index}, {self.eye_fusion_right_index}), "
                f"K={self.num_regions}->{self.num_output_regions}"
            )
        else:
            self.eye_fusion = None
            if self.use_eye_fusion_token:
                print(
                    "--> [ConvNeXtRegionAttention] Eye-fusion mask-union token enabled: "
                    f"indices=({self.eye_fusion_left_index}, {self.eye_fusion_right_index}), "
                    f"K={self.num_regions}->{self.num_output_regions}"
                )

        if self.use_global_visual_bias or self.use_global_feature_concat:
            self.visual_proj = nn.Sequential(
                nn.LayerNorm(self.visual_dim),
                nn.Linear(self.visual_dim, self.embed_dim),
                nn.Dropout(self.dropout_rate),
            )
        else:
            self.visual_proj = None
        if self.use_global_feature_concat:
            print("--> [ConvNeXtRegionAttention] Global feature concat enabled.")

        if self.use_region_relation_tokens:
            self.region_relation_builder = RegionRelationTokenBuilder(
                embed_dim=self.embed_dim,
                relation_pairs=self.region_relation_pairs,
                dropout=float(model_cfg.get("region_relation_dropout", self.dropout_rate)),
            )
            relation_names = ", ".join(pair["name"] for pair in self.region_relation_pairs)
            print(
                "--> [ConvNeXtRegionAttention] Region relation tokens enabled: "
                f"{relation_names}; K={self.num_regions}->{self.num_output_regions}"
            )
        else:
            self.region_relation_builder = None

        if self.use_dynamic_region_weighting:
            self.region_weighter = DynamicRegionWeighter(
                global_dim=self.visual_dim,
                num_regions=self.num_output_regions,
                hidden_dim=self.region_weight_hidden_dim,
                dropout=self.region_weight_dropout,
                temperature=self.region_weight_temperature,
                output_scale=self.region_weight_scale,
                zero_init=self.region_weight_zero_init,
            )
            print(
                "--> [ConvNeXtRegionAttention] Dynamic region weighting enabled: "
                f"K={self.num_output_regions}, hidden_dim={self.region_weight_hidden_dim}, "
                f"temperature={self.region_weight_temperature}, "
                f"scale={self.region_weight_scale}"
            )
        else:
            self.region_weighter = None

        if self.fusion_type == "subgraph":
            self.transformer_encoder = nn.Sequential(*[
                SubGraphFusion(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ])
            print("--> [ConvNeXtRegionAttention] Using SubGraph Fusion.")
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.embed_dim * 2,
                dropout=self.dropout_rate,
                batch_first=True,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_layers,
            )
            print("--> [ConvNeXtRegionAttention] Using standard Transformer encoder.")

        if self.use_region_slot_embed:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_output_regions, self.embed_dim) * 0.02
            )
        else:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, self.num_output_regions, self.embed_dim),
            )

        classifier_input_dim = (
            self.embed_dim * self.num_output_regions
            if self.region_pooling == "concat"
            else self.embed_dim
        )
        if self.use_global_feature_concat:
            classifier_input_dim += self.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(float(model_cfg.get("classifier_dropout1", 0.3))),
            nn.Linear(classifier_input_dim, self.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg.get("classifier_dropout2", 0.2))),
            nn.Linear(self.classifier_hidden_dim, num_classes),
        )
        if self.learnable_logit_fusion:
            init_weight = min(
                max(
                    self.learnable_logit_fusion_init,
                    self.learnable_logit_fusion_min,
                ),
                self.learnable_logit_fusion_max,
            )
            span = self.learnable_logit_fusion_max - self.learnable_logit_fusion_min
            normalized_init = (init_weight - self.learnable_logit_fusion_min) / span
            normalized_init = min(max(normalized_init, 1e-4), 1.0 - 1e-4)
            alpha_init = math.log(normalized_init / (1.0 - normalized_init))
            self.logit_fusion_alpha = nn.Parameter(
                torch.tensor(alpha_init, dtype=torch.float32)
            )
        else:
            self.register_parameter("logit_fusion_alpha", None)

        if self.use_cnn_aux_classifier:
            aux_hidden_dim = int(model_cfg.get("cnn_aux_hidden_dim", self.classifier_hidden_dim))
            aux_dropout = float(model_cfg.get("cnn_aux_dropout", 0.25))
            aux_input_dim = self.visual_dim * 2 if self.cnn_aux_pooling == "avgmax" else self.visual_dim
            self.cnn_aux_classifier = nn.Sequential(
                nn.LayerNorm(aux_input_dim),
                nn.Dropout(aux_dropout),
                nn.Linear(aux_input_dim, aux_hidden_dim),
                nn.GELU(),
                nn.Dropout(aux_dropout),
                nn.Linear(aux_hidden_dim, num_classes),
            )
            print(
                "--> [ConvNeXtRegionAttention] CNN auxiliary classifier enabled: "
                f"pooling={self.cnn_aux_pooling}, input_dim={aux_input_dim}, "
                f"hidden_dim={aux_hidden_dim}, dropout={aux_dropout}, "
                f"use_loss={self.use_cnn_aux_loss}, use_logits={self.use_cnn_aux_logits}, "
                f"logit_weight={self.cnn_aux_logit_weight}"
            )
        else:
            self.cnn_aux_classifier = None
        if self.learnable_logit_fusion:
            print(
                "--> [ConvNeXtRegionAttention] Learnable CNN/region logit "
                "fusion enabled: "
                f"cnn_init={self.current_cnn_logit_weight():.4f}, "
                f"bounds=[{self.learnable_logit_fusion_min:.2f}, "
                f"{self.learnable_logit_fusion_max:.2f}]"
            )

        checkpoint_path = model_cfg.get("checkpoint_path")
        if checkpoint_path:
            self.load_pretrained_backbones(checkpoint_path, device="cpu")
        self.freeze_backbones()

    def load_pretrained_backbones(self, convnext_ckpt_path=None, device="cpu", **kwargs):
        if convnext_ckpt_path is None:
            convnext_ckpt_path = kwargs.get("pretrained_convnext_path")
        checkpoint_path = self.convnext_backbone.resolve_checkpoint_path(convnext_ckpt_path)
        checkpoint = safe_torch_load(checkpoint_path, map_location=device)
        state_dict = strip_known_prefixes(extract_state_dict(checkpoint))

        full_model_prefixes = (
            "convnext_backbone.",
            "region_dict.",
            "alignment.",
            "eye_fusion.",
            "region_relation_builder.",
            "visual_proj.",
            "region_weighter.",
            "cnn_aux_classifier.",
            "transformer_encoder.",
            "classifier.",
            "visual_pos_embed",
            "pos_embed",
        )
        if any(key.startswith(full_model_prefixes) for key in state_dict):
            print(
                "--> [ConvNeXtRegionAttention] Loading full region-attention "
                f"checkpoint: {checkpoint_path}"
            )
            incompatible = self.load_state_dict(state_dict, strict=self.checkpoint_strict)
            if incompatible.missing_keys:
                print(
                    "--> [ConvNeXtRegionAttention] Missing keys after full load: "
                    f"{len(incompatible.missing_keys)}"
                )
            if incompatible.unexpected_keys:
                print(
                    "--> [ConvNeXtRegionAttention] Unexpected keys after full load: "
                    f"{len(incompatible.unexpected_keys)}"
                )
            return

        self.convnext_backbone.load_from_checkpoint(checkpoint_path, device=device)

    def freeze_backbones(self):
        for param in self.convnext_backbone.backbone.features.parameters():
            param.requires_grad = False
        if self.convnext_backbone.source_classifier is not None:
            for param in self.convnext_backbone.source_classifier.parameters():
                param.requires_grad = False
        self.is_frozen = True
        print("[ConvNeXtRegionAttention] ConvNeXt visual backbone FROZEN.")

    def unfreeze_backbones(self):
        for param in self.convnext_backbone.backbone.features.parameters():
            param.requires_grad = True
        if self.logit_fusion in ("attention", "sum") and self.convnext_backbone.source_classifier is not None:
            for param in self.convnext_backbone.source_classifier.parameters():
                param.requires_grad = False
        previous_logit_fusion = self.logit_fusion
        self.logit_fusion = self.finetune_logit_fusion
        self.is_frozen = False
        print("[ConvNeXtRegionAttention] ConvNeXt visual backbone UNFROZEN.")
        if previous_logit_fusion != self.logit_fusion:
            print(
                "[ConvNeXtRegionAttention] "
                f"logit_fusion switched: {previous_logit_fusion} -> {self.logit_fusion}"
            )

    @staticmethod
    def _freeze_norm_layers(module):
        for child in module.modules():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                child.eval()

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            return self

        if self.is_frozen:
            self.convnext_backbone.backbone.features.eval()
            if self.convnext_backbone.source_classifier is not None:
                self.convnext_backbone.source_classifier.eval()
        else:
            if self.freeze_unfrozen_batchnorm:
                self._freeze_norm_layers(self.convnext_backbone.backbone.features)
            if self.logit_fusion in ("attention", "sum") and self.convnext_backbone.source_classifier is not None:
                self.convnext_backbone.source_classifier.eval()
        return self

    def check_unfreeze(self, epoch):
        if (
            self.unfreeze_backbone
            and self.is_frozen
            and self.freeze_epochs > 0
            and epoch >= self.freeze_epochs
        ):
            self.unfreeze_backbones()
            return True
        return False

    def set_epoch(self, epoch_index):
        self.current_epoch_index = int(epoch_index)

    def parameter_role(self, clean_name):
        if clean_name.startswith("convnext_backbone.backbone.features."):
            return "visual"
        return "head"

    @staticmethod
    def _normalize_region_group(value):
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        raise ValueError("region relation group must be an int or a list of ints.")

    def _parse_region_relation_pairs(self, relation_pairs):
        if relation_pairs is None:
            return [
                {"name": "brow_eye", "left": [0], "right": [1, 2]},
                {"name": "eye_nose", "left": [1, 2], "right": [3]},
                {"name": "nose_mouth", "left": [3], "right": [4]},
                {"name": "eyes_mouth", "left": [1, 2], "right": [4]},
            ]

        parsed = []
        for index, pair in enumerate(relation_pairs):
            if isinstance(pair, dict):
                name = str(pair.get("name", f"relation_{index}"))
                left = self._normalize_region_group(pair.get("left"))
                right = self._normalize_region_group(pair.get("right"))
            elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                name = f"relation_{index}"
                left = self._normalize_region_group(pair[0])
                right = self._normalize_region_group(pair[1])
            else:
                raise ValueError(
                    "Each model.region_relation_pairs entry must be either "
                    "{name,left,right} or a two-item list."
                )
            parsed.append({"name": name, "left": left, "right": right})
        return parsed

    def _validate_region_relation_pairs(self, relation_pairs):
        if not relation_pairs:
            raise ValueError(
                "model.use_region_relation_tokens=true requires at least one "
                "region relation pair."
            )
        for pair in relation_pairs:
            for side in ("left", "right"):
                indices = pair[side]
                if not indices:
                    raise ValueError(f"Region relation '{pair['name']}' has empty {side} group.")
                for region_index in indices:
                    if not 0 <= region_index < self.num_regions:
                        raise ValueError(
                            f"Region relation '{pair['name']}' index {region_index} "
                            f"is out of range for num_regions={self.num_regions}."
                        )

    def _pool_region_features(self, encoded, region_weights=None):
        if self.region_pooling == "concat":
            return encoded.reshape(encoded.size(0), -1)
        if region_weights is not None:
            return encoded.sum(dim=1)
        return encoded.mean(dim=1)

    def _apply_dynamic_region_weighting(self, encoded, global_feat):
        if self.region_weighter is None:
            return encoded, None

        region_weights = self.region_weighter(global_feat)
        if region_weights.size(1) != encoded.size(1):
            raise ValueError(
                f"Dynamic region weights K={region_weights.size(1)} do not match "
                f"encoded regions K={encoded.size(1)}."
            )
        return encoded * region_weights.unsqueeze(-1), region_weights

    def _append_eye_fusion_token(self, region_features):
        if self.eye_fusion is None:
            return region_features

        left_eye = region_features[:, self.eye_fusion_left_index, :]
        right_eye = region_features[:, self.eye_fusion_right_index, :]
        eye_pair = torch.cat((left_eye, right_eye), dim=-1)
        eye_token = self.eye_fusion(eye_pair).unsqueeze(1)
        return torch.cat((region_features, eye_token), dim=1)

    def _append_region_relation_tokens(self, region_features):
        if self.region_relation_builder is None:
            return region_features
        return self.region_relation_builder(region_features)

    def _append_eye_union_mask(self, flat_masks):
        if (
            flat_masks is None
            or not self.use_eye_fusion_token
            or self.eye_fusion_mode != "mask_union"
        ):
            return flat_masks

        left_eye_mask = flat_masks[:, self.eye_fusion_left_index, :]
        right_eye_mask = flat_masks[:, self.eye_fusion_right_index, :]
        eye_union_mask = torch.maximum(left_eye_mask, right_eye_mask).unsqueeze(1)
        return torch.cat((flat_masks, eye_union_mask), dim=1)

    def _learnable_cnn_logit_weight(self):
        if not self.learnable_logit_fusion:
            return None

        gate = torch.sigmoid(self.logit_fusion_alpha)
        span = self.learnable_logit_fusion_max - self.learnable_logit_fusion_min
        return self.learnable_logit_fusion_min + span * gate

    def current_cnn_logit_weight(self):
        if not self.learnable_logit_fusion:
            return self.cnn_aux_logit_weight

        with torch.no_grad():
            return float(self._learnable_cnn_logit_weight().detach().cpu().item())

    def current_region_logit_weight(self):
        if not self.learnable_logit_fusion:
            return self.attention_logit_weight
        return 1.0 - self.current_cnn_logit_weight()

    def _combine_logits(self, attention_logits, source_logits=None, cnn_aux_logits=None):
        if self.logit_fusion == "source":
            if source_logits is None:
                raise RuntimeError("logit_fusion='source' needs a checkpoint classifier.")
            return source_logits

        if self.logit_fusion == "sum" and source_logits is not None:
            return (
                self.attention_logit_weight * attention_logits
                + self.source_logit_weight * source_logits
            )

        if self.logit_fusion == "cnn_aux":
            if cnn_aux_logits is None:
                raise RuntimeError("logit_fusion='cnn_aux' needs model.use_cnn_aux_logits: true.")
            return cnn_aux_logits

        if self.logit_fusion == "cnn_aux_sum":
            if cnn_aux_logits is None:
                raise RuntimeError("logit_fusion='cnn_aux_sum' needs model.use_cnn_aux_logits: true.")
            if self.learnable_logit_fusion:
                cnn_weight = self._learnable_cnn_logit_weight().to(
                    device=cnn_aux_logits.device,
                    dtype=cnn_aux_logits.dtype,
                )
                region_weight = 1.0 - cnn_weight
                return region_weight * attention_logits + cnn_weight * cnn_aux_logits
            return (
                self.attention_logit_weight * attention_logits
                + self.cnn_aux_logit_weight * cnn_aux_logits
            )

        return attention_logits

    def _cnn_aux_features(self, global_feat, global_max_feat=None):
        if self.cnn_aux_pooling == "avg":
            return global_feat
        if global_max_feat is None:
            raise RuntimeError("cnn_aux_pooling='avgmax' needs max-pooled ConvNeXt features.")
        return torch.cat((global_feat, global_max_feat), dim=-1)

    def _region_tokens(self, batch_size):
        if not self.use_learnable_clip_region_tokens:
            return self.region_dict(batch_size)

        tokens = self.learned_region_dict(batch_size)
        if self.clip_region_dict is None:
            return tokens

        gamma = self.clip_region_gamma.to(device=tokens.device, dtype=tokens.dtype)
        clip_tokens = self.clip_region_dict(batch_size).to(device=tokens.device, dtype=tokens.dtype)
        return tokens + gamma * clip_tokens

    def _flatten_region_masks(self, region_masks, visual_features):
        if region_masks is None:
            return None

        flat_masks = region_masks.view(region_masks.size(0), self.num_regions, -1)
        if flat_masks.size(2) != visual_features.size(1):
            raise ValueError(
                f"region_masks spatial size {region_masks.shape[2:]} "
                f"flattens to {flat_masks.size(2)} tokens, but the backbone "
                f"produces {visual_features.size(1)} visual tokens. "
                "Make sure mask_size matches the ConvNeXt visual token grid."
            )
        return self._append_eye_union_mask(flat_masks)

    def forward(self, x, region_masks=None):
        batch_size = x.shape[0]
        backbone_outputs = self.convnext_backbone(x)
        if len(backbone_outputs) == 3:
            visual_features, global_feat, pooled_map = backbone_outputs
            global_max_feat = None
        else:
            visual_features, global_feat, pooled_map, global_max_feat = backbone_outputs
        if visual_features.size(1) != self.visual_pos_embed.size(1):
            raise ValueError(
                f"visual_pos_embed expects {self.visual_pos_embed.size(1)} tokens, "
                f"but the backbone returned {visual_features.size(1)}. "
                "Check image_size or pool_visual_tokens in config."
            )
        visual_features = visual_features + self.visual_pos_embed

        flat_masks = self._flatten_region_masks(region_masks, visual_features)
        region_tokens = self._region_tokens(batch_size)
        if self.mask_guided_attention:
            phi_sem, attn_weights = self.alignment(
                region_tokens,
                visual_features,
                region_masks=flat_masks,
            )
        else:
            phi_sem, attn_weights = self.alignment(region_tokens, visual_features)

        hyper_visual = (
            self._append_eye_fusion_token(phi_sem)
            if self.eye_fusion_mode == "post"
            else phi_sem
        )
        hyper_visual = self._append_region_relation_tokens(hyper_visual)
        hyper_visual = hyper_visual + self.pos_embed
        global_context = (
            self.visual_proj(global_feat)
            if (self.use_global_visual_bias or self.use_global_feature_concat)
            else None
        )
        if self.use_global_visual_bias:
            hyper_visual = hyper_visual + global_context.unsqueeze(1)

        encoded = self.transformer_encoder(hyper_visual)
        encoded, region_weights = self._apply_dynamic_region_weighting(encoded, global_feat)
        pooled = self._pool_region_features(encoded, region_weights=region_weights)
        if self.use_global_feature_concat:
            pooled = torch.cat((pooled, global_context), dim=-1)
        attention_logits = self.classifier(pooled)
        cnn_aux_feat = self._cnn_aux_features(global_feat, global_max_feat)
        cnn_aux_logits = (
            self.cnn_aux_classifier(cnn_aux_feat)
            if self.cnn_aux_classifier is not None
            else None
        )

        source_logits = None
        if self.logit_fusion in ("source", "sum"):
            source_logits = self.convnext_backbone.source_logits(pooled_map)
        logits = self._combine_logits(attention_logits, source_logits, cnn_aux_logits)

        attn_norm = F.normalize(attn_weights, p=2, dim=-1)
        sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        mask = torch.eye(sim.size(1), device=sim.device).bool()
        off_diag_sim = sim[:, ~mask]
        if self.ortho_loss_type == "squared_offdiag":
            aux_loss = off_diag_sim.pow(2).mean()
        else:
            aux_loss = off_diag_sim.mean()

        if self.training:
            if cnn_aux_logits is not None:
                return logits, aux_loss, cnn_aux_logits
            return logits, aux_loss

        if self.return_region_weights:
            if self.return_attn:
                return logits, attn_weights, region_weights
            return logits, region_weights

        if self.return_attn:
            return logits, attn_weights

        return logits
