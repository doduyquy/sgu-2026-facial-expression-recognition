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


class ConvNeXtSpatialTokenizer(nn.Module):
    """ConvNeXt ImageNet features -> local visual tokens for region attention."""

    _WEIGHT_ENUMS = {
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
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

    def forward(self, x):
        feat_map = self.backbone.features(x)
        token_map = self.token_pool(feat_map)
        visual_tokens = token_map.flatten(2).transpose(1, 2)
        pooled_map = F.adaptive_avg_pool2d(feat_map, 1)
        global_feat = torch.flatten(pooled_map, 1)
        return visual_tokens, global_feat, pooled_map

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
        self.fusion_type = model_cfg.get("fusion_type", "transformer")
        self.region_pooling = model_cfg.get("region_pooling", "concat").lower()
        self.classifier_hidden_dim = int(model_cfg.get("classifier_hidden_dim", 1024))
        self.ortho_loss_type = model_cfg.get("ortho_loss_type", "squared_offdiag").lower()
        self.logit_fusion = model_cfg.get("logit_fusion", "attention")
        self.finetune_logit_fusion = model_cfg.get("finetune_logit_fusion", self.logit_fusion)
        self.attention_logit_weight = float(model_cfg.get("attention_logit_weight", 1.0))
        self.source_logit_weight = float(model_cfg.get("source_logit_weight", 1.0))
        self.freeze_epochs = int(model_cfg.get("freeze_backbone_epochs", 0))
        self.unfreeze_backbone = bool(model_cfg.get("unfreeze_backbone", False))
        self.unfreeze_backbone_scope = model_cfg.get("unfreeze_backbone_scope", "all").lower()
        self.freeze_unfrozen_batchnorm = bool(model_cfg.get("freeze_unfrozen_batchnorm", False))
        self.current_epoch_index = 0
        self.is_frozen = False
        self.return_attn = False
        self.checkpoint_strict = bool(model_cfg.get("checkpoint_strict", False))

        num_classes = int(data_cfg.get("num_classes", 7))
        if self.region_pooling not in ("mean", "concat"):
            raise ValueError("model.region_pooling must be one of: mean, concat")
        if self.ortho_loss_type not in ("mean_offdiag", "squared_offdiag"):
            raise ValueError("model.ortho_loss_type must be one of: mean_offdiag, squared_offdiag")
        if self.logit_fusion not in ("attention", "source", "sum"):
            raise ValueError("model.logit_fusion must be one of: attention, source, sum")
        if self.finetune_logit_fusion not in ("attention", "source", "sum"):
            raise ValueError("model.finetune_logit_fusion must be one of: attention, source, sum")

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
        if self.use_clip_dictionary:
            clip_model_name = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
            try:
                self.region_dict = CLIPFacialRegionDictionary(
                    num_regions=self.num_regions,
                    embed_dim=self.embed_dim,
                    clip_model_name=clip_model_name,
                )
                print(f"--> [ConvNeXtRegionAttention] CLIP text region tokens: K={self.num_regions}")
            except Exception as exc:
                if not bool(model_cfg.get("clip_fallback_to_learned", True)):
                    raise
                print(
                    "--> [ConvNeXtRegionAttention] CLIP region tokens unavailable; "
                    f"using learned region tokens instead. Reason: {exc}"
                )
                self.region_dict = FacialRegionDictionary(
                    num_regions=self.num_regions,
                    embed_dim=self.embed_dim,
                )
        else:
            self.region_dict = FacialRegionDictionary(
                num_regions=self.num_regions,
                embed_dim=self.embed_dim,
            )
            print(f"--> [ConvNeXtRegionAttention] Learned region tokens: K={self.num_regions}")

        self.alignment = CrossDimSemanticVisualAlignment(
            embed_dim=self.embed_dim,
            visual_dim=self.visual_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
        )

        if self.use_global_visual_bias:
            self.visual_proj = nn.Sequential(
                nn.LayerNorm(self.visual_dim),
                nn.Linear(self.visual_dim, self.embed_dim),
                nn.Dropout(self.dropout_rate),
            )
        else:
            self.visual_proj = None

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
                torch.randn(1, self.num_regions, self.embed_dim) * 0.02
            )
        else:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, self.num_regions, self.embed_dim),
            )

        classifier_input_dim = (
            self.embed_dim * self.num_regions
            if self.region_pooling == "concat"
            else self.embed_dim
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(float(model_cfg.get("classifier_dropout1", 0.3))),
            nn.Linear(classifier_input_dim, self.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg.get("classifier_dropout2", 0.2))),
            nn.Linear(self.classifier_hidden_dim, num_classes),
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
            "visual_proj.",
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

    def _pool_region_features(self, encoded):
        if self.region_pooling == "concat":
            return encoded.reshape(encoded.size(0), -1)
        return encoded.mean(dim=1)

    def _combine_logits(self, attention_logits, source_logits):
        if self.logit_fusion == "source":
            if source_logits is None:
                raise RuntimeError("logit_fusion='source' needs a checkpoint classifier.")
            return source_logits

        if self.logit_fusion == "sum" and source_logits is not None:
            return (
                self.attention_logit_weight * attention_logits
                + self.source_logit_weight * source_logits
            )

        return attention_logits

    def forward(self, x):
        batch_size = x.shape[0]
        visual_features, global_feat, pooled_map = self.convnext_backbone(x)
        if visual_features.size(1) != self.visual_pos_embed.size(1):
            raise ValueError(
                f"visual_pos_embed expects {self.visual_pos_embed.size(1)} tokens, "
                f"but the backbone returned {visual_features.size(1)}. "
                "Check image_size or pool_visual_tokens in config."
            )
        visual_features = visual_features + self.visual_pos_embed

        region_tokens = self.region_dict(batch_size)
        phi_sem, attn_weights = self.alignment(region_tokens, visual_features)

        hyper_visual = phi_sem + self.pos_embed
        if self.use_global_visual_bias:
            phi_visual = self.visual_proj(global_feat).unsqueeze(1)
            hyper_visual = hyper_visual + phi_visual

        encoded = self.transformer_encoder(hyper_visual)
        pooled = self._pool_region_features(encoded)
        attention_logits = self.classifier(pooled)

        source_logits = None
        if self.logit_fusion in ("source", "sum"):
            source_logits = self.convnext_backbone.source_logits(pooled_map)
        logits = self._combine_logits(attention_logits, source_logits)

        attn_norm = F.normalize(attn_weights, p=2, dim=-1)
        sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        mask = torch.eye(sim.size(1), device=sim.device).bool()
        off_diag_sim = sim[:, ~mask]
        if self.ortho_loss_type == "squared_offdiag":
            aux_loss = off_diag_sim.pow(2).mean()
        else:
            aux_loss = off_diag_sim.mean()

        if self.training:
            return logits, aux_loss

        if self.return_attn:
            return logits, attn_weights

        return logits
