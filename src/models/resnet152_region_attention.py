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


class ResNet152SpatialTokenizer(nn.Module):
    """
    ResNet152 feature maps -> visual tokens for region attention.

    Single-scale modes expose layer2/layer3/layer4 tokens directly.
    ``layer3_layer4`` keeps both top stages as separate token sequences, while
    ``layer3_layer4_fused`` first fuses layer3 local detail with layer4
    semantics on the same spatial grid.
    """

    def __init__(self, config, channels=3):
        super().__init__()
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})

        self.num_classes = data_cfg.get("num_classes", 7)
        image_size = data_cfg.get("image_size", 224)
        self.token_grid_size = model_cfg.get("token_grid_size", 3)
        self.pool_visual_tokens = model_cfg.get("pool_visual_tokens", False)
        self.feature_layer = model_cfg.get("feature_layer", "layer4")
        self.multi_scale_tokens = self.feature_layer == "layer3_layer4"
        self.fused_multiscale_tokens = self.feature_layer == "layer3_layer4_fused"
        feature_dims = {
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }
        output_strides = {
            "layer2": 8,
            "layer3": 16,
            "layer4": 32,
        }
        if self.feature_layer not in (
            *feature_dims.keys(),
            "layer3_layer4",
            "layer3_layer4_fused",
        ):
            raise ValueError(
                "model.feature_layer must be one of: "
                "layer2, layer3, layer4, layer3_layer4, layer3_layer4_fused"
            )
        self.feature_dim = (
            model_cfg.get("multiscale_visual_dim", model_cfg.get("embed_dim", 512))
            if self.multi_scale_tokens or self.fused_multiscale_tokens
            else feature_dims[self.feature_layer]
        )
        self.source_feature_dim = (
            2048
            if self.multi_scale_tokens or self.fused_multiscale_tokens
            else self.feature_dim
        )

        self.backbone = models.resnet152(weights=None)
        if channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.backbone.fc = nn.Identity()
        if self.feature_layer == "layer2":
            self.backbone.layer3 = nn.Identity()
            self.backbone.layer4 = nn.Identity()
        elif self.feature_layer == "layer3":
            self.backbone.layer4 = nn.Identity()

        if self.multi_scale_tokens:
            self.layer3_native_grid_size = max(1, image_size // output_strides["layer3"])
            self.layer4_native_grid_size = max(1, image_size // output_strides["layer4"])
            self.layer3_visual_grid_size = (
                self.token_grid_size if self.pool_visual_tokens else self.layer3_native_grid_size
            )
            self.layer4_visual_grid_size = (
                self.token_grid_size if self.pool_visual_tokens else self.layer4_native_grid_size
            )
            self.layer3_token_pool = (
                nn.AdaptiveAvgPool2d((self.token_grid_size, self.token_grid_size))
                if self.pool_visual_tokens
                else nn.Identity()
            )
            self.layer4_token_pool = (
                nn.AdaptiveAvgPool2d((self.token_grid_size, self.token_grid_size))
                if self.pool_visual_tokens
                else nn.Identity()
            )
            self.layer3_token_proj = nn.Sequential(
                nn.LayerNorm(feature_dims["layer3"]),
                nn.Linear(feature_dims["layer3"], self.feature_dim),
            )
            self.layer4_token_proj = nn.Sequential(
                nn.LayerNorm(feature_dims["layer4"]),
                nn.Linear(feature_dims["layer4"], self.feature_dim),
            )
            self.layer3_num_visual_tokens = self.layer3_visual_grid_size ** 2
            self.layer4_num_visual_tokens = self.layer4_visual_grid_size ** 2
            self.num_visual_tokens = (
                self.layer3_num_visual_tokens + self.layer4_num_visual_tokens
            )
        elif self.fused_multiscale_tokens:
            self.fused_grid_size = max(1, image_size // output_strides["layer4"])
            norm_groups = int(model_cfg.get("multiscale_norm_groups", 32))
            while self.feature_dim % norm_groups != 0 and norm_groups > 1:
                norm_groups //= 2
            self.fused_projection = nn.Sequential(
                nn.Conv2d(
                    feature_dims["layer3"] + feature_dims["layer4"],
                    self.feature_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(norm_groups, self.feature_dim),
                nn.GELU(),
            )
            self.num_visual_tokens = self.fused_grid_size ** 2
        else:
            self.native_grid_size = max(1, image_size // output_strides[self.feature_layer])
            self.visual_grid_size = (
                self.token_grid_size if self.pool_visual_tokens else self.native_grid_size
            )
            self.token_pool = (
                nn.AdaptiveAvgPool2d((self.token_grid_size, self.token_grid_size))
                if self.pool_visual_tokens
                else nn.Identity()
            )
            self.num_visual_tokens = self.visual_grid_size ** 2

        self.source_fc = nn.Linear(self.source_feature_dim, self.num_classes)
        self.has_source_classifier = False
        if self.multi_scale_tokens:
            print(
                f"--> [ResNet152Tokenizer] feature_layer={self.feature_layer}, "
                f"feature_dim={self.feature_dim}, tokens="
                f"{self.layer3_num_visual_tokens}+{self.layer4_num_visual_tokens}="
                f"{self.num_visual_tokens}, pool_visual_tokens={self.pool_visual_tokens}"
            )
        elif self.fused_multiscale_tokens:
            print(
                f"--> [ResNet152Tokenizer] feature_layer={self.feature_layer}, "
                f"feature_dim={self.feature_dim}, tokens="
                f"{self.fused_grid_size}x{self.fused_grid_size}, "
                "fusion=layer3_pool_to_layer4+1x1"
            )
        else:
            print(
                f"--> [ResNet152Tokenizer] feature_layer={self.feature_layer}, "
                f"feature_dim={self.feature_dim}, tokens="
                f"{self.visual_grid_size}x{self.visual_grid_size}, "
                f"pool_visual_tokens={self.pool_visual_tokens}"
            )

    def forward_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        if self.feature_layer == "layer2":
            return x

        feat3 = self.backbone.layer3(x)
        if self.feature_layer == "layer3":
            return feat3

        feat4 = self.backbone.layer4(feat3)
        if self.multi_scale_tokens or self.fused_multiscale_tokens:
            return feat3, feat4
        return feat4

    def forward(self, x):
        features = self.forward_features(x)
        if self.multi_scale_tokens:
            feat3, feat4 = features
            layer3_tokens = self.layer3_token_pool(feat3).flatten(2).transpose(1, 2)
            layer4_tokens = self.layer4_token_pool(feat4).flatten(2).transpose(1, 2)
            tokens = torch.cat(
                (
                    self.layer3_token_proj(layer3_tokens),
                    self.layer4_token_proj(layer4_tokens),
                ),
                dim=1,
            )
            global_feat = F.adaptive_avg_pool2d(feat4, 1).flatten(1)
        elif self.fused_multiscale_tokens:
            feat3, feat4 = features
            layer3_map = F.adaptive_avg_pool2d(feat3, feat4.shape[-2:])
            fused_map = torch.cat((layer3_map, feat4), dim=1)
            token_map = self.fused_projection(fused_map)
            tokens = token_map.flatten(2).transpose(1, 2)
            global_feat = F.adaptive_avg_pool2d(feat4, 1).flatten(1)
        else:
            feat_map = features
            token_map = self.token_pool(feat_map)
            tokens = token_map.flatten(2).transpose(1, 2)
            global_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        return tokens, global_feat

    def source_logits(self, global_feat):
        if not self.has_source_classifier:
            return None
        return self.source_fc(global_feat)

    def load_from_checkpoint(self, checkpoint_path, device="cpu"):
        checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
        print(f"--> [ResNet152Tokenizer] Loading source checkpoint: {checkpoint_path}")

        checkpoint = safe_torch_load(checkpoint_path, map_location=device)
        state_dict = strip_known_prefixes(extract_state_dict(checkpoint))

        backbone_state = {}
        source_fc_state = {}
        skipped = []

        backbone_ref = self.backbone.state_dict()
        source_fc_ref = self.source_fc.state_dict()

        for key, value in state_dict.items():
            name = key
            if name.startswith("backbone."):
                name = name[len("backbone."):]

            if name.startswith("fc."):
                fc_name = name[len("fc."):]
                if fc_name in source_fc_ref and source_fc_ref[fc_name].shape == value.shape:
                    source_fc_state[fc_name] = value
                else:
                    skipped.append(key)
                continue

            if name in backbone_ref and backbone_ref[name].shape == value.shape:
                backbone_state[name] = value
            elif key.startswith("head.") or key.startswith("source_fc."):
                skipped.append(key)
            else:
                skipped.append(key)

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if source_fc_state:
            self.source_fc.load_state_dict(source_fc_state, strict=False)
            self.has_source_classifier = True

        print(f"--> [ResNet152Tokenizer] Backbone loaded: {len(backbone_state)} tensors")
        if self.has_source_classifier:
            print("--> [ResNet152Tokenizer] Source fc loaded for baseline/residual logits.")
        elif self.source_feature_dim != 2048:
            print(
                "--> [ResNet152Tokenizer] Source fc is unavailable because the backbone "
                f"is truncated at {self.feature_layer}."
            )
        if missing:
            print(f"--> [ResNet152Tokenizer] Missing backbone keys: {len(missing)}")
        if unexpected:
            print(f"--> [ResNet152Tokenizer] Unexpected backbone keys: {len(unexpected)}")
        if skipped:
            print(f"--> [ResNet152Tokenizer] Skipped checkpoint keys: {len(skipped)}")

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
                    print(f"--> [ResNet152Tokenizer] Using discovered checkpoint: {found}")
                    return found

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


class CrossDimSemanticVisualAlignment(nn.Module):
    """
    Cross-attention where CLIP/text region queries stay at ``embed_dim`` while
    the tokenizer may expose either single-scale or projected multi-scale
    visual keys/values.
    """

    def __init__(self, embed_dim=512, visual_dim=2048, num_heads=4, dropout=0.1):
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
        self.drop_path = DropPath(dropout if dropout > 0. else 0.)

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


class ResNet152RegionAttentionFER(nn.Module):
    """
    Region-token attention head on top of a pretrained ResNet152 backbone.

    The pretrained checkpoint supplies the visual feature extractor. The region
    dictionary, cross-attention, transformer encoder, and classifier are new
    layers and must be trained/fine-tuned before their accuracy is meaningful.
    """

    EMOTION_NAMES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
    REGION_NAMES = ("forehead", "left_eye", "right_eye", "nose", "mouth", "chin")

    def __init__(self, config, channels=3):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        self.embed_dim = model_cfg.get("embed_dim", 512)
        self.num_heads = model_cfg.get("num_heads", 4)
        self.num_regions = model_cfg.get("num_regions", 6)
        self.num_layers = model_cfg.get("num_encoder_layers", 2)
        self.dropout_rate = model_cfg.get("transformer_dropout", 0.1)
        self.token_grid_size = model_cfg.get("token_grid_size", 3)
        self.use_visual_pos_embed = model_cfg.get("use_visual_pos_embed", True)
        self.fusion_type = model_cfg.get("fusion_type", "transformer")
        self.logit_fusion = model_cfg.get("logit_fusion", "attention")
        self.finetune_logit_fusion = model_cfg.get("finetune_logit_fusion", self.logit_fusion)
        self.attention_logit_weight = model_cfg.get("attention_logit_weight", 1.0)
        self.source_logit_weight = model_cfg.get("source_logit_weight", 1.0)
        self.freeze_epochs = model_cfg.get("freeze_backbone_epochs", 0)
        self.unfreeze_backbone = model_cfg.get("unfreeze_backbone", False)
        self.unfreeze_backbone_scope = model_cfg.get("unfreeze_backbone_scope", "all").lower()
        self.freeze_unfrozen_batchnorm = model_cfg.get("freeze_unfrozen_batchnorm", False)
        self.region_pooling = model_cfg.get("region_pooling", "mean").lower()
        self.classifier_hidden_dim = model_cfg.get("classifier_hidden_dim", 512)
        self.ortho_loss_type = model_cfg.get("ortho_loss_type", "mean_offdiag").lower()
        self.use_global_visual_bias = model_cfg.get("use_global_visual_bias", True)
        self.use_region_slot_embed = model_cfg.get("use_region_slot_embed", True)
        self.is_frozen = False
        self.return_attn = False

        num_classes = data_cfg.get("num_classes", 7)
        self.num_classes = num_classes
        if self.unfreeze_backbone_scope not in ("all", "layer3", "layer4", "layer3_layer4"):
            raise ValueError(
                "model.unfreeze_backbone_scope must be one of: "
                "all, layer3, layer4, layer3_layer4"
            )
        if self.logit_fusion not in ("attention", "source", "sum"):
            raise ValueError("model.logit_fusion must be one of: attention, source, sum")
        if self.finetune_logit_fusion not in ("attention", "source", "sum"):
            raise ValueError(
                "model.finetune_logit_fusion must be one of: attention, source, sum"
            )
        if self.region_pooling not in ("mean", "concat"):
            raise ValueError("model.region_pooling must be one of: mean, concat")
        if self.ortho_loss_type not in ("mean_offdiag", "squared_offdiag"):
            raise ValueError(
                "model.ortho_loss_type must be one of: mean_offdiag, squared_offdiag"
            )

        self.res_backbone = ResNet152SpatialTokenizer(config, channels=channels)
        self.visual_dim = self.res_backbone.feature_dim
        if self.logit_fusion == "attention":
            # Pure attention mode never consumes the checkpoint classifier head.
            # Keep it frozen so DDP does not wait for gradients that cannot exist.
            for param in self.res_backbone.source_fc.parameters():
                param.requires_grad = False
        num_visual_tokens = self.res_backbone.num_visual_tokens
        if self.use_visual_pos_embed:
            self.visual_pos_embed = nn.Parameter(
                torch.randn(1, num_visual_tokens, self.visual_dim) * 0.02
            )
        else:
            self.register_buffer("visual_pos_embed", torch.zeros(1, num_visual_tokens, self.visual_dim))

        self.use_clip_dictionary = model_cfg.get("use_clip_dictionary", True)
        if self.use_clip_dictionary:
            clip_model_name = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
            self.region_dict = CLIPFacialRegionDictionary(
                num_regions=self.num_regions,
                embed_dim=self.embed_dim,
                clip_model_name=clip_model_name,
            )
            print(f"--> [ResNet152RegionAttention] CLIP text region tokens: K={self.num_regions}")
        else:
            self.region_dict = FacialRegionDictionary(
                num_regions=self.num_regions,
                embed_dim=self.embed_dim,
            )
            print(f"--> [ResNet152RegionAttention] Learned region tokens: K={self.num_regions}")

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
            print("--> [ResNet152RegionAttention] Using SubGraph Fusion.")
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
            print("--> [ResNet152RegionAttention] Using standard Transformer encoder.")

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
            nn.Dropout(model_cfg.get("classifier_dropout1", 0.3)),
            nn.Linear(classifier_input_dim, self.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_cfg.get("classifier_dropout2", 0.2)),
            nn.Linear(self.classifier_hidden_dim, num_classes),
        )
        self._init_emotion_region_prior(model_cfg, num_classes)

        checkpoint_path = model_cfg.get("checkpoint_path")
        if checkpoint_path:
            self.load_pretrained_backbones(checkpoint_path, device="cpu")

    def _init_emotion_region_prior(self, model_cfg, num_classes):
        prior_cfg = model_cfg.get("emotion_region_prior", {})
        self.use_emotion_region_prior = prior_cfg.get("enabled", False)
        self.emotion_prior_mode = prior_cfg.get("mode", "gate").lower()
        self.emotion_prior_strength = float(prior_cfg.get("strength", 0.25))
        self.emotion_prior_temperature = float(prior_cfg.get("temperature", 1.5))
        self.detach_emotion_prior_prob = prior_cfg.get("detach_emotion_prob", True)
        self.emotion_prior_normalize = prior_cfg.get("normalize", "mean").lower()
        self.emotion_prior_min_gate = float(prior_cfg.get("min_gate", 0.5))
        self.emotion_prior_max_gate = float(prior_cfg.get("max_gate", 1.5))

        if self.emotion_prior_mode not in ("gate", "loss"):
            raise ValueError("model.emotion_region_prior.mode must be 'gate' or 'loss'")
        if self.emotion_prior_temperature <= 0:
            raise ValueError("model.emotion_region_prior.temperature must be > 0")
        if self.emotion_prior_normalize not in ("mean", "none"):
            raise ValueError("model.emotion_region_prior.normalize must be 'mean' or 'none'")
        if self.emotion_prior_min_gate <= 0 or self.emotion_prior_max_gate <= 0:
            raise ValueError("model.emotion_region_prior min_gate/max_gate must be > 0")
        if self.emotion_prior_min_gate > self.emotion_prior_max_gate:
            raise ValueError("model.emotion_region_prior.min_gate cannot exceed max_gate")

        matrix = prior_cfg.get("matrix")
        if matrix is None:
            matrix_tensor = self._default_emotion_region_prior()
        else:
            matrix_tensor = torch.tensor(matrix, dtype=torch.float32)

        expected_shape = (num_classes, self.num_regions)
        if tuple(matrix_tensor.shape) != expected_shape:
            raise ValueError(
                "model.emotion_region_prior.matrix must have shape "
                f"{expected_shape}, got {tuple(matrix_tensor.shape)}"
            )
        self.register_buffer("emotion_region_prior", matrix_tensor, persistent=False)

        if self.use_emotion_region_prior:
            message = (
                "--> [ResNet152RegionAttention] Emotion-region prior enabled: "
                f"mode={self.emotion_prior_mode}, "
                f"temperature={self.emotion_prior_temperature}, "
                f"detach={self.detach_emotion_prior_prob}"
            )
            if self.emotion_prior_mode == "gate":
                message += f", strength={self.emotion_prior_strength}"
            print(message)

    @staticmethod
    def _default_emotion_region_prior():
        # Rows: angry, disgust, fear, happy, sad, surprise, neutral.
        # Cols: forehead, left_eye, right_eye, nose, mouth, chin.
        return torch.tensor(
            [
                [1.0, 0.9, 0.9, 0.3, 0.5, 0.6],
                [0.4, 0.3, 0.3, 1.0, 0.8, 0.3],
                [0.7, 1.0, 1.0, 0.3, 0.8, 0.4],
                [0.1, 0.6, 0.6, 0.2, 1.0, 0.8],
                [0.5, 0.8, 0.8, 0.2, 0.7, 0.4],
                [0.6, 1.0, 1.0, 0.2, 1.0, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )

    def load_pretrained_backbones(self, resnet_ckpt_path=None, device="cpu", **kwargs):
        if resnet_ckpt_path is None:
            resnet_ckpt_path = kwargs.get("pretrained_resnet_path")
        self.res_backbone.load_from_checkpoint(resnet_ckpt_path, device=device)

    def freeze_backbones(self):
        for param in self.res_backbone.backbone.parameters():
            param.requires_grad = False
        if self.res_backbone.has_source_classifier:
            for param in self.res_backbone.source_fc.parameters():
                param.requires_grad = False
        self.is_frozen = True
        print("[ResNet152RegionAttention] ResNet152 backbone FROZEN.")

    def unfreeze_backbones(self):
        for param in self.res_backbone.backbone.parameters():
            param.requires_grad = False

        if self.unfreeze_backbone_scope == "layer4":
            trainable_modules = (self.res_backbone.backbone.layer4,)
            message = "ResNet152 layer4 UNFROZEN."
        elif self.unfreeze_backbone_scope == "layer3":
            trainable_modules = (self.res_backbone.backbone.layer3,)
            message = "ResNet152 layer3 UNFROZEN."
        elif self.unfreeze_backbone_scope == "layer3_layer4":
            trainable_modules = (
                self.res_backbone.backbone.layer3,
                self.res_backbone.backbone.layer4,
            )
            message = "ResNet152 layer3 + layer4 UNFROZEN."
        else:
            trainable_modules = (self.res_backbone.backbone,)
            message = "ResNet152 backbone UNFROZEN."

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

        previous_logit_fusion = self.logit_fusion
        self.logit_fusion = self.finetune_logit_fusion

        if self.logit_fusion in ("attention", "sum"):
            for param in self.res_backbone.source_fc.parameters():
                param.requires_grad = False
        self.is_frozen = False
        print(f"[ResNet152RegionAttention] {message}")
        if previous_logit_fusion != self.logit_fusion:
            print(
                "[ResNet152RegionAttention] "
                f"logit_fusion switched: {previous_logit_fusion} -> {self.logit_fusion}"
            )

    @staticmethod
    def _freeze_batchnorm(module):
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            return self

        if self.is_frozen:
            self.res_backbone.backbone.eval()
            self.res_backbone.source_fc.eval()
        else:
            if self.unfreeze_backbone_scope == "layer4":
                self.res_backbone.backbone.eval()
                self.res_backbone.backbone.layer4.train()
                if self.freeze_unfrozen_batchnorm:
                    self._freeze_batchnorm(self.res_backbone.backbone.layer4)
            elif self.unfreeze_backbone_scope == "layer3":
                self.res_backbone.backbone.eval()
                self.res_backbone.backbone.layer3.train()
                if self.freeze_unfrozen_batchnorm:
                    self._freeze_batchnorm(self.res_backbone.backbone.layer3)
            elif self.unfreeze_backbone_scope == "layer3_layer4":
                self.res_backbone.backbone.eval()
                self.res_backbone.backbone.layer3.train()
                self.res_backbone.backbone.layer4.train()
                if self.freeze_unfrozen_batchnorm:
                    self._freeze_batchnorm(self.res_backbone.backbone.layer3)
                    self._freeze_batchnorm(self.res_backbone.backbone.layer4)
            elif self.unfreeze_backbone_scope == "all":
                if self.freeze_unfrozen_batchnorm:
                    self._freeze_batchnorm(self.res_backbone.backbone)

            if self.logit_fusion in ("attention", "sum"):
                self.res_backbone.source_fc.eval()

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

    def _combine_logits(self, attention_logits, source_logits):
        if self.logit_fusion == "source":
            if source_logits is None:
                raise RuntimeError("logit_fusion='source' needs a checkpoint fc layer.")
            return source_logits

        if self.logit_fusion == "sum" and source_logits is not None:
            return (
                self.attention_logit_weight * attention_logits
                + self.source_logit_weight * source_logits
            )

        return attention_logits

    def _pool_region_features(self, encoded):
        if self.region_pooling == "concat":
            return encoded.reshape(encoded.size(0), -1)
        return encoded.mean(dim=1)

    def _emotion_region_weight(self, coarse_logits):
        emotion_prob = F.softmax(coarse_logits / self.emotion_prior_temperature, dim=-1)
        if self.detach_emotion_prior_prob:
            emotion_prob = emotion_prob.detach()

        prior = self.emotion_region_prior.to(dtype=emotion_prob.dtype)
        region_weight = torch.matmul(emotion_prob, prior)
        if self.emotion_prior_normalize == "mean":
            region_weight = region_weight / region_weight.mean(dim=1, keepdim=True).clamp_min(1e-6)
        return region_weight

    def _region_prior_gate(self, region_weight):
        gate = 1.0 + self.emotion_prior_strength * (region_weight - 1.0)
        return gate.clamp(
            min=self.emotion_prior_min_gate,
            max=self.emotion_prior_max_gate,
        )

    def _apply_emotion_region_prior(self, encoded, coarse_logits=None, region_weight=None):
        if region_weight is None:
            if coarse_logits is None:
                raise ValueError("coarse_logits is required when region_weight is not provided.")
            region_weight = self._emotion_region_weight(coarse_logits)
        gate = self._region_prior_gate(region_weight)
        return encoded * gate.unsqueeze(-1)

    def _apply_region_weight_to_attention(self, attn_weights, region_weight):
        if attn_weights.size(1) != region_weight.size(1):
            return attn_weights
        return attn_weights * region_weight.unsqueeze(-1).to(dtype=attn_weights.dtype)

    @staticmethod
    def _region_importance(encoded):
        scores = encoded.float().norm(p=2, dim=-1)
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, x):
        batch_size = x.shape[0]

        visual_features, global_feat = self.res_backbone(x)  # [B, N, visual_dim], [B, C]
        if visual_features.size(1) != self.visual_pos_embed.size(1):
            raise ValueError(
                f"visual_pos_embed expects {self.visual_pos_embed.size(1)} tokens, "
                f"but the backbone returned {visual_features.size(1)}. "
                "Check image_size, feature_layer, or pool_visual_tokens in config."
            )
        visual_features = visual_features + self.visual_pos_embed

        region_tokens = self.region_dict(batch_size)         # [B, 6, 512]
        phi_sem, attn_weights = self.alignment(
            region_tokens,
            visual_features,
        )                                                    # [B, 6, 512], [B, 6, 49]

        hyper_visual = phi_sem + self.pos_embed
        if self.use_global_visual_bias:
            phi_visual = visual_features.mean(dim=1, keepdim=True)
            phi_visual = self.visual_proj(phi_visual)        # [B, 1, 512]
            hyper_visual = hyper_visual + phi_visual

        encoded = self.transformer_encoder(hyper_visual)     # [B, 6, 512]
        pooled = self._pool_region_features(encoded)
        coarse_logits = self.classifier(pooled)
        attention_logits = coarse_logits
        region_weight = None
        region_importance = None
        if self.use_emotion_region_prior:
            if self.emotion_prior_mode == "gate":
                region_weight = self._emotion_region_weight(coarse_logits)
                encoded = self._apply_emotion_region_prior(encoded, region_weight=region_weight)
                pooled = self._pool_region_features(encoded)
                attention_logits = self.classifier(pooled)
            else:
                region_importance = self._region_importance(encoded)

        source_logits = None
        if self.logit_fusion in ("source", "sum"):
            source_logits = self.res_backbone.source_logits(global_feat)
        logits = self._combine_logits(attention_logits, source_logits)

        attn_norm = F.normalize(attn_weights, p=2, dim=-1)
        sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        mask = torch.eye(sim.size(1), device=sim.device).bool()
        off_diag_sim = sim[:, ~mask]
        if self.ortho_loss_type == "squared_offdiag":
            ortho_loss = off_diag_sim.pow(2).mean()
        else:
            ortho_loss = off_diag_sim.mean()

        if self.training:
            if self.use_emotion_region_prior:
                if self.emotion_prior_mode == "loss":
                    return logits, ortho_loss, coarse_logits, region_importance
                return logits, ortho_loss, coarse_logits
            return logits, ortho_loss

        if self.return_attn:
            if self.use_emotion_region_prior and region_weight is not None:
                attn_weights = self._apply_region_weight_to_attention(attn_weights, region_weight)
            return logits, attn_weights

        return logits


if __name__ == "__main__":
    print("=== Testing ResNet152RegionAttentionFER ===")
    cfg = {
        "data": {"num_classes": 7, "channels": 3},
        "model": {
            "embed_dim": 512,
            "num_heads": 4,
            "num_regions": 6,
            "num_encoder_layers": 1,
            "transformer_dropout": 0.1,
            "token_grid_size": 3,
            "use_clip_dictionary": False,
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet152RegionAttentionFER(cfg, channels=3).to(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    logits = out[0] if isinstance(out, tuple) else out
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (2, 7)
    print("Test passed.")
