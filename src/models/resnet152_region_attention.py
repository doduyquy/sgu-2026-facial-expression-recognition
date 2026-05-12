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
    ResNet152 feature map -> a small grid of visual tokens.
    Default endpoint is layer4 and the default keeps its native 7x7x2048 map.
    The final ResNet avgpool/fc tail is still skipped; attention consumes the
    spatial layer4 tokens directly.
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
        if self.feature_layer not in feature_dims:
            raise ValueError(
                "model.feature_layer must be one of: layer2, layer3, layer4"
            )
        self.feature_dim = feature_dims[self.feature_layer]

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

        self.native_grid_size = max(1, image_size // output_strides[self.feature_layer])
        self.visual_grid_size = (
            self.token_grid_size if self.pool_visual_tokens else self.native_grid_size
        )
        self.token_pool = (
            nn.AdaptiveAvgPool2d((self.token_grid_size, self.token_grid_size))
            if self.pool_visual_tokens
            else nn.Identity()
        )
        self.source_fc = nn.Linear(self.feature_dim, self.num_classes)
        self.has_source_classifier = False
        print(
            f"--> [ResNet152Tokenizer] feature_layer={self.feature_layer}, "
            f"feature_dim={self.feature_dim}, tokens={self.visual_grid_size}x{self.visual_grid_size}, "
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

        x = self.backbone.layer3(x)
        if self.feature_layer == "layer3":
            return x

        x = self.backbone.layer4(x)
        return x

    def forward(self, x):
        feat_map = self.forward_features(x)
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
            elif key.startswith("backbone.") and name in backbone_ref and backbone_ref[name].shape == value.shape:
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
        elif self.feature_layer != "layer4":
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
    Cross-attention where CLIP/text region queries stay at 512-d while
    ResNet152 layer4 visual keys/values stay at 2048-d.
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
        self.is_frozen = False
        self.return_attn = False

        num_classes = data_cfg.get("num_classes", 7)
        if self.unfreeze_backbone_scope not in ("all", "layer4"):
            raise ValueError("model.unfreeze_backbone_scope must be one of: all, layer4")
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
        num_visual_tokens = self.res_backbone.visual_grid_size * self.res_backbone.visual_grid_size
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

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate),
        )

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

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_regions, self.embed_dim) * 0.02)

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

        checkpoint_path = model_cfg.get("checkpoint_path")
        if checkpoint_path:
            self.load_pretrained_backbones(checkpoint_path, device="cpu")

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
        if mode and self.is_frozen:
            self.res_backbone.backbone.eval()
            self.res_backbone.source_fc.eval()
        elif mode and self.unfreeze_backbone and self.unfreeze_backbone_scope == "layer4":
            self.res_backbone.backbone.eval()
            self.res_backbone.backbone.layer4.train()
            if self.freeze_unfrozen_batchnorm:
                self._freeze_batchnorm(self.res_backbone.backbone.layer4)
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

    def forward(self, x):
        batch_size = x.shape[0]

        visual_features, global_feat = self.res_backbone(x)  # [B, 49, 2048], [B, 2048]
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

        phi_visual = visual_features.mean(dim=1, keepdim=True)
        phi_visual = self.visual_proj(phi_visual)            # [B, 1, 512]
        hyper_visual = phi_sem + phi_visual + self.pos_embed

        encoded = self.transformer_encoder(hyper_visual)     # [B, 6, 512]
        if self.region_pooling == "concat":
            pooled = encoded.reshape(encoded.size(0), -1)
        else:
            pooled = encoded.mean(dim=1)
        attention_logits = self.classifier(pooled)

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
            return logits, ortho_loss

        if self.return_attn:
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
