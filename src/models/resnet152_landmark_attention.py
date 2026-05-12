"""
ResNet152 + Landmark-guided Soft Mask Attention for FER.

Extends ResNet152RegionAttentionFER by accepting optional `region_masks`
[B, K, H, W] tensors that bias cross-attention toward anatomically correct
facial regions (eyes, nose, mouth, ...).

When region_masks is None the model behaves identically to the original.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet152_region_attention import (
    ResNet152SpatialTokenizer,
    safe_torch_load,
    extract_state_dict,
    strip_known_prefixes,
)
from .region_attention import (
    CLIPFacialRegionDictionary,
    DropPath,
    FacialRegionDictionary,
    SubGraphFusion,
)


# =====================================================================
# Cross-Attention with Landmark Soft Mask
# =====================================================================
class LandmarkGuidedAlignment(nn.Module):
    """
    Cross-attention where CLIP/text region queries (embed_dim) attend to
    ResNet visual keys/values (visual_dim).

    When `region_masks` is supplied the mask is converted to additive
    log-space bias and injected into the attention score matrix *before*
    softmax, steering each region query toward its anatomical location.

    Shapes (example with layer3, 224x224 input):
        region_tokens  : [B, K, embed_dim]   e.g. [B, 6, 512]
        visual_features: [B, N, visual_dim]  e.g. [B, 196, 1024]
        region_masks   : [B, K, N]           e.g. [B, 6, 196]  (values 0-1)
    """

    def __init__(self, embed_dim=512, visual_dim=1024, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
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

    @staticmethod
    def _build_log_mask(region_masks, num_heads):
        """
        Convert probability masks [B, K, N] (0..1) to additive log-space
        bias [B*num_heads, K, N] compatible with nn.MultiheadAttention.

        mask=1.0  ->  log(1)   =  0    (no change to attention logit)
        mask=0.0  ->  log(eps) ~ -20   (attention prob -> ~0 after softmax)
        """
        log_mask = torch.log(region_masks + 1e-9)           # [B, K, N]
        log_mask = log_mask.repeat_interleave(num_heads, dim=0)  # [B*H, K, N]
        return log_mask

    def forward(self, region_tokens, visual_features, region_masks=None):
        """
        Args:
            region_tokens:   [B, K, embed_dim]
            visual_features: [B, N, visual_dim]
            region_masks:    [B, K, N] float (0..1), or None
        Returns:
            region_enriched: [B, K, embed_dim]
            attn_weights:    [B, K, N]
        """
        attn_mask = None
        if region_masks is not None:
            attn_mask = self._build_log_mask(region_masks, self.num_heads)

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


# =====================================================================
# Main Model
# =====================================================================
class ResNet152LandmarkAttentionFER(nn.Module):
    """
    ResNet152 backbone  +  Landmark-guided Region Attention  +  Classifier.

    Identical to ResNet152RegionAttentionFER except:
    - forward() accepts an optional `region_masks` tensor [B, K, H, W].
    - The masks are flattened to [B, K, N] and fed into LandmarkGuidedAlignment.
    - When masks are None, the model falls back to vanilla cross-attention.
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

        # ── Backbone ──
        self.res_backbone = ResNet152SpatialTokenizer(config, channels=channels)
        self.visual_dim = self.res_backbone.feature_dim
        num_visual_tokens = self.res_backbone.num_visual_tokens

        if self.use_visual_pos_embed:
            self.visual_pos_embed = nn.Parameter(
                torch.randn(1, num_visual_tokens, self.visual_dim) * 0.02
            )
        else:
            self.register_buffer(
                "visual_pos_embed",
                torch.zeros(1, num_visual_tokens, self.visual_dim),
            )

        # ── Region Dictionary (CLIP or Learned) ──
        self.use_clip_dictionary = model_cfg.get("use_clip_dictionary", True)
        if self.use_clip_dictionary:
            clip_model_name = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
            self.region_dict = CLIPFacialRegionDictionary(
                num_regions=self.num_regions,
                embed_dim=self.embed_dim,
                clip_model_name=clip_model_name,
            )
            print(f"--> [LandmarkAttention] CLIP text region tokens: K={self.num_regions}")
        else:
            self.region_dict = FacialRegionDictionary(
                num_regions=self.num_regions,
                embed_dim=self.embed_dim,
            )
            print(f"--> [LandmarkAttention] Learned region tokens: K={self.num_regions}")

        # ── Landmark-guided Cross-Attention ──
        self.alignment = LandmarkGuidedAlignment(
            embed_dim=self.embed_dim,
            visual_dim=self.visual_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
        )

        # ── Visual Projection ──
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate),
        )

        # ── Transformer / SubGraph Encoder ──
        if self.fusion_type == "subgraph":
            self.transformer_encoder = nn.Sequential(*[
                SubGraphFusion(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ])
            print("--> [LandmarkAttention] Using SubGraph Fusion.")
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
            print("--> [LandmarkAttention] Using standard Transformer encoder.")

        # ── Positional Embedding for Region Tokens ──
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_regions, self.embed_dim) * 0.02
        )

        # ── Classifier ──
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

        # ── Load pretrained backbone ──
        checkpoint_path = model_cfg.get("checkpoint_path")
        if checkpoint_path:
            self.load_pretrained_backbones(checkpoint_path, device="cpu")

    # ------------------------------------------------------------------
    # Backbone loading / freezing (identical to original)
    # ------------------------------------------------------------------
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
        print("[LandmarkAttention] ResNet152 backbone FROZEN.")

    def unfreeze_backbones(self):
        for param in self.res_backbone.backbone.parameters():
            param.requires_grad = False

        if self.unfreeze_backbone_scope == "layer4":
            trainable_modules = (self.res_backbone.backbone.layer4,)
            message = "ResNet152 layer4 UNFROZEN."
        elif self.unfreeze_backbone_scope == "layer3":
            trainable_modules = (self.res_backbone.backbone.layer3,)
            message = "ResNet152 layer3 UNFROZEN."
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
        print(f"[LandmarkAttention] {message}")
        if previous_logit_fusion != self.logit_fusion:
            print(
                f"[LandmarkAttention] logit_fusion switched: "
                f"{previous_logit_fusion} -> {self.logit_fusion}"
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

    # ------------------------------------------------------------------
    # Forward with optional landmark masks
    # ------------------------------------------------------------------
    def forward(self, x, region_masks=None):
        """
        Args:
            x:             [B, C, H, W] input image tensor.
            region_masks:  [B, K, Hf, Wf] float tensor (0..1) where
                           K = num_regions,  Hf x Wf = feature map grid size.
                           Each channel is a Gaussian heatmap centered on the
                           anatomical landmark for that region.
                           Pass None to use vanilla (unguided) attention.

        Returns (training):
            logits:     [B, num_classes]
            ortho_loss: scalar
        Returns (eval):
            logits:     [B, num_classes]
            (optionally attn_weights if self.return_attn is True)
        """
        batch_size = x.shape[0]

        # ── 1. Extract visual tokens from ResNet backbone ──
        visual_features, global_feat = self.res_backbone(x)  # [B, N, D_vis], [B, D_vis]
        if visual_features.size(1) != self.visual_pos_embed.size(1):
            raise ValueError(
                f"visual_pos_embed expects {self.visual_pos_embed.size(1)} tokens, "
                f"but the backbone returned {visual_features.size(1)}. "
                "Check image_size, feature_layer, or pool_visual_tokens in config."
            )
        visual_features = visual_features + self.visual_pos_embed

        # ── 2. Prepare landmark masks ──
        flat_masks = None
        if region_masks is not None:
            # region_masks: [B, K, Hf, Wf] -> flatten to [B, K, N]
            flat_masks = region_masks.view(
                batch_size, self.num_regions, -1
            )
            # Validate that flattened spatial dim matches visual token count
            if flat_masks.size(2) != visual_features.size(1):
                raise ValueError(
                    f"region_masks spatial size {region_masks.shape[2:]} "
                    f"flattens to {flat_masks.size(2)} tokens, but backbone "
                    f"produces {visual_features.size(1)} visual tokens. "
                    "Make sure region_masks matches the feature_layer grid size."
                )

        # ── 3. Region tokens + Landmark-guided Cross-Attention ──
        region_tokens = self.region_dict(batch_size)         # [B, K, 512]
        phi_sem, attn_weights = self.alignment(
            region_tokens,
            visual_features,
            region_masks=flat_masks,                         # [B, K, N] or None
        )                                                    # [B, K, 512], [B, K, N]

        # ── 4. Hyper-visual representation ──
        phi_visual = visual_features.mean(dim=1, keepdim=True)
        phi_visual = self.visual_proj(phi_visual)            # [B, 1, 512]
        hyper_visual = phi_sem + phi_visual + self.pos_embed

        # ── 5. Transformer Encoder ──
        encoded = self.transformer_encoder(hyper_visual)     # [B, K, 512]
        if self.region_pooling == "concat":
            pooled = encoded.reshape(encoded.size(0), -1)
        else:
            pooled = encoded.mean(dim=1)
        attention_logits = self.classifier(pooled)

        # ── 6. Optional source logit fusion ──
        source_logits = None
        if self.logit_fusion in ("source", "sum"):
            source_logits = self.res_backbone.source_logits(global_feat)
        logits = self._combine_logits(attention_logits, source_logits)

        # ── 7. Orthogonal diversity loss ──
        attn_norm = F.normalize(attn_weights, p=2, dim=-1)
        sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        eye_mask = torch.eye(sim.size(1), device=sim.device).bool()
        off_diag_sim = sim[:, ~eye_mask]
        if self.ortho_loss_type == "squared_offdiag":
            ortho_loss = off_diag_sim.pow(2).mean()
        else:
            ortho_loss = off_diag_sim.mean()

        if self.training:
            return logits, ortho_loss

        if self.return_attn:
            return logits, attn_weights

        return logits


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print("=== Test 1: Forward WITHOUT landmark masks (vanilla) ===")
    cfg = {
        "data": {"num_classes": 7, "channels": 3, "image_size": 224},
        "model": {
            "embed_dim": 512,
            "num_heads": 4,
            "num_regions": 6,
            "num_encoder_layers": 1,
            "transformer_dropout": 0.1,
            "feature_layer": "layer3",
            "use_clip_dictionary": False,
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet152LandmarkAttentionFER(cfg, channels=3).to(device)
    model.train()

    dummy_img = torch.randn(2, 3, 224, 224, device=device)
    logits, ortho = model(dummy_img)
    print(f"  logits: {logits.shape}, ortho: {ortho.item():.4f}")
    assert logits.shape == (2, 7)

    print("\n=== Test 2: Forward WITH landmark masks ===")
    # layer3 on 224x224 -> 14x14 feature map -> 196 tokens
    dummy_masks = torch.rand(2, 6, 14, 14, device=device)
    logits2, ortho2 = model(dummy_img, region_masks=dummy_masks)
    print(f"  logits: {logits2.shape}, ortho: {ortho2.item():.4f}")
    assert logits2.shape == (2, 7)

    print("\n=== Test 3: Eval mode with return_attn ===")
    model.eval()
    model.return_attn = True
    logits3, attn = model(dummy_img, region_masks=dummy_masks)
    print(f"  logits: {logits3.shape}, attn_weights: {attn.shape}")
    assert logits3.shape == (2, 7)
    assert attn.shape == (2, 6, 196)

    print("\nAll tests passed!")
