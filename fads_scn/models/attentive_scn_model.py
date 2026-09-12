import math
import torch
import torch.nn as nn
from .backbones import FacialBackbone
from .spatial_attention import MultiHeadSpatialAttention
from .latent_graph import LatentGraphReasoner
from .scn_head import SCNHead


class AttentiveSCNFER(nn.Module):
    """
    Pure Image-Based Attentive Self-Cure Network with Latent Dynamic Graph Reasoning for FER.
    Zero dependency on bounding boxes, landmarks, or pre-extracted masks.
    
    Architecture Pipeline:
        Input: Raw grayscale images [B, 1, 48, 48]
          │
          ▼
        FacialBackbone (ResNet50 / ResNet34 adapted for 48x48)
          │ Feature Map F ∈ R^{B × C × 12 × 12}
          ├──► Global Average Pooling + Linear ──► f_global ∈ R^{B × D}
          │
          └──► MultiHeadSpatialAttention ────────► M Soft Regional Tokens {h_m} + Attention Maps
                 │
                 ▼
               Latent Dynamic Graph Reasoner ──► Dual-factor Dynamic Adjacency + Residual GCN
                 │
                 ▼
               Node Importance Readout ─────────► f_graph ∈ R^{B × D}
          │
          ▼
        Fusion: LayerNorm(f_global + gate * f_graph) ──► f_fused ∈ R^{B × D}
          │
          ▼
        SCNHead:
          ├──► Logits z ∈ R^{B × 7}
          └──► Sample Confidence alpha ∈ (0, 1)^{B × 1}
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 7,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_attn_heads: int = 8,
        use_latent_graph: bool = True,
        use_spatial_attention: bool = True,
        graph_mode: str = "dense",
        graph_topk: int = 3,
        graph_self_loop_bias: float = 1.0,
        graph_fusion_mode: str = "legacy_add",
        graph_gate_init: float = 0.01,
        dropout: float = 0.25,
        classifier_type: str = "linear",
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.20,
        use_pretrained: bool = True,
        pretrained_weights_path: str = "",
        stem_init: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.use_spatial_attention = use_spatial_attention
        if use_latent_graph and not use_spatial_attention:
            raise ValueError("use_latent_graph requires use_spatial_attention=true")
        self.use_latent_graph = use_latent_graph
        self.graph_mode = graph_mode
        if graph_fusion_mode not in ("legacy_add", "residual_delta"):
            raise ValueError("graph_fusion_mode must be legacy_add or residual_delta")
        if graph_fusion_mode == "residual_delta" and not use_latent_graph:
            raise ValueError("residual_delta graph fusion requires use_latent_graph=true")
        self.graph_fusion_mode = graph_fusion_mode

        # 1. Backbone adapted for 48x48
        self.backbone = FacialBackbone(
            backbone_name=backbone_name,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
            pretrained_weights_path=pretrained_weights_path,
            target_feat_size=12,
            stem_init=stem_init,
        )

        backbone_out_ch = self.backbone.out_channels

        # 2. Global Stream Projector
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_out_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Local Spatial Attention Stream (Unsupervised discovery of Action Units)
        self.spatial_attention = (
            MultiHeadSpatialAttention(
                in_channels=backbone_out_ch,
                embed_dim=embed_dim,
                num_heads=num_attn_heads,
                dropout=dropout,
            ) if self.use_spatial_attention else None
        )

        # 4. Latent Dynamic Graph Reasoner (Message passing between soft semantic nodes)
        if self.use_latent_graph:
            self.latent_graph = LatentGraphReasoner(
                embed_dim=embed_dim,
                num_nodes=num_attn_heads,
                dropout=dropout,
                graph_mode=graph_mode,
                topk=graph_topk,
                self_loop_bias=graph_self_loop_bias,
            )
        else:
            self.latent_graph = None

        # 5. Fusion Layer
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_gate = (nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
                            if self.use_spatial_attention else None)
        # Only the experimental mode owns this parameter, preserving the
        # state_dict of every existing legacy checkpoint.
        self.graph_residual_gate = (
            nn.Parameter(torch.tensor([graph_gate_init], dtype=torch.float32))
            if self.use_latent_graph and self.graph_fusion_mode == "residual_delta"
            else None
        )

        # 6. SCN Head (classifier + confidence weight)
        self.scn_head = SCNHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            classifier_type=classifier_type,
            cosface_scale=cosface_scale,
            cosface_margin=cosface_margin,
            init_confidence_bias=1.5,
        )

    def _forward_single(self, x: torch.Tensor, targets=None, targets_b=None, lam=1.0):
        # Feature map: [B, C, 12, 12]
        feat_map = self.backbone(x)

        # Global feature: [B, D]
        f_global = self.global_proj(feat_map)

        # Local spatial feature & attention maps & soft node tokens: [B, M, D]
        if self.spatial_attention is not None:
            f_local, attn_maps, div_loss, head_feats = self.spatial_attention(feat_map)
        else:
            f_local = torch.zeros_like(f_global)
            attn_maps, head_feats = None, None
            div_loss = torch.zeros((), device=x.device)

        # Latent Dynamic Graph Reasoning over soft tokens
        if self.use_latent_graph and self.latent_graph is not None:
            # Proposed reliability-sparse graph obtains alpha from global features
            # before graph fusion, avoiding a circular dependence on graph output.
            graph_alpha = (self.scn_head.predict_alpha(f_global)
                           if self.graph_mode == "reliability_sparse" else None)
            graph_context = f_global if self.graph_mode == "contextual_delta" else None
            graph_feats, adj_matrix, sparsity_loss = self.latent_graph(
                head_feats,
                attn_maps,
                global_features=graph_context if graph_context is not None else f_global,
                reliability=graph_alpha,
            )
            graph_gain = self.latent_graph.last_graph_gain
            graph_delta = graph_feats - head_feats.mean(dim=1)
        else:
            adj_matrix = None
            sparsity_loss = torch.tensor(0.0, device=x.device)
            graph_alpha = None
            graph_gain = torch.ones((x.size(0), 1), device=x.device)
            graph_feats = torch.zeros_like(f_local)
            graph_delta = torch.zeros_like(f_local)

        # The legacy formula remains unchanged. The experimental formula uses
        # the graph only as a separately gated relational correction.
        local_gate = torch.sigmoid(self.fusion_gate) if self.fusion_gate is not None else 0.0
        if self.use_latent_graph and self.graph_fusion_mode == "residual_delta":
            graph_gate = self.graph_residual_gate
            local_term = local_gate * f_local
            graph_term = graph_gate * graph_delta
            f_fused = self.fusion_norm(f_global + local_term + graph_term)
        else:
            graph_gate = local_gate if self.use_latent_graph else torch.zeros((), device=x.device)
            local_term = local_gate * f_local
            graph_term = local_gate * graph_feats
            f_fused = self.fusion_norm(f_global + local_term + graph_term)

        # Detached per-image diagnostics reveal node collapse and quantify how
        # strongly graph evidence changes the fused representation.
        eps = torch.finfo(f_global.dtype).eps
        if self.use_latent_graph:
            graph_delta_ratio = graph_delta.norm(dim=-1) / f_local.norm(dim=-1).clamp_min(eps)
            graph_contribution_ratio = graph_term.norm(dim=-1) / local_term.norm(dim=-1).clamp_min(eps)
            safe_adj = adj_matrix.clamp_min(eps)
            adjacency_entropy = -(adj_matrix * safe_adj.log()).sum(dim=-1).mean(dim=-1)
            adjacency_entropy = adjacency_entropy / math.log(max(self.num_attn_heads, 2))
            normalized_nodes = torch.nn.functional.normalize(head_feats, p=2, dim=-1)
            node_similarity_matrix = torch.bmm(normalized_nodes, normalized_nodes.transpose(1, 2))
            off_diagonal = ~torch.eye(
                self.num_attn_heads, dtype=torch.bool, device=x.device
            ).unsqueeze(0)
            node_cosine_similarity = node_similarity_matrix.masked_select(off_diagonal)
            node_cosine_similarity = node_cosine_similarity.view(x.size(0), -1).mean(dim=-1)
        else:
            graph_delta_ratio = torch.zeros(x.size(0), device=x.device, dtype=f_global.dtype)
            graph_contribution_ratio = torch.zeros_like(graph_delta_ratio)
            adjacency_entropy = torch.zeros_like(graph_delta_ratio)
            node_cosine_similarity = torch.zeros_like(graph_delta_ratio)

        if isinstance(graph_gate, torch.Tensor):
            graph_gate_value = graph_gate.reshape(1).expand(x.size(0))
        else:
            graph_gate_value = torch.full(
                (x.size(0),), float(graph_gate), device=x.device, dtype=f_global.dtype
            )

        # SCN Head classifier
        logits, alpha = self.scn_head(f_fused, targets=targets, targets_b=targets_b, lam=lam)
        # The sparse graph's alpha must be available before graph propagation;
        # expose that same reliability estimate to SCN loss.
        if graph_alpha is not None:
            alpha = graph_alpha

        return {
            "logits": logits,
            "alpha": alpha,
            "attn_maps": attn_maps,
            "diversity_loss": div_loss,
            "adj_matrix": adj_matrix,
            "sparsity_loss": sparsity_loss,
            "graph_gain": graph_gain,
            "graph_gate": graph_gate_value.detach(),
            "graph_delta_ratio": graph_delta_ratio.detach(),
            "graph_contribution_ratio": graph_contribution_ratio.detach(),
            "adjacency_entropy": adjacency_entropy.detach(),
            "node_cosine_similarity": node_cosine_similarity.detach(),
            "features": f_fused,
        }

    def forward(self, x: torch.Tensor, targets=None, targets_b=None, lam=1.0, use_tta=None):

        """
        Forward pass with Test-Time Augmentation (TTA).
        Modes for use_tta:
          - False / None during training: Single image standard forward
          - True / 'flip': 2-crop Horizontal Flip TTA (average original + flipped)
          - 'multiscale' / 'multi_scale': 4-crop Multi-Scale TTA (original, flipped, zoom 1.05x, zoom flipped)
        """
        if use_tta is None:
            use_tta = not self.training

        if not self.training and use_tta:
            if use_tta in ("multiscale", "multi_scale"):
                # 4-crop Multi-Scale TTA
                x_orig = x
                x_flip = torch.flip(x, dims=[-1])
                # Zoom 1.05x and center crop back to the original resolution.
                # Works for both the 48x48 and 96x96 ablations.
                height, width = x.shape[-2:]
                zoom_h, zoom_w = max(height + 2, round(height * 1.05)), max(width + 2, round(width * 1.05))
                top, left = (zoom_h - height) // 2, (zoom_w - width) // 2
                x_zoom = torch.nn.functional.interpolate(
                    x, size=(zoom_h, zoom_w), mode="bilinear", align_corners=False
                )[:, :, top:top + height, left:left + width]
                x_zoom_flip = torch.flip(x_zoom, dims=[-1])

                out1 = self._forward_single(x_orig)
                out2 = self._forward_single(x_flip)
                out3 = self._forward_single(x_zoom)
                out4 = self._forward_single(x_zoom_flip)

                avg_logits = 0.25 * (out1["logits"] + out2["logits"] + out3["logits"] + out4["logits"])
                avg_alpha = 0.25 * (out1["alpha"] + out2["alpha"] + out3["alpha"] + out4["alpha"])
                avg_graph_gain = 0.25 * (
                    out1["graph_gain"] + out2["graph_gain"] + out3["graph_gain"] + out4["graph_gain"]
                )

                return {
                    "logits": avg_logits,
                    "alpha": avg_alpha,
                    "attn_maps": out1["attn_maps"],
                    "diversity_loss": out1["diversity_loss"],
                    "adj_matrix": out1["adj_matrix"],
                    "sparsity_loss": out1["sparsity_loss"],
                    "graph_gain": avg_graph_gain,
                    "graph_gate": out1["graph_gate"],
                    "graph_delta_ratio": 0.25 * sum(
                        out["graph_delta_ratio"] for out in (out1, out2, out3, out4)
                    ),
                    "graph_contribution_ratio": 0.25 * sum(
                        out["graph_contribution_ratio"] for out in (out1, out2, out3, out4)
                    ),
                    "adjacency_entropy": 0.25 * sum(
                        out["adjacency_entropy"] for out in (out1, out2, out3, out4)
                    ),
                    "node_cosine_similarity": 0.25 * sum(
                        out["node_cosine_similarity"] for out in (out1, out2, out3, out4)
                    ),
                    "features": out1["features"],
                }
            else:
                # 2-crop Horizontal Flip TTA
                out_orig = self._forward_single(x)
                x_flipped = torch.flip(x, dims=[-1])
                out_flipped = self._forward_single(x_flipped)

                avg_logits = 0.5 * (out_orig["logits"] + out_flipped["logits"])
                avg_alpha = 0.5 * (out_orig["alpha"] + out_flipped["alpha"])
                avg_graph_gain = 0.5 * (out_orig["graph_gain"] + out_flipped["graph_gain"])

                return {
                    "logits": avg_logits,
                    "alpha": avg_alpha,
                    "attn_maps": out_orig["attn_maps"],
                    "diversity_loss": out_orig["diversity_loss"],
                    "adj_matrix": out_orig["adj_matrix"],
                    "sparsity_loss": out_orig["sparsity_loss"],
                    "graph_gain": avg_graph_gain,
                    "graph_gate": out_orig["graph_gate"],
                    "graph_delta_ratio": 0.5 * (
                        out_orig["graph_delta_ratio"] + out_flipped["graph_delta_ratio"]
                    ),
                    "graph_contribution_ratio": 0.5 * (
                        out_orig["graph_contribution_ratio"] + out_flipped["graph_contribution_ratio"]
                    ),
                    "adjacency_entropy": 0.5 * (
                        out_orig["adjacency_entropy"] + out_flipped["adjacency_entropy"]
                    ),
                    "node_cosine_similarity": 0.5 * (
                        out_orig["node_cosine_similarity"] + out_flipped["node_cosine_similarity"]
                    ),
                    "features": out_orig["features"],
                }
        else:
            return self._forward_single(x, targets=targets, targets_b=targets_b, lam=lam)
