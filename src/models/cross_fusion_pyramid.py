import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionImg(nn.Module):
    """POSTER direction: Q from landmark stream, K/V from image stream."""

    def __init__(self, dim, img_tokens, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.img_tokens = img_tokens
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_img = x[:, : self.img_tokens, :]
        x_lm = x[:, self.img_tokens :, :]

        bsz, n_img, dim = x_img.shape
        kv = self.kv(x_img).reshape(bsz, n_img, 2, self.num_heads, dim // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q = x_lm.reshape(bsz, -1, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, n_img, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AttentionLm(nn.Module):
    """POSTER direction: Q from image stream, K/V from landmark stream."""

    def __init__(self, dim, img_tokens, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.img_tokens = img_tokens
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_img = x[:, : self.img_tokens, :]
        x_lm = x[:, self.img_tokens :, :]

        bsz, n_lm, dim = x_lm.shape
        kv = self.kv(x_lm).reshape(bsz, n_lm, 2, self.num_heads, dim // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q = x_img.reshape(bsz, -1, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, n_lm, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PosterBlock(nn.Module):
    def __init__(
        self,
        dim,
        img_tokens,
        lm_tokens,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.img_tokens = img_tokens
        self.lm_tokens = lm_tokens
        self.total_tokens = img_tokens + lm_tokens

        self.norm1 = nn.LayerNorm(dim)
        self.attn_img = AttentionImg(dim, img_tokens=img_tokens, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn_lm = AttentionLm(dim, img_tokens=img_tokens, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        mlp_hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp_img = Mlp(in_features=dim, hidden_features=mlp_hidden, drop=drop)
        self.mlp_lm = Mlp(in_features=dim, hidden_features=mlp_hidden, drop=drop)

        # This matches POSTER token-channel mixing intent.
        self.token_conv = nn.Conv1d(self.total_tokens, self.total_tokens, kernel_size=1)

    def forward(self, x):
        x_img = x[:, : self.img_tokens, :]
        x_lm = x[:, self.img_tokens :, :]

        x_img = x_img + self.drop_path(self.attn_img(self.norm1(x)))
        x_img = x_img + self.drop_path(self.mlp_img(self.norm2(x_img)))

        x_lm = x_lm + self.drop_path(self.attn_lm(self.norm3(x)))
        x_lm = x_lm + self.drop_path(self.mlp_lm(self.norm4(x_lm)))

        x = torch.cat((x_img, x_lm), dim=1)
        x = self.token_conv(x)
        return x


class PosterPyramidBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_tokens,
        lm_tokens,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.total_tokens = img_tokens + lm_tokens

        self.block_l = PosterBlock(
            dim=embed_dim,
            img_tokens=img_tokens,
            lm_tokens=lm_tokens,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
        )
        self.block_m = PosterBlock(
            dim=embed_dim // 2,
            img_tokens=img_tokens,
            lm_tokens=lm_tokens,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
        )
        self.block_s = PosterBlock(
            dim=embed_dim // 4,
            img_tokens=img_tokens,
            lm_tokens=lm_tokens,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
        )

        self.upsample_m = nn.ConvTranspose1d(self.total_tokens, self.total_tokens, kernel_size=2, stride=2)
        self.upsample_s = nn.ConvTranspose1d(self.total_tokens, self.total_tokens, kernel_size=2, stride=2)

    def forward(self, x_l, x_m, x_s):
        x_l = self.block_l(x_l)
        x_m = self.block_m(x_m)
        x_s = self.block_s(x_s)

        x_m = self.upsample_s(x_s) + x_m
        x_l = x_l + self.upsample_m(x_m)
        return x_l, x_m, x_s


class PyramidCrossFusionTransformer(nn.Module):
    """Strict POSTER-style landmark pyramid with embedding-dimension pyramid."""

    def __init__(
        self,
        lm_base_dim=128,
        topk_tokens=49,
        token_selection_mode="topk_softmax",
        use_sinusoidal_pos=True,
        dropout_rate=0.1,
        pyramid_depth=4,
        fusion_dim=256,
        num_heads=8,
    ):
        super().__init__()
        self.topk_tokens = topk_tokens
        self.token_selection_mode = token_selection_mode
        self.use_sinusoidal_pos = use_sinusoidal_pos

        # Poster uses equal token counts in image and landmark streams.
        self.img_tokens = 49
        self.lm_tokens = 49

        self.img_proj = nn.Conv2d(1024, fusion_dim, kernel_size=1)
        self.img_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.lm_proj = nn.Linear(lm_base_dim, fusion_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.img_tokens + 1, fusion_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.downsample_m = nn.Conv1d(self.img_tokens + self.lm_tokens + 2, self.img_tokens + self.lm_tokens + 2, kernel_size=2, stride=2)
        self.downsample_s = nn.Conv1d(self.img_tokens + self.lm_tokens + 2, self.img_tokens + self.lm_tokens + 2, kernel_size=4, stride=4)

        dpr = torch.linspace(0, dropout_rate, pyramid_depth).tolist()
        self.blocks = nn.ModuleList(
            [
                PosterPyramidBlock(
                    embed_dim=fusion_dim,
                    img_tokens=self.img_tokens + 1,
                    lm_tokens=self.lm_tokens + 1,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    qkv_bias=True,
                    drop=dropout_rate,
                    attn_drop=dropout_rate,
                    drop_path_rate=dpr[i],
                )
                for i in range(pyramid_depth)
            ]
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def _select_tokens(self, img_tokens):
        if self.topk_tokens is None or self.topk_tokens <= 0:
            return img_tokens
        if img_tokens.size(1) <= self.topk_tokens:
            return img_tokens

        scores = img_tokens.norm(dim=-1)
        if self.token_selection_mode == "softmax":
            return img_tokens * torch.softmax(scores, dim=1).unsqueeze(-1)
        if self.token_selection_mode == "sigmoid":
            return img_tokens * torch.sigmoid(scores).unsqueeze(-1)

        topk_scores, topk_idx = torch.topk(scores, k=self.topk_tokens, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, img_tokens.size(-1))
        selected = torch.gather(img_tokens, dim=1, index=gather_idx)
        if self.token_selection_mode == "topk_softmax":
            selected = selected * torch.softmax(topk_scores, dim=1).unsqueeze(-1)
        return selected

    @staticmethod
    def _resize_token_count(tokens, target_count):
        if tokens.size(1) == target_count:
            return tokens
        resized = F.interpolate(tokens.transpose(1, 2), size=target_count, mode="linear", align_corners=False)
        return resized.transpose(1, 2)

    def forward(self, x2, x3, x4, lm_base_tokens, landmark_mask=None):
        _ = x2
        _ = x3
        _ = landmark_mask

        img_map = self.img_pool(self.img_proj(x4))
        img_tokens = img_map.flatten(2).transpose(1, 2)
        img_tokens = self._select_tokens(img_tokens)
        img_tokens = self._resize_token_count(img_tokens, self.img_tokens)

        lm_tokens = self.lm_proj(lm_base_tokens)
        lm_tokens = self._resize_token_count(lm_tokens, self.lm_tokens)

        bsz = img_tokens.size(0)
        img_cls = torch.mean(img_tokens, dim=1, keepdim=True)
        lm_cls = torch.mean(lm_tokens, dim=1, keepdim=True)

        img_tokens = torch.cat((img_cls, img_tokens), dim=1)
        img_tokens = self.pos_drop(img_tokens + self.pos_embed)
        lm_tokens = torch.cat((lm_cls, lm_tokens), dim=1)

        x = torch.cat((img_tokens, lm_tokens), dim=1)

        x_l = x
        x_m = self.downsample_m(x)
        x_s = self.downsample_s(x)

        for blk in self.blocks:
            x_l, x_m, x_s = blk(x_l, x_m, x_s)

        x_l = self.norm(x_l)

        # Return cls-like fused token and two auxiliary tokens for compatibility.
        fused_cls = x_l[:, 0, :]
        img_cls_out = x_l[:, 1, :]
        lm_cls_out = x_l[:, self.img_tokens + 1, :]
        return fused_cls, img_cls_out, lm_cls_out
