import torch
import torch.nn as nn


class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)
        # (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        # (B, 1, C) -> (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = avg_out + max_out
        return self.sigmoid(attn)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=5):
#         super().__init__()
#         padding = kernel_size // 2

#         self.conv = nn.Sequential(
#             nn.Conv2d(2, 16, kernel_size, padding=padding, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, kernel_size, padding=padding, bias=False)
#         )

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_map = torch.mean(x, dim=1, keepdim=True)
#         max_map, _ = torch.max(x, dim=1, keepdim=True)

#         attn = torch.cat([avg_map, max_map], dim=1)
#         attn = self.conv(attn)

#         return self.sigmoid(attn)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_in = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        attn = self.sigmoid(self.conv(x_in)) # [B, 1, H, W]
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        attn = self.ca(x) * x
        attn = self.sa(attn)
        return x + attn