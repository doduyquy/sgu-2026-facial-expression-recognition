import torch
import torch.nn as nn

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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size, padding=padding, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.conv(attn)

        return self.sigmoid(attn)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=5):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        attn = self.ca(x) * x
        attn = self.sa(attn) * attn
        return x + attn