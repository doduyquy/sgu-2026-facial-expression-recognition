import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM import ECA, CBAM

# Input:  (B, 1, 48, 48)
# conv1: 3x3, stride=1, pad=1           -> (B, 64, 48, 48)
# pool:  2x2, stride=2                  -> (B, 64, 24, 24)
# layer2: ConvBlock(s=1) + 2 IDs        -> (B, 256, 24, 24)
# layer3: ConvBlock(s=2) + 3 IDs        -> (B, 512, 12, 12)
# layer4: ConvBlock(s=2) + 3 IDs        -> (B, 1024, 6, 6)
# avgpool: AdaptiveAvgPool2d((1,1))     -> (B, 1024, 1, 1)
# flatten                               -> (B, 1024)
# fc                                    -> (B, num_classes)
#Hout = ((Hin + 2*pad - kernel_size) // stride) + 1


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(torch.pi - m))
        self.mm = torch.sin(torch.tensor(torch.pi - m)) * m

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if labels is None:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s
        return logits

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.attn = nn.Identity()

        self.relu = nn.ReLU()
    def forward(self, x):
        shortcut = x    
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        x=self.attn(x)

        x += shortcut
        x = self.relu(x)

        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=2):
        super().__init__()
        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.attn = nn.Identity()

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, F3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(F3)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.attn(x)

        x += shortcut
        x = self.relu(x)

        return x
    
class ResNet50(nn.Module):
    def __init__(self, config, channels=1):
        super().__init__()
        
        # Load from config
        self.num_classes = config['data']['num_classes']
        model_cfg = config.get('model', {})
        self.attention_type = model_cfg.get('attention_type', 'cbam') # 'eca', 'cbam', or None
        self.attention_kernel_size = model_cfg.get('attention_kernel_size', 7)
        self.use_arcface = model_cfg.get('use_arcface', False)
        
        # Arcface params
        arc_cfg = model_cfg.get('arcface', {})
        self.arcface_s = arc_cfg.get('s', 30.0)
        self.arcface_m = arc_cfg.get('m', 0.5)
        self.arcface_easy_margin = arc_cfg.get('easy_margin', False)

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        def get_attn(channels):
            if self.attention_type == 'eca':
                return ECA(channels, kernel_size=3)
            elif self.attention_type == 'cbam':
                return CBAM(channels, kernel_size=self.attention_kernel_size)
            return nn.Identity()

        # Stage 2
        self.layer2 = nn.Sequential(
            ConvBlock(64, [64,64,256], stride=1),
            IdentityBlock(256, [64,64,256]),
            IdentityBlock(256, [64,64,256])
        )
        for i in range(3): self.layer2[i].attn = get_attn(256)

        # Stage 3
        self.layer3 = nn.Sequential(
            ConvBlock(256, [128,128,512]),
            IdentityBlock(512, [128,128,512]),
            IdentityBlock(512, [128,128,512]),
            IdentityBlock(512, [128,128,512])
        )
        for i in range(4): self.layer3[i].attn = get_attn(512)

        # Stage 4
        self.layer4 = nn.Sequential(
            ConvBlock(512, [256,256,1024]),
            IdentityBlock(1024, [256,256,1024]),
            IdentityBlock(1024, [256,256,1024]),
            IdentityBlock(1024, [256,256,1024])
        )
        for i in range(4): self.layer4[i].attn = get_attn(1024)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, self.num_classes)
        if self.use_arcface:
            self.arcface_head = ArcMarginProduct(
                in_features=1024,
                out_features=self.num_classes,
                s=self.arcface_s,
                m=self.arcface_m,
                easy_margin=self.arcface_easy_margin,
            )
        else:
            self.arcface_head = None

    def forward(self, x, labels=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        if self.use_arcface:
            x = self.arcface_head(features, labels)
        else:
            x = self.fc(features)

        return x