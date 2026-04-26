import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from .CBAM import ECA, CBAM

# Input:  (B, 1, 48, 48)
# conv1 + pool: 3x3, stride=1 + 2x2, stride=2  -> (B, 64, 24, 24)
# layer2 (Stage 1 Bottleneck): 3 blocks       -> (B, 256, 24, 24)
# layer3 (Stage 2 Bottleneck): 4 blocks       -> (B, 512, 12, 12)
# layer4 (Stage 3 Bottleneck): 4 blocks       -> (B, 1024, 6, 6)
# If Fusion: Cat(Pool(Layer3), Pool(Layer4))  -> (B, 1536)
# Else: Pool(Layer4)                          -> (B, 1024)
# fc / arcface_head                           -> (B, num_classes)
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
        self.use_fusion = model_cfg.get('use_fusion', False)
        
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
        
        # Input features for classifier
        self.feature_dim = 1024
        if self.use_fusion:
            self.feature_dim = 1024 + 512
            print(f"--> Using Multi-scale Fusion (Stage 3 + Stage 4), Feature dim: {self.feature_dim}")

        self.fc = nn.Linear(self.feature_dim, self.num_classes)
        if self.use_arcface:
            self.arcface_head = ArcMarginProduct(
                in_features=self.feature_dim,
                out_features=self.num_classes,
                s=self.arcface_s,
                m=self.arcface_m,
                easy_margin=self.arcface_easy_margin,
            )
        else:
            self.arcface_head = None

    def forward(self, x, labels=None):
        # input: (B, 1, 48, 48)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        # after stem: (B, 64, 24, 24)

        x = self.layer2(x) # (B, 256, 24, 24)
        
        feat3 = self.layer3(x) # (B, 512, 12, 12)
        feat4 = self.layer4(feat3) # (B, 1024, 6, 6)

        if self.use_fusion:
            # Multi-scale fusion: GAP each stage and concatenate
            p3 = self.avgpool(feat3).flatten(1) # (B, 512)
            p4 = self.avgpool(feat4).flatten(1) # (B, 1024)
            features = torch.cat([p3, p4], dim=1) # (B, 1536)
        else:
            x = self.avgpool(feat4)       # (B, 1024, 1, 1)
            features = torch.flatten(x, 1) # (B, 1024)

        if self.use_arcface:
            x = self.arcface_head(features, labels)
        else:
            x = self.fc(features)

        return x

class ResNet152(nn.Module):
    def __init__(self, config, channels=3):
        super().__init__()
        self.num_classes = config['data']['num_classes']
        self.config = config
        model_cfg = config.get('model', {})
        self.pretrained_checkpoint_path = model_cfg.get('checkpoint_path')
        self.reset_classifier_after_load = model_cfg.get('reset_classifier', True)
        self.freeze_backbone_on_start = model_cfg.get('freeze_backbone', False)
        self.unfreeze_epoch = model_cfg.get('unfreeze_epoch', None)
        self.backbone_frozen = False
        self.feature_dim = 2048

        self.backbone = models.resnet152(weights=None)
        if channels != 3:
            self.backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = self.build_head(model_cfg)

        if self.pretrained_checkpoint_path:
            self.load_from_checkpoint(self.pretrained_checkpoint_path, device='cpu')

        if self.reset_classifier_after_load:
            self.reset_classifier()

        if self.freeze_backbone_on_start:
            self.freeze_backbone()

    def forward(self, x, labels=None):
        features = self.backbone(x)
        return self.head(features)

    def build_head(self, model_cfg):
        head_type = model_cfg.get('head_type', 'mlp')
        hidden_dim = model_cfg.get('head_hidden_dim', 512)
        dropout = model_cfg.get('head_dropout', 0.3)

        if head_type == 'linear':
            return nn.Linear(self.feature_dim, self.num_classes)

        return nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.backbone_frozen:
            self.backbone.eval()
        return self

    def reset_classifier(self):
        self.head = self.build_head(self.config.get('model', {}))
        print("--> [ResNet152] Reset classifier head.")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
        self.backbone_frozen = True
        print("--> [ResNet152] Frozen backbone; training head only.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
        print("--> [ResNet152] Unfrozen full backbone for fine-tuning.")

    def check_unfreeze(self, epoch):
        if self.unfreeze_epoch is None:
            return False
        if self.backbone_frozen and epoch >= self.unfreeze_epoch:
            self.unfreeze_backbone()
            return True
        return False

    def load_from_checkpoint(self, checkpoint_path, device):
        checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
        print(f"--> Loading ResNet152 weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        state_dict = None
        if 'net' in ckpt:
            state_dict = ckpt['net']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') 
            new_state_dict[name] = v

        if any(k.startswith('backbone.') or k.startswith('head.') for k in new_state_dict):
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        else:
            backbone_state_dict = {
                k: v for k, v in new_state_dict.items()
                if not k.startswith('fc.')
            }
            skipped = len(new_state_dict) - len(backbone_state_dict)
            if skipped:
                print(f"--> [ResNet152] Skipped {skipped} checkpoint classifier keys; using custom head.")
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Warning: Missing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys: {len(unexpected_keys)}")
            
        print("--> Weights loaded successfully into self.model.")

    @staticmethod
    def resolve_checkpoint_path(checkpoint_path):
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        basename = os.path.basename(checkpoint_path)
        search_roots = [os.getcwd()]
        if os.path.exists("/kaggle/input"):
            search_roots.insert(0, "/kaggle/input")

        for root in search_roots:
            for current_dir, _, files in os.walk(root):
                if basename in files:
                    found_path = os.path.join(current_dir, basename)
                    print(f"--> [ResNet152] Checkpoint path not found; using discovered file: {found_path}")
                    return found_path

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
