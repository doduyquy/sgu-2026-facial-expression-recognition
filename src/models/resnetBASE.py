import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityBlock(nn.Module):
    def __init__(self, channels, filters):
        super(IdentityBlock,self). __init__()
        F1,F2,F3=filters
        self.conv1=nn.Conv2d(channels,F1,kernel_size=1)
        self.bn1=nn.BatchNorm2d(F1)

        self.conv2=nn.Conv2d(F1,F2,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(F2)
        
        self.conv3=nn.Conv2d(F2,F3,kernel_size=1)
        self.bn3=nn.BatchNorm2d(F3)
        self.attn=nn.Identity()

        self.relu=nn.ReLU()
    def forward(self,x):
        shorcut=x
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        x=x+shorcut # cong phan du
        x=self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=2):
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(f3),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.relu(x + shortcut)
        return x


class Resnet35(nn.Module):

    def __init__(self, config, channels=1):
        super().__init__()
        
        # Lấy thông tin từ config
        self.num_classes = config.get('data', {}).get('num_classes', 7) if config else 7
        
        # 1. Stem
        # Input: (B, channels, 48, 48)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. Stages
        # Mỗi block bottleneck sẽ có dạng [f1, f2, f3]
        # Stage 2: 3 blocks -> Output spatial: 24x24
        self.layer2 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256])
        )

        # Stage 3: 4 blocks -> Output spatial: 12x12
        self.layer3 = nn.Sequential(
            ConvBlock(256, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512])
        )

        # Stage 4: 4 blocks -> Output spatial: 6x6
        self.layer4 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024])
        )
        self.dropout = nn.Dropout(0.3)

        # 3. Head for Classification (Trường hợp dùng độc lập)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        """input: (B, C, 48, 48)
       stage1: (B, 64, 24, 24)
       stage2: (B, 256, 24, 24)
       stage3: (B, 512, 12, 12)
       stage4: (B, 1024, 6, 6)
       sau quá trình trích xuất đặc trưng ta avgpool để giảm kích thước về (B, 1024, 1, 1)
       sau đó flatten để giảm kích thước về (B, 1024)
       cuối cùng ta đưa vào fc để phân loại"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # ResNet Blocks
        x = self.layer2(x)  # -> (B, 256, 24, 24)
        x = self.layer3(x)  # -> (B, 512, 12, 12)
        x = self.layer4(x)  # -> (B, 1024, 6, 6)

        x = self.avgpool(x) # -> (B, 1024, 1, 1)
        x = torch.flatten(x, 1) # -> (B, 1024)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    def extract_region_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer2(x) #[B,256,24,24]
        x = self.layer3(x) #[B,512,12,12]
        x = self.layer4(x)  # [B, 1024, 6, 6]
        x = torch.flatten(x, 2)               # [B, 1024, 36]
        x = x.transpose(1, 2)                 # [B, 36, 1024]
        return x

