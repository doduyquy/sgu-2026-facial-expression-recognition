import torch
import torch.nn as nn

# Input:  (B, 1, 48, 48)
# conv1: 3x3, stride=1, pad=1           -> (B, 64, 48, 48)
# pool:  2x2, stride=2                  -> (B, 64, 24, 24)
# layer2: ConvBlock(s=1) + 2 IDs        -> (B, 256, 24, 24)
# layer3: ConvBlock(s=2) + 3 IDs        -> (B, 512, 12, 12)
# avgpool: AdaptiveAvgPool2d((1,1))     -> (B, 512, 1, 1)
# flatten                               -> (B, 512)
# fc                                    -> (B, num_classes)
#Hout = ((Hin + 2*pad - kernel_size) // stride) + 1

class IdentityBlock(nn.Module): #giữ nguyên kích thước không gian (H x W) và số kênh,tinh chỉnh đặc trưng rồi cộng tắt (residual) với đầu vào.
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        F1,F2,F3 = filters # F1: số kênh của conv1, F2: số kênh của conv2, F3: số kênh của conv3
        #vd 256, [64,64,256]
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1) #256, 64, kernel_size=1
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1) #64, 64, kernel_size=3, padding=1
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1) #64, 256, kernel_size=1
        self.bn3 = nn.BatchNorm2d(F3)

        self.relu = nn.ReLU()
    def forward(self, x):
        shortcut = x    
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))

        x += shortcut
        x = self.relu(x)

        return x
    
class ConvBlock(nn.Module): #thay đổi kích thước/ số kênh đặc trưng,đồng thời chiếu nhánh tắt (shortcut) để khớp kích thước trước khi cộng residual.
    def __init__(self, in_channels, filters, stride=2):
        super().__init__()
        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)

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

        x += shortcut
        x = self.relu(x)

        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=7, in_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        # Stage 2
        self.layer2 = nn.Sequential(
            ConvBlock(64, [64,64,256], stride=1),
            IdentityBlock(256, [64,64,256]),
            IdentityBlock(256, [64,64,256])
        )

        # Stage 3
        self.layer3 = nn.Sequential(
            ConvBlock(256, [128,128,512]),
            IdentityBlock(512, [128,128,512]),
            IdentityBlock(512, [128,128,512]),
            IdentityBlock(512, [128,128,512])
        )
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x