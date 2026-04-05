#impor thu vien de su dung vvg19
import torchvision.models as models
import torch
import torch.nn as nn 

class VGG19(nn.Module):
    # def __init__(self, config, channels=3):
    #     super().__init__()

    #     # config
    #     self.num_classes = config['data']['num_classes']
    #     self.pretrained = config['model'].get('pretrained', True)
    #     self.dropout_dense = config['model'].get('dropout_dense', 0.5)

    #     # load vgg19
    #     self.vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT if self.pretrained else None)
        
    #     if channels != 3: 
    #         self.vgg19.features[0] = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)

    #     # Adapt classifier
    #     self.vgg19.classifier[2] = nn.Dropout(self.dropout_dense)
    #     self.vgg19.classifier[5] = nn.Dropout(self.dropout_dense)
    #     self.vgg19.classifier[6] = nn.Linear(4096, self.num_classes)
    # def forward(self,x):
    #     return self.vgg19(x)



    def __init__(self, config, channels):
        super().__init__()
        # config
        self.num_classes = config['data']['num_classes']
        self.dropout_dense = config['model']['dropout_dense']
        self.dropout_block = config['model']['dropout_block']

        # Block 1: 48x48 -> 24x24
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 2: 24x24 -> 12x12
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 3: 12x12 -> 6x6
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 4: 6x6 -> 3x3
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_dense),
            nn.Linear(512, self.num_classes)
        )

        # Kaiming init (Cực kỳ quan trọng để train từ đầu)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class VGGFusion(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        # config
        self.num_classes = config['data']['num_classes']
        self.dropout_dense = config['model']['dropout_dense']
        self.dropout_block = config['model']['dropout_block']
        self.use_aux = config['model'].get('use_aux', False)
        self.use_attention = config['model'].get('use_attention', False)

        # Block 1: 48x48 -> 24x24
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 2: 24x24 -> 12x12
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 3: 12x12 -> 6x6
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 4: 6x6 -> 3x3
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_block)
        )

        if self.use_attention:
            print("--> Using CBAM (Channel + Spatial Attention) in VGGFusion")
            self.cbam3 = CBAM(256, ratio=16, kernel_size=7)
            self.cbam4 = CBAM(512, ratio=16, kernel_size=3)

        # Auxiliary classifier from Block 3
        if self.use_aux:
            self.aux_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Flatten(),
                nn.Linear(256 * 3 * 3, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_dense),
                nn.Linear(256, self.num_classes)
            )

        # Fusion logic
        self.fusion_pool = nn.AdaptiveAvgPool2d((3, 3)) # To make B3 (6,6) match B4 (3,3)
        
        self.classifier = nn.Sequential(
            nn.Linear((512 + 256) * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_dense),
            nn.Linear(512, self.num_classes)
        )

        # Kaiming init
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        feat_b3 = self.b3(x)
        feat_b4 = self.b4(feat_b3)
        
        
        # Apply Attention BEFORE Fusion (giữ flow của bản 69.3%)
        if self.use_attention:
            feat_b3 = self.cbam3(feat_b3)
            feat_b4 = self.cbam4(feat_b4)
            
        # 1. Auxiliary branch (chỉ lấy khi train và bật use_aux)
        aux_out = None
        if self.training and self.use_aux:
            aux_out = self.aux_classifier(feat_b3)
            
        # 2. Multi-scale Fusion
        # Resize feat_b3 (6x6) về (3x3) để nối với feat_b4
        feat_b3_resized = self.fusion_pool(feat_b3)
        combined = torch.cat([feat_b4, feat_b3_resized], dim=1) # (512+256) = 768 channels
        
        out = torch.flatten(combined, 1)
        out = self.classifier(out)
        
        if self.training and self.use_aux:
            return out, aux_out
            
        return out

if __name__ == "__main__":
    # Test VGGFusion
    print("Testing VGGFusion with Attention...")
    mock_config = {
        'data': {'num_classes': 7},
        'model': {
            'dropout_dense': 0.5,
            'dropout_block': 0.3,
            'use_aux': True,
            'use_attention': True
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGGFusion(mock_config, channels=1).to(device)
    model.train() # To test aux branch
    
    dummy_input = torch.randn(2, 1, 48, 48).to(device)
    out, aux = model(dummy_input)
    
    print(f"Main output shape: {out.shape}")
    print(f"Aux output shape: {aux.shape}")
    
    model.eval()
    out_eval = model(dummy_input)
    print(f"Eval output shape: {out_eval.shape}")
    
    assert out.shape == (2, 7)
    assert aux.shape == (2, 7)
    assert out_eval.shape == (2, 7)
    print("VGGFusion Test Passed!")








