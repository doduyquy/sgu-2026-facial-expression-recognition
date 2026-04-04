#impor thu vien de su dung vvg19
import torchvision.models as models
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
        return x








