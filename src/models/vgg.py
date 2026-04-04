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
        self.dropout_block=config['model']['dropout_block']
        # self.input_size=input_size
        #block1 
        self.b1_conv1=nn.Conv2d(channels,out_channels=64,kernel_size=3, padding=1) #[channel,224,224] -> [64,224,224]
        self.b1_relu=nn.ReLU()
        self.b1_conv2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1) #[64,224,224] -> [64,224,224]
        self.b1_maxpool=nn.MaxPool2d(kernel_size=2,stride=2) #[64,224,224] -> [64,112,112]
        self.b1_dropout = nn.Dropout(self.dropout_block)

        #block2
        self.b2_conv1=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1) #[channel,112,112] -> [128,112,112]
        self.b2_relu=nn.ReLU()
        self.b2_conv2=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1) #[128,112,112] -> [128,112,112]
        self.b2_relu=nn.ReLU()
        self.b2_maxpool=nn.MaxPool2d(kernel_size=2,stride=2) #[128,112,112] -> [128,56,56]
        self.b2_dropout=nn.Dropout(self.dropout_block)

        #block3
        self.b3_conv1=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1) #[128,56,56] -> [256,56,56]
        self.b3_relu=nn.ReLU()
        self.b3_conv2=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1) #[256,56,56] -> [256,56,56]
        self.b3_relu=nn.ReLU()
        self.b3_maxpool=nn.MaxPool2d(kernel_size=2,stride=2) #[256,56,56] -> [256,28,28]
        self.b3_dropout=nn.Dropout(self.dropout_block)

        #block4
        self.b4_conv1=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1) #[256,28,28] -> [512,28,28]
        self.b4_relu=nn.ReLU()
        self.b4_conv2=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1) #[512,28,28] -> [512,28,28]
        self.b4_relu=nn.ReLU()
        self.b4_maxpool=nn.MaxPool2d(kernel_size=2,stride=2) #[512,28,28] -> [512,14,14]
        self.b4_dropout=nn.Dropout(self.dropout_block)

        # Removed Block 5 because 48/2^5 = 1.5. Stay at 3x3.
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()

        # Dense
        self.fc = nn.Linear(512 * 3 * 3, 512)
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(self.dropout_dense)

        # Output
        self.out = nn.Linear(512, self.num_classes)

    def forward(self, x):
        #block1
        out=self.b1_conv1(x)
        out=self.b1_relu(out)
        out=self.b1_conv2(out)
        out=self.b1_relu(out)
        out=self.b1_maxpool(out)
        out=self.b1_dropout(out)

        #block2
        out=self.b2_conv1(out)
        out=self.b2_relu(out)
        out=self.b2_conv2(out)
        out=self.b2_relu(out)
        out=self.b2_maxpool(out)
        out=self.b2_dropout(out)

        #block3
        out=self.b3_conv1(out)
        out=self.b3_relu(out)
        out=self.b3_conv2(out)
        out=self.b3_relu(out)
        out=self.b3_maxpool(out)
        out=self.b3_dropout(out)

        # block 4
        out = self.b4_relu(self.b4_conv1(out))
        out = self.b4_maxpool(self.b4_relu(self.b4_conv2(out)))
        out = self.b4_dropout(out)

        # avgpool
        out = self.avgpool(out)
        out = self.flatten(out)

        # FC
        out = self.fc_relu(self.fc(out))
        out = self.fc_dropout(out)

        # output
        out = self.out(out)
        return out







