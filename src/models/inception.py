import torch 
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5) # ~ laplace smoothing, default: 1e-5
        # inplace=True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class InceptionBlock(nn.Module):
    """Inception Block: 
    Args:
        in_channels     : input channels
        out_1x1         : branch 1
        out_3x3_reduced : branch 2 (1x1)
        out_3x3         : branch 2 (3x3)
        out_5x5_reduced : branch 3 (1x1)
        out_5x5         : branch 3 (5x5)
        out_pool        : branch 4 
    
    Main:
        - branch1:  1x1 | 1S conv (stride=1 + same padding)
        - branch2:  1x1 | 1S conv       ~ 3x3 reduced
                    3x3 | 1S conv
        - branch3:  1x1 | 1S conv
                    5x5 | 1S conv       ~ 5x5 reduced
        - branch4:  3x3 | 1S maxpool
                    1x1 | 1S conv
    Return
        DepthConcat of 4 branches
    """
    


    
    pass


class Inception(nn.Module):

    pass


if __name__ == "main":
    pass