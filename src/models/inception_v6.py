import torch 
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU, without bias """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.conv       = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5) # ~ laplace smoothing, default: 1e-5
        # inplace=True
        self.relu       = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block:
        Flow:
            Squeeze: Global Average Pooling -> BxCx1x1
            Excitation: FC -> ReLU -> FC -> Sigmoid -> BxCx1x1 (scale)
            Scale: input * scale (broadcast)
        Args:
        channels: số channel của input
        reduction: tỉ lệ giảm channel trong FC đầu tiên (bottleneck), default=16
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class InceptionBlock(nn.Module):
    """Inception Block: 
    Main:
        - branch1:  1x1 | 1S conv (stride=1 + same padding)
        - branch2:  1x1 | 1S conv       ~ 3x3 reduced
                    3x3 | 1S conv
        - branch3:  1x1 | 1S conv       ~ 2_3x3 reduced
                    3x3 | 1S conv       
                    3x3 | 1S conv
        - branch4:  3x3 | 1S maxpool
                    1x1 | 1S conv

    Return:
        DepthConcat([b1, b2, b3, b4], dim=1)       DepthConcat of 4 branches

    Args:
        in_channels        : số channel đầu vào
        out_1x1            : số filter branch b1
        out_3x3_reduced    : bottleneck trước Conv 3x3 (branch b2)
        out_3x3            : filter sau Conv 3x3 (branch b2)
        out_2_3x3_reduced  : bottleneck trước 2xConv 3x3 (branch b3)
        out_2_3x3          : filter của 2xConv 3x3 (branch b3)
        out_pool           : filter của Conv 1x1 sau MaxPool (branch b4)
    
    """
    def __init__(
        self,
        in_channels         : int, 
        out_1x1             : int,
        out_3x3_reduced     : int,
        out_3x3             : int,
        out_2_3x3_reduced   : int,
        out_2_3x3           : int,
        out_pool            : int,
    ):
        super().__init__()

        # branch 1
        self.b1 = ConvBlock(in_channels, out_1x1, kernel_size=1, stride=1, padding=0)
        
        # branch 2:
        self.b2 = nn.Sequential(
            ConvBlock(in_channels, out_3x3_reduced, kernel_size=1, stride=1, padding=0),
            ConvBlock(out_3x3_reduced, out_3x3, kernel_size=3, stride=1, padding=1)
        )

        # branch 3:
        self.b3 = nn.Sequential(
            ConvBlock(in_channels, out_2_3x3_reduced, kernel_size=1, stride=1, padding=0),
            ConvBlock(out_2_3x3_reduced, out_2_3x3, kernel_size=3, stride=1, padding=1),
            ConvBlock(out_2_3x3, out_2_3x3, kernel_size=3, stride=1, padding=1)
        ) 

        # branch 4:
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        # depth concat
        return torch.cat([b1, b2, b3, b4], dim=1)
    
class ResidualInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        out_3x3_reduced: int,
        out_3x3: int,
        out_2_3x3_reduced: int,
        out_2_3x3: int,
        out_pool: int,
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()

        self.block = InceptionBlock(
            in_channels=in_channels,
            out_1x1=out_1x1,
            out_3x3_reduced=out_3x3_reduced,
            out_3x3=out_3x3,
            out_2_3x3_reduced=out_2_3x3_reduced,
            out_2_3x3=out_2_3x3,
            out_pool=out_pool,
        )

        out_channels = out_1x1 + out_3x3 + out_2_3x3 + out_pool

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        if use_se:
            self.se = SEBlock(out_channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.block(x)
        out = self.se(out)
        out = out + identity
        return out


class AuxiliaryClassifier(nn.Module):
    """That is sub softmax after Block 3 (224 channels, 48x48)
    Flow:
        AdaptiveAvaragePool(16x16) -> Conv 1x1 -> Flatten -> Dropout -> FC -> Softmax
    """

    def __init__(self, in_channels: int, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        
        self.pool       = nn.AdaptiveAvgPool2d((16, 16))              # 16x16xin_channels (224)
        self.conv       = ConvBlock(in_channels=in_channels, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)
        self.flatten    = nn.Flatten()
        self.dropout    = nn.Dropout(p=dropout_p)
        self.fc         = nn.Linear(128 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class Inception(nn.Module):
    """ All inception net (7 block), has 2: softmax1,2, softmax1 only use when train, and when we inference softmax1 is discarded

    Pipeline:
        Stem  : Conv 3x3 -> BN -> MaxPool 3x3|1S
        B1  : 32  -> 64   (48x48)
        B2  : 64  -> 128  (48x48)
        B3  : 128 -> 224  (48x48)  <- here, auxiliary softmax1 
        B4  : 224 -> 320  (48x48)
        MaxPool stride=2   -> 24x24
        B5  : 320 -> 384  (24x24)
        B6  : 384 -> 448  (24x24)
        MaxPool stride=2   -> 12x12
        B7  : 448 -> 512  (12x12)
        AdaptiveAvgPool(1x1) -> Dropout -> FC -> softmax2
 
    """
    def __init__(self, config):
        super().__init__()

        # get config
        num_classes  = config['data'].get('num_classes', 7)
        dropout_main = config['model'].get('dropout_main', 0.4)
        dropout_aux  = config['model'].get('dropout_aux',  0.3)
        self.use_aux = config['model'].get('use_aux', True)

        use_se = config['model'].get('use_se', False)
        se_reduction = config['model'].get('se_reduction', 16)

        # Tranditional CNN or Stem (Stem?)
        # Bx1x48x48 --> Bx32x48x48
        # self.stem_conv = ConvBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.stem_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.stem = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )

        # --- Inception block 1-4 (keep 48x48) ---
        # Block1: in:32 --> out: 8 + 24 + 16 + 16 = 64
        self.block1 = ResidualInceptionBlock(32,  out_1x1=8,  out_3x3_reduced=16, out_3x3=24,
                                          out_2_3x3_reduced=8,  out_2_3x3=16, out_pool=16, use_se=use_se, se_reduction=se_reduction)
 
        # Block2: in=64,  out=32+48+24+24 = 128
        self.block2 = ResidualInceptionBlock(64,  out_1x1=32, out_3x3_reduced=24, out_3x3=48,
                                          out_2_3x3_reduced=16, out_2_3x3=24, out_pool=24, use_se=use_se, se_reduction=se_reduction)
 
        # Block3: in=128, out=64+96+32+32 = 224
        self.block3 = ResidualInceptionBlock(128, out_1x1=64, out_3x3_reduced=48, out_3x3=96,
                                          out_2_3x3_reduced=24, out_2_3x3=32, out_pool=32, use_se=use_se, se_reduction=se_reduction)
 
        # --- Auxiliary Classifier (softmax1) which input from block3 output 
        if self.use_aux:
            self.auxiliary = AuxiliaryClassifier( in_channels=224,
                                                  num_classes=num_classes,
                                                  dropout_p=dropout_aux)
 
        # Block4: in=224, out=96+128+64+32 = 320
        self.block4 = ResidualInceptionBlock(224, out_1x1=96, out_3x3_reduced=64,  out_3x3=128,
                                          out_2_3x3_reduced=32, out_2_3x3=64,  out_pool=32, use_se=use_se, se_reduction=se_reduction)

        # Giảm spatial lần 1: 48x48 --> 24x24 
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Inception block 5-6 (keep 24x24) ---
        # Block5: in=320, out=128+192+32+32 = 384
        self.block5 = ResidualInceptionBlock(320, out_1x1=128, out_3x3_reduced=96,  out_3x3=192,
                                          out_2_3x3_reduced=24, out_2_3x3=32,  out_pool=32, use_se=use_se, se_reduction=se_reduction)
 
        # Block6: in=384, out=160+224+32+32 = 448
        self.block6 = ResidualInceptionBlock(384, out_1x1=160, out_3x3_reduced=112, out_3x3=224,
                                          out_2_3x3_reduced=24, out_2_3x3=32,  out_pool=32, use_se=use_se, se_reduction=se_reduction)
 
        # Giảm spatial lần 2: 24×24 -> 12×12
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
        # --- Inception Block 7  (with 12×12) ---
        # Block7: in=448, out=192+256+32+32 = 512   # Q choose this number before FC
        self.block7 = ResidualInceptionBlock(448, out_1x1=192, out_3x3_reduced=144, out_3x3=256,
                                          out_2_3x3_reduced=24, out_2_3x3=32,  out_pool=32, use_se=use_se, se_reduction=se_reduction)
 

        # --- Softmax2 - main ouput ---
        self.adap_avg_pool  = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten        = nn.Flatten()      # -> 512
        self.dropout        = nn.Dropout(p=dropout_main)
        self.fc             = nn.Linear(512, num_classes)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x: Tensor Bx1x48x48 (orginal image)
        Returns:
            main_out    :   Tensor Bx7  (*)
            aux_out     :   Tensor Bx7  (only use_aux=True and when we training)
                            tuple(main_out, aux_out) | main_out
        """

        # Stem
        out = self.stem(x)    # B×32×48×48
        # out = self.stem_conv(x)         # B×32×48×48
        # out = self.stem_maxpool(out)    # B×32×48×48  (stride=1, không giảm)
 
        # Blocks 1->3
        out = self.block1(out)          # B×64×48×48
        out = self.block2(out)          # B×128×48×48
        out = self.block3(out)          # B×224×48×48
 
        # Auxiliary branch (only you, training:))
        aux_out = None
        if self.use_aux and self.training:
            aux_out = self.auxiliary(out)   # B×7
 
        # Block 4
        out = self.block4(out)          # B×320×48×48
 
        # Giảm spatial lần 1
        out = self.maxpool1(out)        # B×320×24×24
 
        # Blocks 5->6
        out = self.block5(out)          # B×384×24×24
        out = self.block6(out)          # B×448×24×24
 
        # Giảm spatial lần 2
        out = self.maxpool2(out)        # B×448×12×12
 
        # Block 7
        out = self.block7(out)          # B×512×12×12
 
        # Main out
        out = self.adap_avg_pool(out)   # B×512×1×1
        out = self.flatten(out)         # B×512
        out = self.dropout(out)
        main_out = self.fc(out)         # B×7
 
        if self.use_aux and self.training:
            return main_out, aux_out
 
        return main_out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    config = {
        "data":  {"num_classes": 7},
        "model": {"dropout_main": 0.4, "dropout_aux": 0.3, "use_aux": True},
    }
 
    model = Inception(config).to(device)
    model.train()
 
    x = torch.randn(4, 1, 48, 48, device=device)   # batch=4, grayscale, 48×48
 
    main_out, aux_out = model(x)
    print(f"main output : {main_out.shape}")        # torch.Size([4, 7])
    print(f"aux  output : {aux_out.shape}")         # torch.Size([4, 7])
 
    # Check tunwgf phần
    model.eval()
    with torch.no_grad():
        t = torch.randn(1, 1, 48, 48, device=device)
        t = model.stem_conv(t);    print(f"after stem_conv   : {t.shape}")   # 1×32×48×48
        t = model.stem_maxpool(t); print(f"after stem_pool   : {t.shape}")   # 1×32×48×48
        t = model.block1(t);       print(f"after block1      : {t.shape}")   # 1×64×48×48
        t = model.block2(t);       print(f"after block2      : {t.shape}")   # 1×128×48×48
        t = model.block3(t);       print(f"after block3      : {t.shape}")   # 1×224×48×48
        t = model.block4(t);       print(f"after block4      : {t.shape}")   # 1×320×48×48
        t = model.maxpool1(t);     print(f"after maxpool1    : {t.shape}")   # 1×320×24×24
        t = model.block5(t);       print(f"after block5      : {t.shape}")   # 1×384×24×24
        t = model.block6(t);       print(f"after block6      : {t.shape}")   # 1×448×24×24
        t = model.maxpool2(t);     print(f"after maxpool2    : {t.shape}")   # 1×448×12×12
        t = model.block7(t);       print(f"after block7      : {t.shape}")   # 1×512×12×12
 