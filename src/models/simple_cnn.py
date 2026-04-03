import torch 
import torch.nn as nn

""" Table of SimpleCNN architecture details made by nii

- Formula output shape:             
                                        O = floor((I - K + 2P) / S ) + 1

    + O: output shape   |   I: input shape
    + K: kernel (3x3 or 5x5,...) -> K = 3 | 5
    + P: padding (0, 1,...)
    + S: stride (1, 2, 3,...)

- NOTE: all kernel size is used: 3x3, add relu after conv and dense
===============================================================================
Block name      Layer,              Params,                    Output Shape
===============================================================================
Input,          Input image,        Gray or RGB,               "(C, 48, 48)"
-------------------------------------------------------------------------------
Block 1,        Conv + Conv,        "64 kernels, padding=1",   "(64,48,48)"     
                Max-Pool 1,         "2x2, stride=2",           "(64, 24, 24)"
                Dropout             0.25
-------------------------------------------------------------------------------
Block 2,        Conv + Conv,        "128 kernels, padding=1",  "(128,24,24)"
                Max-Pool 2,         "2x2, stride=2",           "(128, 12, 12)"
                Dropout             0.25
-------------------------------------------------------------------------------
Block 3,        Conv + Conv,        "256 kernels, padding=1",  "(256,12,12)"
                Max-Pool 3,         "2x2, stride=2",           "(256, 6, 6)"
                Dropout             0.25
-------------------------------------------------------------------------------
Flatten                                                        "(9216,)"
-------------------------------------------------------------------------------
Dense                               128 units                  "(128,)"
                Dropout             0.5         
-------------------------------------------------------------------------------
Output                                                         "(7,)" -- logits
===============================================================================
"""

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # config
        self.num_classes = config['data']['num_classes']
        self.dropout_block = config['model']['dropout_block']
        self.dropout_dense = config['model']['dropout_dense']

        # block 1: (default) stride=1
        self.b1_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.b1_relu1 = nn.ReLU()
        self.b1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b1_relu2 = nn.ReLU()
        self.b1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # (64, 24, 24)
        self.b1_dropout = nn.Dropout(self.dropout_block)

        # block 2:
        self.b2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.b2_relu1 = nn.ReLU()
        self.b2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.b2_relu2 = nn.ReLU()
        self.b2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # (128, 12, 12)
        self.b2_dropout = nn.Dropout(self.dropout_block)


        # block 3: 
        self.b3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.b3_relu1 = nn.ReLU()
        self.b3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.b3_relu2 = nn.ReLU()
        self.b3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # (256, 6, 6)
        self.b3_dropout = nn.Dropout(self.dropout_block)

        # flatten
        self.flatten = nn.Flatten()

        # dense
        self.fc = nn.Linear(256*6*6, 128) # (9216, 128) -> 128
        self.fc_batchnorm = nn.BatchNorm1d(128)
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(self.dropout_dense)

        # output: 
        self.out = nn.Linear(128, self.num_classes) # 7

    def forward(self, x):
        # block 1
        out = self.b1_conv1(x)
        out = self.b1_relu1(out)
        out = self.b1_conv2(out)
        out = self.b1_relu2(out)
        out = self.b1_maxpool(out)
        out = self.b1_dropout(out)
        
        # block 2
        out = self.b2_conv1(out)
        out = self.b2_relu1(out)
        out = self.b2_conv2(out)
        out = self.b2_relu2(out)
        out = self.b2_maxpool(out)
        out = self.b2_dropout(out)

        # block 3
        out = self.b3_conv1(out)
        out = self.b3_relu1(out)
        out = self.b3_conv2(out)
        out = self.b3_relu2(out)
        out = self.b3_maxpool(out)
        out = self.b3_dropout(out)

        # flatten
        out = self.flatten(out)

        # dense
        out = self.fc(out)
        out = self.fc_batchnorm(out)
        out = self.fc_relu(out)
        out = self.fc_dropout(out)

        # out 
        out = self.out(out)

        return out



if __name__ == "__main__":
    """Test model which Q has defined"""
    mock_config = {
        'data': {
            'num_classes': 7  # 7 emotions of FER2013
        },
        'model': {
            'dropout_block': 0.25,
            'dropout_dense': 0.5
        }
    }
    print(" Initialize SimpleCNN...")
    model = SimpleCNN(mock_config)

    # Create rand Tensor which is similar to Tensor from DataLoader.
    # Batch=8, Channel=1 (Grayscale), WxH = 48x48
    dummy_input = torch.randn(8, 1, 48, 48)
    
    # Forward pass
    print(f"- Kích thước Input: {dummy_input.shape}")
    output = model(dummy_input)
    # Check output shape: (Batch=8, Classes=7) ? 
    print(f"- Kích thước Output:  {output.shape}") 
    
    if output.shape == (8, 7):
        print("-"*51)
        print("[OK] PASS: Tensor shape oke!")
        print("-"*51)
    else:
        print("-"*51)
        print("[X] FAILED: Output's shape is wrong.")
        print("-"*51)

    # Count: Trainable Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Out network has: {total_params:,} params (learn).")
