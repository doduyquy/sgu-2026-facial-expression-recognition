import torch
import torch.nn as nn

"""
Custom EfficientNet architecture for facial expression recognition

EfficientNetForFER2013

Mô hình CNN được thiết kế theo hướng EfficientNet-inspired cho bài toán
nhận diện cảm xúc khuôn mặt trên tập dữ liệu FER-2013.

Đặc điểm chính:
- Sử dụng MBConv (Mobile Inverted Bottleneck Convolution)
- Áp dụng Depthwise Separable Convolution để giảm số lượng tham số
- Tích hợp Squeeze-and-Excitation (SE) để học trọng số theo kênh
- Sử dụng hàm kích hoạt SiLU (Swish)
- Có kết nối residual trong các block phù hợp
- Sử dụng Global Average Pooling thay cho Flatten truyền thống

Kiến trúc tổng quát:
Input → Stem → MBConv Stages → Head → Global Pooling → Dense → Output

Mô hình được rút gọn và điều chỉnh để phù hợp với:
- Kích thước ảnh nhỏ (48x48)
- Dataset FER-2013
- Huấn luyện từ đầu (không dùng pretrained)

Output:
- Vector logits với 7 lớp cảm xúc
"""


class SqueezeExcitationBlock(nn.Module):
    
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()

        reduced_channels = max(1, in_channels // reduction_ratio)

        self.global_average_pool = nn.AdaptiveAvgPool2d(1)

        self.reduce_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=reduced_channels,
            kernel_size=1,
            bias=True
        )
        self.reduce_activation = nn.SiLU()

        self.expand_conv = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True
        )
        self.channel_gate = nn.Sigmoid()

    def forward(self, x):
        squeeze = self.global_average_pool(x)
        squeeze = self.reduce_conv(squeeze)
        squeeze = self.reduce_activation(squeeze)
        squeeze = self.expand_conv(squeeze)
        squeeze = self.channel_gate(squeeze)

        return x * squeeze


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_ratio=6,
        stride=1,
        kernel_size=3,
        se_reduction_ratio=4,
        dropout_rate=0.0
    ):
        super().__init__()

        self.use_residual_connection = (stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expansion_ratio

        expansion_layers = []
        if expansion_ratio != 1:
            expansion_layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=expanded_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            ])
        else:
            expanded_channels = in_channels

        self.expansion_block = nn.Sequential(*expansion_layers) if expansion_layers else nn.Identity()

        self.depthwise_convolution = nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expanded_channels,
            bias=False
        )
        self.depthwise_batch_normalization = nn.BatchNorm2d(expanded_channels)
        self.depthwise_activation = nn.SiLU()

        self.squeeze_excitation_block = SqueezeExcitationBlock(
            in_channels=expanded_channels,
            reduction_ratio=se_reduction_ratio
        )

        self.projection_convolution = nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.projection_batch_normalization = nn.BatchNorm2d(out_channels)

        self.residual_dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        x = self.expansion_block(x)

        x = self.depthwise_convolution(x)
        x = self.depthwise_batch_normalization(x)
        x = self.depthwise_activation(x)

        x = self.squeeze_excitation_block(x)

        x = self.projection_convolution(x)
        x = self.projection_batch_normalization(x)

        if self.use_residual_connection:
            x = self.residual_dropout(x)
            x = x + identity

        return x


class EfficientNetForFER2013(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config["data"]["num_classes"]

        model_config = config["model"]
        self.stem_channels = model_config.get("stem_channels", 32)
        self.head_channels = model_config.get("head_channels", 128)
        self.classifier_dropout_rate = model_config.get("classifier_dropout_rate", 0.4)
        self.block_dropout_rate = model_config.get("block_dropout_rate", 0.1)

        # Input: (B, 1, 48, 48)
        self.stem_convolution = nn.Conv2d(
            in_channels=1,
            out_channels=self.stem_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.stem_batch_normalization = nn.BatchNorm2d(self.stem_channels)
        self.stem_activation = nn.SiLU()

        # Stage 1 -> (B, 16, 48, 48)
        self.stage_1 = MBConvBlock(
            in_channels=self.stem_channels,
            out_channels=16,
            expansion_ratio=1,
            stride=1,
            kernel_size=3,
            se_reduction_ratio=4,
            dropout_rate=self.block_dropout_rate
        )

        # Stage 2 -> (B, 24, 24, 24)
        self.stage_2_block_1 = MBConvBlock(
            in_channels=16,
            out_channels=24,
            expansion_ratio=6,
            stride=2,
            kernel_size=3,
            se_reduction_ratio=4,
            dropout_rate=self.block_dropout_rate
        )
        self.stage_2_block_2 = MBConvBlock(
            in_channels=24,
            out_channels=24,
            expansion_ratio=6,
            stride=1,
            kernel_size=3,
            se_reduction_ratio=4,
            dropout_rate=self.block_dropout_rate
        )

        # Stage 3 -> (B, 40, 12, 12)
        self.stage_3_block_1 = MBConvBlock(
            in_channels=24,
            out_channels=40,
            expansion_ratio=6,
            stride=2,
            kernel_size=5,
            se_reduction_ratio=4,
            dropout_rate=self.block_dropout_rate
        )
        self.stage_3_block_2 = MBConvBlock(
            in_channels=40,
            out_channels=40,
            expansion_ratio=6,
            stride=1,
            kernel_size=5,
            se_reduction_ratio=4,
            dropout_rate=self.block_dropout_rate
        )

        # Stage 4 -> (B, 80, 6, 6)
        self.stage_4_block_1 = MBConvBlock(
            in_channels=40,
            out_channels=80,
            expansion_ratio=6,
            stride=2,
            kernel_size=3,
            se_reduction_ratio=4,
            dropout_rate=self.block_dropout_rate
        )
        self.stage_4_block_2 = MBConvBlock(
            in_channels=80,
            out_channels=80,
            expansion_ratio=6,
            stride=1,
            kernel_size=3,
            se_reduction_ratio=4,
            dropout_rate=self.block_dropout_rate
        )

        # Head -> (B, head_channels, 6, 6)
        self.head_convolution = nn.Conv2d(
            in_channels=80,
            out_channels=self.head_channels,
            kernel_size=1,
            bias=False
        )
        self.head_batch_normalization = nn.BatchNorm2d(self.head_channels)
        self.head_activation = nn.SiLU()

        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.classifier_dropout = nn.Dropout(self.classifier_dropout_rate)
        self.output_layer = nn.Linear(self.head_channels, self.num_classes)

    def forward(self, x):
        # Stem
        x = self.stem_convolution(x)
        x = self.stem_batch_normalization(x)
        x = self.stem_activation(x)

        # MBConv stages
        x = self.stage_1(x)

        x = self.stage_2_block_1(x)
        x = self.stage_2_block_2(x)

        x = self.stage_3_block_1(x)
        x = self.stage_3_block_2(x)

        x = self.stage_4_block_1(x)
        x = self.stage_4_block_2(x)

        # Head
        x = self.head_convolution(x)
        x = self.head_batch_normalization(x)
        x = self.head_activation(x)

        # Classifier
        x = self.global_average_pooling(x)
        x = self.flatten(x)
        x = self.classifier_dropout(x)
        x = self.output_layer(x)

        return x


if __name__ == "__main__":
    sample_config = {
        "data": {
            "num_classes": 7
        },
        "model": {
            "stem_channels": 32,
            "head_channels": 128,
            "classifier_dropout_rate": 0.4,
            "block_dropout_rate": 0.1
        }
    }

    model = EfficientNetForFER2013(sample_config)

    dummy_input = torch.randn(8, 1, 48, 48)
    dummy_output = model(dummy_input)

    print("Model architecture:")
    print(model)
    print()

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Total trainable parameters: {trainable_parameters:,}")