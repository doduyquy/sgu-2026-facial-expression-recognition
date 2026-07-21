import torch
import timm

print("--- HRNet 48x48 stride=(1,1) ---")
model1 = timm.create_model('hrnet_w18', pretrained=False, features_only=True)
model1.conv1.stride = (1, 1)
dummy1 = torch.randn(2, 3, 48, 48)
feats1 = model1(dummy1)
for i, f in enumerate(feats1):
    print(f"Feature {i}: {f.shape}")
