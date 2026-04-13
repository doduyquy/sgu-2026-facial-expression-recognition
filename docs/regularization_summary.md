# Summary: Anti-Overfitting measures for Hybrid VGG-ResNet

This document details the modifications made to the project on 2026-04-13 to prevent overfitting (currently observed as 75% train vs 68% val accuracy).

## 1. Advanced Data Augmentation
**File**: `src/data/transforms.py`
- **ColorJitter**: Randomly varies brightness and contrast by 20%. This makes the model more robust to lighting variations in the FER2013 dataset.
- **RandomErasing (Probability: 0.3)**: Randomly masks out small patches (pixels) of the image. 
    - **Rationale**: Since facial expressions are often determined by multiple regions (eyes AND mouth), if one region is masked, the model is forced to learn features from the other regions. This prevents "lazy learning" or "over-relying" on a single clear feature.

## 2. Model Regularization (DropPath)
**File**: `src/models/dual_fusion.py`
- **DropPath (Stochastic Depth)**: Implemented and integrated into the `HybridAttentionFusionBlock`.
- **How it works**: During training, it randomly drops whole paths in the residual connections (in this case, the Attention outcome).
- **Rationale**: This is a powerful form of dropout that works well for multi-branch/hybrid architectures. It forces the model to achieve correct classification even if a specific pathway (e.g., the Self-Attention branch) is temporarily disabled.

## 3. Training & Loss Adjustments
**File**: `configs/vgg_resnet_attention.yaml`
- **Weight Decay (0.001)**: Increased by 10x relative to the previous default. This puts a higher penalty on large weights, effectively "simplifying" the model and preventing it from fitting noise.
- **Label Smoothing (0.1)**: Instead of the loss forcing the model to be 100% sure about a label, it now targets 90% and spreads the other 10% across all labels.
    - **Rationale**: Mitigates "over-confidence" and helps the model generalize across similar-looking expressions (like Surprise and Fear).

## 4. Expected Results
- Narrowing the gap between Train and Val Accuracy.
- A smoother, more stable loss curve during Phase 2 (Fine-tuning).
- Expected accuracy boost: **+1.5% to 3%** on the validation set.
