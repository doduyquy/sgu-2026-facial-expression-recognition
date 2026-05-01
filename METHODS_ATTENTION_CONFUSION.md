# Facial Expression Recognition - Attention & Confusion Learning Methods

## Overview
This document describes two priority-3 improvements to reach **74% accuracy** from current **68.18%**.

---

## Method 1: Region Attention Mechanisms 🔍

### Problem
Fear/Sad confusion (49.1% / 55.1% recall) due to:
- Small image size (48x48) loses micro-expressions
- Model doesn't focus on discriminative regions (mouth, eyes, eyebrows)

### Solution: RegionAttention Module
```python
# Location: src/models/attention.py
```

**Key Features:**
- **Spatial Attention**: Focus on specific face regions
  - Eye region (rows 0-2 in 6x6 feature map)
  - Mouth region (rows 4-6 in 6x6 feature map)
  
- **Channel Attention (SE-blocks)**: Learn which channels matter per region

- **Learned Region Weights**: Adaptive combination of regions
  - Eye attention: Distinguish fear/surprise/sad
  - Mouth attention: Distinguish fear/sad/disgust/anger
  - Eyebrow attention: Distinguish anger/fear

**Integration:**
```yaml
# configs/motif_attention_config.yaml
model:
  use_region_attention: true
```

**Expected Impact:**
- Fear recall: 49.1% → ~60%
- Sad recall: 55.1% → ~65%
- Overall: 68.18% → ~70-71%

---

## Method 2: Confusable Pair Learning 🎯

### Problem
Hard emotion pairs not weighted differently:
- Fear ↔ Sad (most confused)
- Sad ↔ Anger (moderate confusion)
- Standard CE loss treats all errors equally

### Solution: ConfusionMatrixLoss
```python
# Location: src/training/confusion_loss.py
```

**Key Features:**

1. **ConfusionMatrixLoss** (Margin-based)
   - Penalize confusing similar emotions
   - Weight pairs by historical confusion: (fear, sad) = 2.5x
   - Margin-based: want logit_true >> logit_confused + margin

2. **ContrastiveConfusionLoss** (Alternative)
   - Push confusable classes apart in logit space
   - Per-sample contrastive weighting

**Usage:**
```yaml
# configs/motif_attention_config.yaml
training:
  loss: "confusion_combined"  # CE + ConfusionMatrixLoss
  confusion_margin: 0.5
  confusion_loss_weight: 0.6  # Weight vs CE
```

**Expected Impact:**
- Fear vs Sad separation: +5-10% recall each
- Overall: 68.18% → ~71-72%

---

## Config: motif_attention_config.yaml

```yaml
model:
  use_region_attention: true  # Enable region attention
  
training:
  loss: "confusion_combined"  # New loss type
  confusion_margin: 0.5
  confusion_loss_weight: 0.6
  
  # Gentle augmentation for 48x48
  label_smoothing: 0.1
  use_class_weights: true
  class_weight_mode: "sqrt_inverse"

augmentation:
  hflip_p: 0.3       # Reduce horizontal flip
  rotation: 8        # Very gentle
  perspective_scale: 0.05
  crop_scale: [0.95, 1.0]  # Tight crop
```

---

## Training Command

### Local (with new attention config):
```bash
python scripts/train.py --config motif_attention_config --env local
```

### Expected Results Timeline:
- **Epoch 20-30**: Both methods should show improvement in fear/sad recall
- **Epoch 50-80**: Fear/Sad confusion should separate clearly
- **Epoch 100-150**: Plateau around 72-74% (region att) + confusion learning

---

## File Structure

```
New Files:
├── src/models/attention.py                # RegionAttention + ConfusionAwareAttention
├── src/training/confusion_loss.py        # ConfusionMatrixLoss + ContrastiveConfusionLoss
└── configs/motif_attention_config.yaml   # Configuration with both methods

Modified Files:
├── src/models/motif_graph_fer.py         # +RegionAttention integration
├── src/training/losses.py                # +confusion_combined loss type
└── src/data/transforms.py                # Gentle augmentation (already done)
```

---

## Debugging & Monitoring

1. **Check Region Attention is active:**
   ```python
   # In trainer.py log:
   if hasattr(model, 'region_attention'):
       print("✓ Region Attention enabled")
   ```

2. **Monitor confusion matrix per epoch:**
   - Track (fear, sad) recall specifically
   - WandB chart: Fear Recall, Sad Recall separately

3. **Loss breakdown:**
   - Log CE loss and Confusion loss separately
   - Should see CE decrease + Confusion loss spike then decrease

---

## Next Steps (If <72% Still)

1. **Focal Loss** (Priority 1):
   ```yaml
   loss: "focal_combined"
   focal_gamma: 2.0
   focal_alpha: [1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.2]
   ```

2. **Ensemble** (Priority 2):
   - Train: Motif + Simple CNN
   - Combine: 0.6*motif + 0.4*cnn

3. **Data Augmentation** (Priority 3):
   - Oversample fear/sad classes
   - Synthetic generation via mixup

---

## References

| Method | Paper |
|--------|-------|
| Region Attention | "CBAM: Convolutional Block Attention Module" (Woo et al., 2018) |
| Confusion Learning | "Class-Balanced Loss Based on Effective Number of Samples" |
| Margin Loss | "Large Margin Cosine Loss for Deep Face Recognition" (Wang et al., 2018) |
