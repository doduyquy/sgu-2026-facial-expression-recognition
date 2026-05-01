"""
Test script for new attention and confusion learning methods
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

def test_region_attention():
    """Test RegionAttention module"""
    print("=" * 60)
    print("Testing RegionAttention...")
    print("=" * 60)
    
    from src.models.attention import RegionAttention
    
    region_att = RegionAttention(feat_dim=128, num_regions=3)
    
    # Mock feature map: (B=4, C=128, H=6, W=6)
    x = torch.randn(4, 128, 6, 6)
    print(f"Input shape: {x.shape}")
    
    y = region_att(x)
    print(f"Output shape: {y.shape}")
    
    assert y.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {y.shape}"
    assert not torch.isnan(y).any(), "NaN values in output!"
    assert not torch.isinf(y).any(), "Inf values in output!"
    
    print("✓ RegionAttention: PASSED")
    print()


def test_confusion_matrix_loss():
    """Test ConfusionMatrixLoss"""
    print("=" * 60)
    print("Testing ConfusionMatrixLoss...")
    print("=" * 60)
    
    from src.training.confusion_loss import ConfusionMatrixLoss
    
    loss_fn = ConfusionMatrixLoss(num_classes=7, margin=0.5)
    
    # Mock data: hard confusion pairs
    B = 8
    logits = torch.randn(B, 7)
    # Emotion indices: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    labels = torch.tensor([2, 4, 0, 1, 2, 4, 0, 6])  # Mix of hard pairs
    
    print(f"Batch size: {B}, Num classes: 7")
    print(f"Labels (hard pairs): {labels.tolist()}")
    
    loss = loss_fn(logits, labels, reduction='mean')
    print(f"Mean loss: {loss.item():.4f}")
    
    assert loss.item() > 0, "Loss should be positive!"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    
    print("✓ ConfusionMatrixLoss: PASSED")
    print()


def test_motif_model_with_attention():
    """Test MotifGraphModel with RegionAttention integrated"""
    print("=" * 60)
    print("Testing MotifGraphModel with RegionAttention...")
    print("=" * 60)
    
    from src.models.motif_graph_fer import MotifGraphModel
    
    config = {
        'feat_dim': 128,
        'num_classes': 7,
        'motifs_per_class': 8,
        'top_k': 4,
        'use_region_attention': True,  # Enable region attention
    }
    
    model = MotifGraphModel(config)
    
    # Check that region attention is initialized
    assert hasattr(model, 'region_attention'), "Model should have region_attention attribute"
    assert model.region_attention is not None, "region_attention should not be None"
    print("✓ Region Attention initialized in model")
    
    # Mock input: (B=2, C=1, H=48, W=48)
    x = torch.randn(2, 1, 48, 48)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (2, 7), f"Expected output shape (2, 7), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in model output!"
    
    print("✓ MotifGraphModel forward pass: PASSED")
    print()


def test_combined_loss():
    """Test combined CE + ConfusionMatrixLoss"""
    print("=" * 60)
    print("Testing Combined CE + Confusion Loss...")
    print("=" * 60)
    
    from src.training.confusion_loss import ConfusionMatrixLoss
    
    # Build combined loss manually
    ce_loss = nn.CrossEntropyLoss()
    conf_loss = ConfusionMatrixLoss(num_classes=7, margin=0.5)
    weight = 0.6
    
    B = 4
    logits = torch.randn(B, 7)
    labels = torch.tensor([2, 4, 0, 1])
    
    l_ce = ce_loss(logits, labels)
    l_conf = conf_loss(logits, labels)
    total_loss = l_ce + weight * l_conf
    
    print(f"CE Loss: {l_ce.item():.4f}")
    print(f"Confusion Loss: {l_conf.item():.4f}")
    print(f"Total Loss (CE + {weight}*Conf): {total_loss.item():.4f}")
    
    assert total_loss.item() > 0, "Total loss should be positive!"
    assert not torch.isnan(total_loss), "Total loss is NaN!"
    
    print("✓ Combined Loss: PASSED")
    print()


def test_config_loading():
    """Test that new config can be loaded"""
    print("=" * 60)
    print("Testing Config Loading...")
    print("=" * 60)
    
    from src.utils.config import load_config
    
    try:
        config = load_config("motif_attention_config", "local")
        print(f"✓ Config loaded successfully")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Loss: {config['training']['loss']}")
        print(f"  - Use Region Attention: {config['model'].get('use_region_attention', False)}")
        print(f"  - Confusion Loss Weight: {config['training'].get('confusion_loss_weight', 'N/A')}")
        
        assert config['model'].get('use_region_attention') == True, "Region attention should be enabled"
        assert config['training']['loss'] == 'confusion_combined', "Loss should be confusion_combined"
        
        print("✓ Config validation: PASSED")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        raise
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  TESTING ATTENTION & CONFUSION LEARNING METHODS".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    try:
        test_region_attention()
        test_confusion_matrix_loss()
        test_motif_model_with_attention()
        test_combined_loss()
        test_config_loading()
        
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 58 + "║")
        print("║" + "  ✓ ALL TESTS PASSED!".ljust(58) + "║")
        print("║" + " " * 58 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\n")
        
    except Exception as e:
        print("\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 58 + "║")
        print("║" + f"  ✗ TEST FAILED: {str(e)[:50]}".ljust(58) + "║")
        print("║" + " " * 58 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\n")
        raise
