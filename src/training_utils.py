"""
Training utilities for FER Advanced Model.

Includes:
- Combined loss functions (CE + Regularization)
- Training loop wrapper
- Metrics computation
- Visualization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# COMBINED LOSS FUNCTIONS
# ============================================================================

class FERCombinedLoss(nn.Module):
    """
    Combined loss for training:
    Loss = CE(logits, labels) + λ_div * diversity_loss + λ_sparse * sparsity_loss
    
    Why combine?
    - CE: Direct classification objective
    - Diversity: Encourages different regions to specialize
    - Sparsity: Prevents diffuse attention (encourages focus)
    
    These auxiliary losses guide the model to learn interpretable regions.
    """
    
    def __init__(self, lambda_diversity=0.1, lambda_sparsity=0.05):
        super().__init__()
        self.lambda_diversity = lambda_diversity
        self.lambda_sparsity = lambda_sparsity
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels, auxiliary):
        """
        Args:
            logits: (B, num_emotions) - Model predictions
            labels: (B,) - Ground truth labels
            auxiliary: dict with 'diversity_loss' and 'sparsity_loss'
        
        Returns:
            total_loss: Scalar
            loss_dict: dict with individual loss components
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Auxiliary losses
        diversity_loss = auxiliary['diversity_loss']
        sparsity_loss = auxiliary['sparsity_loss']
        
        # Combined
        total_loss = (
            ce_loss + 
            self.lambda_diversity * diversity_loss + 
            self.lambda_sparsity * sparsity_loss
        )
        
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, loss_dict


# ============================================================================
# TRAINING WRAPPER
# ============================================================================

class FERTrainer:
    """
    Trainer class for FER model with typical PyTorch training loop.
    
    Features:
    - Train/Val/Test splits
    - Learning rate scheduling
    - Early stopping
    - Metrics tracking
    """
    
    def __init__(self, model, device='cuda', lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss
        self.loss_fn = FERCombinedLoss(lambda_diversity=0.1, lambda_sparsity=0.05)
        
        # Tracking
        self.history = defaultdict(list)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits, auxiliary = self.model(images, return_auxiliary=True)
            loss, loss_dict = self.loss_fn(logits, labels, auxiliary)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def val_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, auxiliary = self.model(images, return_auxiliary=True)
                loss, _ = self.loss_fn(logits, labels, auxiliary)
                
                total_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=100, early_stop_patience=15):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.val_epoch(val_loader)
            
            # Track
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # LR scheduling
            self.scheduler.step(val_loss)
            
            # Log
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                logits, _ = self.model(images, return_auxiliary=True)
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Accuracy
        accuracy = (all_preds == all_labels).mean()
        
        return accuracy, all_preds, all_labels


# ============================================================================
# METRICS & VISUALIZATION
# ============================================================================

def compute_per_class_metrics(predictions, labels, num_classes=7):
    """Compute precision, recall, F1 per class"""
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    metrics = {}
    for cls in range(num_classes):
        tp = ((predictions == cls) & (labels == cls)).sum()
        fp = ((predictions == cls) & (labels != cls)).sum()
        fn = ((predictions != cls) & (labels == cls)).sum()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics[emotion_names[cls]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (labels == cls).sum()
        }
    
    return metrics


def plot_confusion_matrix(predictions, labels, num_classes=7):
    """Plot and save confusion matrix"""
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Compute confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes))
    for pred, label in zip(predictions, labels):
        conf_matrix[label, pred] += 1
    
    # Normalize
    conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    return conf_matrix, conf_matrix_norm


def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Training utilities loaded successfully!")
    print("\nExample usage:")
    print("""
    from src.models.fer_advanced_model import FERAdvancedModel
    from src.training_utils import FERTrainer
    
    # Initialize
    model = FERAdvancedModel()
    trainer = FERTrainer(model, device='cuda')
    
    # Train
    history = trainer.train(train_loader, val_loader, epochs=100)
    
    # Evaluate
    accuracy, predictions, labels = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")
    """)
