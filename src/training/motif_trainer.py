import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.early_stopping import EarlyStopping

class MotifTrainer:
    """
    Dedicated trainer for MotifGraphModel.
    Simplified and robust.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion # CombinedMotifLoss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        self.epochs = config['training'].get('epochs', 100)
        self.best_val_loss = float('inf')
        
        # Early Stopping
        patience = config['training'].get('patience', 10)
        self.early_stopping = EarlyStopping(
            patience=patience, 
            verbose=True, 
            path=os.path.join(save_dir, 'best_motif_model.pt')
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward: MotifGraphModel returns (logits, top_k_idx, centers, scores)
            logits, top_k_idx, _, scores = self.model(images, return_selection=True)
            
            # CombinedMotifLoss forward(logits, targets, scores, top_k_idx, model)
            loss = self.criterion(logits, labels, scores, top_k_idx, model=self.model)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{torch.sum(preds == labels.data).item()/images.size(0):.4f}"})
            
        return running_loss / total, corrects.double() / total

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            logits, top_k_idx, _, scores = self.model(images, return_selection=True)
            loss = self.criterion(logits, labels, scores, top_k_idx, model=self.model)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
        return running_loss / total, corrects.double() / total

    def fit(self):
        train_losses, val_losses = [], []
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Summary Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early Stopping (handles saving best model internally)
            checkpoint_data = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': self.config
            }
            self.early_stopping(val_loss, self.model, extra_data=checkpoint_data)
            
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break
                
        return train_losses, val_losses
