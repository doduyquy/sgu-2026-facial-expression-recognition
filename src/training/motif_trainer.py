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
        
        # SCN Parameters
        self.use_scn = config['training'].get('use_scn', True)
        self.scn_alpha = float(config['training'].get('scn_alpha', 1.0))
        self.scn_rank_lambda = float(config['training'].get('scn_rank_lambda', 0.3))
        self.scn_warmup_epochs = int(config['training'].get('scn_warmup_epochs', 5))
        self.scn_margin = 0.4
        
        # Early Stopping
        patience = config['training'].get('patience', 10)
        self.early_stopping = EarlyStopping(
            patience=patience, 
            verbose=True, 
            path=os.path.join(save_dir, 'best_motif_model.pt')
        )

    def _scn_loss(self, logits, labels, base_loss_per_sample, epoch):
        """ SCN logic: weighted loss + ranking loss """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1) # (B,)
            weights = (1.0 - conf) ** 2
            weights = weights.clamp(min=0.2)
        
        # Weighted main loss
        weighted_loss = (weights * base_loss_per_sample).mean()
        
        # Ranking loss
        if epoch >= self.scn_warmup_epochs:
            B = logits.size(0)
            k = max(2, int(0.2 * B))
            sorted_conf, idx = torch.sort(conf)
            hard_idx, easy_idx = idx[:k], idx[k:]
            
            hard_loss = base_loss_per_sample[hard_idx].mean()
            easy_loss = base_loss_per_sample[easy_idx].mean()
            ranking_loss = torch.relu(easy_loss - hard_loss + self.scn_margin)
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)
            
        total_loss = self.scn_alpha * weighted_loss + self.scn_rank_lambda * ranking_loss
        return total_loss

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
            
            if self.use_scn:
                # 1. Classification part (per-sample)
                ce_none = nn.CrossEntropyLoss(reduction='none')
                l_ce_none = ce_none(logits, labels)
                
                # 2. Motif Consistency part (per-sample)
                l_motif_none = self.criterion.motif(scores, top_k_idx, labels, reduction='none')
                
                # Combined per-sample base loss for SCN to judge
                base_loss_per_sample = l_ce_none + self.criterion.weight * l_motif_none
                
                # SCN re-weighting and ranking
                total_weighted_loss = self._scn_loss(logits, labels, base_loss_per_sample, epoch)
                
                # Diversity loss (global regularizer, doesn't depend on samples)
                l_div = self.model.compute_motif_diversity_loss()
                
                loss = total_weighted_loss + self.criterion.div_weight * l_div
            else:
                # Standard CombinedMotifLoss
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
