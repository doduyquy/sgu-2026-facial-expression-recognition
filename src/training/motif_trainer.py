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
        """ SCN logic: weighted loss + ranking loss (Class-aware Tuning) """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1) # (B,)
            
            # Basic confidence weighting
            weights = (1.0 - conf) ** 2
            
            # (3) SCN Tuning theo class:
            # Tăng trọng số cho Disgust (1) và Fear (2) để mô hình tập trung hơn
            class_focus = torch.ones_like(labels, dtype=torch.float)
            class_focus[labels == 1] = 1.5 # Disgust focus
            class_focus[labels == 2] = 1.2 # Fear focus
            weights = weights * class_focus
            
            weights = weights.clamp(min=0.2, max=2.0)
        
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
        import torchvision.transforms.functional as TF
        self.model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward: MotifGraphModel returns (logits, top_k_idx, centers, scores)
            # MotifGraphModel now uses Gumbel-Softmax during training
            logits, top_k_idx, _, scores = self.model(images, return_selection=True)
            
            # (6) Entropy Regularization: Encourage sharp selection
            # Extract attn_weights indirectly from the model if possible or recalculate
            # For simplicity, we can regularize the confidence scores
            probs = torch.softmax(logits, dim=1)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-9), dim=1))
            
            if self.use_scn:
                ce_none = nn.CrossEntropyLoss(reduction='none')
                l_ce_none = ce_none(logits, labels)
                l_motif_none = self.criterion.motif(scores, top_k_idx, labels, reduction='none')
                base_loss_per_sample = l_ce_none + self.criterion.weight * l_motif_none
                main_loss = self._scn_loss(logits, labels, base_loss_per_sample, epoch)
            else:
                main_loss = self.criterion(logits, labels, scores, top_k_idx, model=self.model)

            # (6) Augment-Consistency Loss (Stability)
            consistency_loss = torch.tensor(0.0, device=self.device)
            if self.config['training'].get('use_consistency', True):
                # Mild augmentation
                angle = float(torch.empty(1).uniform_(-10, 10))
                images_aug = TF.rotate(images, angle)
                
                # Forward aug
                logits_aug, _, _, scores_aug = self.model(images_aug, return_selection=True)
                # MSE loss between scores (ensure same motifs are activated)
                consistency_loss = F.mse_loss(scores, scores_aug)
            
            # Total Loss
            loss = main_loss + 0.01 * entropy_loss + 0.1 * consistency_loss
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "cons": f"{consistency_loss.item():.4f}",
                "acc": f"{torch.sum(preds == labels.data).item()/images.size(0):.4f}"
            })
            
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
