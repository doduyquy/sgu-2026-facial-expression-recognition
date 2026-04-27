import os
import torch
import numpy as np 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics


class Trainer:
    """
    Focused Trainer for MotifGraphModel.
    Handles Forward -> Loss -> Backward -> Optimization.
    Includes SCN (Self-Correcting Network) and Motif-specific strategies.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self._base_criterion = self.criterion
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 10)
        self.model_name = config['model'].get('name', 'motif_graph_fer')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.config = config
        self.path_save_ckpt = save_dir
        
        # SCN Config
        self.use_scn = config['training'].get('use_scn', True)
        self.scn_warmup_epochs = int(config['training'].get('scn_warmup_epochs', 0))
        self.scn_alpha = float(config['training'].get('scn_alpha', 1.0))
        self.scn_rank_lambda = float(config['training'].get('scn_rank_lambda', 0.3))
        self.scn_min_weight = float(config['training'].get('scn_min_weight', 0.2))
        self.scn_margin = float(config['training'].get('scn_margin', 0.4))
        
        self._current_epoch = 0
        self._runtime_use_scn = None

    def _scn_loss(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction='none')
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            weights = (1.0 - conf) ** 2
            weights = weights.clamp(min=self.scn_min_weight)

        loss = (weights * ce).mean()
        sorted_conf, idx = torch.sort(conf)
        B = logits.size(0)
        k = max(2, int(0.2 * B))
        hard_idx = idx[:k]
        easy_idx = idx[k:]
        
        hard_loss = ce[hard_idx].mean() if hard_idx.numel() > 0 else torch.tensor(0.0, device=self.device)
        easy_loss = ce[easy_idx].mean() if easy_idx.numel() > 0 else torch.tensor(0.0, device=self.device)
        
        if self._current_epoch >= self.scn_warmup_epochs:
            ranking_loss = F.relu(easy_loss - hard_loss + self.scn_margin)
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)

        total_loss = (self.scn_alpha * loss) + (self.scn_rank_lambda * ranking_loss)
        return total_loss, {"scn_weight_mean": weights.mean().item(), "scn_rank_loss": ranking_loss.item()}

    def train_one_epoch(self, epoch):
        self.model.train()
        self._current_epoch = epoch
        running_loss = 0.0
        corrects = 0
        total = 0
        
        is_motif_model = self.model_name == 'motif_graph_fer'
        if is_motif_model and hasattr(self.model, 'set_training_progress'):
            self.model.set_training_progress(epoch / max(1, self.epochs))

        pbar = tqdm(self.train_loader)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            if is_motif_model:
                outputs = self.model(images, return_selection=True, targets=labels)
                logits, top_k_idx, _, scores = outputs
                
                # Main loss (CombinedMotifLoss)
                main_loss = self.criterion(logits, labels, scores, top_k_idx, model=self.model)
                
                # 1. Consistency Loss
                consistency_loss = torch.tensor(0.0, device=self.device)
                if self.config['training'].get('use_consistency', True):
                    angle = float(torch.empty(1).uniform_(-10, 10))
                    images_aug = TF.rotate(images, angle)
                    with torch.no_grad():
                        _, _, _, scores_aug = self.model(images_aug, return_selection=True)
                    consistency_loss = F.mse_loss(scores, scores_aug)
                
                # 2. Entropy Regularization
                probs = torch.softmax(logits, dim=1)
                entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-9), dim=1))
                
                # SCN handling if enabled
                if self.use_scn and epoch >= self.scn_warmup_epochs:
                    scn_loss, _ = self._scn_loss(logits, labels)
                    # For motif model, we blend SCN with our motif losses
                    loss = scn_loss + 0.1 * consistency_loss + 0.01 * entropy_loss
                else:
                    loss = main_loss + 0.1 * consistency_loss + 0.01 * entropy_loss
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(corrects.item()/total):.4f}"})
            
        return running_loss / total, corrects.double() / total

    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                
        return running_loss / total, corrects.double() / total

    def fit(self):
        best_acc = 0.0
        train_losses, val_losses = [], []
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate_one_epoch(epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            
            if self.use_wandb:
                log_metrics({
                    "Epoch": epoch,
                    "Train/Loss": train_loss,
                    "Train/Accuracy": train_acc,
                    "Val/Loss": val_loss,
                    "Val/Accuracy": val_acc,
                    "LR": self.optimizer.param_groups[0]['lr']
                }, epoch=epoch)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_env() if hasattr(self.model, 'state_env') else self.model.state_dict(), self.path_save_ckpt)
                print(f"--> Saved best model with val_acc: {val_acc:.4f}")
                
        return train_losses, val_losses