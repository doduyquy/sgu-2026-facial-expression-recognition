import os
import torch
import numpy as np 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from datetime import datetime
from src.utils.logger_wandb import init_wandb, log_metrics, log_heatmap_samples

class Trainer:
    """
    Refactored Trainer for Motif-Graph FER.
    Optimized for PyTorch, Label Noise Robustness (SCN), and Differential LR.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Extract configurations first to avoid NameError
        train_cfg = config.get('training', {})
        
        # Initialize Criterion with Label Smoothing support
        ls = float(train_cfg.get('label_smoothing', 0.0))
        if ls > 0 and isinstance(criterion, torch.nn.CrossEntropyLoss):
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)
        else:
            self.criterion = criterion
            
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.epochs = train_cfg.get('epochs', 100)
        self.patience = train_cfg.get('patience', 10)
        self.run_name = run_name
        self.path_save_ckpt = save_dir
        
        # SCN Hyperparameters
        self.use_scn = train_cfg.get('use_scn', True)
        self.scn_alpha = float(train_cfg.get('scn_alpha', 1.0))
        self.scn_rank_lambda = float(train_cfg.get('scn_rank_lambda', 0.5))
        self.scn_margin = float(train_cfg.get('scn_margin', 0.4))
        self.scn_warmup_epochs = int(train_cfg.get('scn_warmup_epochs', 0))
        
        # Runtime states
        self._current_epoch = 0
        self._runtime_use_scn = self.use_scn
        self._runtime_use_mixup = False
        self._latest_scn_logs = {}

    def _scn_loss(self, logits, labels):
        """
        Correct SCN (Self-Cure Network) implementation for Label Noise:
        1. Suppress low-confidence samples (likely noisy labels).
        2. Ranking loss to maintain margin between easy and hard correctable samples.
        """
        ce = F.cross_entropy(logits, labels, reduction='none')
        
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            
            # SCN Suppressing Function: sigm(beta * (conf - threshold))
            # Rewards high-confidence samples, suppresses low-confidence (noisy) ones.
            # beta=10, threshold=0.45 creates a sharp transition around 0.45 prob.
            weights = torch.sigmoid(10 * (conf - 0.45))
            weights = weights.clamp(min=0.01) # Keep tiny gradient for stability
            
        loss_weighted = (weights * ce).mean()
        
        # Ranking Loss: enforces hard samples to have higher loss than easy samples by at least a margin
        sorted_ce, _ = torch.sort(ce)
        B = ce.size(0)
        k = max(1, int(0.7 * B)) # Đảm bảo luôn có ít nhất 1 easy sample
        easy_loss = sorted_ce[:k].mean()
        hard_tensor = sorted_ce[k:]
        # Tránh lỗi NaN khi batch cuối cùng quá nhỏ (empty tensor)
        hard_loss = hard_tensor.mean() if hard_tensor.numel() > 0 else easy_loss.detach()
        
        # L_rank = max(0, margin - (hard_loss - easy_loss))
        # Note: logic ensures hard_loss is at least 'margin' greater than easy_loss
        ranking_loss = F.relu(self.scn_margin - (hard_loss - easy_loss))
        
        total_loss = (self.scn_alpha * loss_weighted) + (self.scn_rank_lambda * ranking_loss)
        
        logs = {
            "scn/weight_mean": weights.mean().item(),
            "scn/conf_mean": conf.mean().item(),
            "scn/rank_loss": ranking_loss.item()
        }
        return total_loss, logs

    def train_one_epoch(self):
        self.model.train()
        running_loss, corrects, total = 0.0, 0, 0
        scn_acc_logs = {"scn/weight_mean": [], "scn/conf_mean": [], "scn/rank_loss": []}

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # 1. MixUp Staging (Preserve user's curriculum)
            if self._runtime_use_mixup:
                alpha = self.config['training'].get('mixup_alpha', 0.2)
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(images.size(0), device=self.device)
                mixed_images = lam * images + (1 - lam) * images[index]
                
                # Phase 1: Truyền targets=None để bỏ qua tính Consistency Loss nội bộ trên ảnh MixUp
                outputs = self.model(mixed_images, targets=None) 
                logits = outputs if torch.is_tensor(outputs) else outputs[0]
                
                loss = lam * F.cross_entropy(logits, labels) + (1 - lam) * F.cross_entropy(logits, labels[index])
            else:
                # 2. Forward & SCN / Base Loss
                outputs = self.model(images, targets=labels)
                logits = outputs if torch.is_tensor(outputs) else outputs[0]
                
                if self._runtime_use_scn and self._current_epoch >= self.scn_warmup_epochs:
                    loss, scn_logs = self._scn_loss(logits, labels)
                    for k, v in scn_logs.items():
                        scn_acc_logs[k].append(v)
                else:
                    loss = self.criterion(logits, labels)

            # 3. Auxiliary Losses (Motif Diversity, Consistency, Entropy, Offset...)
            aux_losses = getattr(self.model, "get_aux_losses", lambda: {})()
            
            # Kích hoạt các hàm loss đồ thị khi KHÔNG dùng MixUp (Phase 2 & 3)
            if not self._runtime_use_mixup and isinstance(aux_losses, dict):
                aux_weights_config = {
                    "motif_diversity": "motif_diversity_weight",
                    "motif_consistency": "motif_consistency_weight",
                    "offset_reg": "offset_reg_weight",
                    "attn_entropy": "attn_entropy_weight" 
                }
                
                for loss_name, config_key in aux_weights_config.items():
                    if loss_name in aux_losses:
                        weight = float(self.config.get('training', {}).get(config_key, 0.05))
                        if weight > 0.0:
                            loss += weight * aux_losses[loss_name]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        # Log SCN averages
        if scn_acc_logs["scn/weight_mean"]:
            self._latest_scn_logs = {k: np.mean(v) for k, v in scn_acc_logs.items()}
            
        return running_loss / total, corrects.double() / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss, corrects, total = 0.0, 0, 0

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images, targets=labels)
            logits = outputs if torch.is_tensor(outputs) else outputs[0]
            
            loss = self.criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        return running_loss / total, corrects.double() / total

    def fit(self):
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')
        if self.config.get('logging', {}).get('use_wandb', False):
            init_wandb(config=self.config, run_name=self.run_name)

        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0

        for ep in range(self.epochs):
            self._current_epoch = ep
            progress = ep / max(self.epochs - 1, 1)
            
            # --- BẤT DI BẤT DỊCH: Curriculum Strategy ---
            if progress <= 0.06:
                # Phase 1: Warm-up with MixUp
                self._runtime_use_scn = False
                self._runtime_use_mixup = True
                phase_name = "Phase 1: MixUp Warmup"
            elif progress <= 0.7:
                # Phase 2: SCN + Motif Signals
                self._runtime_use_scn = True
                self._runtime_use_mixup = False
                phase_name = "Phase 2: SCN & Motif Signals"
            else:
                # Phase 3: Final Refinement (Higher lambdas if needed)
                self._runtime_use_scn = True
                self._runtime_use_mixup = False
                phase_name = "Phase 3: Refinement"
            
            # Sync progress to model if supported
            if hasattr(self.model, "set_training_progress"):
                self.model.set_training_progress(progress)

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {ep+1}/{self.epochs} [{phase_name}] - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

            if self.config.get('logging', {}).get('use_wandb', False):
                metrics = {
                    "train/loss": train_loss, "train/acc": train_acc,
                    "val/loss": val_loss, "val/acc": val_acc,
                    "lr": self.optimizer.param_groups[0]['lr']
                }
                metrics.update(self._latest_scn_logs)
                log_metrics(metrics, epoch=ep)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 1. Save Best Model based on val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": ep,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "val_acc": val_acc.item() if hasattr(val_acc, 'item') else val_acc,
                    "val_loss": val_loss
                }, self.path_save_ckpt)
                print(f"\t>>> Saved Best Model (Acc: {val_acc:.4f})")
            
            # 2. Early Stopping based on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"\t>>> Val Loss improved: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"\t-!- No loss improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break

        # --- Sau khi huấn luyện xong: Log 10 ảnh đúng/sai của mô hình TỐT NHẤT ---
        if self.config.get('logging', {}).get('use_wandb', False):
            print("\n--> Logging final heatmap visualizations...")
            # Load lại best model weights
            if os.path.exists(self.path_save_ckpt):
                checkpoint = torch.load(self.path_save_ckpt, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.eval()
            images_v, labels_v = next(iter(self.val_loader))
            images_v, labels_v = images_v.to(self.device), labels_v.to(self.device)
            
            with torch.no_grad():
                logits_v = self.model(images_v, targets=labels_v)
                preds_v = torch.argmax(logits_v, dim=1)
                meta_v = getattr(self.model, "_latest_metadata", {})
                log_heatmap_samples(
                    images_v, labels_v, preds_v,
                    meta_v.get("node_attention"),
                    meta_v.get("sampling_grid"),
                    epoch=self.epochs
                )

        return best_val_acc