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
    Optimized for PyTorch, Label Noise Robustness (SCE), and Differential LR.
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
        
        # Runtime states
        self.start_epoch = 0
        self._current_epoch = 0
        
        # Training progress trackers
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self._current_phase = 0 # Track curriculum phase
        self._latest_scn_logs = {} # For backward compatibility in metrics
        
        # --- AMP (Automatic Mixed Precision) ---
        self.scaler = torch.cuda.amp.GradScaler()


    def train_one_epoch(self):
        self.model.train()
        running_loss, corrects, total = 0.0, 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # Tự động chọn device type cho autocast (cuda hoặc cpu)
            device_type = self.device.type
            with torch.amp.autocast(device_type):
                # --- CHIẾN THUẬT MIXUP FADE-OUT ---
                # Phase 1 (Ep < 30): 100% dùng MixUp
                # Phase 2 (Ep 30-60): 50% cơ hội dùng MixUp
                # Phase 3 (Ep > 60): 10% cơ hội dùng MixUp
                use_mixup = False
                if self._current_epoch < 30:
                    use_mixup = True
                elif self._current_epoch < 90 and np.random.rand() < 0.7:
                    use_mixup = True
                elif self._current_epoch >= 90 and np.random.rand() < 0.5:
                    use_mixup = True
                elif self._current_epoch >= 120 and np.random.rand() < 0.2:
                    use_mixup = True

                if use_mixup:
                    alpha = self.config.get('training', {}).get('mixup_alpha', 0.2)
                    lam = np.random.beta(alpha, alpha)
                    index = torch.randperm(images.size(0), device=self.device)
                    mixed_images = lam * images + (1 - lam) * images[index]
                    
                    outputs = self.model(mixed_images, targets=None) 
                    logits = outputs if torch.is_tensor(outputs) else outputs[0]
                    
                    # Tính SCE Loss cho ảnh MixUp
                    loss = lam * self.criterion(logits, labels) + (1 - lam) * self.criterion(logits, labels[index])
                else:
                    outputs = self.model(images, targets=labels)
                    logits = outputs if torch.is_tensor(outputs) else outputs[0]
                    # Tính SCE Loss cho ảnh gốc
                    loss = self.criterion(logits, labels)

                # --- AUXILIARY LOSSES ĐỒ THỊ ---
                if not use_mixup:
                    aux_losses = getattr(self.model, "get_aux_losses", lambda: {})()
                    if isinstance(aux_losses, dict):
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

            # Backward & Step với AMP
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            corrects += (preds == labels).sum().item()
            total += labels.size(0)
            
        return running_loss / total, corrects / total

    def resume_from_checkpoint(self, checkpoint_path):
        """
        Khôi phục trạng thái huấn luyện từ checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint {checkpoint_path} không tìm thấy. Bắt đầu từ đầu.")
            return

        print(f"--> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', -1) + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self._current_phase = checkpoint.get('current_phase', 0)
        
        print(f"    [Resume] Resuming at epoch {self.start_epoch} (Best Val Acc: {self.best_val_acc:.4f})")

    def _set_backbone_frozen(self, freeze: bool):
        """Helper to freeze/unfreeze backbone layers"""
        # Hỗ trợ cả 1 GPU và Đa GPU (DataParallel)
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model

        if not hasattr(actual_model, 'backbone'): 
            return

        for param in actual_model.backbone.parameters():
            param.requires_grad = not freeze

        state = "FROZEN" if freeze else "UNFROZEN"
        print(f"\t>>> [Backbone] Set to {state}")

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
            preds = torch.argmax(logits, dim=1)
            corrects += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, corrects / total

    def fit(self):
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')
        if self.config.get('logging', {}).get('use_wandb', False):
            init_wandb(config=self.config, run_name=self.run_name)

        # Những thông số này đã được khởi tạo trong __init__ hoặc load từ checkpoint
        # Không reset ở đây để hỗ trợ Resume Training
        train_losses, val_losses = [], []

        for ep in range(self.start_epoch, self.epochs):
            self._current_epoch = ep
            progress = ep / max(self.epochs - 1, 1)
            
            # --- BẤT DI BẤT DỊCH: Curriculum Strategy (Nghệ thuật cài số) ---
            # Phase 1: Epoch 1 - 30 (MixUp Warmup 100%)
            if ep < 200:
                if self._current_phase != 1:
                    self._set_backbone_frozen(False)
                    self._current_phase = 1
                phase_name = "Phase 1: MixUp Warmup"

            # Phase 2: Epoch 31 - 90 (Co-Adaptation 70% MixUp)
            elif ep < 300:
                if self._current_phase != 2:
                    self._set_backbone_frozen(False)
                    # HẠ CÁNH MỀM: Kế thừa LR hiện tại và giảm 50%
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print("\t>>> [Phase 2 Init] Co-Adaptation: Kế thừa và giảm 50% LR hiện tại.")
                    self._current_phase = 2
                phase_name = "Phase 2: Co-Adaptation"

            # Phase 3: Epoch 91 - 1000 (Deep Refinement 50% MixUp)
            else:
                if self._current_phase != 3:
                    self._set_backbone_frozen(False)
                    # TINH CHỈNH SÂU: Tiếp tục giảm nhẹ LR
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print("\t>>> [Phase 3 Init] Deep Refinement: Giảm tiếp 50% LR.")
                    self._current_phase = 3
                phase_name = "Phase 3: Deep Refinement"
            
            # Sync progress to model if supported
            if hasattr(self.model, "set_training_progress"):
                self.model.set_training_progress(progress)

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

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
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    "epoch": ep,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "patience_counter": self.patience_counter,
                    "best_val_acc": self.best_val_acc,
                    "best_val_loss": self.best_val_loss,
                    "current_phase": self._current_phase
                }, self.path_save_ckpt)
                print(f"\t>>> Saved Best Model (Acc: {val_acc:.4f})")
            
            # 2. Early Stopping dựa trên val_loss (Không đổi nhãn để tránh nhiễu)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"\t>>> Val Loss improved: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"\t-!- No loss improvement: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
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
                outputs_v = self.model(images_v, targets=labels_v)
                logits_v = outputs_v if torch.is_tensor(outputs_v) else outputs_v[0]
                preds_v = torch.argmax(logits_v, dim=1)
                meta_v = getattr(self.model, "_latest_metadata", {})
                log_heatmap_samples(
                    images_v, labels_v, preds_v,
                    meta_v.get("node_attention"),
                    meta_v.get("sampling_grid"),
                    epoch=self.epochs
                )

        return train_losses, val_losses