import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from ..utils.logger_wandb import init_wandb, log_metrics, log_image_to_wandb
import math

class Trainer:
    """Forward -> Compute loss -> zero_grad -> Backward -> Update weights (step)"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        # keep base criterion available for runtime switching (focal vs base)
        self._base_criterion = self.criterion
        
        # optionally enable label smoothing for CrossEntropy if configured
        ls = float(config.get('training', {}).get('label_smoothing', 0.0)) if isinstance(config, dict) else 0.0
        self.use_scn = config.get('training', {}).get('use_scn', True)
        
        # SCN Conflict: Disable label smoothing if SCN is enabled to preserve confidence signals
        if self.use_scn and ls > 0:
            print("WARNING: SCN is active. Disabling Label Smoothing to preserve confidence signals.")
            ls = 0.0

        if ls and isinstance(self._base_criterion, torch.nn.CrossEntropyLoss):
            try:
                self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)
                self._base_criterion = self.criterion
            except Exception:
                pass

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 10)
        self.model_name = config['model'].get('name', 'simple_cnn')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.config = config
        self.path_save_ckpt = save_dir
        # Ensure save directory exists
        os.makedirs(self.path_save_ckpt, exist_ok=True)
        
        # Resume training support
        self.base_epoch = int(config.get('training', {}).get('base_epoch', 0))
        
        # Aligned keys from motif_config.yaml
        self.motif_diversity_weight = config.get('training', {}).get('motif_diversity_weight', 0.15)
        self.attn_entropy_weight = config.get('training', {}).get('attn_entropy_weight', 0.01)
        self.motif_consistency_weight = config.get('training', {}).get('motif_consistency_weight', 0.05)
        self.offset_reg_weight = config.get('training', {}).get('offset_reg_weight', 0.01)
        
        # Legacy/Other lambdas (hidden or defaults)
        self.landmark_aux_cls_lambda = config['training'].get('landmark_aux_cls_lambda', 0.05)
        
        # === SCN (light) ===
        self.scn_warmup_epochs = int(config['training'].get('scn_warmup_epochs', 0))
        self.scn_alpha = float(config['training'].get('scn_alpha', 1.0))
        self.scn_rank_lambda = float(config['training'].get('scn_rank_lambda', 0.5))
        self.scn_min_weight = float(config['training'].get('scn_min_weight', 0.2))
        self.scn_margin = float(config['training'].get('scn_margin', 0.15)) 
        
        # runtime flags (set by fit staging)
        self._runtime_use_scn = None
        self.mixup_alpha = float(config['training'].get('mixup_alpha', 0.2))
        self._runtime_use_mixup = False

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, dict):
            return outputs.get("logits")
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            return outputs[0]
        return outputs

    def _extract_aux_losses(self, outputs):
        if isinstance(outputs, dict):
            aux = outputs.get("aux_losses", None)
            if isinstance(aux, dict):
                return aux
        getter = getattr(self.model, "get_aux_losses", None)
        if callable(getter):
            aux = getter()
            if isinstance(aux, dict):
                return aux
        return {}

    def _scn_loss(self, logits, labels):
        """
        SCN-light with Confidence-based Ranking
        """
        ce = F.cross_entropy(logits, labels, reduction='none')
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            weights = (1.0 - conf) ** 2
            weights = weights.clamp(min=self.scn_min_weight)

        loss = (weights * ce).mean()

        # Confidence-based Ranking (User Fix)
        sorted_conf, idx = torch.sort(conf)
        B = logits.size(0)
        k = max(2, int(0.2 * B))
        hard_idx = idx[:k]
        easy_idx = idx[k:]

        if hard_idx.numel() > 0 and easy_idx.numel() > 0:
            conf_easy = conf[easy_idx].mean()
            conf_hard = conf[hard_idx].mean()
            ranking_loss = F.relu(conf_hard - conf_easy + self.scn_margin)
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)

        total_loss = (self.scn_alpha * loss) + (self.scn_rank_lambda * ranking_loss)
        logs = {
            "scn_weight_mean": float(weights.mean().cpu().item()),
            "scn_conf_mean": float(conf.mean().cpu().item()),
            "scn_rank_loss": float(ranking_loss.cpu().item()),
        }
        return total_loss, logs

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        _scn_acc = {"scn_weight_mean": [], "scn_conf_mean": [], "scn_rank_loss": []}

        for images, labels in tqdm(self.train_loader, desc=f"Epoch {self._current_epoch}"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            mixup_active = bool(getattr(self, '_runtime_use_mixup', False))
            if mixup_active:
                alpha = self.mixup_alpha
                lam = np.random.beta(alpha, alpha)
                perm = torch.randperm(images.size(0), device=images.device)
                images = lam * images + (1.0 - lam) * images[perm]
                labels_a, labels_b = labels, labels[perm]

            outputs = self.model(images)
            logits = self._extract_logits(outputs)

            if mixup_active:
                cls_loss = lam * self._base_criterion(logits, labels_a) + (1.0 - lam) * self._base_criterion(logits, labels_b)
                scn_logs = None
            else:
                runtime_use_scn = getattr(self, '_runtime_use_scn', self.use_scn)
                if runtime_use_scn:
                    cls_loss, scn_logs = self._scn_loss(logits, labels)
                    for k in _scn_acc: _scn_acc[k].append(scn_logs[k])
                else:
                    cls_loss = self._base_criterion(logits, labels)

            # Motif Branch Loss Calculation
            aux_losses = self._extract_aux_losses(outputs)
            m_div = aux_losses.get("motif_diversity", torch.tensor(0.0, device=self.device))
            m_ent = aux_losses.get("attn_entropy", torch.tensor(0.0, device=self.device))
            m_off = aux_losses.get("offset_reg", torch.tensor(0.0, device=self.device))
            m_con = aux_losses.get("motif_consistency", torch.tensor(0.0, device=self.device))

            # Runtime lambdas
            div_lambda = getattr(self, '_runtime_diversity_lambda', self.motif_diversity_weight)
            ent_lambda = getattr(self, '_runtime_entropy_lambda', self.attn_entropy_weight)
            con_lambda = getattr(self, '_runtime_consistency_lambda', self.motif_consistency_weight)
            off_lambda = getattr(self, '_runtime_offset_reg_lambda', self.offset_reg_weight)

            # Dynamic scaling based on batch confidence
            try:
                probs = F.softmax(logits.detach(), dim=1)
                conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                scale = torch.clamp(((1.0 - conf.mean()) ** 2), 0.5, 1.5)
            except Exception:
                scale = torch.tensor(1.0, device=self.device)

            loss = cls_loss + (float(div_lambda) * m_div * scale) + \
                              (float(ent_lambda) * m_ent * scale) + \
                              (float(con_lambda) * m_con * scale) + \
                              (float(off_lambda) * m_off * scale)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        corrects = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                logits = self._extract_logits(outputs)
                cls_loss = self._base_criterion(logits, labels)

                aux_losses = self._extract_aux_losses(outputs)
                m_div = aux_losses.get("motif_diversity", torch.tensor(0.0, device=self.device))
                m_ent = aux_losses.get("attn_entropy", torch.tensor(0.0, device=self.device))
                m_off = aux_losses.get("offset_reg", torch.tensor(0.0, device=self.device))
                m_con = aux_losses.get("motif_consistency", torch.tensor(0.0, device=self.device))

                div_lambda = getattr(self, '_runtime_diversity_lambda', self.motif_diversity_weight)
                ent_lambda = getattr(self, '_runtime_entropy_lambda', self.attn_entropy_weight)
                con_lambda = getattr(self, '_runtime_consistency_lambda', self.motif_consistency_weight)
                off_lambda = getattr(self, '_runtime_offset_reg_lambda', self.offset_reg_weight)

                loss = cls_loss + (float(div_lambda) * m_div) + \
                                  (float(ent_lambda) * m_ent) + \
                                  (float(con_lambda) * m_con) + \
                                  (float(off_lambda) * m_off)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        return running_loss / total, corrects.double() / total

    def fit(self):
        if self.use_wandb: init_wandb(config=self.config, run_name=self.run_name)
        best_val_acc = 0.0
        for ep in range(self.epochs):
            actual_epoch = self.base_epoch + ep
            self._current_epoch = actual_epoch
            progress = actual_epoch / max(self.epochs + self.base_epoch - 1, 1)
            
            # Phase staging logic
            if progress <= 0.06:
                self._runtime_use_scn, self._runtime_use_mixup = False, True
                self._runtime_diversity_lambda, self._runtime_entropy_lambda = 0.0, 0.0
                self._runtime_consistency_lambda, self._runtime_offset_reg_lambda = 0.0, 0.0
            elif progress <= 0.7:
                self._runtime_use_scn, self._runtime_use_mixup = True, False
                self._runtime_diversity_lambda, self._runtime_entropy_lambda = 0.18, 0.004
                self._runtime_consistency_lambda, self._runtime_offset_reg_lambda = 0.07, 0.01
            else:
                self._runtime_use_scn, self._runtime_use_mixup = True, False
                self._runtime_diversity_lambda, self._runtime_entropy_lambda = 0.30, 0.008
                self._runtime_consistency_lambda, self._runtime_offset_reg_lambda = 0.10, 0.01

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            if self.scheduler: self.scheduler.step(val_loss)
            
            print(f"Epoch {actual_epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            if self.use_wandb:
                log_metrics({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(self.path_save_ckpt, f"{self.model_name}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"--> Saved best model to: {save_path}")
        
        # Log Heatmaps sau khi train xong (User Request)
        print("\n--> Training finished. Generating heatmaps for WandB...")
        self.log_heatmaps(num_images=10)

    def log_heatmaps(self, num_images=10):
        """Trực quan hóa Motif Attention Heatmaps và log lên WandB"""
        if not self.use_wandb: return
        
        self.model.eval()
        images_logged = 0
        
        # Lấy một batch từ val_loader
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                # Lấy heatmaps từ model (đã được lưu trong model._latest_heatmaps)
                heatmaps, _ = self.model.get_landmark_outputs() # (B, K, H_att, W_att)
                if heatmaps is None: break
                
                B, K, H_att, W_att = heatmaps.shape
                
                for i in range(min(B, num_images - images_logged)):
                    img = images[i].cpu().numpy().squeeze() # (H, W)
                    # Chuẩn hóa ảnh gốc để hiển thị
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    
                    # Gộp các heatmap của các motif lại thành 1 heatmap tổng
                    combined_hm = torch.sum(heatmaps[i], dim=0).cpu().numpy()
                    combined_hm = (combined_hm - combined_hm.min()) / (combined_hm.max() - combined_hm.min() + 1e-8)
                    
                    # Resize heatmap về kích thước ảnh gốc
                    hm_resized = np.array(Image.fromarray((combined_hm * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BILINEAR))
                    hm_resized = hm_resized / 255.0
                    
                    # Vẽ bằng matplotlib
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(img, cmap='gray')
                    ax.imshow(hm_resized, cmap='jet', alpha=0.4) # Overlay với độ trong suốt 0.4
                    ax.axis('off')
                    ax.set_title(f"Label: {labels[i].item()}")
                    
                    # Log lên WandB
                    log_image_to_wandb(f"Heatmap/Sample_{images_logged}", fig)
                    plt.close(fig)
                    
                    images_logged += 1
                
                if images_logged >= num_images: break
        print(f"--> Successfully logged {images_logged} heatmaps to WandB.")