import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from ..utils.logger_wandb import init_wandb, log_metrics, log_image_to_wandb
import math

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self._base_criterion = self.criterion
        
        # SCN & Label Smoothing Conflict fix
        ls = float(config.get('training', {}).get('label_smoothing', 0.0))
        self.use_scn = config.get('training', {}).get('use_scn', True)
        if self.use_scn and ls > 0:
            ls = 0.0 # Disable LS if SCN is active to protect confidence signals
            
        if ls and isinstance(self._base_criterion, torch.nn.CrossEntropyLoss):
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 10)
        self.model_name = config['model'].get('name', 'motif_graph_fer')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.path_save_ckpt = save_dir
        os.makedirs(self.path_save_ckpt, exist_ok=True)

        # Resume training
        self.base_epoch = int(config.get('training', {}).get('base_epoch', 0))
        
        # Optimized Weights from config
        self.motif_diversity_weight = config['training'].get('motif_diversity_weight', 0.15)
        self.attn_entropy_weight = config['training'].get('attn_entropy_weight', 0.01)
        self.motif_consistency_weight = config['training'].get('motif_consistency_weight', 0.05)
        self.offset_reg_weight = config['training'].get('offset_reg_weight', 0.01)
        
        # SCN Params
        self.scn_alpha = float(config['training'].get('scn_alpha', 1.0))
        self.scn_rank_lambda = float(config['training'].get('scn_rank_lambda', 0.5))
        self.scn_margin = float(config['training'].get('scn_margin', 0.15))
        self.scn_min_weight = 0.2

    def _scn_loss(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction='none')
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            weights = (1.0 - conf) ** 2
            weights = weights.clamp(min=self.scn_min_weight)

        loss = (weights * ce).mean()

        # Confidence-based Ranking (User Requested Fix)
        sorted_conf, idx = torch.sort(conf)
        B = logits.size(0)
        k = max(2, int(0.2 * B))
        hard_idx = idx[:k]
        easy_idx = idx[k:]

        if hard_idx.numel() > 0 and easy_idx.numel() > 0:
            conf_easy = conf[easy_idx].mean()
            conf_hard = conf[hard_idx].mean()
            # Penalize if hard confidence is not significantly lower than easy confidence
            ranking_loss = F.relu(conf_hard - conf_easy + self.scn_margin)
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)

        total_loss = (self.scn_alpha * loss) + (self.scn_rank_lambda * ranking_loss)
        return total_loss, {"scn_rank_loss": ranking_loss.item()}

    def train_one_epoch(self):
        self.model.train()
        running_loss, corrects, total = 0.0, 0, 0
        # Hiển thị Epoch thực tế (bắt đầu từ 1 hoặc base_epoch + 1)
        display_epoch = self._current_epoch + 1
        pbar = tqdm(self.train_loader, desc=f"Epoch {display_epoch}/{self.epochs + self.base_epoch}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            logits = outputs if not isinstance(outputs, dict) else outputs.get("logits")
            
            if self.use_scn:
                cls_loss, _ = self._scn_loss(logits, labels)
            else:
                cls_loss = self.criterion(logits, labels)

            # Aux losses
            aux = self.model.get_aux_losses() if hasattr(self.model, "get_aux_losses") else {}
            m_div = aux.get("motif_diversity", torch.tensor(0.0, device=self.device))
            m_ent = aux.get("attn_entropy", torch.tensor(0.0, device=self.device))
            m_con = aux.get("motif_consistency", torch.tensor(0.0, device=self.device))
            m_off = aux.get("offset_reg", torch.tensor(0.0, device=self.device))

            loss = cls_loss + (self.motif_diversity_weight * m_div) + \
                              (self.attn_entropy_weight * m_ent) + \
                              (self.motif_consistency_weight * m_con) + \
                              (self.offset_reg_weight * m_off)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            corrects += torch.sum(preds == labels)
            total += labels.size(0)
            
            # Hiển thị loss thời gian thực trên thanh tiến trình
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return running_loss / total, corrects.double() / total

    def validate(self):
        self.model.eval()
        running_loss, corrects, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                logits = outputs if not isinstance(outputs, dict) else outputs.get("logits")
                loss = self._base_criterion(logits, labels)
                running_loss += loss.item() * images.size(0)
                preds = torch.max(logits, 1)[1]
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        return running_loss / total, corrects.double() / total

    def fit(self):
        if self.use_wandb: init_wandb(self.config, self.run_name)
        best_acc = 0.0
        for ep in range(self.epochs):
            self._current_epoch = self.base_epoch + ep
            t_loss, t_acc = self.train_one_epoch()
            v_loss, v_acc = self.validate()
            if self.scheduler: self.scheduler.step(v_loss)
            
            # Print với Epoch bắt đầu từ 1
            display_epoch = self._current_epoch + 1
            print(f"Epoch {display_epoch}/{self.epochs + self.base_epoch} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
            
            if self.use_wandb: 
                log_metrics({"train_acc": t_acc, "train_loss": t_loss, "val_acc": v_acc, "val_loss": v_loss}, epoch=display_epoch)
            
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(self.model.state_dict(), os.path.join(self.path_save_ckpt, f"{self.model_name}_best.pth"))
        
        self.log_heatmaps(num_images=10)

    def log_heatmaps(self, num_images=10):
        """Trực quan hóa Motif Attention Heatmaps và log lên WandB"""
        if not self.use_wandb: return
        self.model.eval()
        images_logged = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _ = self.model(images)
                # Lấy heatmaps từ model (đã được lưu trong model._latest_heatmaps)
                heatmaps, _ = self.model.get_landmark_outputs() 
                if heatmaps is None: break
                
                B = images.size(0)
                for i in range(min(B, num_images - images_logged)):
                    # Chuẩn hóa ảnh gốc (H, W)
                    img = images[i][0].cpu().numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    
                    # Gộp và chuẩn hóa Heatmap tổng
                    hm = torch.sum(heatmaps[i], dim=0).cpu().numpy()
                    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                    
                    # Resize Heatmap về kích thước ảnh gốc bằng PIL
                    hm_resized = np.array(Image.fromarray((hm * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BILINEAR))
                    hm_resized = hm_resized / 255.0
                    
                    # Vẽ Overlay
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(img, cmap='gray')
                    ax.imshow(hm_resized, cmap='jet', alpha=0.4)
                    ax.axis('off')
                    ax.set_title(f"Label: {labels[i].item()}")
                    
                    log_image_to_wandb(f"Heatmap/Sample_{images_logged}", fig)
                    plt.close(fig)
                    images_logged += 1
                if images_logged >= num_images: break
        print(f"--> Successfully logged {images_logged} heatmaps to WandB.")