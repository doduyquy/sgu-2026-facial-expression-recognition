import os
import torch
import numpy as np 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics


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
        # (Cac tham so lambda cu cua landmark branch da duoc loai bo de toi uu hoa)

        # === SCN (light) ===
        self.use_scn = config['training'].get('use_scn', True)
        # default warmup: disabled by default, SCN controlled by phase schedule
        self.scn_warmup_epochs = int(config['training'].get('scn_warmup_epochs', 0))
        self.scn_alpha = float(config['training'].get('scn_alpha', 1.0))
        # ranking influence tuned for FER (raise to emphasize hard/easy separation)
        self.scn_rank_lambda = float(config['training'].get('scn_rank_lambda', 0.5))  # UPDATE: stronger SCN ranking
        self.scn_min_weight = float(config['training'].get('scn_min_weight', 0.2))
        # UPDATE: Giảm margin xuống 0.2 để tránh bùng nổ Gradient
        self.scn_margin = float(config['training'].get('scn_margin', 0.2))
        # runtime flags (set by fit staging)
        self._runtime_use_scn = None
        # mixup defaults
        self.mixup_alpha = float(config['training'].get('mixup_alpha', 0.2))
        self._runtime_use_mixup = False
        self.scaler = GradScaler()

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
        SCN-light:
        - sample weighting theo confidence
        - ranking loss (easy vs hard)
        Returns: total_loss, logs_dict
        """
        # per-sample CE
        ce = F.cross_entropy(logits, labels, reduction='none')  # (B,)

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)
            # stronger focus on hard samples: square the (1 - conf) factor
            weights = (1.0 - conf) ** 2
            weights = weights.clamp(min=self.scn_min_weight)

        # main weighted CE term
        loss = (weights * ce).mean()

        # ranking loss: use percentile split (e.g., 30% hardest) to be robust
        sorted_conf, idx = torch.sort(conf)
        B = logits.size(0)
        # use a smaller percentile split and a minimum of 2 for stability on small batches
        k = max(2, int(0.2 * B))
        hard_idx = idx[:k]
        easy_idx = idx[k:]
        # safe computation in small batches: fallback to zero when empty
        if hard_idx.numel() > 0:
            hard_loss = ce[hard_idx].mean()
        else:
            hard_loss = torch.tensor(0.0, device=self.device)
        if easy_idx.numel() > 0:
            easy_loss = ce[easy_idx].mean()
        else:
            easy_loss = torch.tensor(0.0, device=self.device)
        # margin to enforce separation
        margin = float(getattr(self, 'scn_margin', 0.4))
        # start ranking after SCN warmup (scale with config)
        ranking_start = int(getattr(self, 'scn_warmup_epochs', 0))
        # use >= so that a zero warmup enables ranking immediately
        if getattr(self, '_current_epoch', 0) >= ranking_start:
            ranking_loss = F.relu(easy_loss - hard_loss + margin)
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)

        # combine with alpha scaling
        total_loss = (self.scn_alpha * loss) + (self.scn_rank_lambda * ranking_loss)

        logs = {
            "scn_weight_mean": float(weights.mean().cpu().item()),
            "scn_conf_mean": float(conf.mean().cpu().item()),
            "scn_rank_loss": float(ranking_loss.cpu().item()),
        }
        return total_loss, logs


    # ==========================================================
    # EVENT-DRIVEN / HOOK-BASED MODULAR REFACTORING (Lightning/FastAI style)
    # ==========================================================
    def on_batch_start(self, images, labels):
        """Hook executed at the start of each batch. Handles MixUp logic."""
        mixup_active = bool(getattr(self, '_runtime_use_mixup', False)) and self.model.training
        if mixup_active:
            alpha = float(getattr(self, 'mixup_alpha', 0.2))
            lam = float(np.random.beta(alpha, alpha)) if alpha > 0.0 else 1.0
            perm = torch.randperm(images.size(0), device=images.device)
            images = (lam * images) + ((1.0 - lam) * images[perm])
            labels_a, labels_b = labels, labels[perm]
            return mixup_active, images, labels_a, labels_b, lam
        return False, images, labels, labels, 1.0

    def on_loss_compute(self, logits, labels, labels_a, labels_b, lam, mixup_active, aux_losses):
        """Hook executed to compute total loss including SCN, Auxiliary, and DGS."""
        scn_logs = None
        if mixup_active:
            cls_loss = lam * F.cross_entropy(logits, labels_a) + (1.0 - lam) * F.cross_entropy(logits, labels_b)
        else:
            runtime_use_scn = getattr(self, '_runtime_use_scn', self.use_scn)
            if runtime_use_scn and getattr(self, '_current_epoch', 0) >= getattr(self, 'scn_warmup_epochs', 0):
                try:
                    cls_loss, scn_logs = self._scn_loss(logits, labels)
                except Exception:
                    cls_loss = self._base_criterion(logits, labels)
            else:
                cls_loss = self._base_criterion(logits, labels)
                
        loss = cls_loss
        
        # Aggregate scalar auxiliary losses automatically
        for k, v in aux_losses.items():
            if k not in ["logits_global", "logits_motif"]:
                w = self.config.get('training', {}).get(f'{k}_weight', 0.1)
                loss = loss + float(w) * v
                
        # Dynamic Gate Supervision (DGS)
        l_glob = aux_losses.get("logits_global", None)
        l_mot = aux_losses.get("logits_motif", None)
        if l_glob is not None and l_mot is not None:
            if mixup_active:
                loss_glob = lam * self._base_criterion(l_glob, labels_a) + (1.0 - lam) * self._base_criterion(l_glob, labels_b)
                loss_mot = lam * self._base_criterion(l_mot, labels_a) + (1.0 - lam) * self._base_criterion(l_mot, labels_b)
                loss = loss + 0.3 * loss_glob + 0.3 * loss_mot
            else:
                loss = loss + 0.3 * self.criterion(l_glob, labels)
                loss = loss + 0.3 * self.criterion(l_mot, labels)
                
        return loss, scn_logs

    def on_backward_end(self):
        """Hook executed after backward pass. Handles AMP unscaling and gradient clipping."""
        try:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        except Exception:
            pass


    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        self._latest_scn_logs = None
        _scn_acc = {"scn_weight_mean": [], "scn_conf_mean": [], "scn_rank_loss": []}

        for images, labels, landmarks, statuses in self.train_loader:
            images, labels, landmarks, statuses = images.to(self.device), labels.to(self.device), landmarks.to(self.device), statuses.to(self.device)
            self.optimizer.zero_grad()

            # 1. on_batch_start (MixUp)
            mixup_active, images, labels_a, labels_b, lam = self.on_batch_start(images, labels)

            # 2. Forward pass
            with autocast():
                if hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                    outputs = self.model(images, landmarks=landmarks, statuses=statuses) if mixup_active else self.model(images, targets=labels, landmarks=landmarks, statuses=statuses)
                else:
                    outputs = self.model(images, landmarks=landmarks, statuses=statuses)
                logits = self._extract_logits(outputs)
                aux_losses = self._extract_aux_losses(outputs)

            # 3. on_loss_compute (SCN, Aux, DGS)
            loss, scn_logs = self.on_loss_compute(logits, labels, labels_a, labels_b, lam, mixup_active, aux_losses)
            if scn_logs is not None:
                for k in _scn_acc:
                    _scn_acc[k].append(scn_logs.get(k, 0.0))

            # 4. Backward & on_backward_end
            self.scaler.scale(loss).backward()
            self.on_backward_end()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        if total > 0:
            epoch_loss = running_loss / total
            epoch_acc = corrects.double() / total
        else:
            epoch_loss = 0.0; epoch_acc = torch.tensor(0.0)

        if len(_scn_acc["scn_weight_mean"]) > 0:
            self._latest_scn_logs = {k: float(sum(v)/len(v)) for k, v in _scn_acc.items()}
        else:
            self._latest_scn_logs = None

        return epoch_loss, epoch_acc


    def validate(self):
        self.model.eval()

        running_loss = 0.0
        corrects = 0
        total = 0

        with torch.no_grad():
            for images, labels, landmarks, statuses in self.val_loader:
                images, labels, landmarks, statuses = images.to(self.device), labels.to(self.device), landmarks.to(self.device), statuses.to(self.device)

                # Pass labels to forward for internal loss calculation
                with autocast():
                    if hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                        outputs = self.model(images, targets=labels, landmarks=landmarks, statuses=statuses)
                    else:
                        outputs = self.model(images, landmarks=landmarks, statuses=statuses)
                
                logits = self._extract_logits(outputs)
                cls_loss = self.criterion(logits, labels)
                aux_losses = self._extract_aux_losses(outputs)
                
                loss = cls_loss
                
                # Aggregate scalar auxiliary losses automatically
                for k, v in aux_losses.items():
                    if k not in ["logits_global", "logits_motif"]:
                        w = self.config.get('training', {}).get(f'{k}_weight', 0.1)
                        loss = loss + float(w) * v

                # DYNAMIC GATE SUPERVISION (DGS) for Validation consistency
                l_glob = aux_losses.get("logits_global", None)
                l_mot = aux_losses.get("logits_motif", None)
                if l_glob is not None and l_mot is not None:
                    loss = loss + 0.3 * self.criterion(l_glob, labels)
                    loss = loss + 0.3 * self.criterion(l_mot, labels)

                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(logits, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        return epoch_loss, epoch_acc


    def fit(self):
        """ Fit your model
        Return:
            all_train_loss, all_val_loss
        """
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        if self.use_wandb:
            init_wandb(config=self.config, run_name=self.run_name)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        all_train_loss = []
        all_val_loss = []

        print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.epochs):
            # expose current epoch for runtime gating (SCN warmup etc.)
            self._current_epoch = ep

            # =========================================================
            # BACKBONE FREEZE SCHEDULE (DISABLED FOR RESNET18 CHECKPOINT)
            # =========================================================
            # backbone = getattr(self.model, 'backbone', None) or getattr(self.model, 'resnet', None)
            # if backbone is not None:
            #     main_lr = self.optimizer.param_groups[-1]['lr'] 
            #     if ep == 0:
            #         for param in backbone.parameters():
            #             param.requires_grad = False
            #         print(f"[Phase 1] Epoch {ep+1}: Backbone FROZEN. Only training Motif Graph Head.")
            #     elif ep == 15:
            #         layer4 = getattr(backbone, 'layer4', None)
            #         if layer4 is not None:
            #             for param in layer4.parameters():
            #                 param.requires_grad = True
            #         phase2_lr = main_lr * 0.1 
            #         self.optimizer.param_groups[0]['lr'] = phase2_lr
            #         print(f"[Phase 2] Epoch {ep+1}: Unfreeze backbone.layer4 with lr={phase2_lr:.2e}")
            #     elif ep == 30:
            #         for param in backbone.parameters():
            #             param.requires_grad = True
            #         phase3_lr = main_lr * 0.02 
            #         self.optimizer.param_groups[0]['lr'] = phase3_lr
            #         print(f"[Phase 3] Epoch {ep+1}: Full backbone UNFROZEN with lr={phase3_lr:.2e}")

            progress = ep / self.epochs
            set_progress = getattr(self.model, "set_training_progress", None)
            if callable(set_progress):
                try:
                    set_progress(progress)
                except Exception:
                    pass

            # Áp dụng Lịch trình 3 Giai đoạn (Tuned)
            if progress <= 0.05:
                # Phase 1 (0-5%): SCN OFF, MixUp OFF - Để Motif Head học cách định hướng cơ bản
                self._runtime_use_scn = False
                self._runtime_use_mixup = False
                self._runtime_phase = 1
            elif progress <= 0.25:
                # Phase 2 (5-25%): SCN BẬT NHẸ - Chuẩn bị tinh thần trước khi rã đông
                self._runtime_use_scn = False
                self._runtime_use_mixup = False
                self._runtime_phase = 2
            else:
                # Phase 3 (25-100%): SCN BẬT TOÀN DIỆN - Trấn áp ResNet152
                self._runtime_use_scn = True
                self._runtime_use_mixup = False
                self._runtime_phase = 3


            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            print(
                f"Epoch {ep+1}/{self.epochs} - "
                f"loss: {train_loss:.4f} - accuracy: {train_acc.item():.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc.item():.4f}"
            )
            get_prior = getattr(self.model, "get_current_prior_strength", None)
            if callable(get_prior):
                current_prior = get_prior()


            # wandb log
            if self.use_wandb:
                log_metrics({
                    "Epoch": ep + 1,
                    "Train/Loss": train_loss,
                    "Train/Accuracy": train_acc,
                    "Val/Loss": val_loss,
                    "Val/Accuracy": val_acc,
                    "Learning_Rate": self.optimizer.param_groups[0]['lr']
                }, epoch=ep)
            # log SCN internals if present (use epoch-aggregated self._latest_scn_logs)
            if self.use_wandb and getattr(self, '_latest_scn_logs', None) is not None:
                try:
                    log_metrics(self._latest_scn_logs, epoch=ep)
                except Exception:
                    pass

            # lr scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # SỬA LỖI: Track theo val_loss để nhận diện plateau nhạy bén và mịn màng hơn
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # save checkpoint and early stopping (tracking val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": ep,
                    "val_acc": val_acc.item() if hasattr(val_acc, 'item') else val_acc,
                    "val_loss": val_loss
                }, self.path_save_ckpt)
                print(f"\t--- Save best Accuracy & Update EarlyStopping at ep {ep+1}, val_acc: {val_acc:.4f} ---")
            else:
                patience_counter += 1
                print(f"\t-!- No accuracy improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping triggered at ep={ep+1}")
                    break

        return all_train_loss, all_val_loss



if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    
    print("Test training...")

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 7)
        def forward(self, x):
            return self.fc(x)
        # minimal stubs used by Trainer test
        def get_landmark_outputs(self):
            return None, None
        def get_aux_losses(self):
            return {}

    class DummyDataset(Dataset):
        def __len__(self): return 16
        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0, 7, (1,)).item()

    mock_config = {
        'training': {'epochs': 3, 'patience': 2},
        'path': {'root': '/tmp/'},
        'model': {'name': 'dummy_model'},
        'logging': {'use_wandb': True}
    }

    train_loader = DataLoader(DummyDataset(), batch_size=8)
    val_loader = DataLoader(DummyDataset(), batch_size=8)

    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        scheduler = None
        run_name = "debug_run"
        save_path = "checkpoint.pth"
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            mock_config,
            device,
            run_name,
            save_path,
        )
        print("Fitting...")
        trainer.fit()
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")