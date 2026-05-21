import os
import torch
import numpy as np 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datetime import datetime
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics


class Trainer:
    """Forward -> Compute loss -> zero_grad -> Backward -> Update weights (step)"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self._base_criterion = self.criterion
        
        # Optionally enable label smoothing for CrossEntropy if configured
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

        # Base motif loss weights (from config)
        self.motif_diversity_weight = float(config['training'].get('motif_diversity_weight', 0.05))
        self.motif_consistency_weight = float(config['training'].get('motif_consistency_weight', 0.05))
        self.attn_entropy_weight = float(config['training'].get('attn_entropy_weight', 0.01))
        self.offset_reg_weight = float(config['training'].get('offset_reg_weight', 0.01))
        self.au_contrastive_weight = float(config['training'].get('au_contrastive_weight', 0.03))

        # === SCN (light) ===
        self.use_scn = config['training'].get('use_scn', True)
        self.scn_warmup_epochs = int(config['training'].get('scn_warmup_epochs', 0))
        self.scn_alpha = float(config['training'].get('scn_alpha', 1.0))
        self.scn_rank_lambda = float(config['training'].get('scn_rank_lambda', 0.5))
        self.scn_min_weight = float(config['training'].get('scn_min_weight', 0.2))
        self.scn_margin = float(config['training'].get('scn_margin', 0.6))
        
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

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                images, labels = batch
                return images, labels, None
            if len(batch) == 3:
                images = batch[0]
                if getattr(batch[1], 'ndim', 0) == 3 and getattr(batch[2], 'ndim', 0) == 1:
                    bboxes, labels = batch[1], batch[2]
                elif getattr(batch[1], 'ndim', 0) == 1 and getattr(batch[2], 'ndim', 0) == 3:
                    labels, bboxes = batch[1], batch[2]
                else:
                    labels, bboxes = batch[1], batch[2]
                return images, labels, bboxes
        return batch, None, None

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
        - sample weighting according to confidence
        - ranking loss (easy vs hard)
        """
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
        
        if hard_idx.numel() > 0:
            hard_loss = ce[hard_idx].mean()
        else:
            hard_loss = torch.tensor(0.0, device=self.device)
            
        if easy_idx.numel() > 0:
            easy_loss = ce[easy_idx].mean()
        else:
            easy_loss = torch.tensor(0.0, device=self.device)
            
        margin = float(getattr(self, 'scn_margin', 0.4))
        ranking_start = int(getattr(self, 'scn_warmup_epochs', 0))
        
        if getattr(self, '_current_epoch', 0) >= ranking_start:
            ranking_loss = F.relu(easy_loss - hard_loss + margin)
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
        self._latest_scn_logs = None

        # Fetch scheduled weights for this phase
        w_div = getattr(self, '_runtime_motif_diversity_weight', self.motif_diversity_weight)
        w_consist = getattr(self, '_runtime_motif_consistency_weight', self.motif_consistency_weight)
        w_ent = getattr(self, '_runtime_attn_entropy_weight', self.attn_entropy_weight)
        w_off = getattr(self, '_runtime_offset_reg_weight', self.offset_reg_weight)
        w_contrastive = getattr(self, '_runtime_au_contrastive_weight', self.au_contrastive_weight)

        _scn_acc = {"scn_weight_mean": [], "scn_conf_mean": [], "scn_rank_loss": []}

        for batch in self.train_loader:
            images, labels, bboxes = self._unpack_batch(batch)
            images = images.to(self.device)
            labels = labels.to(self.device)
            if bboxes is not None:
                bboxes = bboxes.to(self.device)
            self.optimizer.zero_grad()

            mixup_active = bool(getattr(self, '_runtime_use_mixup', False)) and self.model.training
            if mixup_active:
                alpha = float(getattr(self, 'mixup_alpha', 0.2))
                if alpha > 0.0:
                    lam = float(np.random.beta(alpha, alpha))
                else:
                    lam = 1.0
                perm = torch.randperm(images.size(0), device=images.device)
                images = (lam * images) + ((1.0 - lam) * images[perm])
                labels_a = labels
                labels_b = labels[perm]

            loss_mode = self.config.get('training', {}).get('loss', 'cross_entropy')

            if bboxes is not None:
                outputs = self.model(images, bboxes)
            elif hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                outputs = self.model(images, targets=labels)
            else:
                outputs = self.model(images)
            logits = self._extract_logits(outputs)

            runtime_use_scn = getattr(self, '_runtime_use_scn', self.use_scn)

            if mixup_active:
                try:
                    cls_loss = lam * F.cross_entropy(logits, labels_a) + (1.0 - lam) * F.cross_entropy(logits, labels_b)
                except Exception:
                    cls_loss = self._base_criterion(logits, labels)
            elif loss_mode == 'semantic_roi_graph' and hasattr(self.model, 'compute_losses'):
                loss_dict = self.model.compute_losses(outputs, labels)
                cls_loss = loss_dict["loss"]
            else:
                if runtime_use_scn and getattr(self, '_current_epoch', 0) >= getattr(self, 'scn_warmup_epochs', 0):
                    try:
                        cls_loss, scn_logs = self._scn_loss(logits, labels)
                        _scn_acc["scn_weight_mean"].append(scn_logs.get("scn_weight_mean", 0.0))
                        _scn_acc["scn_conf_mean"].append(scn_logs.get("scn_conf_mean", 0.0))
                        _scn_acc["scn_rank_loss"].append(scn_logs.get("scn_rank_loss", 0.0))
                    except Exception:
                        cls_loss = self._base_criterion(logits, labels)
                else:
                    cls_loss = self._base_criterion(logits, labels)

            loss = cls_loss

            # Extract and add scheduled motif losses
            aux_losses = self._extract_aux_losses(outputs)
            
            if "motif_diversity" in aux_losses:
                loss = loss + w_div * aux_losses["motif_diversity"]
            if "motif_consistency" in aux_losses:
                loss = loss + w_consist * aux_losses["motif_consistency"]
            if "attn_entropy" in aux_losses:
                loss = loss + w_ent * aux_losses["attn_entropy"]
            if "offset_reg" in aux_losses:
                loss = loss + w_off * aux_losses["offset_reg"]
            if "au_contrastive" in aux_losses:
                loss = loss + w_contrastive * aux_losses["au_contrastive"]

            # Fallback for other unrecognized auxiliary losses
            for k, v in aux_losses.items():
                if k not in ["motif_diversity", "motif_consistency", "attn_entropy", "offset_reg", "au_contrastive"]:
                    w_other = self.config.get('training', {}).get(f'{k}_weight', 0.1)
                    loss = loss + float(w_other) * v

            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            except Exception:
                pass
                
            self.optimizer.step()
            if hasattr(self, 'ema_model'):
                self.ema_model.update_parameters(self.model)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        if total > 0:
            epoch_loss = running_loss / total
            epoch_acc = corrects.double() / total
        else:
            epoch_loss = 0.0
            epoch_acc = torch.tensor(0.0)

        try:
            if len(_scn_acc["scn_weight_mean"]) > 0:
                self._latest_scn_logs = {
                    "scn_weight_mean": float(sum(_scn_acc["scn_weight_mean"]) / len(_scn_acc["scn_weight_mean"])),
                    "scn_conf_mean": float(sum(_scn_acc["scn_conf_mean"]) / len(_scn_acc["scn_conf_mean"])),
                    "scn_rank_loss": float(sum(_scn_acc["scn_rank_loss"]) / len(_scn_acc["scn_rank_loss"])),
                }
            else:
                self._latest_scn_logs = None
        except Exception:
            self._latest_scn_logs = None

        return epoch_loss, epoch_acc

    def validate(self):
        eval_model = getattr(self, 'ema_model', self.model)
        eval_model.eval()

        running_loss = 0.0
        corrects = 0
        total = 0

        # Fetch scheduled weights for validation
        w_div = getattr(self, '_runtime_motif_diversity_weight', self.motif_diversity_weight)
        w_consist = getattr(self, '_runtime_motif_consistency_weight', self.motif_consistency_weight)
        w_ent = getattr(self, '_runtime_attn_entropy_weight', self.attn_entropy_weight)
        w_off = getattr(self, '_runtime_offset_reg_weight', self.offset_reg_weight)
        w_contrastive = getattr(self, '_runtime_au_contrastive_weight', self.au_contrastive_weight)

        with torch.no_grad():
            for batch in self.val_loader:
                images, labels, bboxes = self._unpack_batch(batch)
                images = images.to(self.device)
                labels = labels.to(self.device)
                if bboxes is not None:
                    bboxes = bboxes.to(self.device)

                loss_mode = self.config.get('training', {}).get('loss', 'cross_entropy')

                if bboxes is not None:
                    outputs = eval_model(images, bboxes)
                elif hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                    outputs = eval_model(images, targets=labels)
                else:
                    outputs = eval_model(images)
                
                logits = self._extract_logits(outputs)
                if loss_mode == 'semantic_roi_graph' and hasattr(eval_model, 'compute_losses'):
                    loss_dict = eval_model.compute_losses(outputs, labels)
                    cls_loss = loss_dict["loss"]
                else:
                    cls_loss = self.criterion(logits, labels)
                
                loss = cls_loss
                aux_losses = self._extract_aux_losses(outputs)
                
                if "motif_diversity" in aux_losses:
                    loss = loss + w_div * aux_losses["motif_diversity"]
                if "motif_consistency" in aux_losses:
                    loss = loss + w_consist * aux_losses["motif_consistency"]
                if "attn_entropy" in aux_losses:
                    loss = loss + w_ent * aux_losses["attn_entropy"]
                if "offset_reg" in aux_losses:
                    loss = loss + w_off * aux_losses["offset_reg"]
                if "au_contrastive" in aux_losses:
                    loss = loss + w_contrastive * aux_losses["au_contrastive"]

                for k, v in aux_losses.items():
                    if k not in ["motif_diversity", "motif_consistency", "attn_entropy", "offset_reg", "au_contrastive"]:
                        w_other = self.config.get('training', {}).get(f'{k}_weight', 0.1)
                        loss = loss + float(w_other) * v

                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(logits, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        return epoch_loss, epoch_acc

    def fit(self):
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        if self.use_wandb:
            init_wandb(config=self.config, run_name=self.run_name)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        all_train_loss = []
        all_val_loss = []

        print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

        for ep in range(self.epochs):
            self._current_epoch = ep
            progress = ep / max(self.epochs - 1, 1)
            
            set_progress = getattr(self.model, "set_training_progress", None)
            if callable(set_progress):
                try:
                    set_progress(progress)
                except Exception:
                    pass

            if progress <= 0.7:
                # Phase 2: Mixup off, SCN active, Motif weights at configured values
                self._runtime_motif_diversity_weight = self.motif_diversity_weight
                self._runtime_motif_consistency_weight = self.motif_consistency_weight
                self._runtime_attn_entropy_weight = self.attn_entropy_weight
                self._runtime_offset_reg_weight = self.offset_reg_weight
                self._runtime_use_scn = False
                self._runtime_use_mixup = False
                self._runtime_phase = 2
            else:
                # Phase 3: Fine-tuning. Slightly boost diversity and consistency weights to optimize clusters
                self._runtime_motif_diversity_weight = self.motif_diversity_weight * 1.5
                self._runtime_motif_consistency_weight = self.motif_consistency_weight * 1.5
                self._runtime_attn_entropy_weight = self.attn_entropy_weight
                self._runtime_offset_reg_weight = self.offset_reg_weight
                self._runtime_use_scn = True
                self._runtime_use_mixup = False
                self._runtime_phase = 3

            # Warmup au_contrastive_weight to prevent cold start issues with random spatial attention
            if ep < 5:
                self._runtime_au_contrastive_weight = 0.0
            elif ep < 10:
                self._runtime_au_contrastive_weight = self.au_contrastive_weight * ((ep - 4) / 5.0)
            else:
                self._runtime_au_contrastive_weight = self.au_contrastive_weight

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            print(
                f"Epoch {ep+1}/{self.epochs} - "
                f"loss: {train_loss:.4f} - accuracy: {train_acc.item():.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc.item():.4f}"
            )

            if self.use_wandb:
                log_metrics({
                    "Epoch": ep + 1,
                    "Train/Loss": train_loss,
                    "Train/Accuracy": train_acc,
                    "Val/Loss": val_loss,
                    "Val/Accuracy": val_acc,
                    "Learning_Rate": self.optimizer.param_groups[0]['lr']
                }, epoch=ep)
                if getattr(self, '_latest_scn_logs', None) is not None:
                    try:
                        log_metrics(self._latest_scn_logs, epoch=ep)
                    except Exception:
                        pass

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_state_dict = self.ema_model.module.state_dict() if hasattr(self, 'ema_model') else self.model.state_dict()
                torch.save({
                    "model_state_dict": save_state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": ep,
                    "val_acc": val_acc.item() if hasattr(val_acc, 'item') else val_acc,
                    "val_loss": val_loss
                }, self.path_save_ckpt)
                print(f"\t--- Save best Accuracy at ep {ep+1}, val_acc: {val_acc:.4f}, path: {self.path_save_ckpt} ---")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"\t--- Best Loss updated: {val_loss:.4f} ---")
            else:
                patience_counter += 1
                print(f"\t-!- No loss improvement: {patience_counter}/{self.patience}")
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
        'logging': {'use_wandb': False}
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