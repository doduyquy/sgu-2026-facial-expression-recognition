import os
import torch
import numpy as np 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datetime import datetime
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics
from src.models.semantic_roi_graph_losses import compute_semantic_roi_graph_losses


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
            if len(batch) == 4:
                images = batch[0]
                labels = batch[1]
                bboxes = batch[2]
                semantic_meta = batch[3]
                return images, labels, bboxes, semantic_meta
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
        # Accumulate per-component loss values for epoch-level WandB logging
        self._latest_loss_components: dict = {}

        # Fetch scheduled weights for this phase
        w_div = getattr(self, '_runtime_motif_diversity_weight', self.motif_diversity_weight)
        w_consist = getattr(self, '_runtime_motif_consistency_weight', self.motif_consistency_weight)
        w_ent = getattr(self, '_runtime_attn_entropy_weight', self.attn_entropy_weight)
        w_off = getattr(self, '_runtime_offset_reg_weight', self.offset_reg_weight)
        w_contrastive = getattr(self, '_runtime_au_contrastive_weight', self.au_contrastive_weight)

        _scn_acc = {"scn_weight_mean": [], "scn_conf_mean": [], "scn_rank_loss": []}
        _component_accum: dict = {}

        for batch in self.train_loader:
            unpacked = self._unpack_batch(batch)
            if len(unpacked) == 4:
                images, labels, bboxes, semantic_meta = unpacked
            else:
                images, labels, bboxes = unpacked
                semantic_meta = None
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

            def compute_loss_and_backward(is_first_step=True):
                if bboxes is not None:
                    if isinstance(semantic_meta, dict) and "region_mask" in semantic_meta:
                        region_mask = semantic_meta["region_mask"].to(self.device)
                        region_confidence = semantic_meta.get("region_confidence", None)
                        if region_confidence is not None:
                            region_confidence = region_confidence.to(self.device)
                        outputs = self.model(
                            images,
                            bboxes,
                            region_mask=region_mask,
                            region_confidence=region_confidence,
                        )
                    else:
                        outputs = self.model(images, bboxes)
                elif hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                    outputs = self.model(images, targets=labels)
                else:
                    outputs = self.model(images)
                    
                logits = self._extract_logits(outputs)
                if logits is None:
                    raise ValueError("Model outputs do not contain 'logits'.")

                runtime_use_scn = getattr(self, '_runtime_use_scn', self.use_scn)

                if mixup_active:
                    try:
                        cls_loss = lam * F.cross_entropy(logits, labels_a) + (1.0 - lam) * F.cross_entropy(logits, labels_b)
                    except Exception:
                        cls_loss = self._base_criterion(logits, labels)
                elif loss_mode == 'semantic_roi_graph':
                    # Use the standalone loss function that reads weights from config
                    class_weights = getattr(self._base_criterion, 'weight', None)
                    loss_dict = compute_semantic_roi_graph_losses(self.model, outputs, labels, class_weights=class_weights)
                    cls_loss = loss_dict["loss"]
                    # Accumulate component values for later WandB logging
                    if is_first_step:
                        for _k, _v in loss_dict.items():
                            if _k != "loss" and torch.is_tensor(_v):
                                _component_accum.setdefault(_k, []).append(float(_v.item()))
                        # Log fusion gate / scale stats
                        _gate = outputs.get("structure_gate")
                        if _gate is not None:
                            _component_accum.setdefault("_fusion_gate_mean", []).append(float(_gate.mean().detach().cpu()))
                            _component_accum.setdefault("_fusion_gate_min", []).append(float(_gate.min().detach().cpu()))
                            _component_accum.setdefault("_fusion_gate_max", []).append(float(_gate.max().detach().cpu()))
                        _fs = outputs.get("fusion_scale")
                        if _fs is not None:
                            _component_accum.setdefault("_fusion_scale", []).append(float(_fs.detach().cpu()))
                else:
                    if runtime_use_scn and getattr(self, '_current_epoch', 0) >= getattr(self, 'scn_warmup_epochs', 0):
                        try:
                            cls_loss, scn_logs = self._scn_loss(logits, labels)
                            if is_first_step:
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

                if loss_mode == 'semantic_roi_graph':
                    skip_aux = {"motif_diversity", "motif_consistency", "au_contrastive"}
                else:
                    skip_aux = set()

                if "motif_diversity" in aux_losses and "motif_diversity" not in skip_aux:
                    loss = loss + w_div * aux_losses["motif_diversity"]
                if "motif_consistency" in aux_losses and "motif_consistency" not in skip_aux:
                    loss = loss + w_consist * aux_losses["motif_consistency"]
                if "attn_entropy" in aux_losses and "attn_entropy" not in skip_aux:
                    loss = loss + w_ent * aux_losses["attn_entropy"]
                if "offset_reg" in aux_losses and "offset_reg" not in skip_aux:
                    loss = loss + w_off * aux_losses["offset_reg"]
                if "au_contrastive" in aux_losses and "au_contrastive" not in skip_aux:
                    loss = loss + w_contrastive * aux_losses["au_contrastive"]

                # Fallback for other unrecognized auxiliary losses
                for k, v in aux_losses.items():
                    if k in skip_aux:
                        continue
                    if k not in ["motif_diversity", "motif_consistency", "attn_entropy", "offset_reg", "au_contrastive"]:
                        w_other = self.config.get('training', {}).get(f'{k}_weight', 0.1)
                        loss = loss + float(w_other) * v

                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                except Exception:
                    pass
                    
                return loss, logits

            # --- Execute Forward/Backward ---
            loss, logits = compute_loss_and_backward(is_first_step=True)
            
            from src.training.optimizer import SAM
            if isinstance(self.optimizer, SAM):
                # 1. Update weights with perturbed gradients
                self.optimizer.first_step(zero_grad=True)
                # 2. Re-evaluate loss at perturbed position
                compute_loss_and_backward(is_first_step=False)
                # 3. Un-perturb weights and take the final step
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
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

        # Average the per-component loss values
        self._latest_loss_components = {
            k: float(sum(v) / len(v)) for k, v in _component_accum.items() if len(v) > 0
        }

        return epoch_loss, epoch_acc

    @staticmethod
    def _compute_metrics(all_preds, all_labels, num_classes: int = 7):
        """Return macro-F1, balanced accuracy, per-class recall, per-class F1."""
        try:
            from sklearn.metrics import (
                f1_score, balanced_accuracy_score, recall_score,
            )
            macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
            bal_acc = float(balanced_accuracy_score(all_labels, all_preds))
            per_recall = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes))).tolist()
            per_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes))).tolist()
        except ImportError:
            # Fallback: pure numpy/torch
            all_preds_t = np.array(all_preds)
            all_labels_t = np.array(all_labels)
            per_recall, per_f1 = [], []
            for c in range(num_classes):
                tp = ((all_preds_t == c) & (all_labels_t == c)).sum()
                fp = ((all_preds_t == c) & (all_labels_t != c)).sum()
                fn = ((all_preds_t != c) & (all_labels_t == c)).sum()
                rec = tp / (tp + fn + 1e-8)
                prec = tp / (tp + fp + 1e-8)
                per_recall.append(float(rec))
                per_f1.append(float(2 * prec * rec / (prec + rec + 1e-8)))
            macro_f1 = float(np.mean(per_f1))
            bal_acc = float(np.mean(per_recall))
        return macro_f1, bal_acc, per_recall, per_f1

    def validate(self):
        eval_model = getattr(self, 'ema_model', self.model)
        eval_model.eval()

        running_loss = 0.0
        corrects = 0
        corrects_motif = 0
        corrects_fused = 0
        total = 0
        all_preds, all_preds_motif, all_preds_fused, all_labels = [], [], [], []

        # Fetch scheduled weights for validation
        w_div = getattr(self, '_runtime_motif_diversity_weight', self.motif_diversity_weight)
        w_consist = getattr(self, '_runtime_motif_consistency_weight', self.motif_consistency_weight)
        w_ent = getattr(self, '_runtime_attn_entropy_weight', self.attn_entropy_weight)
        w_off = getattr(self, '_runtime_offset_reg_weight', self.offset_reg_weight)
        w_contrastive = getattr(self, '_runtime_au_contrastive_weight', self.au_contrastive_weight)

        with torch.no_grad():
            for batch in self.val_loader:
                unpacked = self._unpack_batch(batch)
                if len(unpacked) == 4:
                    images, labels, bboxes, semantic_meta = unpacked
                else:
                    images, labels, bboxes = unpacked
                    semantic_meta = None
                images = images.to(self.device)
                labels = labels.to(self.device)
                if bboxes is not None:
                    bboxes = bboxes.to(self.device)

                loss_mode = self.config.get('training', {}).get('loss', 'cross_entropy')

                if bboxes is not None:
                    if isinstance(semantic_meta, dict) and "region_mask" in semantic_meta:
                        region_mask = semantic_meta["region_mask"].to(self.device)
                        region_confidence = semantic_meta.get("region_confidence", None)
                        if region_confidence is not None:
                            region_confidence = region_confidence.to(self.device)
                        outputs = eval_model(
                            images,
                            bboxes,
                            region_mask=region_mask,
                            region_confidence=region_confidence,
                        )
                    else:
                        outputs = eval_model(images, bboxes)
                elif hasattr(self.model, 'forward') and 'targets' in self.model.forward.__code__.co_varnames:
                    outputs = eval_model(images, targets=labels)
                else:
                    outputs = eval_model(images)

                logits = self._extract_logits(outputs)
                if logits is None:
                    raise ValueError("Model outputs do not contain 'logits'. When returning a dict from forward(), include a 'logits' key with classification scores.")
                if loss_mode == 'semantic_roi_graph':
                    class_weights = getattr(self._base_criterion, 'weight', None)
                    loss_dict = compute_semantic_roi_graph_losses(eval_model, outputs, labels, class_weights=class_weights)
                    cls_loss = loss_dict["loss"]
                else:
                    cls_loss = self.criterion(logits, labels)

                loss = cls_loss
                aux_losses = self._extract_aux_losses(outputs)

                if loss_mode == 'semantic_roi_graph':
                    skip_aux = {"motif_diversity", "motif_consistency", "au_contrastive"}
                else:
                    skip_aux = set()

                if "motif_diversity" in aux_losses and "motif_diversity" not in skip_aux:
                    loss = loss + w_div * aux_losses["motif_diversity"]
                if "motif_consistency" in aux_losses and "motif_consistency" not in skip_aux:
                    loss = loss + w_consist * aux_losses["motif_consistency"]
                if "attn_entropy" in aux_losses and "attn_entropy" not in skip_aux:
                    loss = loss + w_ent * aux_losses["attn_entropy"]
                if "offset_reg" in aux_losses and "offset_reg" not in skip_aux:
                    loss = loss + w_off * aux_losses["offset_reg"]
                if "au_contrastive" in aux_losses and "au_contrastive" not in skip_aux:
                    loss = loss + w_contrastive * aux_losses["au_contrastive"]

                for k, v in aux_losses.items():
                    if k in skip_aux:
                        continue
                    if k not in ["motif_diversity", "motif_consistency", "attn_entropy", "offset_reg", "au_contrastive"]:
                        w_other = self.config.get('training', {}).get(f'{k}_weight', 0.1)
                        loss = loss + float(w_other) * v

                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(logits, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                # Branch-level accuracy
                if isinstance(outputs, dict):
                    lm = outputs.get("logits_motif")
                    lf = outputs.get("logits_fused")
                    if lm is not None:
                        pm = torch.max(lm, dim=1)[1]
                        corrects_motif += torch.sum(pm == labels.data)
                        all_preds_motif.extend(pm.cpu().tolist())
                    if lf is not None:
                        pf = torch.max(lf, dim=1)[1]
                        corrects_fused += torch.sum(pf == labels.data)
                        all_preds_fused.extend(pf.cpu().tolist())

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        # Extended metrics
        num_classes = self.config.get('model', {}).get('num_classes', 7)
        macro_f1, bal_acc, per_recall, per_f1 = self._compute_metrics(all_preds, all_labels, num_classes)

        self._latest_val_metrics = {
            "Val/Accuracy_Final": float(epoch_acc),
            "Val/MacroF1_Final": macro_f1,
            "Val/BalancedAccuracy": bal_acc,
        }
        for i, (r, f) in enumerate(zip(per_recall, per_f1)):
            self._latest_val_metrics[f"Val/Recall_Class{i}"] = r
            self._latest_val_metrics[f"Val/F1_Class{i}"] = f

        if all_preds_motif:
            mf1_m, _, _, _ = self._compute_metrics(all_preds_motif, all_labels, num_classes)
            acc_m = float(corrects_motif.double() / total)
            self._latest_val_metrics["Val/Accuracy_Motif"] = acc_m
            self._latest_val_metrics["Val/MacroF1_Motif"] = mf1_m
        if all_preds_fused:
            mf1_f, _, _, _ = self._compute_metrics(all_preds_fused, all_labels, num_classes)
            acc_f = float(corrects_fused.double() / total)
            self._latest_val_metrics["Val/Accuracy_Fused"] = acc_f
            self._latest_val_metrics["Val/MacroF1_Fused"] = mf1_f

        return epoch_loss, epoch_acc

    def fit(self):
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        if self.use_wandb:
            init_wandb(config=self.config, run_name=self.run_name)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_selection_score = -float("inf")
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
                wandb_metrics = {
                    "Epoch": ep + 1,
                    "Train/Loss": train_loss,
                    "Train/Accuracy": train_acc,
                    "Val/Loss": val_loss,
                    "Val/Accuracy": val_acc,
                    "Learning_Rate": self.optimizer.param_groups[0]['lr']
                }
                # Branch-level & extended val metrics
                for _mk, _mv in getattr(self, '_latest_val_metrics', {}).items():
                    wandb_metrics[_mk] = _mv
                # Per-component loss averages
                loss_component_map = {
                    "loss_ce": "Loss/CE",
                    "loss_micro_motif_diversity": "Loss/MicroDiversity",
                    "loss_macro_motif_diversity": "Loss/MacroDiversity",
                    "loss_contrastive": "Loss/RegionContrastive",
                    "loss_semantic_consistency": "Loss/SemanticConsistency",
                    "loss_compositional_program_consistency": "Loss/CompositionalProgram",
                    "loss_program_diversity": "Loss/ProgramDiversity",
                    "loss_fused_aux_ce": "Loss/FusedAuxCE",
                    "loss_semantic_disentanglement": "Loss/Disentanglement",
                    "loss_region_coordination": "Loss/RegionCoordination",
                    "loss_topology_alignment": "Loss/TopologyAlignment",
                    "loss_program_sparsity": "Loss/ProgramSparsity",
                }
                for _lk, _lname in loss_component_map.items():
                    if _lk in getattr(self, '_latest_loss_components', {}):
                        wandb_metrics[_lname] = self._latest_loss_components[_lk]
                # Fusion gate / scale stats
                _fusion_map = {
                    "_fusion_gate_mean": "Fusion/GateMean",
                    "_fusion_gate_min": "Fusion/GateMin",
                    "_fusion_gate_max": "Fusion/GateMax",
                    "_fusion_scale": "Fusion/Scale",
                }
                for _fk, _fn in _fusion_map.items():
                    if _fk in getattr(self, '_latest_loss_components', {}):
                        wandb_metrics[_fn] = self._latest_loss_components[_fk]
                log_metrics(wandb_metrics, epoch=ep)
                if getattr(self, '_latest_scn_logs', None) is not None:
                    try:
                        log_metrics(self._latest_scn_logs, epoch=ep)
                    except Exception:
                        pass

            # Calculate a selection score that balances accuracy and macro F1
            # to prevent the model from ignoring difficult minority classes
            val_macro_f1 = getattr(self, '_latest_val_metrics', {}).get("Val/MacroF1_Final", 0.0)
            selection_score = 0.5 * (val_acc.item() if hasattr(val_acc, 'item') else float(val_acc)) + 0.5 * val_macro_f1

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_val_acc = val_acc
                patience_counter = 0
                save_state_dict = self.ema_model.module.state_dict() if hasattr(self, 'ema_model') else self.model.state_dict()
                torch.save({
                    "model_state_dict": save_state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": ep,
                    "val_acc": val_acc.item() if hasattr(val_acc, 'item') else val_acc,
                    "selection_score": selection_score,
                    "macro_f1": getattr(self, '_latest_val_metrics', {}).get("Val/MacroF1_Final", None),
                    "balanced_accuracy": getattr(self, '_latest_val_metrics', {}).get("Val/BalancedAccuracy", None),
                    "val_loss": val_loss
                }, self.path_save_ckpt)
                print(
                    f"\t--- Save best hybrid score at ep {ep+1}, "
                    f"score: {selection_score:.4f}, "
                    f"val_acc: {val_acc:.4f}, "
                    f"macro_f1: {getattr(self, '_latest_val_metrics', {}).get('Val/MacroF1_Final', 0.0):.4f}, "
                    f"path: {self.path_save_ckpt} ---"
                )
            else:
                patience_counter += 1
                print(f"\t-!- No score improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping triggered at ep={ep+1}")
                    break

            # Log val_loss for monitoring (no longer used for early stopping)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\t    [info] Best val_loss updated: {val_loss:.4f}")

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