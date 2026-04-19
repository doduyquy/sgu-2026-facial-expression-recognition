import os
import torch
import numpy as np 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datetime import datetime
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
        # Tuned defaults to avoid over-constraint (Sunset suggestions)
        self.landmark_diversity_lambda = config['training'].get('landmark_diversity_lambda', 0.25)
        # keep entropy off by default for low-res FER unless explicitly enabled
        self.landmark_entropy_lambda = config['training'].get('landmark_entropy_lambda', 0.0)
        # keep edge_align disabled by default
        self.landmark_edge_align_lambda = config['training'].get('landmark_edge_align_lambda', 0.0)
        # per-keypoint edge consistency - disabled by default for SOTA simplicity
        self.landmark_edge_consistency_lambda = config['training'].get('landmark_edge_consistency_lambda', 0.0)
        # disable heavy regularizers by default (keep code but no loss contribution)
        self.landmark_edge_conv_reg_lambda = config['training'].get('landmark_edge_conv_reg_lambda', 0.0)
        self.landmark_edge_tv_lambda = config['training'].get('landmark_edge_tv_lambda', 0.0)
        # augment consistency disabled by default (expensive / can harm alignment)
        self.landmark_augment_consistency_lambda = config['training'].get('landmark_augment_consistency_lambda', 0.0)
        # probability to run augment-consistency per batch (to save compute). Disabled by default.
        self.landmark_augment_consistency_prob = config['training'].get('landmark_augment_consistency_prob', 0.0)
        # Target entropy for attention maps; regularize toward this value (abs diff)
        self.landmark_target_entropy = config['training'].get('landmark_target_entropy', 2.0)
        # auxiliary classification head weight for landmark features (lighter default)
        self.landmark_aux_cls_lambda = config['training'].get('landmark_aux_cls_lambda', 0.05)
        # optional positional supervision (upper/lower face guidance) - off by default
        self.landmark_pos_sup_lambda = config['training'].get('landmark_pos_sup_lambda', 0.0)
        # heatmap overlap penalty default
        self.landmark_overlap_lambda = config['training'].get('landmark_overlap_lambda', 0.05)
        # auxiliary logits consistency (KL) weight: disabled by default (can destabilize)
        self.landmark_aux_consistency_lambda = config['training'].get('landmark_aux_consistency_lambda', 0.0)
        # focal loss removed to avoid conflict with SCN; use base criterion only
        # === SCN (light) ===
        self.use_scn = config['training'].get('use_scn', True)
        # default warmup: disabled by default, SCN controlled by phase schedule
        self.scn_warmup_epochs = int(config['training'].get('scn_warmup_epochs', 0))
        self.scn_alpha = float(config['training'].get('scn_alpha', 1.0))
        # ranking influence tuned for FER (raise to emphasize hard/easy separation)
        self.scn_rank_lambda = float(config['training'].get('scn_rank_lambda', 0.2))
        self.scn_min_weight = float(config['training'].get('scn_min_weight', 0.2))
        # margin for ranking loss
        self.scn_margin = float(config['training'].get('scn_margin', 0.2))
        # runtime flags (set by fit staging)
        self._runtime_use_scn = None
        # mixup defaults
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
            # normalize confidence (relative within batch) then squash (unused)
            # conf_norm = (conf - conf.mean()) / (conf.std() + 1e-6)
            # softer weighting: low-conf (hard) -> higher weight, but not overly aggressive
            weights = (1.0 - conf)
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
        margin = float(getattr(self, 'scn_margin', 0.1))
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


    def train_one_epoch(self):
        self.model.train()

        running_loss = 0.0
        corrects = 0
        total = 0
        # reset latest scn logs for this epoch
        self._latest_scn_logs = None

        # accumulator for scn metrics across batches
        _scn_acc = {"scn_weight_mean": [], "scn_conf_mean": [], "scn_rank_loss": []}

        # runtime lambdas (may be set by fit() for staged schedule)
        div_lambda = getattr(self, '_runtime_diversity_lambda', self.landmark_diversity_lambda)
        # entropy and overlap lambdas (used to shape heatmaps)
        entropy_lambda = getattr(self, '_runtime_entropy_lambda', self.landmark_entropy_lambda)
        overlap_lambda = getattr(self, '_runtime_overlap_lambda', self.landmark_overlap_lambda)
        edge_consistency_lambda = getattr(self, '_runtime_edge_consistency_lambda', self.landmark_edge_consistency_lambda)
        # augment consistency intentionally disabled to avoid destabilizing landmarks on small images
        augment_lambda = 0.0
        aux_cls_lambda = getattr(self, '_runtime_aux_cls_lambda', self.landmark_aux_cls_lambda)
        pos_sup_lambda = getattr(self, '_runtime_pos_sup_lambda', self.landmark_pos_sup_lambda)
        # convert lambdas to tensors to avoid dtype/interop issues when combining with torch tensors
        div_lambda_t = torch.tensor(float(div_lambda), device=self.device)
        entropy_lambda_t = torch.tensor(float(entropy_lambda), device=self.device)
        overlap_lambda_t = torch.tensor(float(overlap_lambda), device=self.device)
        edge_consistency_lambda_t = torch.tensor(float(edge_consistency_lambda), device=self.device)
        augment_lambda_t = torch.tensor(float(augment_lambda), device=self.device)
        aux_cls_lambda_t = torch.tensor(float(aux_cls_lambda), device=self.device)

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            try:
                # debug: check label range per user's request
                print("labels range:", labels.min().item(), labels.max().item())
            except Exception:
                pass

            self.optimizer.zero_grad()

            # MixUp: disabled by default in FER pipeline (SCN preferred)
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

            outputs = self.model(images)
            logits = self._extract_logits(outputs)

            # batch confidence used to scale landmark diversity: low-confidence batches
            # should emphasize landmark regularizers more (helps hard samples)
            try:
                probs_batch = F.softmax(logits, dim=1)
                conf_batch = probs_batch.gather(1, labels.unsqueeze(1)).squeeze(1)
                conf_batch_mean = conf_batch.mean()
            except Exception:
                conf_batch_mean = torch.tensor(0.0, device=self.device)

            # determine effective runtime flag for SCN (set by fit phases if present)
            runtime_use_scn = getattr(self, '_runtime_use_scn', self.use_scn)

            # If mixup is active, compute mixup-style CE and skip SCN ranking (SCN needs hard labels)
            if mixup_active:
                try:
                    cls_loss = lam * F.cross_entropy(logits, labels_a) + (1.0 - lam) * F.cross_entropy(logits, labels_b)
                    scn_logs = None
                except Exception:
                    cls_loss = self._base_criterion(logits, labels)
                    scn_logs = None
            else:
                # apply SCN-light after warmup epochs if enabled by runtime flag
                if runtime_use_scn and getattr(self, '_current_epoch', 0) >= getattr(self, 'scn_warmup_epochs', 0):
                    try:
                        cls_loss, scn_logs = self._scn_loss(logits, labels)
                        # accumulate scn logs for epoch-level summary
                        try:
                            _scn_acc["scn_weight_mean"].append(scn_logs.get("scn_weight_mean", 0.0))
                            _scn_acc["scn_conf_mean"].append(scn_logs.get("scn_conf_mean", 0.0))
                            _scn_acc["scn_rank_loss"].append(scn_logs.get("scn_rank_loss", 0.0))
                        except Exception:
                            pass
                    except Exception:
                        # fallback to base criterion
                        cls_loss = self._base_criterion(logits, labels)
                else:
                    # use base criterion when SCN not active
                    cls_loss = self._base_criterion(logits, labels)
            aux_losses = self._extract_aux_losses(outputs)

            # (no target) use raw entropy directly for both train and val

            div_loss = aux_losses.get("landmark_diversity", torch.tensor(0.0, device=self.device))
            try:
                # scale all landmark auxiliary losses by batch confidence (detached)
                # scale = (1 - conf); multiply div/overlap/entropy to emphasize hard batches
                scale = (1.0 - conf_batch_mean).detach()
                try:
                    scale = torch.clamp(scale, 0.5, 1.5)
                except Exception:
                    scale = torch.tensor(max(0.5, min(1.5, float(scale))), device=self.device)
                div_loss = div_loss * scale
                overlap_loss = overlap_loss * scale
                entropy_loss = entropy_loss * scale
            except Exception:
                pass
            entropy_loss = aux_losses.get(
                "landmark_entropy",
                aux_losses.get("landmark_sparsity", torch.tensor(0.0, device=self.device)),
            )
            overlap_loss = aux_losses.get("landmark_overlap", torch.tensor(0.0, device=self.device))
            # (entropy regularization removed) keep raw value if needed elsewhere
            heatmaps_now, _ = self.model.get_landmark_outputs()
            if heatmaps_now is not None:
                try:
                    _, _, H_att, W_att = heatmaps_now.shape
                    denom = float(np.log(max(2, H_att * W_att)))
                    if denom <= 0:
                        denom = 1e-6
                except Exception:
                    denom = 1.0
            else:
                denom = 1.0
            edge_align_loss = aux_losses.get("landmark_edge_align", torch.tensor(0.0, device=self.device))
            edge_consistency_loss = aux_losses.get("landmark_edge_consistency", torch.tensor(0.0, device=self.device))
            pos_sup_loss = aux_losses.get("landmark_pos_supervision", torch.tensor(0.0, device=self.device))
            edge_conv_reg = aux_losses.get("landmark_edge_conv_reg", torch.tensor(0.0, device=self.device))
            edge_tv = aux_losses.get("landmark_edge_tv", torch.tensor(0.0, device=self.device))
            # Compose base loss (classification + landmark auxes) using runtime lambdas
            # conv/TV regularizers are intentionally excluded from the main loss to avoid over-constraint
            loss = (
                cls_loss
                + (div_lambda_t * div_loss)
                + (edge_consistency_lambda_t * edge_consistency_loss)
                + (torch.tensor(float(pos_sup_lambda), device=self.device) * pos_sup_loss)
            )
            # include overlap and entropy aux signals (light-weight) to shape heatmaps
            try:
                if overlap_lambda_t.item() > 0.0:
                    loss = loss + (overlap_lambda_t * overlap_loss)
                if entropy_lambda_t.item() > 0.0:
                    loss = loss + (entropy_lambda_t * entropy_loss)
            except Exception:
                pass

            # Auxiliary classification on landmark features (encourage feat_k to be useful)
            aux_logits_getter = getattr(self.model, 'get_landmark_aux_logits', None)
            if callable(aux_logits_getter):
                aux_logits = aux_logits_getter()
            else:
                aux_logits = None
            if aux_logits is not None:
                try:
                    if aux_cls_lambda_t.item() > 0.0:
                        aux_cls_loss = F.cross_entropy(aux_logits, labels)
                        loss = loss + (aux_cls_lambda_t * aux_cls_loss)
                    # KL consistency: make aux logits follow main logits' decision
                    aux_consistency_lambda = getattr(self, '_runtime_aux_consistency_lambda', self.landmark_aux_consistency_lambda)
                    aux_consistency_lambda_t = torch.tensor(float(aux_consistency_lambda), device=self.device)
                    if aux_consistency_lambda_t.item() > 0.0:
                        # safer: guide main prediction with aux (aux -> main)
                        p_main = F.softmax(logits.detach(), dim=1)
                        kl = F.kl_div(F.log_softmax(aux_logits, dim=1), p_main, reduction='batchmean')
                        loss = loss + (aux_consistency_lambda_t * kl)
                except Exception:
                    pass

            # Augment-consistency: pred(Aug(x)) ≈ Aug(pred(x)) using heatmaps
            # probabilistically run augment-consistency to save compute (and only when enabled)
            if augment_lambda > 0.0 and getattr(self.model, 'use_learned_landmark_branch', False) and (np.random.rand() < getattr(self, 'landmark_augment_consistency_prob', 0.3)):
                try:
                    # get latest landmark heatmaps from original forward
                    heatmaps_orig, coords_orig = self.model.get_landmark_outputs()
                    if heatmaps_orig is not None:
                        # build an augmented batch (same random params for whole batch)
                        bsz, c, H, W = heatmaps_orig.shape
                        # sample milder random affine params to avoid heavy misalignment on small images
                        angle = float(np.random.uniform(-5, 5))
                        max_tx = max(1, int(0.05 * W))
                        max_ty = max(1, int(0.05 * H))
                        translate = (int(np.random.randint(-max_tx, max_tx + 1)), int(np.random.randint(-max_ty, max_ty + 1)))
                        scale = 1.0
                        shear = 0.0

                        # apply same transform to input images
                        images_aug = torch.stack([TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, fill=0) for img in images])

                        # forward pass on augmented images without updating grads
                        # use eval() to keep BN/dropout behavior stable for consistency signal
                        was_training = self.model.training
                        self.model.eval()
                        with torch.no_grad():
                            _ = self.model(images_aug)
                        heatmaps_aug, coords_aug = self.model.get_landmark_outputs()
                        if was_training:
                            # restore train mode if we started in train
                            self.model.train()

                        if heatmaps_aug is not None:
                            # transform original heatmaps (detach to use as pseudo-target)
                            heatmaps_orig_det = heatmaps_orig.detach()
                            transformed = []
                            for i in range(heatmaps_orig_det.size(0)):
                                # heatmaps_orig_det[i]: (K, H, W) -> apply TF.affine per-channel
                                chs = []
                                for k in range(heatmaps_orig_det.size(1)):
                                    hm = heatmaps_orig_det[i, k:k+1]
                                    hm_t = TF.affine(hm, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
                                    chs.append(hm_t)
                                transformed.append(torch.cat(chs, dim=0))
                            transformed = torch.stack(transformed, dim=0).to(heatmaps_aug.dtype)

                            augment_consistency_loss = F.l1_loss(heatmaps_aug, transformed, reduction='mean')
                            loss = loss + (augment_lambda * augment_consistency_loss)
                except Exception:
                    # if any issue with augment or TF, skip augment consistency for this batch
                    pass
            loss.backward()
            try:
                # gradient clipping to stabilize training when combining SCN and landmark auxes
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            except Exception:
                pass
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        # finalize SCN logs (mean across batches) if any
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
        self.model.eval()

        running_loss = 0.0
        corrects = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                logits = self._extract_logits(outputs)
                cls_loss = self.criterion(logits, labels)
                aux_losses = self._extract_aux_losses(outputs)
                div_loss = aux_losses.get("landmark_diversity", torch.tensor(0.0, device=self.device))
                # entropy auxiliary is present but not used as an explicit regularizer
                entropy_loss = aux_losses.get(
                    "landmark_entropy",
                    aux_losses.get("landmark_sparsity", torch.tensor(0.0, device=self.device)),
                )
                overlap_loss = aux_losses.get("landmark_overlap", torch.tensor(0.0, device=self.device))
                edge_align_loss = aux_losses.get("landmark_edge_align", torch.tensor(0.0, device=self.device))
                edge_consistency_loss = aux_losses.get("landmark_edge_consistency", torch.tensor(0.0, device=self.device))
                edge_conv_reg = aux_losses.get("landmark_edge_conv_reg", torch.tensor(0.0, device=self.device))
                edge_tv = aux_losses.get("landmark_edge_tv", torch.tensor(0.0, device=self.device))
                # Use runtime lambdas if scheduled by fit(), otherwise fall back to configured defaults
                div_lambda = getattr(self, '_runtime_diversity_lambda', self.landmark_diversity_lambda)
                edge_consistency_lambda = getattr(self, '_runtime_edge_consistency_lambda', self.landmark_edge_consistency_lambda)
                # convert to tensors to avoid type-mixing errors
                div_lambda_t = torch.tensor(float(div_lambda), device=self.device)
                edge_consistency_lambda_t = torch.tensor(float(edge_consistency_lambda), device=self.device)
                entropy_lambda_t = torch.tensor(float(getattr(self, '_runtime_entropy_lambda', self.landmark_entropy_lambda)), device=self.device)
                overlap_lambda_t = torch.tensor(float(getattr(self, '_runtime_overlap_lambda', self.landmark_overlap_lambda)), device=self.device)
                loss = (
                    cls_loss
                    + (div_lambda_t * div_loss)
                    + (edge_consistency_lambda_t * edge_consistency_loss)
                )
                try:
                    if overlap_lambda_t.item() > 0.0:
                        loss = loss + (overlap_lambda_t * overlap_loss)
                    if entropy_lambda_t.item() > 0.0:
                        loss = loss + (entropy_lambda_t * entropy_loss)
                except Exception:
                    pass
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
        patience_counter = 0
        all_train_loss = []
        all_val_loss = []

        print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.epochs):
            # expose current epoch for runtime gating (SCN warmup etc.)
            self._current_epoch = ep
            progress = ep / max(self.epochs - 1, 1)
            set_progress = getattr(self.model, "set_training_progress", None)
            if callable(set_progress):
                try:
                    set_progress(progress)
                except Exception:
                    pass

            # apply 3-phase staged lambda schedule tuned for noisy FER datasets
            # Phase 1: very early (0-20%): SCN OFF, MixUp OFF
            # Phase 2: (20-70%): SCN ON, stronger landmark signals
            # Phase 3: (70-100%): heavy refinement for landmark branch
            if progress <= 0.2:
                # Phase 1 (0-20%): conservative — no MixUp, SCN off
                self._runtime_diversity_lambda = 0.0
                self._runtime_entropy_lambda = 0.0
                self._runtime_overlap_lambda = 0.0
                self._runtime_augment_lambda = 0.0
                self._runtime_edge_consistency_lambda = 0.0
                self._runtime_aux_cls_lambda = 0.0
                self._runtime_aux_consistency_lambda = 0.0
                self._runtime_use_scn = False
                self._runtime_use_mixup = False
                self._runtime_phase = 1
            elif progress <= 0.7:
                # Phase 2 (20-70%): enable SCN and stronger landmark auxiliaries
                self._runtime_diversity_lambda = 0.18
                self._runtime_entropy_lambda = 0.004
                self._runtime_overlap_lambda = 0.07
                self._runtime_augment_lambda = 0.0
                self._runtime_edge_consistency_lambda = 0.0
                self._runtime_aux_cls_lambda = 0.05
                self._runtime_aux_consistency_lambda = 0.0
                self._runtime_use_scn = True
                self._runtime_use_mixup = False
                self._runtime_phase = 2
            else:
                # Phase 3 (70-100%): strong refinement — increase landmark lambdas
                self._runtime_diversity_lambda = 0.30
                self._runtime_entropy_lambda = 0.008
                self._runtime_overlap_lambda = 0.10
                self._runtime_augment_lambda = 0.0
                self._runtime_edge_consistency_lambda = 0.0
                self._runtime_aux_cls_lambda = 0.05
                self._runtime_aux_consistency_lambda = 0.0
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
                if current_prior is not None:
                    print(f"\tlandmark_prior_strength(now): {current_prior:.4f}")

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
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": ep
                }, self.path_save_ckpt)
                print(f"\t--- Save best at ep {ep+1}, val_loss: {val_loss:.4f}, path: {self.path_save_ckpt} ---")

            else:
                patience_counter += 1
                print(f"\t-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping at ep={ep+1}")
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