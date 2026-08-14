import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import device
from torch.amp import GradScaler, autocast
import os
import numpy as np 
from datetime import datetime
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics
from src.training.losses import confidence_soft_target_loss, inception_loss
from src.training.occlusion import RegionOcclusionGenerator
from src.training.optimizer import build_scheduler, build_optimizer
from .sam import SAM

class Trainer:
    """Forward -> Compute loss -> zero_grad -> Backward -> Update weights (step)"""
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        device,
        run_name,
        save_dir,
        start_epoch=0,
        best_score=None,
        best_val_loss=None,
        best_val_acc=None,
        patience_counter=0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 20)
        self.start_epoch = int(start_epoch)
        self.resume_best_score = best_score
        self.resume_best_val_loss = best_val_loss
        self.resume_best_val_acc = best_val_acc
        self.resume_patience_counter = int(patience_counter)
        self.model_name = config['model'].get('name', 'simple_cnn')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.config = config
        self.path_save_ckpt = save_dir
        self.monitor = config['training'].get('monitor', 'val_loss')
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.is_main_process = self.rank == 0
        self.use_amp = (
            config.get('training', {}).get('use_amp', False)
            and self.device.type == 'cuda'
            and not isinstance(self.optimizer, SAM)
        )
        self.grad_scaler = GradScaler(self.device.type, enabled=self.use_amp)
        train_cfg = config.get('training', {})
        self.grad_clip_norm = train_cfg.get('grad_clip_norm')
        self.skip_nonfinite_batches = train_cfg.get('skip_nonfinite_batches', True)
        self.skipped_nonfinite_batches = 0
        soft_target_cfg = config.get('training', {}).get('confidence_soft_targets', {})
        self.use_confidence_soft_targets = soft_target_cfg.get('enabled', False)
        self.soft_target_max_mix = soft_target_cfg.get('max_mix', 0.2)
        self.soft_target_min_confidence = soft_target_cfg.get('min_confidence', 0.55)
        self.soft_target_confidence_power = soft_target_cfg.get('confidence_power', 1.0)
        self.soft_target_label_smoothing = config.get('training', {}).get('label_smoothing', 0.0)
        self.soft_target_class_weights = getattr(self.criterion, 'weight', None)
        occlusion_cfg = train_cfg.get('occlusion_consistency', {})
        self.use_occlusion_consistency = occlusion_cfg.get('enabled', False)
        self.occlusion_start_epoch = max(int(occlusion_cfg.get('start_epoch', 1)), 1)
        self.occlusion_full_weight_epoch = max(
            int(occlusion_cfg.get('full_weight_epoch', self.occlusion_start_epoch)),
            self.occlusion_start_epoch,
        )
        self.occlusion_masked_ce_weight = float(occlusion_cfg.get('masked_ce_weight', 0.5))
        self.occlusion_consistency_weight = float(occlusion_cfg.get('consistency_weight', 0.3))
        self.occlusion_temperature = float(occlusion_cfg.get('temperature', 2.0))
        self.occlusion_generator = (
            RegionOcclusionGenerator(occlusion_cfg)
            if self.use_occlusion_consistency
            else None
        )
        if self.monitor not in ['val_loss', 'val_accuracy']:
            raise ValueError("training.monitor must be either 'val_loss' or 'val_accuracy'")
        if self.occlusion_masked_ce_weight < 0.0:
            raise ValueError(
                "training.occlusion_consistency.masked_ce_weight must be >= 0."
            )
        if self.occlusion_consistency_weight < 0.0:
            raise ValueError(
                "training.occlusion_consistency.consistency_weight must be >= 0."
            )
        if self.occlusion_temperature <= 0.0:
            raise ValueError("training.occlusion_consistency.temperature must be > 0.")
        if self.use_confidence_soft_targets:
            if config.get('training', {}).get('loss', 'cross_entropy') != 'cross_entropy':
                raise ValueError(
                    "training.confidence_soft_targets currently supports "
                    "training.loss='cross_entropy' only."
                )
            if self.is_main_process:
                print(
                    "[Trainer] Confidence soft targets enabled: "
                    f"max_mix={self.soft_target_max_mix}, "
                    f"min_confidence={self.soft_target_min_confidence}, "
                    f"confidence_power={self.soft_target_confidence_power}."
                )
        if self.grad_clip_norm is not None and self.is_main_process:
            print(f"[Trainer] Gradient clipping enabled: max_norm={self.grad_clip_norm}.")
        if self.skip_nonfinite_batches and self.is_main_process:
            print("[Trainer] Non-finite batches will be skipped before optimizer updates.")
        if self.use_occlusion_consistency and self.is_main_process:
            print(
                "[Trainer] Region occlusion consistency enabled: "
                f"start_epoch={self.occlusion_start_epoch}, "
                f"full_weight_epoch={self.occlusion_full_weight_epoch}, "
                f"masked_ce_weight={self.occlusion_masked_ce_weight}, "
                f"consistency_weight={self.occlusion_consistency_weight}, "
                f"temperature={self.occlusion_temperature}."
            )
        if (
            config.get('training', {}).get('use_amp', False)
            and self.device.type == 'cuda'
            and isinstance(self.optimizer, SAM)
            and self.is_main_process
        ):
            print("[Trainer] AMP requested, but disabled for SAM to keep its two-step gradient flow correct.")

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _current_lrs(self):
        return [float(group.get('lr', 0.0)) for group in self.optimizer.param_groups]

    @staticmethod
    def _lr_metric_dict(prefix, lrs):
        metrics = {}
        for idx, lr in enumerate(lrs):
            metrics[f"{prefix}/Group_{idx}"] = lr
        if lrs:
            metrics[f"{prefix}/Head"] = lrs[0]
            metrics[f"{prefix}/Min"] = min(lrs)
            metrics[f"{prefix}/Max"] = max(lrs)
            metrics[f"{prefix}/Mean"] = sum(lrs) / len(lrs)
        if len(lrs) > 1:
            metrics[f"{prefix}/Visual_Extractor"] = lrs[1]
        return metrics

    def _logit_fusion_metric_dict(self):
        base_model = self._unwrap_model()
        if not getattr(base_model, "learnable_logit_fusion", False):
            return {}
        if not hasattr(base_model, "current_cnn_logit_weight"):
            return {}

        cnn_weight = float(base_model.current_cnn_logit_weight())
        region_weight = float(base_model.current_region_logit_weight())
        return {
            "LogitFusion/CNN_Weight": cnn_weight,
            "LogitFusion/Region_Weight": region_weight,
        }

    def _classification_loss(self, logits, labels):
        if not self.use_confidence_soft_targets:
            return self.criterion(logits, labels)

        return confidence_soft_target_loss(
            logits,
            labels,
            max_mix=self.soft_target_max_mix,
            min_confidence=self.soft_target_min_confidence,
            confidence_power=self.soft_target_confidence_power,
            label_smoothing=self.soft_target_label_smoothing,
            class_weights=self.soft_target_class_weights,
        )

    def _prior_alignment_loss(self, region_importance, labels):
        prior_cfg = self.config.get('model', {}).get('emotion_region_prior', {})
        loss_type = prior_cfg.get('alignment_loss', 'kl').lower()
        base_model = self._unwrap_model()
        prior_matrix = getattr(base_model, 'emotion_region_prior', None)
        if prior_matrix is None:
            return region_importance.sum() * 0.0

        target = prior_matrix.to(
            device=region_importance.device,
            dtype=region_importance.dtype,
        )[labels]
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        region_importance = region_importance / region_importance.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

        if loss_type == 'mse':
            return F.mse_loss(region_importance, target)
        if loss_type == 'kl':
            return F.kl_div(
                region_importance.clamp_min(1e-6).log(),
                target,
                reduction='batchmean',
            )
        raise ValueError("model.emotion_region_prior.alignment_loss must be 'kl' or 'mse'")

    def _forward_model(self, images, region_masks=None):
        if region_masks is not None:
            return self.model(images, region_masks=region_masks)
        return self.model(images)

    def _extract_logits(self, outputs):
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _split_batch_outputs(self, outputs, batch_size):
        if not isinstance(outputs, tuple):
            return outputs[:batch_size], outputs[batch_size:]

        clean_items = []
        masked_items = []
        for item in outputs:
            if (
                isinstance(item, torch.Tensor)
                and item.dim() > 0
                and item.size(0) == batch_size * 2
            ):
                clean_items.append(item[:batch_size])
                masked_items.append(item[batch_size:])
            else:
                clean_items.append(item)
                masked_items.append(item)

        return tuple(clean_items), tuple(masked_items)

    def _supervised_loss_from_outputs(self, outputs, labels):
        def aux_classification_weight():
            model_cfg = self.config.get('model', {})
            if 'cnn_aux_loss_weight' in model_cfg:
                return float(model_cfg.get('cnn_aux_loss_weight', 0.0))
            if 'coarse_aux_loss_weight' in model_cfg:
                return float(model_cfg.get('coarse_aux_loss_weight', 0.0))
            prior_cfg = model_cfg.get('emotion_region_prior', {})
            return float(prior_cfg.get('coarse_aux_loss_weight', 0.0))

        if isinstance(outputs, tuple):
            if (
                len(outputs) == 4
                and isinstance(outputs[1], torch.Tensor)
                and outputs[1].dim() == 0
                and isinstance(outputs[2], torch.Tensor)
                and isinstance(outputs[3], torch.Tensor)
            ):
                logits, ortho_loss, coarse_logits, region_importance = outputs
                main_loss = self._classification_loss(logits, labels)
                ortho_weight = self.config.get('model', {}).get('ortho_loss_weight', 0.1)
                prior_cfg = self.config.get('model', {}).get('emotion_region_prior', {})
                coarse_weight = aux_classification_weight()
                prior_weight = float(prior_cfg.get('prior_alignment_loss_weight', 0.0))
                coarse_aux_loss = self._classification_loss(coarse_logits, labels)
                prior_alignment_loss = self._prior_alignment_loss(region_importance, labels)
                loss = main_loss + ortho_weight * ortho_loss
                if coarse_weight > 0.0:
                    loss = loss + coarse_weight * coarse_aux_loss
                if prior_weight > 0.0:
                    loss = loss + prior_weight * prior_alignment_loss
                return loss, logits, ortho_loss, coarse_aux_loss, prior_alignment_loss

            if (
                len(outputs) == 3
                and isinstance(outputs[1], torch.Tensor)
                and outputs[1].dim() == 0
                and isinstance(outputs[2], torch.Tensor)
            ):
                logits, ortho_loss, coarse_logits = outputs
                main_loss = self._classification_loss(logits, labels)
                ortho_weight = self.config.get('model', {}).get('ortho_loss_weight', 0.1)
                coarse_weight = aux_classification_weight()
                coarse_aux_loss = self._classification_loss(coarse_logits, labels)
                loss = main_loss + ortho_weight * ortho_loss
                if coarse_weight > 0.0:
                    loss = loss + coarse_weight * coarse_aux_loss
                return loss, logits, ortho_loss, coarse_aux_loss, None

            if (
                len(outputs) == 2
                and isinstance(outputs[1], torch.Tensor)
                and outputs[1].dim() == 0
            ):
                logits, aux_loss = outputs
                main_loss = self._classification_loss(logits, labels)
                ortho_weight = self.config.get('model', {}).get('ortho_loss_weight', 0.1)
                return main_loss + ortho_weight * aux_loss, logits, aux_loss, None, None

            main_out, aux_out = outputs
            loss = inception_loss(main_out, aux_out, labels, criterion=self.criterion)
            return loss, main_out, None, None, None

        loss = self._classification_loss(outputs, labels)
        return loss, outputs, None, None, None

    def _occlusion_weight_scale(self, epoch_index):
        if not self.use_occlusion_consistency:
            return 0.0

        epoch_number = epoch_index + 1
        if epoch_number < self.occlusion_start_epoch:
            return 0.0
        if epoch_number >= self.occlusion_full_weight_epoch:
            return 1.0

        ramp_length = max(
            self.occlusion_full_weight_epoch - self.occlusion_start_epoch + 1,
            1,
        )
        return (epoch_number - self.occlusion_start_epoch + 1) / ramp_length

    def _consistency_kl(self, clean_logits, masked_logits):
        temperature = self.occlusion_temperature
        clean_probs = F.softmax(clean_logits.detach() / temperature, dim=-1)
        masked_log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
        return F.kl_div(
            masked_log_probs,
            clean_probs,
            reduction='batchmean',
        ) * (temperature ** 2)

    def _compute_batch_loss(
        self,
        images,
        labels,
        region_masks,
        epoch_index,
        occlusion_batch=None,
    ):
        occlusion_scale = self._occlusion_weight_scale(epoch_index)
        if occlusion_scale <= 0.0 or self.occlusion_generator is None:
            outputs = self._forward_model(images, region_masks=region_masks)
            loss, logits, aux_loss, coarse_aux_loss, prior_alignment_loss = (
                self._supervised_loss_from_outputs(outputs, labels)
            )
            return loss, logits, aux_loss, coarse_aux_loss, prior_alignment_loss, occlusion_batch

        if occlusion_batch is None:
            occlusion_batch = self.occlusion_generator(images)

        masked_images, applied_mask = occlusion_batch
        combined_images = torch.cat((images, masked_images), dim=0)
        combined_region_masks = (
            torch.cat((region_masks, region_masks), dim=0)
            if region_masks is not None
            else None
        )
        combined_outputs = self._forward_model(
            combined_images,
            region_masks=combined_region_masks,
        )
        outputs, masked_outputs = self._split_batch_outputs(
            combined_outputs,
            batch_size=images.size(0),
        )
        loss, logits, aux_loss, coarse_aux_loss, prior_alignment_loss = (
            self._supervised_loss_from_outputs(outputs, labels)
        )
        masked_logits = self._extract_logits(masked_outputs)

        if not applied_mask.any().item():
            # Keep the masked half in the graph, but add no extra loss when this
            # local batch happened to receive no occlusion rectangles.
            loss = loss + masked_logits.sum() * 0.0
            return loss, logits, aux_loss, coarse_aux_loss, prior_alignment_loss, occlusion_batch

        masked_logits = masked_logits[applied_mask]
        clean_logits = logits[applied_mask]
        masked_labels = labels[applied_mask]

        masked_ce = self._classification_loss(masked_logits, masked_labels)
        consistency_kl = self._consistency_kl(clean_logits, masked_logits)
        loss = loss + occlusion_scale * (
            self.occlusion_masked_ce_weight * masked_ce
            + self.occlusion_consistency_weight * consistency_kl
        )
        return loss, logits, aux_loss, coarse_aux_loss, prior_alignment_loss, occlusion_batch

    def _reduce_epoch_stats(
        self,
        loss_sum,
        corrects,
        total,
        ortho_sum=0.0,
        coarse_aux_sum=0.0,
        prior_alignment_sum=0.0,
    ):
        stats = torch.tensor(
            [
                loss_sum,
                float(corrects),
                float(total),
                ortho_sum,
                coarse_aux_sum,
                prior_alignment_sum,
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if self.is_distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total = max(stats[2].item(), 1.0)
        epoch_loss = stats[0].item() / total
        epoch_acc = stats[1] / total
        epoch_ortho = stats[3].item() / total if stats[3].item() > 0 else 0.0
        epoch_coarse_aux = stats[4].item() / total if stats[4].item() > 0 else 0.0
        epoch_prior_alignment = stats[5].item() / total if stats[5].item() > 0 else 0.0
        return (
            epoch_loss,
            epoch_acc.to(device=self.device),
            epoch_ortho,
            epoch_coarse_aux,
            epoch_prior_alignment,
        )

    def _sync_stop_flag(self, should_stop):
        if not self.is_distributed:
            return should_stop

        stop_tensor = torch.tensor(int(should_stop), device=self.device)
        dist.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())


    def train_one_epoch(self, epoch_index=0):
        self.model.train()

        running_loss = 0.0
        running_ortho = 0.0
        running_coarse_aux = 0.0
        running_prior_alignment = 0.0
        corrects = 0
        total = 0

        for batch in self.train_loader:
            if len(batch) == 3:
                images, labels, region_masks = batch
                region_masks = region_masks.to(self.device)
            else:
                images, labels = batch
                region_masks = None
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                (
                    loss,
                    logits,
                    aux_loss,
                    coarse_aux_loss,
                    prior_alignment_loss,
                    occlusion_batch,
                ) = self._compute_batch_loss(
                    images=images,
                    labels=labels,
                    region_masks=region_masks,
                    epoch_index=epoch_index,
                )

            if aux_loss is not None:
                running_ortho += aux_loss.item() * images.size(0)
            if coarse_aux_loss is not None:
                running_coarse_aux += coarse_aux_loss.item() * images.size(0)
            if prior_alignment_loss is not None:
                running_prior_alignment += prior_alignment_loss.item() * images.size(0)

            if not torch.isfinite(loss).item():
                if self.skip_nonfinite_batches:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.skipped_nonfinite_batches += 1
                    if self.is_main_process:
                        print("[Trainer] Skipping batch with non-finite loss.")
                    continue
                raise FloatingPointError("Encountered non-finite training loss.")


            if isinstance(self.optimizer, SAM):
                # ── SAM Step 1 ──
                loss.backward()
                if self.grad_clip_norm is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(total_norm).item():
                        if self.skip_nonfinite_batches:
                            self.optimizer.zero_grad(set_to_none=True)
                            self.skipped_nonfinite_batches += 1
                            if self.is_main_process:
                                print("[Trainer] Skipping SAM batch with non-finite gradient norm.")
                            continue
                        raise FloatingPointError("Encountered non-finite SAM gradient norm.")
                self.optimizer.first_step(zero_grad=True)

                # ── SAM Step 2 ──
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss_2, _, _, _, _, _ = self._compute_batch_loss(
                        images=images,
                        labels=labels,
                        region_masks=region_masks,
                        epoch_index=epoch_index,
                        occlusion_batch=occlusion_batch,
                    )
                
                if not torch.isfinite(loss_2).item():
                    if self.skip_nonfinite_batches:
                        self.optimizer.zero_grad(set_to_none=True)
                        self.skipped_nonfinite_batches += 1
                        if self.is_main_process:
                            print("[Trainer] Skipping SAM second step with non-finite loss.")
                        continue
                    raise FloatingPointError("Encountered non-finite SAM second-step loss.")

                loss_2.backward()
                if self.grad_clip_norm is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(total_norm).item():
                        if self.skip_nonfinite_batches:
                            self.optimizer.zero_grad(set_to_none=True)
                            self.skipped_nonfinite_batches += 1
                            if self.is_main_process:
                                print("[Trainer] Skipping SAM second step with non-finite gradient norm.")
                            continue
                        raise FloatingPointError("Encountered non-finite SAM second-step gradient norm.")
                self.optimizer.second_step(zero_grad=True)
            else:
                # ── Standard Optimizer ──
                if self.use_amp:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    if self.grad_clip_norm is not None:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip_norm,
                            error_if_nonfinite=False,
                        )
                        if not torch.isfinite(total_norm).item():
                            if self.skip_nonfinite_batches:
                                self.optimizer.zero_grad(set_to_none=True)
                                self.grad_scaler.update()
                                self.skipped_nonfinite_batches += 1
                                if self.is_main_process:
                                    print("[Trainer] Skipping AMP batch with non-finite gradient norm.")
                                continue
                            raise FloatingPointError("Encountered non-finite AMP gradient norm.")
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip_norm,
                            error_if_nonfinite=False,
                        )
                        if not torch.isfinite(total_norm).item():
                            if self.skip_nonfinite_batches:
                                self.optimizer.zero_grad(set_to_none=True)
                                self.skipped_nonfinite_batches += 1
                                if self.is_main_process:
                                    print("[Trainer] Skipping batch with non-finite gradient norm.")
                                continue
                            raise FloatingPointError("Encountered non-finite gradient norm.")
                    self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        return self._reduce_epoch_stats(
            running_loss,
            corrects,
            total,
            running_ortho,
            running_coarse_aux,
            running_prior_alignment,
        )


    def validate(self):
        self.model.eval()

        running_loss = 0.0
        corrects = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    images, labels, region_masks = batch
                    region_masks = region_masks.to(self.device)
                else:
                    images, labels = batch
                    region_masks = None
                images, labels = images.to(self.device), labels.to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    if region_masks is not None:
                        outputs = self.model(images, region_masks=region_masks)
                    else:
                        outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss, epoch_acc, _, _, _ = self._reduce_epoch_stats(running_loss, corrects, total)
        return epoch_loss, epoch_acc


    def fit(self):
        """ Fit your model
        Return:
            all_train_loss, all_val_loss
        """
        if self.is_main_process:
            print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        if self.use_wandb and self.is_main_process:
            init_wandb(config=self.config, run_name=self.run_name)

        best_score = (
            self.resume_best_score
            if self.resume_best_score is not None
            else (float("inf") if self.monitor == 'val_loss' else -float("inf"))
        )
        best_val_loss = (
            self.resume_best_val_loss
            if self.resume_best_val_loss is not None
            else float("inf")
        )
        best_val_acc = (
            self.resume_best_val_acc
            if self.resume_best_val_acc is not None
            else -float("inf")
        )
        patience_counter = self.resume_patience_counter
        all_train_loss = []
        all_val_loss = []
        self.history = []

        if self.is_main_process:
            if self.start_epoch > 0:
                print(
                    f'\n--> Resuming training from epoch {self.start_epoch + 1}/{self.epochs} '
                    f'(restored best_score={best_score:.4f}) with {self.device} device. Start...\n'
                )
            else:
                print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.start_epoch, self.epochs):
            if self.is_distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(ep)

            # ── Transfer Learning: kiểm tra có cần mở băng backbone không ──
            base_model = self._unwrap_model()
            if hasattr(base_model, 'set_epoch'):
                base_model.set_epoch(ep)
            phase_transitioned = False
            if hasattr(base_model, 'check_unfreeze'):
                should_rebuild = base_model.check_unfreeze(ep)
                if should_rebuild:
                    phase_transitioned = True
                    finetune_lr = self.config['training'].get('finetune_lr')
                    visual_extractor_lr = self.config['training'].get('visual_extractor_lr')
                    if visual_extractor_lr is None:
                        # Legacy fallback: use one small LR for every trainable parameter.
                        finetune_lr = finetune_lr if finetune_lr is not None else 1e-5
                        old_lr = self.config['training']['lr']
                        self.config['training']['lr'] = finetune_lr
                        self.optimizer = build_optimizer(self.model, self.config)
                        self.config['training']['lr'] = old_lr
                        rebuild_msg = f"finetune_lr={finetune_lr}"
                    else:
                        old_lr = self.config['training']['lr']
                        head_lr = finetune_lr if finetune_lr is not None else old_lr
                        self.config['training']['lr'] = head_lr
                        self.optimizer = build_optimizer(self.model, self.config)
                        self.config['training']['lr'] = old_lr
                        rebuild_msg = (
                            f"head_lr={head_lr}, "
                            f"visual_extractor_lr={visual_extractor_lr}"
                        )
                    
                    # REBUILD scheduler to link to NEW optimizer
                    self.scheduler = build_scheduler(self.optimizer, self.config)
                    
                    # RESET bộ đếm Early Stopping để Phase 2 được chạy đủ
                    patience_counter = 0
                    if self.is_main_process:
                        print(
                            "[Trainer] Rebuilt optimizer & scheduler with "
                            f"{rebuild_msg} and reset patience."
                        )

            (
                train_loss,
                train_acc,
                train_ortho_loss,
                train_coarse_aux_loss,
                train_prior_alignment_loss,
            ) = self.train_one_epoch(epoch_index=ep)
            val_loss, val_acc = self.validate()
            val_acc_value = float(val_acc.item())
            train_acc_value = float(train_acc.item())

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            logit_fusion_metrics = self._logit_fusion_metric_dict()
            logit_fusion_text = ""
            if logit_fusion_metrics:
                logit_fusion_text = (
                    " - fusion_w: "
                    f"cnn={logit_fusion_metrics['LogitFusion/CNN_Weight']:.4f}, "
                    f"region={logit_fusion_metrics['LogitFusion/Region_Weight']:.4f}"
                )

            if self.is_main_process:
                print(
                    f"Epoch {ep+1}/{self.epochs} - "
                    f"loss: {train_loss:.4f} "
                    f"(ortho: {train_ortho_loss:.4f}, "
                    f"coarse_aux: {train_coarse_aux_loss:.4f}, "
                    f"prior_align: {train_prior_alignment_loss:.4f}) - "
                    f"accuracy: {train_acc_value:.4f} - "
                    f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc_value:.4f}"
                    f"{logit_fusion_text}"
                )

            # lr scheduler
            lr_before_scheduler = self._current_lrs()
            scheduler_stepped = False
            if self.scheduler is not None:
                scheduler_stepped = True
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            lr_after_scheduler = self._current_lrs()
            scheduler_reduced = any(
                after < before
                for before, after in zip(lr_before_scheduler, lr_after_scheduler)
            )

            # save checkpoint
            current_score = val_loss if self.monitor == 'val_loss' else val_acc.item()
            improved = current_score < best_score if self.monitor == 'val_loss' else current_score > best_score
            best_val_loss = min(best_val_loss, float(val_loss))
            best_val_acc = max(best_val_acc, val_acc_value)

            should_stop = False
            if improved:
                best_score = current_score
                patience_counter = 0

                if self.is_main_process:
                    torch.save({
                        "model_state_dict": self._unwrap_model().state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": ep,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc_value,
                        "monitor": self.monitor,
                        "best_score": best_score,
                        "logit_fusion_metrics": logit_fusion_metrics,
                    }, self.path_save_ckpt)
                    print(
                        f"\t--- Save best at ep {ep+1}, "
                        f"val_loss: {val_loss:.4f}, val_accuracy: {val_acc_value:.4f}, "
                        f"monitor: {self.monitor}, path: {self.path_save_ckpt} ---"
                    )

            else:
                patience_counter += 1
                if self.is_main_process:
                    print(f"\t-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    if self.is_main_process:
                        print(f"\t-_- Early stopping at ep={ep+1}")
                    should_stop = True

            if self.is_main_process:
                history_row = {
                    "epoch": ep + 1,
                    "train_loss": float(train_loss),
                    "train_accuracy": train_acc_value,
                    "train_ortho_loss": float(train_ortho_loss),
                    "train_coarse_aux_loss": float(train_coarse_aux_loss),
                    "train_prior_alignment_loss": float(train_prior_alignment_loss),
                    "val_loss": float(val_loss),
                    "val_accuracy": val_acc_value,
                    "best_val_loss": float(best_val_loss),
                    "best_val_accuracy": float(best_val_acc),
                    "monitor": self.monitor,
                    "best_score": float(best_score),
                    "improved": int(improved),
                    "patience_counter": int(patience_counter),
                    "lr_head": lr_after_scheduler[0] if lr_after_scheduler else 0.0,
                    "lr_visual_extractor": (
                        lr_after_scheduler[1]
                        if len(lr_after_scheduler) > 1
                        else 0.0
                    ),
                    "phase_transitioned": int(phase_transitioned),
                    "skipped_nonfinite_batches": int(self.skipped_nonfinite_batches),
                }
                history_row.update(logit_fusion_metrics)
                self.history.append(history_row)

            # wandb log
            if self.use_wandb and self.is_main_process:
                finetune_scope = getattr(base_model, "unfreeze_backbone_scope", "backbone")
                current_phase = (
                    f"finetune_{finetune_scope}"
                    if getattr(base_model, "unfreeze_backbone", False)
                    and not getattr(base_model, "is_frozen", False)
                    else "frozen_backbone"
                )
                train_cfg = self.config.get('training', {})
                metrics = {
                    "Epoch": ep + 1,
                    "Train/Loss": float(train_loss),
                    "Train/Accuracy": train_acc_value,
                    "Train/Ortho_Loss": float(train_ortho_loss),
                    "Train/Coarse_Aux_Loss": float(train_coarse_aux_loss),
                    "Train/Prior_Alignment_Loss": float(train_prior_alignment_loss),
                    "Val/Loss": float(val_loss),
                    "Val/Accuracy": val_acc_value,
                    "Best/Val_Loss": best_val_loss,
                    "Best/Val_Accuracy": best_val_acc,
                    "Best/Monitor_Score": float(best_score),
                    "Checkpoint/Improved": int(improved),
                    "EarlyStopping/Patience_Counter": patience_counter,
                    "EarlyStopping/Patience": self.patience,
                    "Learning_Rate": lr_after_scheduler[0] if lr_after_scheduler else 0.0,
                    "Learning_Rate/Head": lr_after_scheduler[0] if lr_after_scheduler else 0.0,
                    "Learning_Rate/Visual_Extractor": (
                        lr_after_scheduler[1]
                        if len(lr_after_scheduler) > 1
                        else 0.0
                    ),
                    "Scheduler/Stepped": int(scheduler_stepped),
                    "Scheduler/LR_Reduced": int(scheduler_reduced),
                    "Scheduler/Factor": float(train_cfg.get('lr_factor', 0.0)),
                    "Scheduler/Patience": int(train_cfg.get('lr_patience', 0)),
                    "Training/AMP_Enabled": int(self.use_amp),
                    "Training/Backbone_Finetune_Active": int(current_phase != "frozen_backbone"),
                    "Training/Phase_Transition": int(phase_transitioned),
                    "Training/Grad_Clip_Norm": float(self.grad_clip_norm or 0.0),
                    "Training/Skipped_Nonfinite_Batches": int(self.skipped_nonfinite_batches),
                }
                metrics.update(logit_fusion_metrics)
                metrics.update(self._lr_metric_dict("Learning_Rate/Before_Scheduler", lr_before_scheduler))
                metrics.update(self._lr_metric_dict("Learning_Rate/After_Scheduler", lr_after_scheduler))
                log_metrics(metrics, epoch=ep)

            if self._sync_stop_flag(should_stop):
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

    class DummyDataset(Dataset):
        def __len__(self): return 16
        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0, 7, (1,)).item()

    mock_config = {
        'training': {'epochs': 3, 'patience': 2},
        'path': {'root': '/tmp/'},
        'model': {'name': 'dummy_model'}
    }

    train_loader = DataLoader(DummyDataset(), batch_size=8)
    val_loader = DataLoader(DummyDataset(), batch_size=8)

    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, mock_config, device)
        print("Fitting...")
        trainer.fit()
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
