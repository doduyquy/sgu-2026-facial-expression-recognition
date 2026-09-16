import os
from pathlib import Path
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from ..evaluation.evaluator import evaluate_model, plot_confusion_matrix
from ..data.dataset import EMOTION_NAMES, build_transforms


def build_adamw_param_groups(
    model: nn.Module,
    weight_decay: float,
    backbone_lr: float = None,
    head_lr: float = None,
):
    """Build AdamW groups split by backbone/head and decay/no-decay rules."""
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, norm_types):
            for param in module.parameters(recurse=False):
                no_decay_ids.add(id(param))

    grouped_params = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        scope = "backbone" if name == "backbone" or name.startswith("backbone.") else "head"
        no_decay = id(param) in no_decay_ids or name.endswith(".bias") or param.ndim <= 1
        grouped_params[f"{scope}_{'no_decay' if no_decay else 'decay'}"].append(param)

    groups = []
    for group_name, params in grouped_params.items():
        if not params:
            continue
        group = {
            "params": params,
            "weight_decay": 0.0 if group_name.endswith("no_decay") else weight_decay,
            "group_name": group_name,
        }
        if group_name.startswith("backbone") and backbone_lr is not None:
            group["lr"] = backbone_lr
        elif group_name.startswith("head") and head_lr is not None:
            group["lr"] = head_lr
        groups.append(group)
    return groups


def build_warmup_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    start_factor: float = 0.1,
    eta_min: float = 1e-6,
):
    """Linear warmup followed by cosine decay, stepped once per optimizer update."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0.0 < start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")

    def make_lr_lambda(base_lr: float):
        min_factor = min(1.0, eta_min / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                progress = step / warmup_steps
                return start_factor + (1.0 - start_factor) * progress

            cosine_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / cosine_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return lr_lambda

    return LambdaLR(
        optimizer,
        lr_lambda=[make_lr_lambda(group["lr"]) for group in optimizer.param_groups],
    )


class ModelEMA:
    """EMA parameters with optional decay warmup and live model buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_updates: int = 0):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.warmup_updates = max(0, int(warmup_updates))
        self.num_updates = 0

    def current_decay(self) -> float:
        if self.warmup_updates <= 1:
            return self.decay
        progress = min(1.0, max(0.0, (self.num_updates - 1) / (self.warmup_updates - 1)))
        return self.decay * progress

    def update(self, model: nn.Module):
        self.num_updates += 1
        decay = self.current_decay()
        model_params = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, ema_param in self.module.named_parameters():
                model_param = model_params[name].detach()
                ema_param.mul_(decay).add_(model_param, alpha=1.0 - decay)

            # Running statistics and integer counters are not trainable parameters.
            # Mirror them during training; selected checkpoints are recalibrated below.
            for name, ema_buffer in self.module.named_buffers():
                ema_buffer.copy_(model_buffers[name].detach())


@torch.no_grad()
def recalibrate_batch_norm(model: nn.Module, dataloader, device, max_batches: int = 0) -> int:
    """Recompute BatchNorm running statistics without updating weights or enabling dropout."""
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    bn_layers = [module for module in model.modules() if isinstance(module, bn_types) and module.track_running_stats]
    if not bn_layers:
        model.eval()
        return 0

    model.eval()
    original_momenta = {}
    for layer in bn_layers:
        original_momenta[layer] = layer.momentum
        layer.reset_running_stats()
        layer.momentum = None  # cumulative average over calibration batches
        layer.train()

    processed_batches = 0
    try:
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            model(images, use_tta=False)
            processed_batches += 1
            if max_batches > 0 and processed_batches >= max_batches:
                break
    finally:
        for layer in bn_layers:
            layer.momentum = original_momenta[layer]
        model.eval()

    return processed_batches


def build_bn_calibration_loader(train_loader, cfg: dict):
    """Use the training images with deterministic validation preprocessing for BN calibration."""
    dataset = train_loader.dataset
    if not hasattr(dataset, "transform"):
        return train_loader

    calibration_dataset = copy.copy(dataset)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    calibration_dataset.transform = build_transforms(
        "val",
        input_size=data_cfg.get("input_size", 48),
        in_channels=model_cfg.get("in_channels", 1),
        normalization=data_cfg.get("normalization", "symmetric"),
    )

    return DataLoader(
        calibration_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        collate_fn=train_loader.collate_fn,
        pin_memory=train_loader.pin_memory,
        drop_last=False,
        worker_init_fn=train_loader.worker_init_fn,
    )


class AttentiveSCNTrainer:
    """
    Complete Trainer for Attentive-SCN on FER2013.
    Features:
    - Pure image training (zero bounding box overhead)
    - Self-Cure sample-weighted loss with rank regularization
    - Dynamic noise relabeling after warmup
    - Exponential Moving Average (EMA) with optional decay warmup
    - Epoch cosine or per-update warmup + cosine scheduling
    - Best model selection by Hybrid Score (val_acc * macro_f1)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader,
        val_loader,
        test_loader=None,
        cfg: dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.cfg = cfg or {}
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        train_cfg = self.cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 150)
        self.lr = train_cfg.get("lr", 0.0003)
        self.backbone_lr = train_cfg.get("backbone_lr", self.lr)
        self.head_lr = train_cfg.get("head_lr", self.lr)
        self.weight_decay = train_cfg.get("weight_decay", 0.001)
        self.clip_grad_norm = train_cfg.get("clip_grad_norm", 2.0)
        self.patience = train_cfg.get("patience", 35)
        self.eval_test_on_best_epoch = train_cfg.get("eval_test_on_best_epoch", False)

        # SCN parameters
        scn_cfg = self.cfg.get("scn", {})
        self.rank_warmup_epochs = scn_cfg.get("rank_warmup_epochs", 0)
        self.enable_relabel = scn_cfg.get("enable_relabel", False)
        self.relabel_epoch = scn_cfg.get("relabel_epoch", 20)
        self.relabel_threshold = scn_cfg.get("relabel_threshold", 0.90)
        self.relabel_max_alpha = scn_cfg.get("relabel_max_alpha", 0.20)

        # Output dir
        self.output_dir = Path(train_cfg.get("output_dir", "outputs/fads_scn"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mixup Augmentation parameters
        data_cfg = self.cfg.get("data", {})
        self.use_mixup = data_cfg.get("use_mixup", True)
        self.mixup_alpha = data_cfg.get("mixup_alpha", 0.2)
        self.mixup_prob = data_cfg.get("mixup_prob", 0.5)

        # Optimizer & Scheduler
        optimizer_params = self.model.parameters()
        if train_cfg.get("no_weight_decay_norm_bias", True):
            optimizer_params = build_adamw_param_groups(
                self.model,
                self.weight_decay,
                backbone_lr=self.backbone_lr,
                head_lr=self.head_lr,
            )
            optimizer_weight_decay = 0.0
        else:
            optimizer_weight_decay = self.weight_decay
        self.optimizer = AdamW(
            optimizer_params,
            lr=self.lr,
            weight_decay=optimizer_weight_decay,
        )
        scheduler_type = train_cfg.get("scheduler", "cosine_annealing")
        eta_min = train_cfg.get("eta_min", 1e-6)
        self.scheduler_step_per_batch = scheduler_type == "warmup_cosine"
        if scheduler_type == "warmup_cosine":
            warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
            total_steps = max(1, self.epochs * len(self.train_loader))
            warmup_steps = min(warmup_epochs * len(self.train_loader), max(0, total_steps - 1))
            self.scheduler = build_warmup_cosine_scheduler(
                self.optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                start_factor=float(train_cfg.get("warmup_start_factor", 0.1)),
                eta_min=eta_min,
            )
        elif scheduler_type == "cosine_annealing_warm_restart":
            t_0 = train_cfg.get("T_0", 30)
            t_mult = train_cfg.get("T_mult", 2)
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=eta_min,
            )
        else:
            # Default: Smooth CosineAnnealingLR across full epochs (no shock restart)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=eta_min,
            )

        # EMA
        use_ema = train_cfg.get("use_ema", True)
        ema_decay = train_cfg.get("ema_decay", 0.999)
        ema_warmup_updates = train_cfg.get("ema_warmup_updates", 0)
        self.ema = (
            ModelEMA(self.model, decay=ema_decay, warmup_updates=ema_warmup_updates)
            if use_ema else None
        )
        self.ema_bn_recalibrate = train_cfg.get("ema_bn_recalibrate", False)
        self.ema_bn_recalibrate_batches = int(train_cfg.get("ema_bn_recalibrate_batches", 0))

        # Tracking
        self.best_score = 0.0
        self.best_val_acc = 0.0
        self.best_macro_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        relabelled_this_epoch = 0

        for images, targets, indices in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            B = images.size(0)

            # Mixup Data Augmentation
            if self.use_mixup and np.random.rand() < self.mixup_prob and self.mixup_alpha > 0 and B > 1:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                if lam < 0.5:
                    lam = 1.0 - lam
                index = torch.randperm(B, device=self.device)
                mixed_images = lam * images + (1.0 - lam) * images[index]
                targets_b = targets[index]
            else:
                mixed_images = images
                targets_b = None
                lam = 1.0
            mixup_active = targets_b is not None and lam < 1.0

            self.optimizer.zero_grad()

            outputs = self.model(
                mixed_images,
                targets=targets,
                targets_b=targets_b,
                lam=lam,
                use_tta=False,
            )
            loss_dict = self.criterion(
                outputs,
                targets,
                targets_b=targets_b,
                lam=lam,
                current_epoch=epoch,
                rank_warmup_epochs=self.rank_warmup_epochs,
            )
            loss = loss_dict["loss"]

            loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model)

            if self.scheduler_step_per_batch:
                self.scheduler.step()

            # SCN Dynamic Relabeling (safe mode: only when enable_relabel is True)
            if self.enable_relabel and not mixup_active and epoch >= self.relabel_epoch:
                with torch.no_grad():
                    probs = torch.softmax(outputs["logits"], dim=-1)
                    max_probs, pred_classes = torch.max(probs, dim=-1)
                    alphas = outputs["alpha"].view(-1)

                    for i in range(B):
                        if (
                            max_probs[i].item() > self.relabel_threshold
                            and pred_classes[i].item() != targets[i].item()
                            and alphas[i].item() < self.relabel_max_alpha
                        ):
                            idx = int(indices[i].item())
                            new_lbl = int(pred_classes[i].item())
                            if hasattr(self.train_loader.dataset, "update_label"):
                                self.train_loader.dataset.update_label(idx, new_lbl)
                                relabelled_this_epoch += 1

            preds = torch.argmax(outputs["logits"], dim=-1)
            if targets_b is not None and lam < 1.0:
                correct_step = (lam * (preds == targets).float() + (1.0 - lam) * (preds == targets_b).float()).sum().item()
            else:
                correct_step = (preds == targets).sum().item()
            correct += correct_step
            total_loss += loss.item() * B
            total_samples += B

        if not self.scheduler_step_per_batch:
            self.scheduler.step()
        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc, relabelled_this_epoch

    def fit(self):
        print(f"\n[START] Starting Attentive-SCN Training on {self.device}")
        print(
            f"Total Epochs: {self.epochs} | Batch Size: {self.train_loader.batch_size} | "
            f"Backbone LR: {self.backbone_lr} | Head LR: {self.head_lr}"
        )
        if self.use_mixup:
            print(f"Data Augmentation: Mixup enabled (alpha={self.mixup_alpha}, prob={self.mixup_prob})")
        print(f"Output Directory: {self.output_dir}\n")

        for epoch in range(self.epochs):
            train_loss, train_acc, relabelled = self.train_one_epoch(epoch)

            # Evaluate with EMA model
            eval_model = self.ema.module if self.ema is not None else self.model

            val_metrics = evaluate_model(eval_model, self.val_loader, self.device, use_tta=True, criterion=self.criterion)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]
            hybrid_score = val_metrics["hybrid_score"]

            print(
                f"Ep {epoch+1:03d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}% Score: {hybrid_score:.4f}"
                + (f" | Relabelled: {relabelled}" if relabelled > 0 else "")
            )

            # Check for best model
            if hybrid_score > self.best_score:
                self.best_score = hybrid_score
                self.best_val_acc = val_acc
                self.best_macro_f1 = val_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0

                best_path = self.output_dir / "attentive_scn_best.pth"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": eval_model.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "macro_f1": val_f1,
                        "hybrid_score": hybrid_score,
                        "config": self.cfg,
                    },
                    best_path,
                )
                print(f"  [BEST] New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, F1: {val_f1*100:.2f}% -> {best_path}")

                if self.eval_test_on_best_epoch and self.test_loader is not None:
                    test_metrics = evaluate_model(eval_model, self.test_loader, self.device, use_tta=True, criterion=self.criterion)
                    test_loss = test_metrics["loss"]
                    print(
                        f"  [Test Set @ Ep {epoch+1}] Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']*100:.2f}% "
                        f"| F1: {test_metrics['macro_f1']*100:.2f}%"
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {self.patience} epochs without improvement.")
                    break

        # ============================================================
        # Post-Training: Load Best Checkpoint & Export Confusion Matrices
        # ============================================================
        best_path = self.output_dir / "attentive_scn_best.pth"
        if best_path.exists():
            print(f"\n[EVALUATION] Training finished. Loading best model from {best_path} to export confusion matrices...")
            checkpoint = torch.load(best_path, map_location=self.device)
            eval_model = self.model
            eval_model.load_state_dict(checkpoint["state_dict"])
            eval_model.eval()

            # Re-evaluate the score-selected checkpoint. Optionally compare it
            # against the same weights with freshly calibrated BN statistics.
            val_metrics = evaluate_model(eval_model, self.val_loader, self.device, use_tta=True, criterion=self.criterion)
            if self.ema_bn_recalibrate:
                calibration_loader = build_bn_calibration_loader(self.train_loader, self.cfg)
                calibration_batches = recalibrate_batch_norm(
                    eval_model,
                    calibration_loader,
                    self.device,
                    max_batches=self.ema_bn_recalibrate_batches,
                )
                recalibrated_metrics = evaluate_model(
                    eval_model,
                    self.val_loader,
                    self.device,
                    use_tta=True,
                    criterion=self.criterion,
                )
                print(
                    f"  [BN Recalibration] {calibration_batches} batches | "
                    f"Original Score: {val_metrics['hybrid_score']:.4f} | "
                    f"Recalibrated Score: {recalibrated_metrics['hybrid_score']:.4f}"
                )

                # Keep checkpoint selection strictly score-based.
                if recalibrated_metrics["hybrid_score"] > val_metrics["hybrid_score"]:
                    val_metrics = recalibrated_metrics
                    checkpoint.update(
                        {
                            "state_dict": eval_model.state_dict(),
                            "val_loss": val_metrics["loss"],
                            "val_acc": val_metrics["accuracy"],
                            "macro_f1": val_metrics["macro_f1"],
                            "hybrid_score": val_metrics["hybrid_score"],
                            "bn_recalibrated": True,
                            "bn_calibration_batches": calibration_batches,
                        }
                    )
                    torch.save(checkpoint, best_path)
                    print("  [SELECTED] BN-recalibrated checkpoint selected by validation hybrid score.")
                else:
                    eval_model.load_state_dict(checkpoint["state_dict"])
                    eval_model.eval()
                    print("  [SELECTED] Original checkpoint retained by validation hybrid score.")

            self.best_score = val_metrics["hybrid_score"]
            self.best_val_acc = val_metrics["accuracy"]
            self.best_macro_f1 = val_metrics["macro_f1"]

            # 1. Export Best Validation Confusion Matrix
            cm_val_path = self.output_dir / "confusion_matrix_val_best.png"
            try:
                plot_confusion_matrix(
                    val_metrics["confusion_matrix"],
                    EMOTION_NAMES,
                    cm_val_path,
                    title=f"Val Confusion Matrix (Best Ep {self.best_epoch} | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['macro_f1']*100:.2f}%)",
                )
                print(f"  [Exported] Val Confusion Matrix -> {cm_val_path.name}")
            except Exception as e:
                print(f"  [Warning] Could not export Val confusion matrix: {e}")

            # 2. Export Best Test Confusion Matrix
            cm_test_path = None
            if self.test_loader is not None:
                test_metrics = evaluate_model(eval_model, self.test_loader, self.device, use_tta=True, criterion=self.criterion)
                cm_test_path = self.output_dir / "confusion_matrix_test_best.png"
                try:
                    plot_confusion_matrix(
                        test_metrics["confusion_matrix"],
                        EMOTION_NAMES,
                        cm_test_path,
                        title=f"Test Confusion Matrix (Best Ep {self.best_epoch} | Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']*100:.2f}% | F1: {test_metrics['macro_f1']*100:.2f}%)",
                    )
                    print(f"  [Exported] Test Confusion Matrix -> {cm_test_path.name}")
                except Exception as e:
                    print(f"  [Warning] Could not export Test confusion matrix: {e}")

            print(
                f"\n[DONE] Training Complete! Best Epoch: {self.best_epoch} | "
                f"Best Val Acc: {self.best_val_acc*100:.2f}% | Best Macro F1: {self.best_macro_f1*100:.2f}%\n"
                f"Saved Artifacts in {self.output_dir}:\n"
                f"  - Best Weights: {best_path.name}\n"
                f"  - Val CM: {cm_val_path.name}\n"
                + (f"  - Test CM: {cm_test_path.name}\n" if cm_test_path is not None else "")
            )
        else:
            print(
                f"\n[DONE] Training Complete! Best Epoch: {self.best_epoch} | "
                f"Best Val Acc: {self.best_val_acc*100:.2f}% | Best Macro F1: {self.best_macro_f1*100:.2f}%\n"
            )
