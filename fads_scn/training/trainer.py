import os
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from ..evaluation.evaluator import evaluate_model, plot_confusion_matrix
from ..data.dataset import EMOTION_NAMES


class ModelEMA:
    """Exponential Moving Average of model parameters with BN buffer synchronization."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_param, model_param in zip(self.module.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def sync_bn(self, model: nn.Module):
        """Copy running mean and var from model to EMA before validation."""
        for ema_buf, model_buf in zip(self.module.buffers(), model.buffers()):
            ema_buf.copy_(model_buf)


class AttentiveSCNTrainer:
    """
    Complete Trainer for Attentive-SCN on FER2013.
    Features:
    - Pure image training (zero bounding box overhead)
    - Self-Cure sample-weighted loss with rank regularization
    - Dynamic noise relabeling after warmup
    - Exponential Moving Average (EMA) with BN synchronization
    - CosineAnnealingWarmRestarts scheduler
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
        self.weight_decay = train_cfg.get("weight_decay", 0.001)
        self.clip_grad_norm = train_cfg.get("clip_grad_norm", 2.0)
        self.patience = train_cfg.get("patience", 35)

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
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler_type = train_cfg.get("scheduler", "cosine_annealing")
        eta_min = train_cfg.get("eta_min", 1e-6)
        if scheduler_type == "cosine_annealing_warm_restart":
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
        self.ema = ModelEMA(self.model, decay=ema_decay) if use_ema else None

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

            self.optimizer.zero_grad()

            outputs = self.model(mixed_images, use_tta=False)
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

            # SCN Dynamic Relabeling (safe mode: only when enable_relabel is True)
            if self.enable_relabel and epoch >= self.relabel_epoch:
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

        self.scheduler.step()
        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc, relabelled_this_epoch

    def fit(self):
        print(f"\n[START] Starting Attentive-SCN Training on {self.device}")
        print(f"Total Epochs: {self.epochs} | Batch Size: {self.train_loader.batch_size} | LR: {self.lr}")
        if self.use_mixup:
            print(f"Data Augmentation: Mixup enabled (alpha={self.mixup_alpha}, prob={self.mixup_prob})")
        print(f"Output Directory: {self.output_dir}\n")

        for epoch in range(self.epochs):
            train_loss, train_acc, relabelled = self.train_one_epoch(epoch)

            # Evaluate with EMA model
            eval_model = self.ema.module if self.ema is not None else self.model
            if self.ema is not None:
                self.ema.sync_bn(self.model)

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

                # Optional: evaluate test set immediately on best checkpoint
                if self.test_loader is not None:
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

            # 1. Export Best Validation Confusion Matrix
            val_metrics = evaluate_model(eval_model, self.val_loader, self.device, use_tta=True, criterion=self.criterion)
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
