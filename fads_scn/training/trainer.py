import os
from pathlib import Path
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from ..evaluation.evaluator import evaluate_model, plot_confusion_matrix
from ..evaluation.weighted_flip_tta import WeightedHorizontalFlipTTASweep
from ..data.dataset import EMOTION_NAMES


def build_adamw_param_groups(model: nn.Module, weight_decay: float):
    """Apply weight decay only to regular matrix/conv weights, not norm or bias terms."""
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

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_ids or name.endswith(".bias") or param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


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
        self.eval_test_on_best_epoch = train_cfg.get("eval_test_on_best_epoch", False)

        # Checkpoint selection stays TTA-free. TTA is selected only after the
        # best checkpoint is frozen, using validation and then applied to test.
        eval_cfg = self.cfg.get("evaluation", {})
        self.validation_tta = eval_cfg.get("validation_tta", False)
        self.post_train_weighted_flip_tta_sweep = eval_cfg.get("post_train_weighted_flip_tta_sweep", True)
        self.weighted_flip_weights = eval_cfg.get("weighted_flip_weights", None)
        self.tta_selection_metric = eval_cfg.get("tta_selection_metric", "accuracy")

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
        self.mix_mode = data_cfg.get("mix_mode", "standard")
        self.mixaugment_real_weight = data_cfg.get("mixaugment_real_weight", 0.5)
        if self.mix_mode not in ("standard", "mixaugment"):
            raise ValueError("data.mix_mode must be standard or mixaugment")
        if not 0.0 <= self.mixaugment_real_weight <= 1.0:
            raise ValueError("data.mixaugment_real_weight must be in [0, 1]")

        # Optimizer & Scheduler
        optimizer_params = self.model.parameters()
        if train_cfg.get("no_weight_decay_norm_bias", True):
            optimizer_params = build_adamw_param_groups(self.model, self.weight_decay)
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

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        eval_model,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        hybrid_score: float,
        selection_criterion: str,
        graph_diagnostics: dict = None,
    ):
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": eval_model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "macro_f1": val_f1,
                "hybrid_score": hybrid_score,
                "selection_criterion": selection_criterion,
                "graph_diagnostics": graph_diagnostics or {},
                "config": self.cfg,
            },
            path,
        )

    def _evaluate_saved_checkpoint(self, checkpoint_path: Path, checkpoint_label: str):
        """Sweep TTA on validation and evaluate one frozen checkpoint on test."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        eval_model = self.model
        eval_model.load_state_dict(checkpoint["state_dict"])
        eval_model.eval()
        epoch = checkpoint.get("epoch", "?")

        val_metrics = evaluate_model(
            eval_model,
            self.val_loader,
            self.device,
            use_tta=self.validation_tta,
            criterion=self.criterion,
        )
        graph_diag = val_metrics.get("graph_diagnostics", {})
        if graph_diag:
            print(
                f"  [{checkpoint_label} GRAPH] gate={graph_diag['graph_gate']:.4f} "
                f"delta/local={graph_diag['graph_delta_ratio']:.3f} "
                f"contribution={graph_diag['graph_contribution_ratio']:.3f} "
                f"adj_entropy={graph_diag['adjacency_entropy']:.3f} "
                f"node_cosine={graph_diag['node_cosine_similarity']:.3f}"
            )
        cm_val_path = self.output_dir / f"confusion_matrix_val_{checkpoint_label}.png"
        try:
            plot_confusion_matrix(
                val_metrics["confusion_matrix"],
                EMOTION_NAMES,
                cm_val_path,
                title=(
                    f"Val CM ({checkpoint_label}, Ep {epoch} | Loss: {val_metrics['loss']:.4f} | "
                    f"Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['macro_f1']*100:.2f}%)"
                ),
            )
            print(f"  [{checkpoint_label}] Exported val CM -> {cm_val_path.name}")
        except Exception as error:
            print(f"  [{checkpoint_label}] Warning: could not export val CM: {error}")

        cm_test_path = None
        tta_sweep_path = None
        test_metrics = None
        if self.test_loader is not None:
            if self.post_train_weighted_flip_tta_sweep:
                sweep = WeightedHorizontalFlipTTASweep.sweep_and_apply(
                    eval_model,
                    self.val_loader,
                    self.test_loader,
                    self.device,
                    flip_weights=self.weighted_flip_weights,
                    selection_metric=self.tta_selection_metric,
                )
                for result in sweep["validation_results"]:
                    print(
                        f"  [{checkpoint_label} VAL TTA] orig={result['original_weight']:.1f} "
                        f"flip={result['flip_weight']:.1f} | Acc={result['accuracy']*100:.2f}% "
                        f"F1={result['macro_f1']*100:.2f}% Score={result['hybrid_score']:.4f}"
                    )
                print(
                    f"  [{checkpoint_label} SELECTED TTA] metric={sweep['selection_metric']} | "
                    f"orig={sweep['selected_original_weight']:.1f}, flip={sweep['selected_flip_weight']:.1f}"
                )
                test_metrics = sweep["test_metrics"]
                tta_sweep_path = self.output_dir / f"weighted_flip_tta_selection_{checkpoint_label}.json"
                serializable_results = [
                    {
                        key: result[key]
                        for key in ("original_weight", "flip_weight", "loss", "accuracy", "macro_f1", "hybrid_score")
                    }
                    for result in sweep["validation_results"]
                ]
                with open(tta_sweep_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "checkpoint": checkpoint_path.name,
                            "checkpoint_epoch": epoch,
                            "selection_split": "val",
                            "selection_metric": sweep["selection_metric"],
                            "selected_original_weight": sweep["selected_original_weight"],
                            "selected_flip_weight": sweep["selected_flip_weight"],
                            "graph_diagnostics": graph_diag,
                            "validation_results": serializable_results,
                            "test_metrics": {
                                key: test_metrics[key]
                                for key in ("loss", "accuracy", "macro_f1", "hybrid_score", "per_class_acc")
                            },
                        },
                        handle,
                        indent=2,
                    )
            else:
                test_metrics = evaluate_model(
                    eval_model,
                    self.test_loader,
                    self.device,
                    use_tta=self.validation_tta,
                    criterion=self.criterion,
                )

            cm_test_path = self.output_dir / f"confusion_matrix_test_{checkpoint_label}.png"
            try:
                plot_confusion_matrix(
                    test_metrics["confusion_matrix"],
                    EMOTION_NAMES,
                    cm_test_path,
                    title=(
                        f"Test CM ({checkpoint_label}, Ep {epoch} | Loss: {test_metrics['loss']:.4f} | "
                        f"Acc: {test_metrics['accuracy']*100:.2f}% | F1: {test_metrics['macro_f1']*100:.2f}%)"
                    ),
                )
                print(f"  [{checkpoint_label}] Exported test CM -> {cm_test_path.name}")
            except Exception as error:
                print(f"  [{checkpoint_label}] Warning: could not export test CM: {error}")

        return {
            "checkpoint": checkpoint_path,
            "val_cm": cm_val_path,
            "test_cm": cm_test_path,
            "tta_selection": tta_sweep_path,
            "test_metrics": test_metrics,
        }

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

            if mixup_active and self.mix_mode == "mixaugment":
                # MixAugment keeps the supervised signal of real facial images
                # while also learning from virtual Mixup examples. In contrast,
                # standard Mixup replaces the real-image loss for this batch.
                real_outputs = self.model(images, targets=targets, use_tta=False)
                real_loss_dict = self.criterion(
                    real_outputs,
                    targets,
                    current_epoch=epoch,
                    rank_warmup_epochs=self.rank_warmup_epochs,
                )
                mix_outputs = self.model(
                    mixed_images,
                    targets=targets,
                    targets_b=targets_b,
                    lam=lam,
                    use_tta=False,
                )
                mix_loss_dict = self.criterion(
                    mix_outputs,
                    targets,
                    targets_b=targets_b,
                    lam=lam,
                    current_epoch=epoch,
                    rank_warmup_epochs=self.rank_warmup_epochs,
                )
                loss = (
                    self.mixaugment_real_weight * real_loss_dict["loss"]
                    + (1.0 - self.mixaugment_real_weight) * mix_loss_dict["loss"]
                )
                # All training metrics and any optional relabeling use real
                # images only; virtual Mixup images have no single hard label.
                outputs = real_outputs
            else:
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

            # SCN Dynamic Relabeling (safe mode: only when enable_relabel is True)
            real_labels_available = not mixup_active or self.mix_mode == "mixaugment"
            if self.enable_relabel and real_labels_available and epoch >= self.relabel_epoch:
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
            if targets_b is not None and lam < 1.0 and self.mix_mode != "mixaugment":
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
        print(f"Validation TTA during training: {self.validation_tta}")
        if self.use_mixup:
            print(
                f"Data Augmentation: {self.mix_mode} enabled "
                f"(alpha={self.mixup_alpha}, prob={self.mixup_prob})"
            )
        print(f"Output Directory: {self.output_dir}\n")

        for epoch in range(self.epochs):
            train_loss, train_acc, relabelled = self.train_one_epoch(epoch)

            # Evaluate with EMA model
            eval_model = self.ema.module if self.ema is not None else self.model
            if self.ema is not None:
                self.ema.sync_bn(self.model)

            val_metrics = evaluate_model(
                eval_model,
                self.val_loader,
                self.device,
                use_tta=self.validation_tta,
                criterion=self.criterion,
            )
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]
            hybrid_score = val_metrics["hybrid_score"]
            graph_diag = val_metrics.get("graph_diagnostics", {})
            graph_log = ""
            if graph_diag:
                graph_log = (
                    f" | GraphGate={graph_diag['graph_gate']:.4f}"
                    f" Delta/Local={graph_diag['graph_delta_ratio']:.3f}"
                    f" Contribution={graph_diag['graph_contribution_ratio']:.3f}"
                    f" AdjH={graph_diag['adjacency_entropy']:.3f}"
                    f" NodeCos={graph_diag['node_cosine_similarity']:.3f}"
                )

            print(
                f"Ep {epoch+1:03d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}% Score: {hybrid_score:.4f}"
                + graph_log
                + (f" | Relabelled: {relabelled}" if relabelled > 0 else "")
            )

            # Score controls early stopping; lowest validation loss is tracked
            # independently and never changes the stopping criterion.
            score_improved = hybrid_score > self.best_score
            if score_improved:
                self.best_score = hybrid_score
                self.best_val_acc = val_acc
                self.best_macro_f1 = val_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0

                best_path = self.output_dir / "attentive_scn_best.pth"
                self._save_checkpoint(
                    best_path,
                    epoch,
                    eval_model,
                    val_loss,
                    val_acc,
                    val_f1,
                    hybrid_score,
                    selection_criterion="hybrid_score",
                    graph_diagnostics=graph_diag,
                )
                print(f"  [BEST SCORE] Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, F1: {val_f1*100:.2f}% -> {best_path}")

                if self.eval_test_on_best_epoch and self.test_loader is not None:
                    test_metrics = evaluate_model(
                        eval_model,
                        self.test_loader,
                        self.device,
                        use_tta=self.validation_tta,
                        criterion=self.criterion,
                    )
                    test_loss = test_metrics["loss"]
                    print(
                        f"  [Test Set @ Ep {epoch+1}] Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']*100:.2f}% "
                        f"| F1: {test_metrics['macro_f1']*100:.2f}%"
                    )

            if not score_improved:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {self.patience} epochs without improvement.")
                    break

        best_path = self.output_dir / "attentive_scn_best.pth"
        if best_path.exists():
            print("\n[EVALUATION] Sweeping validation TTA for the best-score checkpoint...")
            artifact = self._evaluate_saved_checkpoint(best_path, "best_score")
            print(
                f"\n[DONE] Training Complete! Best-score epoch: {self.best_epoch} | "
                f"Best-score Val Acc: {self.best_val_acc*100:.2f}% | Best-score Macro F1: {self.best_macro_f1*100:.2f}%\n"
                f"Saved Artifacts in {self.output_dir}:"
            )
            print(f"  - {artifact['checkpoint'].name}")
            print(f"  - {artifact['val_cm'].name}")
            if artifact["test_cm"] is not None:
                print(f"  - {artifact['test_cm'].name}")
            if artifact["tta_selection"] is not None:
                print(f"  - {artifact['tta_selection'].name}")
        else:
            print(
                f"\n[DONE] Training Complete! Best Epoch: {self.best_epoch} | "
                f"Best Val Acc: {self.best_val_acc*100:.2f}% | Best Macro F1: {self.best_macro_f1*100:.2f}%\n"
            )
