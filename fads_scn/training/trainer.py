import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .ema import ModelEMA, recalibrate_bn
from .optimizer import build_optimizer, build_scheduler, build_adamw_param_groups
from ..data.dataset import EMOTION_NAMES, build_clean_train_loader
from ..evaluation.evaluator import evaluate_model, plot_confusion_matrix
from ..runtime import tta_mode


def selection_key(metrics, metric="accuracy"):
    if metric not in ("accuracy", "macro_f1", "hybrid_score"):
        raise ValueError(f"Unsupported selection_metric: {metric}")
    return metrics[metric], metrics["macro_f1"], -metrics["nll"]


def serializable_metrics(metrics):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metrics.items()}


class AttentiveSCNTrainer:
    """Compare live/EMA weights on validation and export one best checkpoint."""

    def __init__(self, model, criterion, train_loader, val_loader, test_loader=None, cfg=None, device="cpu"):
        self.cfg = copy.deepcopy(cfg or {})
        self.device = torch.device(device)
        self.model, self.criterion = model.to(self.device), criterion.to(self.device)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        tr, scn, data = self.cfg.setdefault("training", {}), self.cfg.get("scn", {}), self.cfg.get("data", {})
        self.epochs, self.patience = int(tr.get("epochs", 60)), int(tr.get("patience", 12))
        self.clip_grad_norm = tr.get("clip_grad_norm", 2.0)
        self.rank_warmup_epochs = scn.get("rank_warmup_epochs", 8)
        if scn.get("enable_relabel", False):
            raise ValueError("Permanent relabeling is disabled; use SCN without rewriting labels")
        self.use_mixup, self.mixup_alpha, self.mixup_prob = data.get("use_mixup", False), data.get("mixup_alpha", 0.2), data.get("mixup_prob", 0.5)
        self.metric = tr.get("selection_metric", "accuracy")
        if self.metric not in ("accuracy", "macro_f1", "hybrid_score"):
            raise ValueError("Invalid selection_metric")
        self.val_tta = tta_mode(tr.get("val_tta", "none"))
        self.compare_raw = tr.get("compare_raw_and_ema", True)
        self.bn_mode, self.bn_batches = tr.get("ema_bn_mode", "average"), int(tr.get("bn_calibration_batches", 0))
        if self.bn_mode not in ("average", "recalibrate"):
            raise ValueError("ema_bn_mode must be average or recalibrate")
        self.optimizer = build_optimizer(self.model, tr)
        self.scheduler, self.scheduler_interval = build_scheduler(self.optimizer, tr, len(train_loader))
        self.ema = (ModelEMA(self.model, tr.get("ema_decay", 0.999), tr.get("ema_warmup_updates", 2000))
                    if tr.get("use_ema", True) else None)
        self.clean_train_loader = (build_clean_train_loader(self.cfg, train_loader.dataset)
                                   if self.ema is not None and self.bn_mode == "recalibrate" else None)
        self.amp_enabled = bool(tr.get("use_amp", True) and self.device.type == "cuda")
        dtype = tr.get("amp_dtype", "float16")
        if dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be float16 or bfloat16")
        self.amp_dtype = getattr(torch, dtype)
        if self.amp_enabled and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16; select float16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        backbone, seed = self.cfg.get("model", {}).get("backbone", "model"), self.cfg.get("seed", {}).get("random_seed", 42)
        run = f"{backbone}_seed{seed}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
        self.output_dir = Path(tr.get("output_dir", "outputs/fads_scn_convnext")) / run
        self.output_dir.mkdir(parents=True, exist_ok=False)
        tr["output_dir"] = str(self.output_dir)
        self.cfg["runtime"] = {"torch_version": str(torch.__version__), "device": str(self.device)}
        with (self.output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.cfg, stream, sort_keys=False)
        self.best_key, self.best_epoch, self.patience_counter = (-math.inf,) * 3, 0, 0
        self.global_step, self.history = 0, []

    def train_one_epoch(self, epoch):
        self.model.train()
        totals = torch.zeros(7, device=self.device)
        alpha_sum, alpha_count = torch.zeros_like(totals), torch.zeros_like(totals)
        noisy_count, ranked_count = torch.zeros_like(totals), torch.zeros_like(totals)
        correct_alpha = torch.zeros(4, device=self.device)
        sample_count = 0
        for images, targets, _ in self.train_loader:
            images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            size, targets_b, lam = images.size(0), None, 1.0
            if self.use_mixup and self.mixup_alpha > 0 and size > 1 and np.random.rand() < self.mixup_prob:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                lam = max(lam, 1 - lam)
                permutation = torch.randperm(size, device=self.device)
                images, targets_b = lam * images + (1 - lam) * images[permutation], targets[permutation]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                outputs = self.model(images, targets=targets, targets_b=targets_b, lam=lam, use_tta=False)
                losses = self.criterion(outputs, targets, targets_b=targets_b, lam=lam,
                                        current_epoch=epoch, rank_warmup_epochs=self.rank_warmup_epochs)
                loss = losses["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= previous_scale:
                if self.ema is not None:
                    self.ema.update(self.model)
                self.global_step += 1
                if self.scheduler_interval == "batch":
                    self.scheduler.step()
                elif self.scheduler_interval == "restart":
                    self.scheduler.step(self.global_step / len(self.train_loader))
            preds = outputs["logits"].detach().argmax(-1)
            correct = (preds == targets).float()
            if targets_b is not None:
                correct = lam * correct + (1 - lam) * (preds == targets_b).float()
            totals[0] += loss.detach() * size
            totals[1] += correct.sum()
            for index, name in enumerate(("base_ce", "weighted_ce", "rank_loss", "div_loss", "sparsity_loss"), 2):
                totals[index] += losses[name] * size
            if targets_b is None:
                alpha = outputs["alpha"].detach().float().flatten()
                alpha_sum.scatter_add_(0, targets, alpha)
                alpha_count.scatter_add_(0, targets, torch.ones_like(alpha))
                noisy_count.scatter_add_(0, targets, losses["noisy_mask"].float())
                ranked_count.scatter_add_(0, targets, losses["ranked_mask"].float())
                good = preds == targets
                correct_alpha += torch.stack((alpha[good].sum(), good.sum(), alpha[~good].sum(), (~good).sum()))
            sample_count += size
        vals = (totals / max(1, sample_count)).cpu().tolist()
        def class_means(numerator, denominator):
            nums, dens = numerator.cpu().tolist(), denominator.cpu().tolist()
            return {name: n / d if d else None for name, n, d in zip(EMOTION_NAMES, nums, dens)}
        ac = correct_alpha.cpu().tolist()
        result = dict(zip(("loss", "accuracy", "base_ce", "weighted_ce", "rank_loss", "div_loss", "sparsity_loss"), vals))
        result.update(alpha_by_class=class_means(alpha_sum, alpha_count),
                      noisy_fraction_by_class=class_means(noisy_count, ranked_count),
                      alpha_correct=ac[0] / ac[1] if ac[1] else None, alpha_wrong=ac[2] / ac[3] if ac[3] else None)
        if not math.isfinite(result["loss"]):
            raise FloatingPointError("Non-finite training loss; checkpoint was not updated")
        return result

    def _validate(self, model, source, epoch):
        backup = ({name: buf.clone() for name, buf in model.named_buffers()}
                  if source == "ema" and self.bn_mode == "recalibrate" else None)
        try:
            if backup is not None:
                recalibrate_bn(model, self.clean_train_loader, self.device, self.bn_batches)
            metrics = evaluate_model(model, self.val_loader, self.device, use_tta=self.val_tta)
            key = selection_key(metrics, self.metric)
            if not all(math.isfinite(value) for value in key):
                raise FloatingPointError("Non-finite validation metrics")
            if key > self.best_key:
                self.best_key, self.best_epoch = key, epoch + 1
                state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                checkpoint = {"epoch": epoch + 1, "state_dict": state, "config": self.cfg,
                              "val_acc": metrics["accuracy"], "val_loss": metrics["nll"],
                              "macro_f1": metrics["macro_f1"], "hybrid_score": metrics["hybrid_score"],
                              "weights_source": source, "selection_metric": self.metric, "tta": self.val_tta,
                              "bn_recalibrated": backup is not None}
                tmp = self.output_dir / "attentive_scn_best.tmp"
                torch.save(checkpoint, tmp)
                tmp.replace(self.output_dir / "attentive_scn_best.pth")
                print(f"  [BEST {source}] Acc={metrics['accuracy']:.2%} F1={metrics['macro_f1']:.2%} NLL={metrics['nll']:.4f}")
            return metrics
        finally:
            if backup is not None:
                with torch.no_grad():
                    for name, buf in model.named_buffers():
                        buf.copy_(backup[name])

    def fit(self):
        print(f"[START] {self.epochs} epochs | selection={self.metric} | validation TTA={self.val_tta}")
        print(f"Output Directory: {self.output_dir}")
        for epoch in range(self.epochs):
            lrs = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
            train, before = self.train_one_epoch(epoch), self.best_key
            record = {"epoch": epoch + 1, "lr": lrs, "train": train}
            if self.ema is None or self.compare_raw:
                record["raw"] = serializable_metrics(self._validate(self.model, "raw", epoch))
            if self.ema is not None:
                record["ema"] = serializable_metrics(self._validate(self.ema.module, "ema", epoch))
            if self.scheduler_interval == "metric":
                self.scheduler.step(max(record[source][self.metric] for source in ("raw", "ema") if source in record))
            self.patience_counter = 0 if self.best_key > before else self.patience_counter + 1
            self.history.append(record)
            with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            summary = " | ".join(f"{source} Acc={record[source]['accuracy']:.2%} NLL={record[source]['nll']:.4f}"
                                 for source in ("raw", "ema") if source in record)
            print(f"Ep {epoch+1:03d}/{self.epochs} | Train Loss={train['loss']:.4f} Acc={train['accuracy']:.2%} | {summary}")
            if self.patience_counter >= self.patience:
                print(f"Early stopping: {self.patience} epochs without improvement")
                break
        best_path = self.output_dir / "attentive_scn_best.pth"
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        for split, loader in (("val", self.val_loader), ("test", self.test_loader)):
            if loader is None or (split == "test" and not self.cfg["training"].get("evaluate_test_at_end", False)):
                continue
            metrics = evaluate_model(self.model, loader, self.device, use_tta=self.val_tta)
            with (self.output_dir / f"metrics_{split}.json").open("w", encoding="utf-8") as stream:
                json.dump(serializable_metrics(metrics), stream, indent=2, allow_nan=False)
            plot_confusion_matrix(metrics["confusion_matrix"], EMOTION_NAMES,
                                  self.output_dir / f"confusion_matrix_{split}_best.png",
                                  title=f"{split.upper()} | {checkpoint['weights_source']} | Acc {metrics['accuracy']:.2%}")
        print(f"[DONE] Best epoch {self.best_epoch}: {best_path}")
        return best_path
