"""
Trainer class cho GNN FER-2013.

Hỗ trợ:
    - train_one_epoch / validate
    - Early stopping (theo val macro_f1)
    - Best checkpoint save
    - WandB logging
    - 3 phase staged schedule (tùy chọn)
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import Dict

from src.evaluation.metrics import compute_classification_metrics


def flush_stdio() -> None:
    """Flush stdout/stderr trước khi vòng lặp DataLoader bắt đầu."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


class Trainer:
    """
    Forward -> Compute loss -> zero_grad -> Backward -> Update weights
    Best model chọn theo val_macro_f1 (khác CNN dùng val_loss).
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config: dict,
        device,
        run_name: str,
        save_dir: str,
    ):
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion.to(device) if hasattr(criterion, "to") else criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.run_name     = run_name
        self.save_dir     = save_dir

        self.epochs     = config["training"].get("epochs", 30)
        self.patience   = config["training"].get(
            "early_stopping_patience",
            config["training"].get("patience", 10),
        )
        self.model_name = config["model"].get("name", "mlp_baseline")
        self.use_wandb  = config["logging"].get("use_wandb", False)
        self.config     = config
        self.num_classes = config.get("model", {}).get(
            "num_classes",
            config.get("data", {}).get("num_classes", 7),
        )
        self.grad_clip_norm = config["training"].get("grad_clip_norm")
        self._logged_candidate_x_stats = False

    # ------------------------------------------------------------------
    #  Core loops
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> Dict:
        self.model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_motif_loss = 0.0
        y_true, y_pred = [], []

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            batch = self._move_batch_to_device(batch)
            x = batch["x"]
            y = self._get_labels(batch)
            self._maybe_log_candidate_x_stats(batch)

            self.optimizer.zero_grad()
            model_out = self._forward_batch(batch, x)
            logits = self._extract_logits(model_out)
            loss_out = self._compute_loss(logits, y, batch, model_out=model_out)
            loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out
            loss.backward()
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            if isinstance(loss_out, dict):
                running_cls_loss += float(loss_out.get("cls_loss", loss).detach().item()) * x.size(0)
                running_motif_loss += float(loss_out.get("motif_loss", loss.new_tensor(0.0)).detach().item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_classification_metrics(y_true, y_pred)
        pred_count = np.bincount(np.array(y_pred, dtype=np.int64), minlength=self.num_classes).tolist()

        out = {
            "loss": epoch_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "pred_count": pred_count,
        }
        if running_cls_loss > 0 or running_motif_loss > 0:
            out["cls_loss"] = running_cls_loss / len(self.train_loader.dataset)
            out["motif_loss"] = running_motif_loss / len(self.train_loader.dataset)
        return out

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_motif_loss = 0.0
        y_true, y_pred = [], []

        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            batch = self._move_batch_to_device(batch)
            x = batch["x"]
            y = self._get_labels(batch)

            model_out = self._forward_batch(batch, x)
            logits = self._extract_logits(model_out)
            loss_out = self._compute_loss(logits, y, batch, model_out=model_out)
            loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out

            running_loss += loss.item() * x.size(0)
            if isinstance(loss_out, dict):
                running_cls_loss += float(loss_out.get("cls_loss", loss).detach().item()) * x.size(0)
                running_motif_loss += float(loss_out.get("motif_loss", loss.new_tensor(0.0)).detach().item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_classification_metrics(y_true, y_pred)
        pred_count = np.bincount(np.array(y_pred, dtype=np.int64), minlength=self.num_classes).tolist()

        out = {
            "loss": epoch_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "pred_count": pred_count,
        }
        if running_cls_loss > 0 or running_motif_loss > 0:
            out["cls_loss"] = running_cls_loss / len(self.val_loader.dataset)
            out["motif_loss"] = running_motif_loss / len(self.val_loader.dataset)
        return out

    # ------------------------------------------------------------------
    #  Dispatch helper
    # ------------------------------------------------------------------

    def _move_batch_to_device(self, batch):
        if not isinstance(batch, dict):
            return batch
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _get_labels(self, batch: dict) -> torch.Tensor:
        y = batch.get("y", batch.get("label"))
        if y is None:
            raise KeyError("Batch must contain 'y' or 'label'")
        return y.to(self.device).long()

    def _extract_logits(self, model_out):
        if isinstance(model_out, dict):
            return model_out["logits"]
        return model_out

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor, batch: dict, model_out=None):
        try:
            loss_out = self.criterion(logits, y, batch=batch)
        except TypeError:
            loss_out = self.criterion(logits, y)
        aux_loss = None
        if isinstance(model_out, dict):
            aux_loss = model_out.get("aux_loss")
        if aux_loss is None:
            return loss_out
        if isinstance(loss_out, dict):
            total = loss_out["loss"] + aux_loss
            out = dict(loss_out)
            out["loss"] = total
            out["motif_loss"] = out.get("motif_loss", logits.new_tensor(0.0)) + aux_loss
            return out
        return loss_out + aux_loss

    def _forward_batch(self, batch: dict, x: torch.Tensor) -> torch.Tensor:
        """
        Dispatch forward theo loại batch:
          - Motif batch: có motif_score_vector/match_scores → model(batch)
          - GNN batch : có 'edge_index' và 'edge_valid' → model(x, edge_index, edge_valid, mask)
          - MLP batch : có 'mask'                       → model(x, mask=mask)
          - Plain      : chỉ có 'x'                      → model(x)
        """
        if "candidate_x" in batch:
            return self.model(batch)
        if {"motif_score_vector", "match_scores", "matched_class"}.issubset(batch.keys()):
            return self.model(batch)
        elif "edge_index" in batch and "edge_valid" in batch:
            # GNN mode
            edge_index = batch["edge_index"]   # [B, 2, E]
            edge_valid = batch["edge_valid"]   # [B, E]
            mask = batch.get("mask")
            return self.model(x, edge_index=edge_index, edge_valid=edge_valid, mask=mask)
        elif "mask" in batch:
            # MLP subgraph mode
            mask = batch["mask"]
            return self.model(x, mask=mask)
        else:
            # Plain MLP mode
            return self.model(x)

    def _maybe_log_candidate_x_stats(self, batch: dict) -> None:
        if self._logged_candidate_x_stats or "candidate_x" not in batch:
            return
        candidate_x = batch["candidate_x"].detach().float()
        mask = batch.get("candidate_mask")
        if torch.is_tensor(mask) and mask.any():
            values = candidate_x[mask.bool()]
        else:
            values = candidate_x.reshape(-1, candidate_x.shape[-1])
        if values.numel() == 0:
            print("--- candidate_x batch stats before projection: empty", flush=True)
        else:
            print(
                "--- candidate_x batch stats before projection: "
                f"min={values.min().item():.6f} "
                f"max={values.max().item():.6f} "
                f"mean={values.mean().item():.6f} "
                f"std={values.std(unbiased=False).item():.6f}",
                flush=True,
            )
        self._logged_candidate_x_stats = True

    # ------------------------------------------------------------------
    #  fit
    # ------------------------------------------------------------------

    def fit(self):
        """
        Train model, early stop theo val_macro_f1.
        Returns: all_train_metrics, all_val_metrics (list of dicts)
        """
        print(
            f"\n--> Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)}",
            flush=True,
        )
        print(f"--> Start training: {self.epochs} epochs | device: {self.device}\n", flush=True)
        flush_stdio()

        if self.use_wandb:
            from src.utils.logger_wandb import init_wandb
            init_wandb(config=self.config, run_name=self.run_name)

        ckpt_parent = Path(self.save_dir).parent
        if str(ckpt_parent) not in ("", "."):
            os.makedirs(ckpt_parent, exist_ok=True)

        monitor_key = self.config.get("training", {}).get(
            "monitor",
            self.config.get("scheduler", {}).get("monitor", "val_macro_f1"),
        )
        monitor_mode = self.config.get("scheduler", {}).get("mode", "max")
        best_monitor = -float("inf") if monitor_mode == "max" else float("inf")
        patience_counter  = 0
        all_train = []
        all_val   = []

        for ep in range(self.epochs):
            train_m = self.train_one_epoch()
            val_m   = self.validate()

            all_train.append(train_m)
            all_val.append(val_m)

            print(
                f"Epoch {ep+1:3d}/{self.epochs} | "
                f"loss: {train_m['loss']:.4f}  acc: {train_m['accuracy']:.4f}  macro_f1: {train_m['macro_f1']:.4f} | "
                f"val_loss: {val_m['loss']:.4f}  val_acc: {val_m['accuracy']:.4f}  val_macro_f1: {val_m['macro_f1']:.4f}"
            )
            if "cls_loss" in train_m or "motif_loss" in train_m:
                print(
                    f"          cls_loss: {train_m.get('cls_loss', 0.0):.4f}  "
                    f"motif_loss: {train_m.get('motif_loss', 0.0):.4f} | "
                    f"val_cls_loss: {val_m.get('cls_loss', 0.0):.4f}  "
                    f"val_motif_loss: {val_m.get('motif_loss', 0.0):.4f}"
                )
            print(f"          val pred_count: {val_m.get('pred_count')}")

            # WandB log
            if self.use_wandb:
                from src.utils.logger_wandb import log_metrics
                payload = {
                    "Epoch": ep + 1,
                    "Train/Loss": train_m["loss"],
                    "Train/Accuracy": train_m["accuracy"],
                    "Train/MacroF1": train_m["macro_f1"],
                    "Val/Loss": val_m["loss"],
                    "Val/Accuracy": val_m["accuracy"],
                    "Val/MacroF1": val_m["macro_f1"],
                    "Learning_Rate": self.optimizer.param_groups[0]["lr"],
                }
                if "cls_loss" in train_m:
                    payload["Train/ClsLoss"] = train_m["cls_loss"]
                    payload["Val/ClsLoss"] = val_m.get("cls_loss", 0.0)
                if "motif_loss" in train_m:
                    payload["Train/MotifLoss"] = train_m["motif_loss"]
                    payload["Val/MotifLoss"] = val_m.get("motif_loss", 0.0)
                for class_id, count in enumerate(val_m.get("pred_count", [])):
                    payload[f"Val/PredCount/class_{class_id}"] = count
                log_metrics(payload, epoch=ep)

            # LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self._monitor_value(val_m, getattr(self.scheduler, "monitor_key", monitor_key)))
                else:
                    self.scheduler.step()

            current_monitor = self._monitor_value(val_m, monitor_key)
            improved = current_monitor > best_monitor if monitor_mode == "max" else current_monitor < best_monitor
            if improved:
                best_monitor = current_monitor
                patience_counter  = 0
                torch.save({
                    "model_state_dict":    self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch":               ep,
                    "best_monitor":        best_monitor,
                    "monitor_key":         monitor_key,
                    "best_val_macro_f1":   val_m["macro_f1"],
                }, self.save_dir)
                print(
                    f"\t--- Saved best  ep={ep+1}  {monitor_key}={best_monitor:.4f}  "
                    f"val_macro_f1={val_m['macro_f1']:.4f}  -> {self.save_dir}"
                )
            else:
                patience_counter += 1
                print(f"\t-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping at ep={ep+1}")
                    break

        return all_train, all_val

    def _monitor_value(self, val_m: Dict, monitor_key: str) -> float:
        key = str(monitor_key)
        aliases = {
            "val_loss": "loss",
            "loss": "loss",
            "val_macro_f1": "macro_f1",
            "macro_f1": "macro_f1",
            "val_acc": "accuracy",
            "val_accuracy": "accuracy",
            "accuracy": "accuracy",
            "val_weighted_f1": "weighted_f1",
            "weighted_f1": "weighted_f1",
        }
        metric_key = aliases.get(key, key.replace("val_", ""))
        if metric_key not in val_m:
            raise KeyError(f"Monitor metric {monitor_key!r} not found in val metrics: {list(val_m.keys())}")
        return float(val_m[metric_key])
