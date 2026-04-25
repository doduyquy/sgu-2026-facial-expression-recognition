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
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.run_name     = run_name
        self.save_dir     = save_dir

        self.epochs     = config["training"].get("epochs", 30)
        self.patience   = config["training"].get("patience", 10)
        self.model_name = config["model"].get("name", "mlp_baseline")
        self.use_wandb  = config["logging"].get("use_wandb", False)
        self.config     = config

    # ------------------------------------------------------------------
    #  Core loops
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> Dict:
        self.model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            self.optimizer.zero_grad()
            logits = self._forward_batch(batch, x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_classification_metrics(y_true, y_pred)

        return {
            "loss": epoch_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []

        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            logits = self._forward_batch(batch, x)
            loss = self.criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_classification_metrics(y_true, y_pred)

        return {
            "loss": epoch_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }

    # ------------------------------------------------------------------
    #  Dispatch helper
    # ------------------------------------------------------------------

    def _forward_batch(self, batch: dict, x: torch.Tensor) -> torch.Tensor:
        """
        Dispatch forward theo loại batch:
          - GNN batch : có 'edge_index' và 'edge_valid' → model(x, edge_index, edge_valid, mask)
          - MLP batch : có 'mask'                       → model(x, mask=mask)
          - Plain      : chỉ có 'x'                      → model(x)
        """
        if "edge_index" in batch and "edge_valid" in batch:
            # GNN mode
            edge_index = batch["edge_index"].to(self.device)   # [B, 2, E]
            edge_valid = batch["edge_valid"].to(self.device)   # [B, E]
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            return self.model(x, edge_index=edge_index, edge_valid=edge_valid, mask=mask)
        elif "mask" in batch:
            # MLP subgraph mode
            mask = batch["mask"].to(self.device)
            return self.model(x, mask=mask)
        else:
            # Plain MLP mode
            return self.model(x)

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

        best_val_macro_f1 = -1.0
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

            # WandB log
            if self.use_wandb:
                from src.utils.logger_wandb import log_metrics
                log_metrics({
                    "Epoch": ep + 1,
                    "Train/Loss": train_m["loss"],
                    "Train/Accuracy": train_m["accuracy"],
                    "Train/MacroF1": train_m["macro_f1"],
                    "Val/Loss": val_m["loss"],
                    "Val/Accuracy": val_m["accuracy"],
                    "Val/MacroF1": val_m["macro_f1"],
                    "Learning_Rate": self.optimizer.param_groups[0]["lr"],
                }, epoch=ep)

            # LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_m["loss"])
                else:
                    self.scheduler.step()

            # Early stopping — best by val_macro_f1
            if val_m["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_m["macro_f1"]
                patience_counter  = 0
                torch.save({
                    "model_state_dict":    self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch":               ep,
                    "best_val_macro_f1":   best_val_macro_f1,
                }, self.save_dir)
                print(f"\t--- Saved best  ep={ep+1}  val_macro_f1={best_val_macro_f1:.4f}  -> {self.save_dir}")
            else:
                patience_counter += 1
                print(f"\t-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping at ep={ep+1}")
                    break

        return all_train, all_val
