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
        self.max_train_batches = self._optional_positive_int(config["training"].get("max_train_batches"))
        self.max_val_batches = self._optional_positive_int(config["training"].get("max_val_batches"))
        self._logged_candidate_x_stats = False
        self._logged_full_graph_stats = False
        self._logged_batch_limits = False

    # ------------------------------------------------------------------
    #  Core loops
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> Dict:
        self.model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_motif_loss = 0.0
        samples_seen = 0
        running_grad_norm = 0.0
        grad_norm_batches = 0
        d4a_acc = self._new_d4a_accumulator()
        y_true, y_pred = [], []

        self._maybe_log_batch_limits()
        train_total = self._progress_total(self.train_loader, self.max_train_batches)
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Train", leave=False, total=train_total)):
            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break
            batch = self._move_batch_to_device(batch)
            x = batch["x"]
            y = self._get_labels(batch)
            self._maybe_log_candidate_x_stats(batch)

            self.optimizer.zero_grad()
            model_out = self._forward_batch(batch, x)
            self._maybe_log_full_graph_stats(batch, model_out)
            logits = self._extract_logits(model_out)
            self._update_d4a_accumulator(d4a_acc, model_out, batch, logits)
            loss_out = self._compute_loss(logits, y, batch, model_out=model_out)
            loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out
            loss.backward()
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
            else:
                grad_norm = self._grad_norm()
            running_grad_norm += float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            grad_norm_batches += 1
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            samples_seen += int(x.size(0))
            if isinstance(loss_out, dict):
                running_cls_loss += float(loss_out.get("cls_loss", loss).detach().item()) * x.size(0)
                running_motif_loss += float(loss_out.get("motif_loss", loss.new_tensor(0.0)).detach().item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

        denom = max(1, samples_seen)
        epoch_loss = running_loss / denom
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
            out["cls_loss"] = running_cls_loss / denom
            out["motif_loss"] = running_motif_loss / denom
        if grad_norm_batches > 0:
            out["grad_norm"] = running_grad_norm / grad_norm_batches
        out.update(self._finalize_d4a_accumulator(d4a_acc))
        return out

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_motif_loss = 0.0
        samples_seen = 0
        d4a_acc = self._new_d4a_accumulator()
        y_true, y_pred = [], []

        self._maybe_log_batch_limits()
        val_total = self._progress_total(self.val_loader, self.max_val_batches)
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Val", leave=False, total=val_total)):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break
            batch = self._move_batch_to_device(batch)
            x = batch["x"]
            y = self._get_labels(batch)

            model_out = self._forward_batch(batch, x)
            logits = self._extract_logits(model_out)
            self._update_d4a_accumulator(d4a_acc, model_out, batch, logits)
            loss_out = self._compute_loss(logits, y, batch, model_out=model_out)
            loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out

            running_loss += loss.item() * x.size(0)
            samples_seen += int(x.size(0))
            if isinstance(loss_out, dict):
                running_cls_loss += float(loss_out.get("cls_loss", loss).detach().item()) * x.size(0)
                running_motif_loss += float(loss_out.get("motif_loss", loss.new_tensor(0.0)).detach().item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        denom = max(1, samples_seen)
        epoch_loss = running_loss / denom
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
            out["cls_loss"] = running_cls_loss / denom
            out["motif_loss"] = running_motif_loss / denom
        out.update(self._finalize_d4a_accumulator(d4a_acc))
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

    @staticmethod
    def _optional_positive_int(value) -> int | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"", "none", "null"}:
            return None
        out = int(value)
        if out <= 0:
            return None
        return out

    def _maybe_log_batch_limits(self) -> None:
        if self._logged_batch_limits:
            return
        if self.max_train_batches is not None:
            print(
                f"WARNING/SMOKE: max_train_batches={self.max_train_batches}; "
                "train metrics are subset metrics, not full-dataset metrics.",
                flush=True,
            )
        if self.max_val_batches is not None:
            print(
                f"WARNING/SMOKE: max_val_batches={self.max_val_batches}; "
                "validation metrics and pred_count are subset metrics, not full-dataset metrics.",
                flush=True,
            )
        self._logged_batch_limits = True

    @staticmethod
    def _progress_total(loader, limit: int | None) -> int | None:
        try:
            loader_len = len(loader)
        except TypeError:
            loader_len = None
        if limit is None:
            return loader_len
        return min(int(limit), loader_len) if loader_len is not None else int(limit)

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

    def _grad_norm(self) -> float:
        total_sq = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total_sq += float(torch.sum(grad * grad).cpu())
        return total_sq ** 0.5

    def _new_d4a_accumulator(self) -> dict:
        return {"weight": 0.0, "metrics": {}}

    def _add_d4a_metric(self, acc: dict, name: str, value, weight: int) -> None:
        if value is None:
            return
        try:
            scalar = float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(scalar):
            scalar = 0.0
        acc["metrics"][name] = acc["metrics"].get(name, 0.0) + scalar * weight

    def _update_d4a_accumulator(self, acc: dict, model_out, batch: dict, logits: torch.Tensor) -> None:
        if not isinstance(model_out, dict) or "slot_assignments" not in model_out:
            return

        assignments = model_out["slot_assignments"].detach().float()
        B = int(assignments.shape[0])
        weight = max(1, B)
        acc["weight"] += weight

        node_mask = batch.get("node_mask")
        if torch.is_tensor(node_mask):
            node_mask = node_mask.to(device=assignments.device).bool()
        else:
            node_mask = torch.ones(assignments.shape[:2], dtype=torch.bool, device=assignments.device)
        mask_f = node_mask.to(dtype=assignments.dtype)

        entropy = -(assignments.clamp_min(1e-8) * assignments.clamp_min(1e-8).log()).sum(dim=-1)
        valid_entropy = entropy[node_mask] if node_mask.any() else entropy.reshape(-1)

        slot_mass = model_out.get("slot_mass")
        if torch.is_tensor(slot_mass):
            slot_mass_values = slot_mass.detach().float()
        else:
            num_slots = assignments.shape[-1] - (1 if getattr(self.model, "use_null_slot", False) else 0)
            slot_mass_values = (assignments[:, :, :num_slots] * mask_f.unsqueeze(-1)).sum(dim=1)

        slot_gates = model_out.get("slot_gates")
        slot_gate_values = slot_gates.detach().float() if torch.is_tensor(slot_gates) else None

        if assignments.shape[-1] > 0 and getattr(self.model, "use_null_slot", False):
            null_values = assignments[:, :, -1]
            null_mass = (null_values * mask_f).sum() / mask_f.sum().clamp_min(1.0)
            motif_mass_total = (assignments[:, :, :-1].sum(dim=-1) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
        else:
            null_mass = assignments.new_tensor(0.0)
            motif_mass_total = (assignments.sum(dim=-1) * mask_f).sum() / mask_f.sum().clamp_min(1.0)

        self._add_d4a_metric(acc, "d4a/null_mass", model_out.get("null_mass", null_mass), weight)
        self._add_d4a_metric(acc, "d4a/motif_mass_total", model_out.get("motif_mass_total", motif_mass_total), weight)
        self._add_d4a_metric(acc, "d4a/assignment_entropy_mean", valid_entropy.mean(), weight)
        self._add_d4a_metric(acc, "d4a/assignment_entropy_std", valid_entropy.std(unbiased=False), weight)
        self._add_d4a_metric(acc, "d4a/slot_mass_mean", slot_mass_values.mean(), weight)
        self._add_d4a_metric(acc, "d4a/slot_mass_min", slot_mass_values.min(), weight)
        self._add_d4a_metric(acc, "d4a/slot_mass_max", slot_mass_values.max(), weight)
        self._add_d4a_metric(acc, "d4a/slot_mass_std", slot_mass_values.std(unbiased=False), weight)
        self._add_d4a_metric(acc, "d4a/active_slot_count_soft", model_out.get("active_slot_count_soft"), weight)
        if slot_gate_values is not None:
            self._add_d4a_metric(acc, "d4a/slot_gate_mean", slot_gate_values.mean(), weight)
            self._add_d4a_metric(acc, "d4a/slot_gate_min", slot_gate_values.min(), weight)
            self._add_d4a_metric(acc, "d4a/slot_gate_max", slot_gate_values.max(), weight)

        logits_bad = (~torch.isfinite(logits.detach())).sum()
        assignments_bad = (~torch.isfinite(assignments)).sum()
        self._add_d4a_metric(acc, "d4a/logits_nan_count", logits_bad, weight)
        self._add_d4a_metric(acc, "d4a/assignments_nan_count", assignments_bad, weight)

    def _finalize_d4a_accumulator(self, acc: dict) -> dict:
        weight = float(acc.get("weight", 0.0))
        if weight <= 0:
            return {}
        return {name: total / weight for name, total in acc["metrics"].items()}

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
        if "node_features" in batch and "edge_index" in batch and "edge_attr" in batch:
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
            print("--- candidate_x batch stats after normalize before projection: empty", flush=True)
        else:
            print(
                "--- candidate_x batch stats after normalize before projection: "
                f"min={values.min().item():.6f} "
                f"max={values.max().item():.6f} "
                f"mean={values.mean().item():.6f} "
                f"std={values.std(unbiased=False).item():.6f}",
                flush=True,
            )
        self._logged_candidate_x_stats = True

    def _maybe_log_full_graph_stats(self, batch: dict, model_out) -> None:
        if self._logged_full_graph_stats or "node_features" not in batch:
            return
        node_features = batch["node_features"].detach().float()
        edge_attr = batch.get("edge_attr")
        node_mask = batch.get("node_mask")
        if torch.is_tensor(node_mask) and node_mask.any():
            node_values = node_features[node_mask.bool()]
        else:
            node_values = node_features.reshape(-1, node_features.shape[-1])
        print(
            "--- full_graph batch node_features stats: "
            f"min={node_values.min().item():.6f} "
            f"max={node_values.max().item():.6f} "
            f"mean={node_values.mean().item():.6f} "
            f"std={node_values.std(unbiased=False).item():.6f}",
            flush=True,
        )
        if torch.is_tensor(edge_attr):
            edge_values = edge_attr.detach().float().reshape(-1, edge_attr.shape[-1])
            print(
                "--- full_graph batch edge_attr stats: "
                f"min={edge_values.min().item():.6f} "
                f"max={edge_values.max().item():.6f} "
                f"mean={edge_values.mean().item():.6f} "
                f"std={edge_values.std(unbiased=False).item():.6f}",
                flush=True,
            )
        if isinstance(model_out, dict) and "slot_assignments" in model_out:
            assignments = model_out["slot_assignments"].detach().float()
            entropy = -(assignments.clamp_min(1e-8) * assignments.clamp_min(1e-8).log()).sum(dim=-1)
            if torch.is_tensor(node_mask) and node_mask.any():
                entropy_value = entropy[node_mask.bool()].mean()
            else:
                entropy_value = entropy.mean()
            null_mass = model_out.get("null_mass")
            mean_gate = model_out.get("slot_gates")
            active_slots = model_out.get("active_slot_count_soft")
            null_value = float(null_mass.detach().cpu()) if torch.is_tensor(null_mass) else float(null_mass or 0.0)
            gate_value = float(mean_gate.detach().float().mean().cpu()) if torch.is_tensor(mean_gate) else 0.0
            active_value = (
                float(active_slots.detach().cpu())
                if torch.is_tensor(active_slots)
                else float(active_slots or 0.0)
            )
            print(
                "--- D4A slot stats: "
                f"assignment_entropy={entropy_value.item():.6f} "
                f"null_slot_mass={null_value:.6f} "
                f"mean_slot_gate={gate_value:.6f} "
                f"active_slot_count_soft={active_value:.6f}",
                flush=True,
            )
        self._logged_full_graph_stats = True

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
            if "d4a/null_mass" in train_m or "d4a/null_mass" in val_m:
                print(
                    "          D4A train: "
                    f"null={train_m.get('d4a/null_mass', 0.0):.4f} "
                    f"motif={train_m.get('d4a/motif_mass_total', 0.0):.4f} "
                    f"entropy={train_m.get('d4a/assignment_entropy_mean', 0.0):.4f} "
                    f"entropy_std={train_m.get('d4a/assignment_entropy_std', 0.0):.4f} "
                    f"slot_mass_mean={train_m.get('d4a/slot_mass_mean', 0.0):.2f} "
                    f"slot_mass_min={train_m.get('d4a/slot_mass_min', 0.0):.2f} "
                    f"slot_mass_max={train_m.get('d4a/slot_mass_max', 0.0):.2f} "
                    f"slot_mass_std={train_m.get('d4a/slot_mass_std', 0.0):.2f} "
                    f"gate={train_m.get('d4a/slot_gate_mean', 0.0):.4f} "
                    f"gate_min={train_m.get('d4a/slot_gate_min', 0.0):.4f} "
                    f"gate_max={train_m.get('d4a/slot_gate_max', 0.0):.4f} "
                    f"active={train_m.get('d4a/active_slot_count_soft', 0.0):.2f} "
                    f"bad_logits={train_m.get('d4a/logits_nan_count', 0.0):.1f} "
                    f"bad_assign={train_m.get('d4a/assignments_nan_count', 0.0):.1f} "
                    f"grad_norm={train_m.get('grad_norm', 0.0):.4f}"
                )
                print(
                    "          D4A val  : "
                    f"null={val_m.get('d4a/null_mass', 0.0):.4f} "
                    f"motif={val_m.get('d4a/motif_mass_total', 0.0):.4f} "
                    f"entropy={val_m.get('d4a/assignment_entropy_mean', 0.0):.4f} "
                    f"entropy_std={val_m.get('d4a/assignment_entropy_std', 0.0):.4f} "
                    f"slot_mass_mean={val_m.get('d4a/slot_mass_mean', 0.0):.2f} "
                    f"slot_mass_min={val_m.get('d4a/slot_mass_min', 0.0):.2f} "
                    f"slot_mass_max={val_m.get('d4a/slot_mass_max', 0.0):.2f} "
                    f"slot_mass_std={val_m.get('d4a/slot_mass_std', 0.0):.2f} "
                    f"gate={val_m.get('d4a/slot_gate_mean', 0.0):.4f} "
                    f"gate_min={val_m.get('d4a/slot_gate_min', 0.0):.4f} "
                    f"gate_max={val_m.get('d4a/slot_gate_max', 0.0):.4f} "
                    f"active={val_m.get('d4a/active_slot_count_soft', 0.0):.2f} "
                    f"bad_logits={val_m.get('d4a/logits_nan_count', 0.0):.1f} "
                    f"bad_assign={val_m.get('d4a/assignments_nan_count', 0.0):.1f}"
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
                if "grad_norm" in train_m:
                    payload["Train/GradNorm"] = train_m["grad_norm"]
                for key, value in train_m.items():
                    if key.startswith("d4a/"):
                        payload[f"Train/{key}"] = value
                for key, value in val_m.items():
                    if key.startswith("d4a/"):
                        payload[f"Val/{key}"] = value
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
