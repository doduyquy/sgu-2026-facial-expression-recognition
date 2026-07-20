import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rafdb_mask_dataset import (  # noqa: E402
    CLASS_FOLDERS,
    CLASS_NAMES,
    build_rafdb_mask_loaders,
    resolve_rafdb_root,
)
from src.models import get_model  # noqa: E402
from src.training.losses import build_loss  # noqa: E402
from src.training.optimizer import build_optimizer, build_scheduler  # noqa: E402
from src.training.sam import SAM  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint does not contain a valid model state dict.")


def strip_known_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        cleaned[name] = value
    return cleaned


def resolve_checkpoint_path(checkpoint_path):
    if Path(checkpoint_path).exists():
        return checkpoint_path

    basename = Path(checkpoint_path).name
    search_roots = [Path.cwd()]
    if Path("/kaggle/input").exists():
        search_roots.insert(0, Path("/kaggle/input"))

    for root in search_roots:
        for current_dir, _, files in os.walk(root):
            if basename in files:
                found = str(Path(current_dir) / basename)
                print(f"--> Using discovered init checkpoint: {found}")
                return found

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_model_init_checkpoint(model, checkpoint_path, strict=True):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    print(f"--> Loading model init checkpoint: {checkpoint_path}")
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = strip_known_prefixes(extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if incompatible.missing_keys:
        print(f"--> Init checkpoint missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"--> Init checkpoint unexpected keys: {len(incompatible.unexpected_keys)}")
    print("--> Model init checkpoint loaded.")


def print_parameter_summary(model, label="Model"):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_percent = 100.0 * trainable_params / total_params if total_params else 0.0
    print(
        f"--> [{label}] Parameters: total={total_params:,}, "
        f"trainable={trainable_params:,} ({trainable_percent:.2f}%), "
        f"frozen={frozen_params:,}"
    )


def resolve_config_path(path):
    raw_path = Path(path)
    candidates = [raw_path]
    if raw_path.suffix == "":
        candidates.extend([raw_path.with_suffix(".yaml"), raw_path.with_suffix(".yml")])
    if not raw_path.is_absolute():
        candidates.extend(
            [
                PROJECT_ROOT / raw_path,
                PROJECT_ROOT / "configs" / raw_path,
                PROJECT_ROOT / "configs" / raw_path.with_suffix(".yaml"),
                PROJECT_ROOT / "configs" / raw_path.with_suffix(".yml"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Config file not found. Tried: {tried}")


def deep_update(base, override):
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(path):
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_name = config.pop("_base_", None)
    if base_name:
        base_path = (config_path.parent / base_name).resolve()
        if not base_path.exists():
            base_path = PROJECT_ROOT / "configs" / base_name
        config = deep_update(load_yaml_config(base_path), config)
    print(f"--> Config: {config_path}")
    return config


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


def supervised_loss(outputs, labels, criterion, config):
    logits = extract_logits(outputs)
    loss = criterion(logits, labels)
    aux_loss = None
    coarse_aux_loss = None

    if isinstance(outputs, tuple):
        model_cfg = config.get("model", {})
        ortho_weight = float(model_cfg.get("ortho_loss_weight", 0.1))
        aux_loss = outputs[1] if len(outputs) > 1 and torch.is_tensor(outputs[1]) and outputs[1].dim() == 0 else None
        coarse_logits = outputs[2] if len(outputs) > 2 and torch.is_tensor(outputs[2]) else None
        if aux_loss is not None:
            loss = loss + ortho_weight * aux_loss
        if coarse_logits is not None:
            coarse_weight = float(model_cfg.get("cnn_aux_loss_weight", 0.0))
            if coarse_weight > 0.0:
                coarse_aux_loss = criterion(coarse_logits, labels)
                loss = loss + coarse_weight * coarse_aux_loss

    return loss, logits, aux_loss, coarse_aux_loss


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    aux_sum = 0.0
    coarse_sum = 0.0
    grad_clip_norm = config.get("training", {}).get("grad_clip_norm")
    skip_nonfinite = bool(config.get("training", {}).get("skip_nonfinite_batches", True))

    for images, labels, region_masks in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        region_masks = region_masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(images, region_masks=region_masks)
        loss, logits, aux_loss, coarse_aux_loss = supervised_loss(outputs, labels, criterion, config)
        if not torch.isfinite(loss).item():
            if skip_nonfinite:
                continue
            raise FloatingPointError("Encountered non-finite training loss.")

        if isinstance(optimizer, SAM):
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm), error_if_nonfinite=False)
            optimizer.first_step(zero_grad=True)

            second_outputs = model(images, region_masks=region_masks)
            second_loss, _, _, _ = supervised_loss(second_outputs, labels, criterion, config)
            if not torch.isfinite(second_loss).item():
                if skip_nonfinite:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise FloatingPointError("Encountered non-finite SAM second-step loss.")
            second_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm), error_if_nonfinite=False)
            optimizer.second_step(zero_grad=True)
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm), error_if_nonfinite=False)
            optimizer.step()

        batch_size = labels.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        if aux_loss is not None:
            aux_sum += float(aux_loss.item()) * batch_size
        if coarse_aux_loss is not None:
            coarse_sum += float(coarse_aux_loss.item()) * batch_size

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "ortho_loss": aux_sum / max(total, 1),
        "coarse_aux_loss": coarse_sum / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

    for images, labels, region_masks in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        region_masks = region_masks.to(device, non_blocking=True)
        logits = extract_logits(model(images, region_masks=region_masks))
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    return {
        "split": split_name,
        "loss": total_loss / max(total, 1),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_class_f1],
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_metrics(output_dir, prefix, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_text = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    report_dict = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_f1": dict(zip(CLASS_NAMES, metrics["per_class_f1"])),
        "confusion_matrix": metrics["confusion_matrix"],
    }
    (output_dir / f"{prefix}_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / f"{prefix}_classification_report.txt").write_text(report_text, encoding="utf-8")

    with (output_dir / f"{prefix}_classification_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_or_average", "precision", "recall", "f1-score", "support"])
        for key, values in report_dict.items():
            if isinstance(values, dict):
                writer.writerow([key, values.get("precision"), values.get("recall"), values.get("f1-score"), values.get("support")])

    cm = np.array(metrics["confusion_matrix"], dtype=int)
    with (output_dir / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, cm):
            writer.writerow([name, *row.tolist()])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{prefix} confusion matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("--> matplotlib not installed; skipped confusion matrix PNG.")
    return summary


def save_history(output_dir, history):
    if not history:
        return
    json_path = output_dir / "training_history.json"
    csv_path = output_dir / "training_history.csv"
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def format_monitor_value(monitor, metrics):
    if monitor == "val_loss":
        return metrics["loss"]
    if monitor == "val_accuracy":
        return metrics["accuracy"]
    return metrics["macro_f1"]


def write_class_counts(output_dir, counts):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "class_counts.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "class_name", "train_raw", "internal_train", "val", "test"])
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow([CLASS_FOLDERS[idx], name, counts["train_raw"][idx], counts["internal_train"][idx], counts["val"][idx], counts["test"][idx]])
    return path


def main():
    parser = argparse.ArgumentParser(description="Train RAF-DB with mask-guided region attention.")
    parser.add_argument("--config", required=True, help="Path to RAF-DB mask-guided YAML config.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    seed = int(config.get("seed", {}).get("random_seed", config.get("seed", 42)))
    set_seed(seed)

    root = resolve_rafdb_root(config.get("data", {}).get("root", "auto"))
    output_dir = Path(config.get("paths", {}).get("output_dir", config.get("output_dir", "/kaggle/working/outputs/rafdb_mask_guided_c")))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"

    train_loader, val_loader, test_loader, class_to_idx, counts = build_rafdb_mask_loaders(config, root)
    counts_path = write_class_counts(output_dir, counts)
    print(f"--> Saved class counts: {counts_path}")
    print(f"--> ImageFolder class_to_idx: {class_to_idx}")

    requested_device = str(config.get("device", "auto")).lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("--> CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(f"--> Device: {device}")
    print(f"--> Output dir: {output_dir}")

    model = get_model(name=config["model"]["name"], config=config).to(device)
    init_checkpoint_path = config.get("training", {}).get("init_checkpoint_path")
    if init_checkpoint_path:
        init_strict = bool(config.get("training", {}).get("init_checkpoint_strict", True))
        load_model_init_checkpoint(model, init_checkpoint_path, strict=init_strict)
    print_parameter_summary(model)

    criterion = build_loss(config=config)
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 70))
    patience = int(train_cfg.get("patience", 15))
    monitor = str(train_cfg.get("monitor", "val_macro_f1"))
    if monitor not in ("val_macro_f1", "val_accuracy", "val_loss"):
        raise ValueError("training.monitor must be one of: val_macro_f1, val_accuracy, val_loss")

    best_score = float("inf") if monitor == "val_loss" else -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        base_model = model.module if hasattr(model, "module") else model
        if hasattr(base_model, "set_epoch"):
            base_model.set_epoch(epoch - 1)
        if hasattr(base_model, "check_unfreeze") and base_model.check_unfreeze(epoch - 1):
            finetune_lr = train_cfg.get("finetune_lr", train_cfg.get("lr"))
            visual_lr = train_cfg.get("visual_extractor_lr")
            old_lr = train_cfg.get("lr")
            train_cfg["lr"] = finetune_lr
            optimizer = build_optimizer(model=model, config=config)
            train_cfg["lr"] = old_lr
            scheduler = build_scheduler(optimizer=optimizer, config=config)
            print(f"--> Rebuilt optimizer after unfreeze: head_lr={finetune_lr}, visual_lr={visual_lr}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config)
        val_metrics = evaluate(model, val_loader, criterion, device, "val")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        if monitor == "val_loss":
            current_score = val_metrics["loss"]
            improved = current_score < best_score
        elif monitor == "val_accuracy":
            current_score = val_metrics["accuracy"]
            improved = current_score > best_score
        else:
            current_score = val_metrics["macro_f1"]
            improved = current_score > best_score

        if improved:
            best_score = current_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "class_to_idx": class_to_idx,
                    "class_names": CLASS_NAMES,
                    "best_score": float(best_score),
                    "monitor": monitor,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": float(lr),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_ortho_loss": train_metrics["ortho_loss"],
            "train_coarse_aux_loss": train_metrics["coarse_aux_loss"],
            "train_prior_align_loss": 0.0,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_score": float(best_score),
            "improved": int(improved),
        }
        history.append(row)
        save_history(output_dir, history)
        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {train_metrics['loss']:.4f} "
            f"(ortho: {train_metrics['ortho_loss']:.4f}, "
            f"coarse_aux: {train_metrics['coarse_aux_loss']:.4f}, "
            f"prior_align: 0.0000) - "
            f"accuracy: {train_metrics['accuracy']:.4f} - "
            f"val_loss: {val_metrics['loss']:.4f} - "
            f"val_accuracy: {val_metrics['accuracy']:.4f} - "
            f"val_macro_f1: {val_metrics['macro_f1']:.4f}"
        )
        if improved:
            print(
                f"\t--- Save best at ep {epoch}, "
                f"val_loss: {val_metrics['loss']:.4f}, "
                f"val_accuracy: {val_metrics['accuracy']:.4f}, "
                f"val_macro_f1: {val_metrics['macro_f1']:.4f}, "
                f"monitor: {monitor}, "
                f"score: {format_monitor_value(monitor, val_metrics):.4f}, "
                f"path: {best_path} ---"
            )
        else:
            print(f"\t-!- No improvement: {stale_epochs}/{patience}")

        if patience > 0 and stale_epochs >= patience:
            print(f"--> Early stopping after {patience} epochs without {monitor} improvement.")
            break

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    print(f"\n--> Loading best checkpoint for final one-time test: {best_path}")
    checkpoint = safe_torch_load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, "test")
    test_summary = save_metrics(output_dir, "test", test_metrics)

    manifest = {
        "config": args.config,
        "dataset_root": str(root),
        "output_dir": str(output_dir),
        "best_model": str(best_path),
        "best_epoch": int(best_epoch),
        "monitor": monitor,
        "best_score": float(best_score),
        "test_accuracy": test_summary["accuracy"],
        "test_macro_f1": test_summary["macro_f1"],
        "class_names": CLASS_NAMES,
        "class_to_idx": class_to_idx,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nFinal test report")
    print(f"best_epoch: {best_epoch}")
    print(f"test_accuracy: {test_summary['accuracy']:.6f}")
    print(f"test_macro_f1: {test_summary['macro_f1']:.6f}")
    print(f"--> Saved best checkpoint: {best_path}")
    print(f"--> Saved reports under: {output_dir}")


if __name__ == "__main__":
    main()
