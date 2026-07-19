import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


RAFDB_FOLDER_TO_NAME = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}
CLASS_FOLDERS = [str(i) for i in range(1, 8)]
CLASS_NAMES = [RAFDB_FOLDER_TO_NAME[str(i)] for i in range(1, 8)]


def resolve_config_path(path):
    raw_path = Path(path)
    candidates = [raw_path]
    if raw_path.suffix == "":
        candidates.extend(
            [
                raw_path.with_suffix(".yaml"),
                raw_path.with_suffix(".yml"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Config file not found. Tried: {tried}")


def load_yaml(path):
    config_path = resolve_config_path(path)
    print(f"--> Config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_dataset_root(config):
    raw_root = str(config.get("data", {}).get("root", "auto"))
    if raw_root.lower() != "auto":
        root = Path(raw_root)
        if not root.exists():
            raise FileNotFoundError(f"Configured RAF-DB root does not exist: {root}")
        return root

    candidates = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.glob("*/DATASET"))
        candidates.extend(input_root.glob("*/*/DATASET"))

    candidates.extend(Path.cwd().glob("*/DATASET"))
    candidates.extend(Path.cwd().glob("DATASET"))

    valid = [path for path in candidates if (path / "train").is_dir() and (path / "test").is_dir()]
    if not valid:
        searched = "/kaggle/input/*/DATASET, /kaggle/input/*/*/DATASET, ./DATASET"
        raise FileNotFoundError(f"Could not auto-find RAF-DB DATASET root. Searched: {searched}")

    valid = sorted(set(path.resolve() for path in valid), key=lambda p: str(p))
    print(f"--> Auto-found RAF-DB root: {valid[0]}")
    return valid[0]


def validate_imagefolder_classes(dataset, split_name):
    classes = list(dataset.classes)
    if classes != CLASS_FOLDERS:
        raise ValueError(
            f"{split_name} folders must be exactly {CLASS_FOLDERS}, got {classes}. "
            "RAF-DB folder ids are part of the label mapping."
        )


def build_transforms(config):
    data_cfg = config.get("data", {})
    train_aug_cfg = config.get("augmentation", {}).get("train", {})
    image_size = int(data_cfg.get("image_size", 224))
    resize_size = int(data_cfg.get("resize_size", max(image_size, int(round(image_size * 1.14)))))
    mean = data_cfg.get("mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("std", [0.229, 0.224, 0.225])

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if not bool(train_aug_cfg.get("enabled", True)):
        return eval_transform, eval_transform

    ops = [
        transforms.RandomResizedCrop(
            image_size,
            scale=tuple(train_aug_cfg.get("random_resized_crop_scale", [0.8, 1.0])),
        ),
        transforms.RandomHorizontalFlip(p=float(train_aug_cfg.get("hflip_prob", 0.5))),
    ]
    if float(train_aug_cfg.get("rotation_degrees", 0)) > 0:
        ops.append(transforms.RandomRotation(float(train_aug_cfg["rotation_degrees"])))
    if train_aug_cfg.get("color_jitter", None):
        jitter = train_aug_cfg["color_jitter"]
        ops.append(
            transforms.ColorJitter(
                brightness=float(jitter.get("brightness", 0.0)),
                contrast=float(jitter.get("contrast", 0.0)),
                saturation=float(jitter.get("saturation", 0.0)),
                hue=float(jitter.get("hue", 0.0)),
            )
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if float(train_aug_cfg.get("random_erasing_prob", 0.0)) > 0:
        ops.append(transforms.RandomErasing(p=float(train_aug_cfg["random_erasing_prob"])))

    return transforms.Compose(ops), eval_transform


def count_by_target(targets):
    counts = Counter(int(target) for target in targets)
    return [counts.get(idx, 0) for idx in range(len(CLASS_NAMES))]


def write_class_counts(output_dir, train_counts, internal_train_counts, val_counts, test_counts):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "class_counts.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "class_name", "train_raw", "internal_train", "val", "test"])
        for idx, name in enumerate(CLASS_NAMES):
            writer.writerow(
                [
                    CLASS_FOLDERS[idx],
                    name,
                    train_counts[idx],
                    internal_train_counts[idx],
                    val_counts[idx],
                    test_counts[idx],
                ]
            )
    return path


def print_class_counts(train_counts, internal_train_counts, val_counts, test_counts):
    print("\nClass counts")
    print("folder  class       train_raw  internal_train  val  test")
    for idx, name in enumerate(CLASS_NAMES):
        print(
            f"{CLASS_FOLDERS[idx]:>6}  {name:<10}  "
            f"{train_counts[idx]:>9}  {internal_train_counts[idx]:>14}  "
            f"{val_counts[idx]:>3}  {test_counts[idx]:>4}"
        )


def build_datasets(config, root):
    train_transform, eval_transform = build_transforms(config)
    train_dir = root / "train"
    test_dir = root / "test"

    split_source = ImageFolder(train_dir)
    validate_imagefolder_classes(split_source, "train")
    test_dataset = ImageFolder(test_dir, transform=eval_transform)
    validate_imagefolder_classes(test_dataset, "test")

    targets = np.array(split_source.targets)
    indices = np.arange(len(targets))
    val_fraction = float(config.get("data", {}).get("val_fraction", 0.1))
    seed = int(config.get("seed", 42))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )

    train_dataset_aug = ImageFolder(train_dir, transform=train_transform)
    val_dataset_eval = ImageFolder(train_dir, transform=eval_transform)
    validate_imagefolder_classes(train_dataset_aug, "train")
    validate_imagefolder_classes(val_dataset_eval, "validation")

    train_subset = Subset(train_dataset_aug, train_indices.tolist())
    val_subset = Subset(val_dataset_eval, val_indices.tolist())

    counts = {
        "train_raw": count_by_target(targets),
        "internal_train": count_by_target(targets[train_indices]),
        "val": count_by_target(targets[val_indices]),
        "test": count_by_target(test_dataset.targets),
    }

    return train_subset, val_subset, test_dataset, split_source.class_to_idx, counts


def build_loaders(config, train_dataset, val_dataset, test_dataset):
    data_cfg = config.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(data_cfg.get("drop_last", False)),
    )
    eval_loader_kwargs = {
        "batch_size": int(data_cfg.get("eval_batch_size", batch_size)),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return train_loader, DataLoader(val_dataset, **eval_loader_kwargs), DataLoader(test_dataset, **eval_loader_kwargs)


def set_child(parent, child_name, child):
    if isinstance(parent, nn.Sequential) and str(child_name).isdigit():
        parent[int(child_name)] = child
    else:
        setattr(parent, child_name, child)


def find_last_linear(module):
    found = None
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            found = (module, name, child)
        deeper = find_last_linear(child)
        if deeper is not None:
            found = deeper
    return found


def resolve_weights(arch, model_cfg):
    if not bool(model_cfg.get("pretrained", False)):
        return None
    weights_name = model_cfg.get("weights", "DEFAULT")
    if weights_name in (None, "none", "None", False):
        return None
    if hasattr(models, "get_model_weights"):
        weights_enum = models.get_model_weights(arch)
        return getattr(weights_enum, weights_name)
    return weights_name


def build_model(config):
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch", "resnet18")
    builder = getattr(models, arch, None)
    if builder is None:
        raise ValueError(f"torchvision.models has no architecture named '{arch}'")

    weights = resolve_weights(arch, model_cfg)
    loaded_pretrained = weights is not None
    try:
        model = builder(weights=weights)
    except Exception as exc:
        if not bool(model_cfg.get("fallback_to_random_weights", True)) or weights is None:
            raise
        print(f"--> Could not load torchvision weights ({exc}). Falling back to random init.")
        model = builder(weights=None)
        loaded_pretrained = False

    found = find_last_linear(model)
    if found is None:
        raise ValueError(f"Could not find final Linear layer in {arch}")
    parent, child_name, old_linear = found
    set_child(parent, child_name, nn.Linear(old_linear.in_features, len(CLASS_NAMES)))
    print(f"--> Model: {arch}, pretrained={loaded_pretrained}, num_classes={len(CLASS_NAMES)}")
    return model


def build_optimizer_and_scheduler(config, model):
    train_cfg = config.get("training", {})
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "reduce_lr_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 3)),
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 10)),
            gamma=float(train_cfg.get("gamma", 0.1)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("epochs", 30)),
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, show_progress=False):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in tqdm(loader, desc="train", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name, show_progress=False):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, desc=split_name, leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
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
        "loss": total_loss / max(total, 1),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_class_f1],
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_checkpoint(path, model, optimizer, epoch, config, class_to_idx, val_metrics):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "class_to_idx": class_to_idx,
        "folder_to_class_name": RAFDB_FOLDER_TO_NAME,
        "best_val_macro_f1": float(val_metrics["macro_f1"]),
        "best_val_accuracy": float(val_metrics["accuracy"]),
    }
    torch.save(payload, path)


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
                writer.writerow(
                    [
                        key,
                        values.get("precision"),
                        values.get("recall"),
                        values.get("f1-score"),
                        values.get("support"),
                    ]
                )

    cm = np.array(metrics["confusion_matrix"], dtype=int)
    with (output_dir / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, cm):
            writer.writerow([name, *row.tolist()])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{prefix} confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close(fig)
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


def maybe_validate_mask_config(config):
    mask_cfg = config.get("mask", {})
    if bool(mask_cfg.get("enabled", False)):
        raise NotImplementedError(
            "This RAF-DB ImageFolder baseline intentionally does not load masks. "
            "Set mask.enabled=false, or create a separate paired image+mask dataset after precomputing RAF-DB masks."
        )
    print("--> Mask: disabled. ImageFolder baseline trains on RGB images only.")


def main():
    parser = argparse.ArgumentParser(description="Train RAF-DB from DATASET/train/1..7 with ImageFolder.")
    parser.add_argument("--config", required=True, help="Path to RAF-DB YAML config.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    maybe_validate_mask_config(config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    root = resolve_dataset_root(config)
    output_dir = Path(config.get("output_dir", "/kaggle/working/outputs/rafdb_imagefolder"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"

    train_dataset, val_dataset, test_dataset, class_to_idx, counts = build_datasets(config, root)
    print_class_counts(counts["train_raw"], counts["internal_train"], counts["val"], counts["test"])
    counts_path = write_class_counts(
        output_dir,
        counts["train_raw"],
        counts["internal_train"],
        counts["val"],
        counts["test"],
    )
    print(f"--> Saved class counts: {counts_path}")
    print(f"--> ImageFolder class_to_idx: {class_to_idx}")

    train_loader, val_loader, test_loader = build_loaders(config, train_dataset, val_dataset, test_dataset)
    requested_device = str(config.get("device", "auto")).lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("--> CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(f"--> Device: {device}")
    print(f"--> Output dir: {output_dir}")

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("training", {}).get("label_smoothing", 0.0)))
    optimizer, scheduler = build_optimizer_and_scheduler(config, model)
    use_amp = bool(config.get("training", {}).get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})
    show_progress = bool(log_cfg.get("progress_bar", False))
    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("patience", 8))
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            show_progress=show_progress,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, "val", show_progress=show_progress)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["macro_f1"])
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, config, class_to_idx, val_metrics)
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_val_macro_f1": best_macro_f1,
            "improved": int(improved),
        }
        history.append(row)
        save_history(output_dir, history)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {train_metrics['loss']:.4f}  "
            f"accuracy: {train_metrics['accuracy']:.4f} - "
            f"val_loss: {val_metrics['loss']:.4f} - "
            f"val_accuracy: {val_metrics['accuracy']:.4f}"
        )

        if patience > 0 and stale_epochs >= patience:
            print(f"--> Early stopping after {patience} epochs without validation macro-F1 improvement.")
            break

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    print(f"\n--> Loading best checkpoint for final one-time test: {best_path}")
    checkpoint = safe_torch_load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, "test", show_progress=show_progress)
    test_summary = save_metrics(output_dir, "test", test_metrics)

    manifest = {
        "config": args.config,
        "dataset_root": str(root),
        "output_dir": str(output_dir),
        "best_model": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_macro_f1),
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
    print("per_class_f1:")
    for class_name, value in test_summary["per_class_f1"].items():
        print(f"  {class_name:<10}: {value:.6f}")
    print("confusion_matrix:")
    print(np.array(test_summary["confusion_matrix"], dtype=int))
    print(f"--> Saved best checkpoint: {best_path}")
    print(f"--> Saved reports under: {output_dir}")


if __name__ == "__main__":
    main()
