import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataloader import build_dataloader
from src.evaluation.evaluator import evaluate_and_show
from src.models import get_model
from src.training.losses import build_loss
from src.utils.config import load_config
from src.utils.seed import set_seed


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        param.requires_grad = trainable

    for param in model.model.fc.parameters():
        param.requires_grad = True


def reset_classifier(model):
    old_fc = model.model.fc
    model.model.fc = nn.Linear(old_fc.in_features, old_fc.out_features)
    nn.init.normal_(model.model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.model.fc.bias)
    print("--> Reset classifier head.")


def build_finetune_optimizer(model, backbone_lr, classifier_lr, weight_decay):
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("model.fc."):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    desc = "Train" if train else "Val"

    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to RMN ResNet152 checkpoint")
    parser.add_argument("--config", type=str, default="resnet152_eval")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--reset-classifier", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.env)
    config["model"]["name"] = "resnet152"
    config["data"]["image_size"] = 224
    config["data"]["channels"] = 3
    config["data"]["normalize"] = False
    config["data"]["batch_size"] = args.batch_size
    config["training"]["epochs"] = args.epochs
    config["training"]["loss"] = "cross_entropy"
    config["training"]["label_smoothing"] = args.label_smoothing
    config["logging"]["use_wandb"] = False

    set_seed(config["seed"].get("random_seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    data_path = config.get("data_path", "dataset/fer13-split")
    if args.env == "local":
        data_path = os.path.abspath(data_path)
    print(f"--> Data path: {data_path}")

    train_loader, val_loader, test_loader = build_dataloader(config, data_path)

    model = get_model(name="resnet152", config=config).to(device)
    model.load_from_checkpoint(args.ckpt, device)

    if args.reset_classifier:
        reset_classifier(model)
        model.to(device)

    criterion = build_loss(config).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", "checkpoints", "resnet152_finetune")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"resnet152_finetune_{timestamp}_best.pth")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    optimizer = None
    current_phase = None

    for epoch in range(args.epochs):
        freeze_backbone = epoch < args.freeze_epochs
        phase = "head-only" if freeze_backbone else "full-finetune"
        if phase != current_phase:
            set_backbone_trainable(model, trainable=not freeze_backbone)
            optimizer = build_finetune_optimizer(
                model,
                backbone_lr=args.backbone_lr,
                classifier_lr=args.classifier_lr,
                weight_decay=args.weight_decay,
            )
            current_phase = phase
            print(f"--> Switch to phase: {phase}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch + 1}/{args.epochs} [{phase}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        is_best = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": config,
                    "source_checkpoint": args.ckpt,
                },
                best_path,
            )
            print(f"--> Saved best checkpoint: {best_path}")

    print(f"\nBest val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")

    if not args.no_eval:
        print("\n" + "=" * 50)
        print("Evaluate best checkpoint on test set")
        print("=" * 50)
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        eval_dir = os.path.join("outputs", "evaluation_resnet152_finetune")
        os.makedirs(eval_dir, exist_ok=True)
        evaluate_and_show(model, test_loader, os.path.join(data_path, "test.csv"), device, eval_dir)


if __name__ == "__main__":
    main()
