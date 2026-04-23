import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.graph_vector_dataset import GraphVectorDataset
from features.graph_vectorizer import GraphVectorizer
from models.mlp_baseline import MLPBaseline
from training.trainer_mlp_baseline import train_one_epoch, evaluate


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(train_path, val_path, test_path, batch_size):
    vectorizer = GraphVectorizer(
        use_mean=True,
        use_std=True,
        use_max=True,
    )

    train_dataset = GraphVectorDataset(train_path, vectorizer)
    val_dataset = GraphVectorDataset(val_path, vectorizer)
    test_dataset = GraphVectorDataset(test_path, vectorizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    input_dim = train_dataset.get_input_dim()
    num_classes = 7

    return train_loader, val_loader, test_loader, input_dim, num_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_graphs", type=str, required=True)
    parser.add_argument("--val_graphs", type=str, required=True)
    parser.add_argument("--test_graphs", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, input_dim, num_classes = build_loaders(
        args.train_graphs,
        args.val_graphs,
        args.test_graphs,
        args.batch_size,
    )

    print(f"Input dim: {input_dim}")
    print(f"Num classes: {num_classes}")

    model = MLPBaseline(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=(64, 32),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_macro_f1 = -1.0
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"[Train] loss={train_metrics['loss']:.4f} | "
            f"acc={train_metrics['accuracy']:.4f} | "
            f"macro_f1={train_metrics['macro_f1']:.4f}"
        )

        print(
            f"[Val]   loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                },
                best_ckpt_path,
            )
            print(f"--> Saved best model to: {best_ckpt_path}")

    print("\n===== Load best model and evaluate on test =====")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"[Test] loss={test_metrics['loss']:.4f} | "
        f"acc={test_metrics['accuracy']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f} | "
        f"weighted_f1={test_metrics['weighted_f1']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])

    with open(os.path.join(args.save_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"loss: {test_metrics['loss']:.6f}\n")
        f.write(f"accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"macro_f1: {test_metrics['macro_f1']:.6f}\n")
        f.write(f"weighted_f1: {test_metrics['weighted_f1']:.6f}\n")
        f.write("confusion_matrix:\n")
        f.write(str(test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()