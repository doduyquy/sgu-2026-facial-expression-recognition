#!/usr/bin/env python
import argparse
import csv
import json
import re
from html import escape
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge phase-1 and phase-2 training histories and plot end-to-end train/validation curves."
    )
    parser.add_argument("--phase1-history", default=None, help="Phase 1 training_history.json/csv")
    parser.add_argument("--phase2-history", default=None, help="Phase 2 training_history.json/csv")
    parser.add_argument("--phase1-log", default=None, help="Phase 1 Kaggle/stdout log txt")
    parser.add_argument("--phase2-log", default=None, help="Phase 2 Kaggle/stdout log txt")
    parser.add_argument(
        "--output-dir",
        default="outputs/training_curves/merged_two_stage",
        help="Directory where merged CSV/JSON/PNG files will be saved.",
    )
    return parser.parse_args()


def load_history(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported history file: {path}. Use .json or .csv")

    cleaned = []
    for idx, row in enumerate(rows, start=1):
        cleaned.append(
            {
                "epoch": int(float(row.get("epoch", idx))),
                "train_loss": to_float(row.get("train_loss")),
                "train_accuracy": to_float(row.get("train_accuracy")),
                "val_loss": to_float(row.get("val_loss")),
                "val_accuracy": to_float(row.get("val_accuracy")),
            }
        )
    return cleaned


def load_log_history(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    pattern = re.compile(
        r"Epoch\s+(\d+)\s*/\s*(\d+)\s*-\s*"
        r"loss:\s*([0-9.eE+-]+).*?"
        r"accuracy:\s*([0-9.eE+-]+)\s*-\s*"
        r"val_loss:\s*([0-9.eE+-]+)\s*-\s*"
        r"val_accuracy:\s*([0-9.eE+-]+)"
    )

    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        epoch, _, train_loss, train_acc, val_loss, val_acc = match.groups()
        rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )

    if not rows:
        raise ValueError(f"No epoch metrics found in log: {path}")
    return rows


def load_history_or_log(history_path, log_path, label):
    if history_path:
        return load_history(history_path)
    if log_path:
        return load_log_history(log_path)
    raise ValueError(f"Provide either --{label}-history or --{label}-log.")


def to_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def merge_histories(phase1_history, phase2_history):
    merged = []
    for row in phase1_history:
        merged.append({**row, "epoch": len(merged) + 1, "stage": "Phase 1 CNN ImageNet"})
    for row in phase2_history:
        merged.append({**row, "epoch": len(merged) + 1, "stage": "Phase 2 Attention"})
    return merged


def save_table(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_history.csv"
    json_path = output_dir / "training_history.json"
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "stage"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path


def plot_curves(rows, phase1_len, output_dir):
    epochs = [row["epoch"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]
    train_accs = [row["train_accuracy"] for row in rows]
    val_accs = [row["val_accuracy"] for row in rows]

    if plt is None:
        return plot_curves_svg(rows, phase1_len, output_dir)

    has_acc = all(value is not None for value in train_accs + val_accs)
    if has_acc:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        loss_ax, acc_ax = axes
    else:
        fig, loss_ax = plt.subplots(1, 1, figsize=(8, 6))
        acc_ax = None

    loss_ax.plot(epochs, train_losses, marker="o", markersize=3, label="Train loss", color="royalblue")
    loss_ax.plot(epochs, val_losses, marker="x", markersize=3, label="Validation loss", color="darkorange")
    loss_ax.axvline(x=phase1_len + 0.5, color="red", linestyle="--", alpha=0.8, label="Phase 2 start")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    if acc_ax is not None:
        acc_ax.plot(epochs, train_accs, marker="o", markersize=3, label="Train accuracy", color="seagreen")
        acc_ax.plot(epochs, val_accs, marker="x", markersize=3, label="Validation accuracy", color="crimson")
        acc_ax.axvline(x=phase1_len + 0.5, color="red", linestyle="--", alpha=0.8, label="Phase 2 start")
        acc_ax.set_title("Accuracy")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.grid(True, alpha=0.3)
        acc_ax.legend()

    fig.suptitle("Two-Stage Training Curves")
    fig.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_curves_svg(rows, phase1_len, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    width = 1400
    height = 620
    margin_left = 72
    margin_right = 28
    margin_top = 64
    margin_bottom = 72
    gap = 70
    panel_width = (width - margin_left - margin_right - gap) / 2
    panel_height = height - margin_top - margin_bottom

    epochs = [row["epoch"] for row in rows]
    series = {
        "Train loss": ([row["train_loss"] for row in rows], "#4169e1"),
        "Validation loss": ([row["val_loss"] for row in rows], "#d97706"),
        "Train accuracy": ([row["train_accuracy"] for row in rows], "#2e8b57"),
        "Validation accuracy": ([row["val_accuracy"] for row in rows], "#dc143c"),
    }

    def clean(values):
        return [float(value) for value in values if value is not None]

    def scale_x(epoch, x0):
        if len(epochs) == 1:
            return x0 + panel_width / 2
        return x0 + (epoch - min(epochs)) / (max(epochs) - min(epochs)) * panel_width

    def scale_y(value, y0, y_min, y_max):
        if y_max == y_min:
            return y0 + panel_height / 2
        return y0 + panel_height - (value - y_min) / (y_max - y_min) * panel_height

    def polyline(values, x0, y0, y_min, y_max):
        points = []
        for epoch, value in zip(epochs, values):
            if value is None:
                continue
            points.append(f"{scale_x(epoch, x0):.2f},{scale_y(value, y0, y_min, y_max):.2f}")
        return " ".join(points)

    def draw_panel(title, y_label, names, x0):
        y0 = margin_top
        values = []
        for name in names:
            values.extend(clean(series[name][0]))
        y_min = min(values)
        y_max = max(values)
        pad = max((y_max - y_min) * 0.08, 0.001)
        y_min -= pad
        y_max += pad

        items = []
        items.append(f'<text x="{x0 + panel_width / 2:.1f}" y="34" text-anchor="middle" class="title">{escape(title)}</text>')
        items.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" class="panel"/>')
        for tick in range(6):
            frac = tick / 5
            y = y0 + panel_height - frac * panel_height
            value = y_min + frac * (y_max - y_min)
            items.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_width:.1f}" y2="{y:.1f}" class="grid"/>')
            items.append(f'<text x="{x0 - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.3f}</text>')

        for tick in range(0, len(epochs), max(1, len(epochs) // 6)):
            epoch = epochs[tick]
            x = scale_x(epoch, x0)
            items.append(f'<line x1="{x:.1f}" y1="{y0 + panel_height:.1f}" x2="{x:.1f}" y2="{y0 + panel_height + 5:.1f}" class="axis"/>')
            items.append(f'<text x="{x:.1f}" y="{y0 + panel_height + 23:.1f}" text-anchor="middle" class="tick">{epoch}</text>')

        boundary_x = scale_x(phase1_len + 0.5, x0)
        items.append(
            f'<line x1="{boundary_x:.1f}" y1="{y0:.1f}" x2="{boundary_x:.1f}" y2="{y0 + panel_height:.1f}" class="phase"/>'
        )
        items.append(
            f'<text x="{boundary_x + 7:.1f}" y="{y0 + 16:.1f}" class="phase-label">Phase 2 start</text>'
        )

        legend_y = y0 + panel_height + 48
        legend_x = x0
        for name in names:
            color = series[name][1]
            values_for_name = series[name][0]
            items.append(
                f'<polyline points="{polyline(values_for_name, x0, y0, y_min, y_max)}" '
                f'fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            items.append(f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 24:.1f}" y2="{legend_y:.1f}" stroke="{color}" stroke-width="3"/>')
            items.append(f'<text x="{legend_x + 31:.1f}" y="{legend_y + 4:.1f}" class="legend">{escape(name)}</text>')
            legend_x += 180

        items.append(f'<text x="{x0 + panel_width / 2:.1f}" y="{height - 14}" text-anchor="middle" class="axis-label">Epoch</text>')
        items.append(
            f'<text transform="translate({x0 - 52:.1f},{y0 + panel_height / 2:.1f}) rotate(-90)" '
            f'text-anchor="middle" class="axis-label">{escape(y_label)}</text>'
        )
        return "\n".join(items)

    left_x = margin_left
    right_x = margin_left + panel_width + gap
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 20px Arial, sans-serif; fill: #111827; }}
  .panel {{ fill: #ffffff; stroke: #d1d5db; stroke-width: 1; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .axis {{ stroke: #6b7280; stroke-width: 1; }}
  .tick {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
  .legend {{ font: 13px Arial, sans-serif; fill: #111827; }}
  .axis-label {{ font: 13px Arial, sans-serif; fill: #374151; }}
  .phase {{ stroke: #ef4444; stroke-width: 2; stroke-dasharray: 7 5; }}
  .phase-label {{ font: 12px Arial, sans-serif; fill: #ef4444; }}
</style>
<rect width="100%" height="100%" fill="#f9fafb"/>
<text x="{width / 2:.1f}" y="24" text-anchor="middle" class="title">Two-Stage Training Curves From Kaggle Logs</text>
{draw_panel("Loss", "Loss", ["Train loss", "Validation loss"], left_x)}
{draw_panel("Accuracy", "Accuracy", ["Train accuracy", "Validation accuracy"], right_x)}
</svg>
"""
    svg_path = output_dir / "training_curves.svg"
    svg_path.write_text(svg, encoding="utf-8")
    return svg_path


def main():
    args = get_args()
    phase1_history = load_history_or_log(args.phase1_history, args.phase1_log, "phase1")
    phase2_history = load_history_or_log(args.phase2_history, args.phase2_log, "phase2")
    merged = merge_histories(phase1_history, phase2_history)

    output_dir = Path(args.output_dir)
    csv_path, json_path = save_table(merged, output_dir)
    plot_path = plot_curves(merged, len(phase1_history), output_dir)

    print("Merged two-stage training curves created.")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
