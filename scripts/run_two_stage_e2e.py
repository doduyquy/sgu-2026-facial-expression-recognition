#!/usr/bin/env python
import os
import sys
import glob
import argparse
import subprocess
import yaml
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Run 2-stage training in a single execution and merge histories.")
    parser.add_argument("--env", type=str, default="kaggle", choices=["local", "kaggle"])
    return parser.parse_args()

def find_latest_history_dir(output_dir, model_name):
    pattern = os.path.join(output_dir, "training_curves", f"{model_name}_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def merge_and_plot_histories(phase1_dir, phase2_dir, output_dir):
    phase1_json = os.path.join(phase1_dir, "training_history.json")
    phase2_json = os.path.join(phase2_dir, "training_history.json")
    
    if not os.path.exists(phase1_json) or not os.path.exists(phase2_json):
        print(f"[WARN] Could not find history files: {phase1_json} or {phase2_json}")
        return
        
    with open(phase1_json, "r", encoding="utf-8") as f:
        phase1_history = json.load(f)
    with open(phase2_json, "r", encoding="utf-8") as f:
        phase2_history = json.load(f)
        
    merged = []
    # Add phase 1 history
    for row in phase1_history:
        merged.append({
            "epoch": row["epoch"],
            "train_loss": row.get("train_loss"),
            "train_accuracy": row.get("train_accuracy"),
            "val_loss": row.get("val_loss"),
            "val_accuracy": row.get("val_accuracy"),
            "stage": "Stage 1 (Backbone)"
        })
        
    phase1_epochs = len(phase1_history)
    # Add phase 2 history (shifting epoch indices)
    for row in phase2_history:
        merged.append({
            "epoch": row["epoch"] + phase1_epochs,
            "train_loss": row.get("train_loss"),
            "train_accuracy": row.get("train_accuracy"),
            "val_loss": row.get("val_loss"),
            "val_accuracy": row.get("val_accuracy"),
            "stage": "Stage 2 (Attention)"
        })
        
    # Create output directory for merged results
    merged_dir = os.path.join(output_dir, "training_curves", "merged_two_stage")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(merged_dir, "training_history.csv")
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "stage"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
        
    # Save JSON
    json_path = os.path.join(merged_dir, "training_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        
    # Plotting
    epochs = [row["epoch"] for row in merged]
    train_losses = [row["train_loss"] for row in merged]
    val_losses = [row["val_loss"] for row in merged]
    train_accs = [row["train_accuracy"] for row in merged]
    val_accs = [row["val_accuracy"] for row in merged]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss Plot
    axes[0].plot(epochs, train_losses, marker="o", markersize=3, label="Train Loss", color="royalblue")
    axes[0].plot(epochs, val_losses, marker="x", markersize=3, label="Val Loss", color="orange")
    axes[0].axvline(x=phase1_epochs, color="red", linestyle="--", alpha=0.8, label="Stage 2 Start")
    axes[0].set_title("Loss History (2-Stage)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Accuracy Plot
    axes[1].plot(epochs, train_accs, marker="o", markersize=3, label="Train Accuracy", color="royalblue")
    axes[1].plot(epochs, val_accs, marker="x", markersize=3, label="Val Accuracy", color="orange")
    axes[1].axvline(x=phase1_epochs, color="red", linestyle="--", alpha=0.8, label="Stage 2 Start")
    axes[1].set_title("Accuracy History (2-Stage)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    fig.suptitle("Merged Two-Stage Training History (Stage 1: Backbone -> Stage 2: Attention Head)", fontsize=14)
    fig.tight_layout()
    
    plot_path = os.path.join(merged_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print("\n" + "="*60)
    print(">>> [SUCCESS] Merged two-stage history created!")
    print(f"--> CSV: {csv_path}")
    print(f"--> Plot: {plot_path}")
    print("="*60 + "\n")

def main():
    args = get_args()
    print(f"--> Starting End-to-End Two-Stage training in 1 run (env: {args.env})...")
    
    # 1. Check GPU count to decide DDP or single GPU
    import torch
    num_gpus = torch.cuda.device_count()
    print(f"--> Found {num_gpus} CUDA GPUs.")
    
    # Base command prefix
    if num_gpus >= 2:
        cmd_prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={num_gpus}", "-m", "scripts.train"]
    else:
        cmd_prefix = [sys.executable, "-m", "scripts.train"]
        
    # Phase 1: Train clean ConvNeXt-Tiny FER2013
    phase1_config = "two_stage_convnext_proposed/phase1_cnn_sam_from_start"
    phase1_cmd = cmd_prefix + ["--env", args.env, "--config", phase1_config]
    
    print("\n" + "="*60)
    print(">>> STAGE 1: Training clean ConvNeXt backbone on FER2013...")
    print("Command:", " ".join(phase1_cmd))
    print("="*60 + "\n")
    
    subprocess.check_call(phase1_cmd)
    
    # 2. Find the best checkpoint from Phase 1
    # Resolve output directory from config/env
    project_root = Path(__file__).resolve().parents[1]
    env_yaml_path = project_root / "configs" / "env.yaml"
    with open(env_yaml_path, "r", encoding="utf-8") as f:
        env_data = yaml.safe_load(f)
    output_dir = env_data[args.env].get("output_dir", "outputs")
    # Make absolute path if relative
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
        
    print(f"--> Looking for checkpoint in output_dir: {output_dir}")
    checkpoint_pattern = os.path.join(output_dir, "checkpoints", "convnext_tiny_fer2013", "*_best.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"Could not find any Phase 1 best checkpoints matching pattern: {checkpoint_pattern}")
        
    best_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    # Convert backslashes for windows compatibility in YAML
    best_checkpoint_path = Path(best_checkpoint).as_posix()
    print(f"--> [SUCCESS] Found best Phase 1 checkpoint: {best_checkpoint_path}")
    
    # 3. Create temporary config for Phase 2 pointing to the checkpoint
    configs_dir = project_root / "configs"
    phase2_temp_name = "two_stage_convnext_proposed/phase2_temp.yaml"
    phase2_temp_path = configs_dir / phase2_temp_name
    
    temp_config_content = {
        "_base_": "phase2_proposed_from_phase1_cnn_stable.yaml",
        "model": {
            "checkpoint_path": best_checkpoint_path
        }
    }
    
    print(f"--> Generating temporary config for Phase 2: {phase2_temp_path}")
    phase2_temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(phase2_temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(temp_config_content, f)
        
    # Phase 2: Train Proposed Attention Module
    phase2_cmd = cmd_prefix + ["--env", args.env, "--config", phase2_temp_name.replace(".yaml", "")]
    
    print("\n" + "="*60)
    print(">>> STAGE 2: Training Proposed Attention Module on top of Phase 1 backbone...")
    print("Command:", " ".join(phase2_cmd))
    print("="*60 + "\n")
    
    try:
        subprocess.check_call(phase2_cmd)
        print("\n" + "="*60)
        print(">>> [SUCCESS] End-to-End 2-Stage training completed successfully!")
        print("="*60 + "\n")
    finally:
        # Clean up temporary config file
        if phase2_temp_path.exists():
            print(f"--> Cleaning up temporary config: {phase2_temp_path}")
            os.remove(phase2_temp_path)
            
    # 4. Find the latest history folders and merge them
    print("--> Finding training histories to merge...")
    p1_hist_dir = find_latest_history_dir(output_dir, "convnext_tiny_fer2013")
    p2_hist_dir = find_latest_history_dir(output_dir, "convnext_tiny_mask_guided_region_attention")
    
    if p1_hist_dir and p2_hist_dir:
        print(f"--> Found Phase 1 history in: {p1_hist_dir}")
        print(f"--> Found Phase 2 history in: {p2_hist_dir}")
        merge_and_plot_histories(p1_hist_dir, p2_hist_dir, output_dir)
    else:
        print("[WARN] Could not locate history directories for merging.")

if __name__ == "__main__":
    main()
