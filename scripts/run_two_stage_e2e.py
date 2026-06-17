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
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "phase1", "phase2"],
        help="Run both stages, only phase 1, or only phase 2 from a saved phase-1 checkpoint.",
    )
    parser.add_argument(
        "--phase1-config",
        type=str,
        default="two_stage_convnext_proposed/phase1_cnn_sam_from_start",
        help="Config used for the clean CNN phase.",
    )
    parser.add_argument(
        "--phase2-base-config",
        type=str,
        default="phase2_proposed_from_phase1_cnn_stable.yaml",
        help="Base config file inside configs/two_stage_convnext_proposed for phase 2.",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default=None,
        help="Best phase-1 checkpoint to initialize phase 2. If omitted, the latest local phase-1 checkpoint is used.",
    )
    parser.add_argument(
        "--phase1-history-json",
        type=str,
        default=None,
        help="Phase-1 training_history.json used to create the merged two-stage plot in a later Kaggle session.",
    )
    parser.add_argument(
        "--phase2-temp-name",
        type=str,
        default="two_stage_convnext_proposed/phase2_temp.yaml",
        help="Temporary phase-2 config path relative to configs/.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merged history creation after phase 2.",
    )
    return parser.parse_args()

def find_latest_history_dir(output_dir, model_name):
    pattern = os.path.join(output_dir, "training_curves", f"{model_name}_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def history_json_from_dir(history_dir):
    if not history_dir:
        return None
    history_json = os.path.join(history_dir, "training_history.json")
    return history_json if os.path.exists(history_json) else None

def resolve_output_dir(project_root, env):
    env_yaml_path = project_root / "configs" / "env.yaml"
    with open(env_yaml_path, "r", encoding="utf-8") as f:
        env_data = yaml.safe_load(f)
    output_dir = env_data[env].get("output_dir", "outputs")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    return output_dir

def find_latest_phase1_checkpoint(output_dir):
    checkpoint_pattern = os.path.join(output_dir, "checkpoints", "convnext_tiny_fer2013", "*_best.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(
            "Could not find any Phase 1 best checkpoints. "
            f"Pattern: {checkpoint_pattern}. "
            "If this is a fresh Kaggle session, pass --phase1-checkpoint pointing to the checkpoint dataset path."
        )
    return max(checkpoint_files, key=os.path.getmtime)

def merge_and_plot_history_jsons(phase1_json, phase2_json, output_dir):
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

def merge_and_plot_histories(phase1_dir, phase2_dir, output_dir):
    phase1_json = history_json_from_dir(phase1_dir)
    phase2_json = history_json_from_dir(phase2_dir)
    if not phase1_json or not phase2_json:
        print(f"[WARN] Could not find history files in: {phase1_dir} or {phase2_dir}")
        return
    merge_and_plot_history_jsons(phase1_json, phase2_json, output_dir)

def main():
    args = get_args()
    print(f"--> Starting Two-Stage training runner (mode: {args.mode}, env: {args.env})...")
    
    # 1. Check GPU count to decide DDP or single GPU
    import torch
    num_gpus = torch.cuda.device_count()
    print(f"--> Found {num_gpus} CUDA GPUs.")
    
    # Base command prefix
    if num_gpus >= 2:
        cmd_prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={num_gpus}", "-m", "scripts.train"]
    else:
        cmd_prefix = [sys.executable, "-m", "scripts.train"]

    project_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_output_dir(project_root, args.env)
        
    # Phase 1: Train clean ConvNeXt-Tiny FER2013
    phase1_config = args.phase1_config
    phase1_cmd = cmd_prefix + ["--env", args.env, "--config", phase1_config]
    
    if args.mode in {"full", "phase1"}:
        print("\n" + "="*60)
        print(">>> STAGE 1: Training clean ConvNeXt backbone on FER2013...")
        print("Command:", " ".join(phase1_cmd))
        print("="*60 + "\n")
        
        subprocess.check_call(phase1_cmd)

        if args.mode == "phase1":
            best_checkpoint = find_latest_phase1_checkpoint(output_dir)
            phase1_history_dir = find_latest_history_dir(output_dir, "convnext_tiny_fer2013")
            phase1_history_json = history_json_from_dir(phase1_history_dir)
            print("\n" + "="*60)
            print(">>> [SUCCESS] Phase 1 completed.")
            print(f"--> Best Phase 1 checkpoint: {Path(best_checkpoint).as_posix()}")
            if phase1_history_json:
                print(f"--> Phase 1 history JSON: {Path(phase1_history_json).as_posix()}")
                print("--> Save both the checkpoint and this JSON as a Kaggle dataset for a merged plot in phase 2.")
            else:
                print("--> [WARN] Phase 1 history JSON was not found.")
            print("="*60 + "\n")
            return
    
    # 2. Find the best checkpoint from Phase 1
    if args.phase1_checkpoint:
        best_checkpoint = args.phase1_checkpoint
        print(f"--> Using provided Phase 1 checkpoint: {best_checkpoint}")
    else:
        print(f"--> Looking for checkpoint in output_dir: {output_dir}")
        best_checkpoint = find_latest_phase1_checkpoint(output_dir)

    # Convert backslashes for windows compatibility in YAML
    best_checkpoint_path = Path(best_checkpoint).as_posix()
    print(f"--> [SUCCESS] Found best Phase 1 checkpoint: {best_checkpoint_path}")
    
    # 3. Create temporary config for Phase 2 pointing to the checkpoint
    configs_dir = project_root / "configs"
    phase2_temp_name = args.phase2_temp_name
    phase2_temp_path = configs_dir / phase2_temp_name
    
    temp_config_content = {
        "_base_": args.phase2_base_config,
        "model": {
            "checkpoint_path": best_checkpoint_path
        }
    }
    
    print(f"--> Generating temporary config for Phase 2: {phase2_temp_path}")
    phase2_temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(phase2_temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(temp_config_content, f)
        
    # Phase 2: Train Proposed Attention Module
    phase2_config_arg = phase2_temp_name[:-5] if phase2_temp_name.endswith(".yaml") else phase2_temp_name
    phase2_cmd = cmd_prefix + ["--env", args.env, "--config", phase2_config_arg]
    
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

    if args.no_merge:
        return
            
    # 4. Find the latest history folders and merge them
    print("--> Finding training histories to merge...")
    p1_hist_dir = find_latest_history_dir(output_dir, "convnext_tiny_fer2013")
    p2_hist_dir = find_latest_history_dir(output_dir, "convnext_tiny_mask_guided_region_attention")
    p1_history_json = args.phase1_history_json or history_json_from_dir(p1_hist_dir)
    p2_history_json = history_json_from_dir(p2_hist_dir)
    
    if p1_history_json and p2_history_json:
        print(f"--> Found Phase 1 history JSON: {p1_history_json}")
        print(f"--> Found Phase 2 history in: {p2_hist_dir}")
        merge_and_plot_history_jsons(p1_history_json, p2_history_json, output_dir)
    else:
        print("[WARN] Could not locate history JSON files for merging.")
        print("[WARN] Phase 2 still has its own training_curves.png under outputs/training_curves/.")

if __name__ == "__main__":
    main()
