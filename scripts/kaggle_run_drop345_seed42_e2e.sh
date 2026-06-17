#!/usr/bin/env bash
set -e

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

REPO_DIR="/kaggle/working/sgu-2026-facial-expression-recognition"
if [ -d "$REPO_DIR" ]; then
  cd "$REPO_DIR"
else
  echo "[INFO] $REPO_DIR not found, using current directory: $(pwd)"
fi

if [ ! -f "scripts/train.py" ]; then
  echo "[ERROR] scripts/train.py not found. Please run this script from the repo root."
  exit 1
fi

CONFIG="version_best_reproduce/reproduce_7512_mask_guided_7462_learned_only_drop345_seed42_kaggle"
BAD_INDEX="/kaggle/input/datasets/lhngphc/345-train-error/bad_row_indices_drop345_mediapipe_failed.txt"
OUT_DIR="/kaggle/working/outputs/seed_sweep_7512/drop345_seed_42"
LOG_DIR="$OUT_DIR/logs"
LOG_FILE="$LOG_DIR/train_drop345_seed42.log"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "[1/5] Check bad-row index file"
echo "============================================================"
if [ ! -f "$BAD_INDEX" ]; then
  echo "[ERROR] Missing bad index file:"
  echo "$BAD_INDEX"
  echo "Please Add Input: lhngphc/345-train-error"
  exit 1
fi

BAD_COUNT=$(grep -cve '^[[:space:]]*$' "$BAD_INDEX")
echo "[OK] bad index file exists: $BAD_INDEX"
echo "[OK] non-empty line count: $BAD_COUNT"
if [ "$BAD_COUNT" != "345" ]; then
  echo "[ERROR] Expected 345 bad row indices, got $BAD_COUNT"
  exit 1
fi

echo "============================================================"
echo "[2/5] Check merged config"
echo "============================================================"
python - <<'PY'
from src.utils.config import load_config

cfg = load_config(
    "version_best_reproduce/reproduce_7512_mask_guided_7462_learned_only_drop345_seed42_kaggle",
    env="kaggle",
)

checks = {
    "seed": cfg["seed"]["random_seed"],
    "use_clean_filter": cfg["data"].get("use_clean_filter"),
    "bad_row_indices_path": cfg["data"].get("bad_row_indices_path"),
    "output_dir": cfg["paths"].get("output_dir"),
    "use_wandb": cfg["logging"].get("use_wandb"),
    "use_clip_dictionary": cfg["model"].get("use_clip_dictionary"),
    "use_learnable_clip_region_tokens": cfg["model"].get("use_learnable_clip_region_tokens"),
    "mask_dir": cfg["model"].get("mask_dir"),
}
for key, value in checks.items():
    print(f"{key}={value}")

assert checks["seed"] == 42
assert checks["use_clean_filter"] is True
assert checks["use_wandb"] is False
assert checks["use_clip_dictionary"] is False
assert checks["use_learnable_clip_region_tokens"] is False
assert checks["bad_row_indices_path"] == "/kaggle/input/datasets/lhngphc/345-train-error/bad_row_indices_drop345_mediapipe_failed.txt"
print("[OK] merged config is correct")
PY

echo "============================================================"
echo "[3/5] Compile smoke test"
echo "============================================================"
python -m py_compile \
  src/data/dataset.py \
  src/data/dataloader.py \
  src/data/dataset_unet_mask.py \
  src/data/dataset_landmark.py \
  src/evaluation/evaluator.py \
  scripts/train.py
echo "[OK] compile smoke test passed"

echo "============================================================"
echo "[4/5] Train + evaluate"
echo "============================================================"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
echo "[INFO] CUDA device count: $GPU_COUNT"

if [ "$GPU_COUNT" -ge 2 ]; then
  echo "[INFO] Running DDP with 2 GPUs"
  torchrun --standalone --nproc_per_node=2 -m scripts.train \
    --env kaggle \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"
else
  echo "[INFO] Running single GPU / fallback"
  python -m scripts.train \
    --env kaggle \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"
fi

echo "============================================================"
echo "[5/5] Summarize and zip output"
echo "============================================================"
echo "[INFO] Expected filter log:"
grep -n "Filtered 345 bad rows" "$LOG_FILE" || {
  echo "[WARN] Did not find exact 'Filtered 345 bad rows' in log. Please inspect:"
  echo "$LOG_FILE"
}

echo "[INFO] Best checkpoints:"
find "$OUT_DIR" -type f -name "*best*.pth" -print || true

echo "[INFO] Training curves:"
find "$OUT_DIR" -type f \( -name "training_curves.png" -o -name "training_history.csv" -o -name "training_history.json" \) -print || true

echo "[INFO] Evaluation figures:"
find "$OUT_DIR" -type f \( -name "confusion_matrix.png" -o -name "correct_preds*.png" -o -name "wrong_preds*.png" \) -print || true

ZIP_PATH="/kaggle/working/drop345_seed42_outputs.zip"
cd /kaggle/working
zip -r "$ZIP_PATH" "outputs/seed_sweep_7512/drop345_seed_42" >/dev/null
echo "[OK] Zipped output: $ZIP_PATH"

echo "DONE"
