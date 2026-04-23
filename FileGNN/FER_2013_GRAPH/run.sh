#! /bin/bash

#  --- CHANGE EXP PARAMS HERE, ONLY THIS FILE! --- 
WANDB_ENTITY="phucga15062005" # team wandb  --> NOT CHANGE!
WANDB_PROJECT="A3Net_Research_And_Develop" # Project wandb --> NOT CHANGE!
# Lấy WANDB_NAME từ environment (nếu được set trên Kaggle), ngược lại dùng tên mặc định
WANDB_NAME="${WANDB_NAME:-team-baseline-WarmupCosine-AdamW-$(date '+%Y%m%d-%H%M')}"

RESUME_PATH="${RESUME_PATH:-}"  # Đã sửa lỗi bash syntax: thêm dấu '-' sau dấu ':'
                # Điền path .pth -> resume (train tiếp) từ checkpoints
# --------------------------------------------------

# update for dataset path in kaggle IU_XRAY_RRG
# đọc DATASET_PATH từ environment (set trên kaggle), nếu không có thì dùng local path
DATASET_PATH="${DATASET_PATH:-data/fer_split}"

python -m scripts.build_graph_cache \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --test_csv data/test.csv \
  --save_dir artifacts/graph_cache
