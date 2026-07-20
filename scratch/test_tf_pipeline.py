import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset_tf import build_datasets
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_csv = "../dataset/train.csv"
if not os.path.exists(train_csv):
    print("Run on Kaggle, skip local test")
    sys.exit(0)

try:
    train_ds, val_ds, test_ds = build_datasets(
        train_csv=train_csv,
        val_csv="../dataset/val.csv",
        image_size=48,
        batch_size=8,
        bbox_col="bboxes"
    )
    for batch in train_ds.take(1):
        if len(batch) >= 3:
            img, lbl, bbox = batch[0], batch[1], batch[2]
            print("With bbox", img.shape, bbox.shape)
        else:
            img, lbl = batch[0], batch[1]
            print("No bbox", img.shape)
    print("Test passed!")
except Exception as e:
    import traceback
    traceback.print_exc()
