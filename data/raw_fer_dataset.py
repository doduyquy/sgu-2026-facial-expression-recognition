"""
data/raw_fer_dataset.py — Raw CSV reader for FER-2013 split files.

Responsibilities
----------------
* Parse train.csv / val.csv / test.csv
* Decode the space-delimited 'pixels' column → numpy array (H, W)
* Return RawSample objects
* Provide class distribution / summary utilities

This module has NO graph logic. It is purely the raw data layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.raw_types import RawSample

log = logging.getLogger(__name__)

# FER-2013 canonical emotion labels
EMOTION_NAMES: Dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


class RawFERDataset(Dataset):
    """
    PyTorch-compatible Dataset over one FER-2013 split CSV.

    Parameters
    ----------
    csv_path    : path to train.csv / val.csv / test.csv
    split       : logical split name ("train" | "val" | "test")
    image_size  : expected image side length (default 48)
    validate    : if True, call sample.validate() on every __getitem__

    Usage
    -----
    >>> ds = RawFERDataset("data/fer13-split/train.csv", split="train")
    >>> sample: RawSample = ds[0]
    >>> print(sample)
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        image_size: int = 48,
        validate: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split = split
        self.image_size = image_size
        self.validate = validate

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        log.info("Loading %s …", self.csv_path)
        self._df = pd.read_csv(self.csv_path)
        self._validate_columns()
        log.info("  → %d samples loaded", len(self._df))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_columns(self) -> None:
        required = {"emotion", "pixels"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} missing columns: {missing}")
        self._df["emotion"] = self._df["emotion"].astype(int)
        if "Usage" not in self._df.columns:
            self._df["Usage"] = self.split

    def _parse_pixels(self, pixel_str: str) -> np.ndarray:
        """Decode space-separated pixel string → float32 (H, W) in [0, 255]."""
        expected = self.image_size * self.image_size
        arr = np.fromstring(str(pixel_str), sep=" ", dtype=np.float32)
        if arr.size != expected:
            raise ValueError(
                f"Pixel count mismatch: expected {expected}, got {arr.size}"
            )
        return arr.reshape(self.image_size, self.image_size)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> RawSample:
        row = self._df.iloc[idx]
        image = self._parse_pixels(row["pixels"])
        sample = RawSample(
            sample_id=int(idx),
            label=int(row["emotion"]),
            split=self.split,
            usage=str(row["Usage"]),
            image=image,
        )
        if self.validate:
            sample.validate(self.image_size, self.image_size)
        return sample

    def __iter__(self) -> Iterator[RawSample]:
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # Analytics / summary
    # ------------------------------------------------------------------

    def class_distribution(self) -> Dict[int, int]:
        """Return {label: count} sorted by label."""
        return (
            self._df["emotion"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    def class_distribution_named(self) -> Dict[str, int]:
        """Return {emotion_name: count}."""
        return {
            EMOTION_NAMES[k]: v
            for k, v in self.class_distribution().items()
        }

    def summary(self) -> Dict:
        dist = self.class_distribution()
        return {
            "split": self.split,
            "csv_path": str(self.csv_path),
            "num_samples": len(self),
            "image_size": self.image_size,
            "class_distribution": {
                f"{k} ({EMOTION_NAMES[k]})": v for k, v in dist.items()
            },
            "class_balance": {
                EMOTION_NAMES[k]: f"{v/len(self)*100:.1f}%"
                for k, v in dist.items()
            },
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"RawFERDataset — split: {s['split']}")
        print(f"  CSV   : {s['csv_path']}")
        print(f"  Total : {s['num_samples']} samples")
        print(f"  Classes:")
        for k, v in s["class_distribution"].items():
            bal = s["class_balance"][k.split(" ")[1].strip("()")]
            print(f"    {k:20s}: {v:5d}  ({bal})")
        print(f"{'='*50}\n")
