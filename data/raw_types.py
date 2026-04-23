"""
data/raw_types.py — Raw data contracts for the FER-2013 pipeline.

This module defines the OUTPUT TYPE of the raw CSV layer.
No graph logic here — only plain numpy arrays and primitive scalars.

Downstream (canonical graph builder) must accept RawSample as its input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RawSample:
    """
    One parsed row from a FER-2013 split CSV file.

    Attributes
    ----------
    sample_id   : integer index within the split file (0-based)
    label       : emotion class index  [0..6]
    split       : "train" | "val" | "test"
    usage       : raw Usage string from CSV (e.g. "Training", "PublicTest", …)
    image       : float32 array shape (H, W) — pixel values in [0,255]
                  (raw, un-normalized — normalization is the graph layer's job)
    metadata    : optional dict for extra CSV columns or debug info
    """

    sample_id: int
    label: int
    split: str
    usage: str
    image: np.ndarray                          # shape (H, W), dtype float32, [0,255]
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate(self, expected_height: int = 48, expected_width: int = 48) -> None:
        """Raise ValueError if the sample is malformed."""
        if self.image.ndim != 2:
            raise ValueError(
                f"sample_id={self.sample_id}: image must be 2-D, "
                f"got shape {self.image.shape}"
            )
        h, w = self.image.shape
        if (h, w) != (expected_height, expected_width):
            raise ValueError(
                f"sample_id={self.sample_id}: expected ({expected_height},{expected_width}), "
                f"got ({h},{w})"
            )
        if not np.issubdtype(self.image.dtype, np.floating):
            raise ValueError(
                f"sample_id={self.sample_id}: image dtype must be float, "
                f"got {self.image.dtype}"
            )
        if self.label < 0 or self.label > 6:
            raise ValueError(
                f"sample_id={self.sample_id}: label {self.label} out of [0,6]"
            )

    def __repr__(self) -> str:
        return (
            f"RawSample(id={self.sample_id}, label={self.label}, "
            f"split={self.split!r}, image_shape={self.image.shape})"
        )
