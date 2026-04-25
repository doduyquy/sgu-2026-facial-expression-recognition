"""I/O helpers for pixel-preserving motif banks."""

from __future__ import annotations

from pathlib import Path

import torch

from src.motif_v2.types import PixelMotifBank


def save_pixel_motif_bank(bank: PixelMotifBank, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)


def load_pixel_motif_bank(path: str | Path) -> PixelMotifBank:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pixel motif bank not found: {path}")
    try:
        bank = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bank = torch.load(path, map_location="cpu")
    if not isinstance(bank, PixelMotifBank):
        raise TypeError(f"Expected PixelMotifBank in {path}, got {type(bank)}")
    return bank
