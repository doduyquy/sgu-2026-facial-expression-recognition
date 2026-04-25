"""I/O helpers for motif banks."""

from __future__ import annotations

from pathlib import Path

import torch

from src.motif.motif_types import MotifBank


def save_motif_bank(bank: MotifBank, path: str | Path) -> None:
    """Save a MotifBank with torch.save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)


def load_motif_bank(path: str | Path) -> MotifBank:
    """Load a MotifBank on CPU."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motif bank not found: {path}")
    try:
        bank = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bank = torch.load(path, map_location="cpu")
    if not isinstance(bank, MotifBank):
        raise TypeError(f"Expected MotifBank in {path}, got {type(bank)}")
    return bank
