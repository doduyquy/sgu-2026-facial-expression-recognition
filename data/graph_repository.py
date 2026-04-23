"""
data/graph_repository.py — Writer and Reader for the canonical graph repository.

Repository layout on disk
--------------------------
  <repo_root>/
    shared/
      shared_graph.pt          ← SharedGraphStructure
    train/
      chunk_000.pt             ← List[PixelGraphSample]
      chunk_001.pt
      …
    val/
      chunk_000.pt
      …
    test/
      chunk_000.pt
      …
    manifest.pt                ← metadata about the full repo

Design goals
------------
* Writer never keeps a full split in RAM — it streams and flushes per chunk.
* Reader lazy-loads chunks (no eager loading of entire split).
* No graph reconstruction from CSV at read time — repository is the source of truth.
* Compatible with Kaggle: repo_root can be any absolute path
  (e.g. /kaggle/input/fer-graph-repo/graph_repo).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from configs.graph_config import GraphConfig
from data.graph_types import PixelGraphSample, SharedGraphStructure

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.pt"
SHARED_DIR = "shared"
SHARED_FILENAME = "shared_graph.pt"
CHUNK_PATTERN = "chunk_{idx:03d}.pt"


# ===========================================================================
# GraphRepositoryWriter
# ===========================================================================

class GraphRepositoryWriter:
    """
    Streams PixelGraphSamples into the chunked repository.

    Usage
    -----
    >>> writer = GraphRepositoryWriter(repo_root="artifacts/graph_repo", config=cfg)
    >>> writer.write_shared(shared)
    >>> with writer.open_split("train") as sw:
    ...     for sample in stream:
    ...         sw.add(sample)
    """

    def __init__(self, repo_root: str | Path, config: GraphConfig) -> None:
        self.repo_root = Path(repo_root)
        self.config = config
        self.chunk_size = config.chunk_size
        self._manifest: Dict = {
            "version": config.version,
            "chunk_size": config.chunk_size,
            "splits": {},
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # ------------------------------------------------------------------
    # Shared graph
    # ------------------------------------------------------------------

    def write_shared(self, shared: SharedGraphStructure) -> Path:
        """Save SharedGraphStructure to shared/shared_graph.pt."""
        out_dir = self.repo_root / SHARED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / SHARED_FILENAME
        torch.save(shared, out_path)
        log.info("Saved shared graph → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Split writer context manager
    # ------------------------------------------------------------------

    def open_split(self, split: str) -> "_SplitWriter":
        """
        Return a context manager that streams samples into chunks.

        Usage
        -----
        >>> with writer.open_split("train") as sw:
        ...     for sample in samples:
        ...         sw.add(sample)
        """
        split_dir = self.repo_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        return _SplitWriter(
            split_dir=split_dir,
            split=split,
            chunk_size=self.chunk_size,
            manifest=self._manifest,
        )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def save_manifest(self) -> Path:
        out_path = self.repo_root / MANIFEST_FILENAME
        torch.save(self._manifest, out_path)
        log.info("Saved manifest → %s", out_path)
        return out_path


class _SplitWriter:
    """Internal context manager for streaming one split."""

    def __init__(
        self,
        split_dir: Path,
        split: str,
        chunk_size: int,
        manifest: Dict,
    ) -> None:
        self._dir = split_dir
        self._split = split
        self._chunk_size = chunk_size
        self._manifest = manifest
        self._buf: List[PixelGraphSample] = []
        self._chunk_idx = 0
        self._total = 0
        self._chunk_paths: List[str] = []

    def add(self, sample: PixelGraphSample) -> None:
        """Add one sample. Flushes automatically when chunk_size is reached."""
        self._buf.append(sample)
        self._total += 1
        if len(self._buf) >= self._chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        name = CHUNK_PATTERN.format(idx=self._chunk_idx)
        path = self._dir / name
        torch.save(self._buf, path)
        log.debug("  Flushed chunk %s → %d samples", path.name, len(self._buf))
        self._chunk_paths.append(str(path))
        self._chunk_idx += 1
        self._buf = []

    def __enter__(self) -> "_SplitWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._flush()   # flush remaining samples
        self._manifest["splits"][self._split] = {
            "num_samples": self._total,
            "num_chunks": self._chunk_idx,
            "chunk_files": self._chunk_paths,
        }
        log.info(
            "Split %s complete: %d samples in %d chunks",
            self._split, self._total, self._chunk_idx,
        )


# ===========================================================================
# GraphRepositoryReader
# ===========================================================================

class GraphRepositoryReader:
    """
    Reads from the canonical graph repository.

    Lazy-loads chunks — never loads a full split into memory at once.

    Parameters
    ----------
    repo_root : path to the root of the repository.
                On Kaggle: "/kaggle/input/fer-graph-repo/graph_repo"
                Locally  : "artifacts/graph_repo"

    Usage
    -----
    >>> reader = GraphRepositoryReader("artifacts/graph_repo")
    >>> shared = reader.load_shared()
    >>> for sample in reader.iter_split("train"):
    ...     process(sample)
    """

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Repository root not found: {self.repo_root}")
        self._manifest: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Shared graph
    # ------------------------------------------------------------------

    def load_shared(self) -> SharedGraphStructure:
        """Load SharedGraphStructure from shared/shared_graph.pt."""
        path = self.repo_root / SHARED_DIR / SHARED_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"shared_graph.pt not found: {path}")
        shared = torch.load(path, map_location="cpu", weights_only=False)
        log.info("Loaded shared graph from %s", path)
        return shared

    # ------------------------------------------------------------------
    # Chunk access
    # ------------------------------------------------------------------

    def chunk_paths(self, split: str) -> List[Path]:
        """Return sorted list of chunk file paths for a split."""
        split_dir = self.repo_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        paths = sorted(split_dir.glob("chunk_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No chunk files found in {split_dir}")
        return paths

    def num_chunks(self, split: str) -> int:
        return len(self.chunk_paths(split))

    def load_chunk(self, split: str, chunk_idx: int) -> List[PixelGraphSample]:
        """Load one specific chunk by index."""
        paths = self.chunk_paths(split)
        if chunk_idx >= len(paths):
            raise IndexError(
                f"chunk_idx={chunk_idx} out of range "
                f"(split '{split}' has {len(paths)} chunks)"
            )
        return torch.load(paths[chunk_idx], map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_split(
        self, split: str, start_chunk: int = 0
    ) -> Iterator[PixelGraphSample]:
        """
        Lazy iterate over ALL samples in a split, chunk by chunk.
        Memory footprint ≤ one chunk at a time.
        """
        for path in self.chunk_paths(split)[start_chunk:]:
            chunk: List[PixelGraphSample] = torch.load(
                path, map_location="cpu", weights_only=False
            )
            yield from chunk

    def iter_chunks(
        self, split: str
    ) -> Iterator[Tuple[int, List[PixelGraphSample]]]:
        """Iterate (chunk_idx, List[PixelGraphSample]) pairs."""
        for idx, path in enumerate(self.chunk_paths(split)):
            chunk = torch.load(path, map_location="cpu", weights_only=False)
            yield idx, chunk

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def load_manifest(self) -> Dict:
        if self._manifest is not None:
            return self._manifest
        path = self.repo_root / MANIFEST_FILENAME
        if not path.exists():
            log.warning("manifest.pt not found at %s", path)
            return {}
        self._manifest = torch.load(path, map_location="cpu", weights_only=False)
        return self._manifest

    def split_info(self, split: str) -> Dict:
        manifest = self.load_manifest()
        return manifest.get("splits", {}).get(split, {})

    def num_samples(self, split: str) -> Optional[int]:
        info = self.split_info(split)
        return info.get("num_samples")

    # ------------------------------------------------------------------
    # Convenience: list available splits
    # ------------------------------------------------------------------

    def available_splits(self) -> List[str]:
        """Return split names based on existing subdirectories."""
        return sorted([
            p.name for p in self.repo_root.iterdir()
            if p.is_dir() and p.name not in (SHARED_DIR, "__pycache__")
            and any(p.glob("chunk_*.pt"))
        ])

    def summary(self) -> Dict:
        manifest = self.load_manifest()
        result = {
            "repo_root": str(self.repo_root),
            "version": manifest.get("version", "unknown"),
            "built_at": manifest.get("built_at", "unknown"),
            "chunk_size": manifest.get("chunk_size", "unknown"),
            "splits": {},
        }
        for split in self.available_splits():
            info = self.split_info(split)
            n_chunks = self.num_chunks(split)
            result["splits"][split] = {
                "num_samples": info.get("num_samples", "unknown"),
                "num_chunks": n_chunks,
            }
        return result
