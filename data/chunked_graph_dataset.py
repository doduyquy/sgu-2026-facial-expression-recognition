"""
data/chunked_graph_dataset.py — PyTorch Dataset over the canonical graph repository.

This is the primary Dataset interface for ALL downstream tasks:
  * Baseline MLP (reads node_features as flat vector)
  * GNN training (reads ResolvedPixelGraph)
  * Subgraph generation
  * Motif pipeline

Design
------
* Reads from the graph repository (chunks), never from raw CSV.
* Lazy chunk loading — a chunk is loaded only when an index within it is first accessed.
  A simple LRU cache (size=1 by default) prevents re-loading the same chunk repeatedly
  during sequential or batched access.
* Optionally resolves graphs (SharedGraphStructure + PixelGraphSample → ResolvedPixelGraph)
  when resolve=True.
* Kaggle-compatible: just point repo_root to the Kaggle dataset path.

Usage on Kaggle
---------------
>>> from data.chunked_graph_dataset import ChunkedGraphDataset
>>> ds = ChunkedGraphDataset(
...     repo_root="/kaggle/input/fer-graph-repo/graph_repo",
...     split="train",
...     resolve=True,   # ← produces ResolvedPixelGraph
... )
>>> resolved = ds[0]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import Dataset

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.graph_types import (
    PixelGraphSample,
    ResolvedPixelGraph,
    SharedGraphStructure,
)

log = logging.getLogger(__name__)


class ChunkedGraphDataset(Dataset):
    """
    PyTorch Dataset over one split of the graph repository.

    Parameters
    ----------
    repo_root   : root directory of the repository
                  (local: "artifacts/graph_repo"
                   Kaggle: "/kaggle/input/fer-graph-repo/graph_repo")
    split       : "train" | "val" | "test"
    resolve     : if True, __getitem__ returns ResolvedPixelGraph;
                  if False, returns raw PixelGraphSample
    transform   : optional callable applied to each item after resolution
    cache_chunks: number of chunks to keep in memory (default 1 = only current chunk)
    """

    def __init__(
        self,
        repo_root: Union[str, Path],
        split: str,
        resolve: bool = True,
        transform: Optional[Callable] = None,
        cache_chunks: int = 1,
    ) -> None:
        self.split = split
        self.resolve = resolve
        self.transform = transform
        self._cache_chunks = max(1, cache_chunks)

        self._reader = GraphRepositoryReader(repo_root)
        self._chunk_paths = self._reader.chunk_paths(split)

        # Build index: (chunk_idx, local_idx) for each global sample index
        self._index: List[tuple[int, int]] = self._build_global_index()

        # Load shared graph and create resolver if needed
        self._shared: Optional[SharedGraphStructure] = None
        self._resolver: Optional[GraphResolver] = None
        if resolve:
            self._shared = self._reader.load_shared()
            self._resolver = GraphResolver(self._shared)

        # Simple chunk cache: {chunk_idx: List[PixelGraphSample]}
        self._chunk_cache: dict[int, List[PixelGraphSample]] = {}

        log.info(
            "ChunkedGraphDataset: split=%s, samples=%d, chunks=%d, resolve=%s",
            split, len(self), len(self._chunk_paths), resolve,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_global_index(self) -> List[tuple[int, int]]:
        """
        Map global_idx → (chunk_idx, local_idx).
        We load each chunk header to get its size, then release it.
        """
        index = []
        for chunk_idx, path in enumerate(self._chunk_paths):
            chunk: List[PixelGraphSample] = torch.load(
                path, map_location="cpu", weights_only=False
            )
            for local_idx in range(len(chunk)):
                index.append((chunk_idx, local_idx))
            # Don't keep in cache yet — let __getitem__ handle caching
        return index

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(
        self, idx: int
    ) -> Union[PixelGraphSample, ResolvedPixelGraph]:
        chunk_idx, local_idx = self._index[idx]
        chunk = self._get_chunk(chunk_idx)
        sample: PixelGraphSample = chunk[local_idx]

        if self.resolve and self._resolver is not None:
            item = self._resolver.resolve(sample)
        else:
            item = sample

        if self.transform is not None:
            item = self.transform(item)

        return item

    # ------------------------------------------------------------------
    # Chunk cache
    # ------------------------------------------------------------------

    def _get_chunk(self, chunk_idx: int) -> List[PixelGraphSample]:
        if chunk_idx not in self._chunk_cache:
            # Evict oldest if cache is full
            if len(self._chunk_cache) >= self._cache_chunks:
                oldest_key = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest_key]

            path = self._chunk_paths[chunk_idx]
            self._chunk_cache[chunk_idx] = torch.load(
                path, map_location="cpu", weights_only=False
            )
        return self._chunk_cache[chunk_idx]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def shared(self) -> Optional[SharedGraphStructure]:
        """Return the shared graph structure (only available if resolve=True)."""
        return self._shared

    @property
    def num_node_features(self) -> int:
        """Infer from first sample."""
        if len(self) == 0:
            return 0
        sample: PixelGraphSample = self._get_chunk(0)[0]
        return sample.num_node_features

    @property
    def num_edge_features(self) -> int:
        """Total edge feature count (static + dynamic) if resolve=True, else dynamic only."""
        if len(self) == 0:
            return 0
        if self.resolve and self._shared is not None:
            sample: PixelGraphSample = self._get_chunk(0)[0]
            return self._shared.num_static_features + sample.num_dynamic_features
        sample = self._get_chunk(0)[0]
        return sample.num_dynamic_features

    @property
    def num_classes(self) -> int:
        return 7  # FER-2013 canonical

    def labels(self) -> List[int]:
        """Return list of all labels (iterates all chunks — use sparingly)."""
        result = []
        for sample in self._reader.iter_split(self.split):
            result.append(sample.label)
        return result

    def sample_ids(self) -> List[int]:
        """Return list of all graph_ids (iterates all chunks)."""
        result = []
        for sample in self._reader.iter_split(self.split):
            result.append(sample.graph_id)
        return result

    def summary(self) -> dict:
        return {
            "split": self.split,
            "num_samples": len(self),
            "num_chunks": len(self._chunk_paths),
            "resolve": self.resolve,
            "num_node_features": self.num_node_features,
            "num_edge_features": self.num_edge_features,
            "num_classes": self.num_classes,
        }
