"""
data/__init__.py — Public API for the canonical graph data pipeline.

Import order matters: raw_types → graph_types → builders → repository → dataset.
"""

from data.raw_types import RawSample
from data.raw_fer_dataset import RawFERDataset, EMOTION_NAMES
from data.graph_types import SharedGraphStructure, PixelGraphSample, ResolvedPixelGraph
from data.shared_graph_builder import SharedGraphBuilder
from data.canonical_graph_builder import CanonicalGraphBuilder
from data.graph_resolver import GraphResolver
from data.graph_repository import GraphRepositoryWriter, GraphRepositoryReader
from data.chunked_graph_dataset import ChunkedGraphDataset

__all__ = [
    # Raw layer
    "RawSample",
    "RawFERDataset",
    "EMOTION_NAMES",
    # Graph types
    "SharedGraphStructure",
    "PixelGraphSample",
    "ResolvedPixelGraph",
    # Builders
    "SharedGraphBuilder",
    "CanonicalGraphBuilder",
    # Resolver
    "GraphResolver",
    # Repository I/O
    "GraphRepositoryWriter",
    "GraphRepositoryReader",
    # Dataset
    "ChunkedGraphDataset",
]
