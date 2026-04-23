"""
src/data/dataloader.py — DataLoader factory dùng canonical graph repository.

Đọc từ graph repository (chunks), không bao giờ đọc từ CSV hay *_graphs.pt kiểu cũ.

Hỗ trợ 2 chế độ đầu vào cho downstream:
    1. "graph_vector"  — flatten node_features → vector → MLP Baseline
    2. "resolved"      — full ResolvedPixelGraph → GNN (future)

Cả 2 chế độ đều đọc từ ChunkedGraphDataset trỏ vào graph_repo_path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup — để import được data package từ project root
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.chunked_graph_dataset import ChunkedGraphDataset
from data.graph_repository import GraphRepositoryReader
from data.graph_types import PixelGraphSample


# ===========================================================================
# Public API
# ===========================================================================

def build_dataloader(
    config: dict,
    graph_repo_path: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build DataLoaders từ canonical graph repository.

    Parameters
    ----------
    config         : full config dict (từ load_config)
    graph_repo_path: đường dẫn đến thư mục gốc của graph repository.
                     Local  : "artifacts/graph_repo"
                     Kaggle : "/kaggle/input/fer-graph-repo/graph_repo"

    Returns
    -------
    train_loader, val_loader, test_loader, input_dim
    """
    # Validate repo exists
    _validate_repo(graph_repo_path)

    mode       = config.get("dataloader_mode", "graph_vector")
    batch_size = config.get("training", {}).get("batch_size",
                 config.get("data", {}).get("batch_size", 128))
    num_workers = config.get("num_workers",
                  config.get("data", {}).get("num_workers", 0))

    print(f"--- Graph repo     : {graph_repo_path}")
    print(f"--- Dataloader mode: [{mode}]")

    if mode == "graph_vector":
        return _build_graph_vector_loaders(
            graph_repo_path, config, batch_size, num_workers
        )
    elif mode == "resolved":
        return _build_resolved_loaders(
            graph_repo_path, config, batch_size, num_workers
        )
    else:
        raise ValueError(
            f"dataloader_mode không hợp lệ: {mode!r}. "
            f"Chọn 'graph_vector' hoặc 'resolved'."
        )


# ===========================================================================
# Mode 1: Graph Vector (MLP Baseline)
# Flatten node_features của mỗi PixelGraphSample → 1 vector
# ===========================================================================

def _build_graph_vector_loaders(
    repo_path: str,
    config: dict,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Chuyển mỗi PixelGraphSample → flat vector bằng GraphVectorizer,
    wrap thành GraphVectorDatasetFromRepo, rồi build DataLoader.
    """
    from src.features.graph_vectorizer import GraphVectorizer

    vectorizer = GraphVectorizer(use_mean=True, use_std=True, use_max=True)

    train_ds = GraphVectorDatasetFromRepo(repo_path, "train", vectorizer)
    val_ds   = GraphVectorDatasetFromRepo(repo_path, "val",   vectorizer)
    test_ds  = GraphVectorDatasetFromRepo(repo_path, "test",  vectorizer)

    input_dim = train_ds.input_dim

    print(f"--- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"--- Input dim (graph vector): {input_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, input_dim


# ===========================================================================
# Mode 2: Resolved (GNN — future)
# Returns ResolvedPixelGraph objects; caller provides custom collate_fn
# ===========================================================================

def _build_resolved_loaders(
    repo_path: str,
    config: dict,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Dùng ChunkedGraphDataset với resolve=True.
    Mỗi item là ResolvedPixelGraph — caller cần custom collate_fn.
    """
    train_ds = ChunkedGraphDataset(repo_path, "train", resolve=True)
    val_ds   = ChunkedGraphDataset(repo_path, "val",   resolve=True)
    test_ds  = ChunkedGraphDataset(repo_path, "test",  resolve=True)

    input_dim = train_ds.num_node_features

    print(f"--- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"--- Node feature dim: {input_dim}")
    print("--- NOTE: Dùng custom collate_fn cho ResolvedPixelGraph với GNN.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,  # pin_memory=False với custom objects
        collate_fn=_identity_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        collate_fn=_identity_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        collate_fn=_identity_collate,
    )
    return train_loader, val_loader, test_loader, input_dim


def _identity_collate(batch):
    """Pass-through collate — batch là List[ResolvedPixelGraph]."""
    return batch


# ===========================================================================
# GraphVectorDatasetFromRepo
# Đọc từ ChunkedGraphDataset (resolve=False) rồi vectorize on-the-fly
# ===========================================================================

class GraphVectorDatasetFromRepo(Dataset):
    """
    Dataset cho MLP Baseline.

    Đọc PixelGraphSample từ graph repository (không resolve),
    vectorize node_features → 1D vector dùng GraphVectorizer.

    Output mỗi sample:
        {
            "x": FloatTensor [D],   # graph-level vector
            "y": LongTensor  [],    # label 0-6
        }
    """

    def __init__(
        self,
        repo_path: str,
        split: str,
        vectorizer,  # GraphVectorizer
    ) -> None:
        self._ds = ChunkedGraphDataset(repo_path, split, resolve=False)
        self._vectorizer = vectorizer
        self._input_dim: int | None = None

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        sample: PixelGraphSample = self._ds[idx]
        # node_features: Tensor [N, d] — vectorize to [D]
        x = self._vectorizer.transform_from_tensor(sample.node_features)
        return {
            "x": x,
            "y": torch.tensor(sample.label, dtype=torch.long),
        }

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            sample: PixelGraphSample = self._ds[0]
            node_feature_dim = sample.num_node_features
            self._input_dim = self._vectorizer.infer_output_dim(node_feature_dim)
        return self._input_dim


# ===========================================================================
# Validation helper
# ===========================================================================

def _validate_repo(repo_path: str) -> None:
    """Kiểm tra repo tồn tại và có shared_graph.pt."""
    from data.graph_repository import SHARED_DIR, SHARED_FILENAME
    shared_path = Path(repo_path) / SHARED_DIR / SHARED_FILENAME
    if not shared_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Không tìm thấy graph repository tại: {repo_path}\n"
            f"  Thiếu file: {shared_path}\n\n"
            f"  Local  : Chạy scripts/build_graph_repository.py trước:\n"
            f"    python scripts/build_graph_repository.py \\\n"
            f"      --train_csv data/train.csv \\\n"
            f"      --val_csv   data/val.csv \\\n"
            f"      --test_csv  data/test.csv \\\n"
            f"      --repo_root artifacts/graph_repo\n\n"
            f"  Kaggle : Upload artifacts/graph_repo/ lên Kaggle dataset 'fer-graph-repo',\n"
            f"           set graph_repo_path: '/kaggle/input/fer-graph-repo/graph_repo' trong base.yaml.\n"
        )
