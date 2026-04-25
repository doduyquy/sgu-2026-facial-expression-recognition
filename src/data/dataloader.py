"""
src/data/dataloader.py — DataLoader factory dùng canonical graph repository.

Đọc từ graph repository (chunks), không bao giờ đọc từ CSV hay *_graphs.pt kiểu cũ.

Hỗ trợ 3 chế độ đầu vào cho downstream:
    1. "graph_vector"         — flatten node_features → vector → MLP Baseline
    2. "subgraph_descriptor"  — bag of subgraph descriptors → Subgraph MLP Baseline
    3. "resolved"             — full ResolvedPixelGraph → GNN (future)

Cả 3 chế độ đều đọc từ ChunkedGraphDataset / graph repository.
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
from data.graph_types import PixelGraphSample
from src.data.subgraph_dataset import SubgraphDescriptorDataset
from src.data.precomputed_subgraph_graph_dataset import PrecomputedSubgraphGraphDataset


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
                     Local  : "artifacts/graph_repo_v2"
                     Kaggle : "/kaggle/input/fer-graph-repo-v2/graph_repo_v2"

    Returns
    -------
    train_loader, val_loader, test_loader, input_dim
    """
    mode       = config.get("dataloader_mode", "graph_vector")
    batch_size = config.get("training", {}).get("batch_size",
                 config.get("data", {}).get("batch_size", 128))
    num_workers = config.get("num_workers",
                  config.get("data", {}).get("num_workers", 0))

    print(f"--- Dataloader mode: [{mode}]")

    if mode == "graph_vector":
        _validate_repo(graph_repo_path)
        print(f"--- Graph repo     : {graph_repo_path}")
        return _build_graph_vector_loaders(
            graph_repo_path, config, batch_size, num_workers
        )
    elif mode == "subgraph_descriptor":
        _validate_repo(graph_repo_path)
        print(f"--- Graph repo     : {graph_repo_path}")
        return _build_subgraph_descriptor_loaders(
            graph_repo_path, config, batch_size, num_workers
        )
    elif mode == "resolved":
        _validate_repo(graph_repo_path)
        print(f"--- Graph repo     : {graph_repo_path}")
        return _build_resolved_loaders(
            graph_repo_path, config, batch_size, num_workers
        )
    elif mode == "precomputed_subgraph_graph":
        # dataset_path được resolve từ subgraph_dataset_path trong config
        subgraph_dataset_path = config.get(
            "subgraph_dataset_path",
            "artifacts/subgraph_graph_dataset_v2",
        )
        print(f"--- Precomputed dataset: {subgraph_dataset_path}")
        model_name = config.get("model", {}).get("name", "subgraph_mlp_baseline")
        use_gnn = "gnn" in model_name
        return _build_precomputed_loaders(
            subgraph_dataset_path, config, batch_size, num_workers, use_gnn=use_gnn
        )
    else:
        raise ValueError(
            f"dataloader_mode không hợp lệ: {mode!r}. "
            f"Chọn 'graph_vector', 'subgraph_descriptor', 'resolved' hoặc 'precomputed_subgraph_graph'."
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


# ===========================================================================
# Mode 2: Subgraph Descriptor Baseline
# Resolved graph -> candidate subgraphs -> descriptors [K, D]
# ===========================================================================

def _build_subgraph_descriptor_loaders(
    repo_path: str,
    config: dict,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build loaders for the subgraph-first baseline.
    """
    data_cfg = config.get("data", {})
    dataset_kwargs = {
        "num_subgraphs": data_cfg.get("num_subgraphs", 16),
        "subgraph_radius": data_cfg.get("subgraph_radius", 1),
        "seed_stride": data_cfg.get("seed_stride", 4),
        "max_candidates": data_cfg.get("max_candidates", 64),
        "max_nodes_per_subgraph": data_cfg.get("max_nodes_per_subgraph"),
    }

    train_ds = SubgraphDescriptorDataset(repo_path=repo_path, split="train", **dataset_kwargs)
    val_ds = SubgraphDescriptorDataset(repo_path=repo_path, split="val", **dataset_kwargs)
    test_ds = SubgraphDescriptorDataset(repo_path=repo_path, split="test", **dataset_kwargs)

    input_dim = train_ds.input_dim

    print(f"--- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"--- Input dim (subgraph descriptor): {input_dim}")
    print(
        f"--- Subgraph config: K={dataset_kwargs['num_subgraphs']} | "
        f"radius={dataset_kwargs['subgraph_radius']} | "
        f"stride={dataset_kwargs['seed_stride']} | "
        f"max_candidates={dataset_kwargs['max_candidates']}"
    )

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
# Mode 4: Precomputed Subgraph Graph (MLP + GNN shared dataset)
# ===========================================================================

def _build_precomputed_loaders(
    dataset_path: str,
    config: dict,
    batch_size: int,
    num_workers: int,
    use_gnn: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build DataLoaders từ precomputed subgraph-level graph dataset.

    Dùng cùng dataset cho cả SubgraphMLPBaseline và SubgraphGNNBaseline.
    Khi use_gnn=True, dùng collate_fn_gnn để batch edge_index đúng cách.
    """
    from pathlib import Path as _Path

    dp = _Path(dataset_path)

    def _pt(split: str) -> str:
        return str(dp / f"{split}_subgraph_graph.pt")

    train_ds = PrecomputedSubgraphGraphDataset(_pt("train"))
    val_ds   = PrecomputedSubgraphGraphDataset(_pt("val"))
    test_ds  = PrecomputedSubgraphGraphDataset(_pt("test"))

    input_dim = train_ds.input_dim

    print(f"--- Subgraph dataset : {dataset_path}")
    print(f"--- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"--- Input dim (descriptor): {input_dim}  |  K={train_ds.num_subgraphs}")
    print(f"--- Collate mode: {'GNN (pad edge_index)' if use_gnn else 'MLP (default)'}")

    collate = _collate_fn_gnn if use_gnn else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, val_loader, test_loader, input_dim


def _collate_fn_gnn(batch):
    """
    Custom collate_fn cho GNN mode.

    Vì mỗi sample có edge_index kích thước khác nhau [2, E_i],
    cần pad thành tensor đồng nhất [B, 2, E_max] và tạo edge_valid mask [B, E_max].

    Output dict keys:
        x          : [B, K, D]
        mask       : [B, K]
        edge_index : [B, 2, E_max]  — padded với 0
        edge_attr  : [B, E_max, 1]  — padded với 0
        edge_valid : [B, E_max]     — 1=valid, 0=pad
        centers    : [B, K, 2]
        y          : [B]
    """
    xs      = torch.stack([s["x"]       for s in batch])   # [B, K, D]
    masks   = torch.stack([s["mask"]    for s in batch])   # [B, K]
    centers = torch.stack([s["centers"] for s in batch])   # [B, K, 2]
    ys      = torch.stack([s["y"]       for s in batch])   # [B]
    graph_ids = torch.tensor([int(s["graph_id"]) for s in batch], dtype=torch.long)

    edge_indices = [s["edge_index"] for s in batch]   # list of [2, E_i]
    edge_attrs   = [s["edge_attr"]  for s in batch]   # list of [E_i, 1]

    E_max = max(ei.shape[1] for ei in edge_indices) if edge_indices else 0

    B = len(batch)
    edge_index_pad = torch.zeros(B, 2, max(E_max, 1), dtype=torch.long)
    edge_attr_pad  = torch.zeros(B, max(E_max, 1), 1,  dtype=torch.float32)
    edge_valid     = torch.zeros(B, max(E_max, 1),      dtype=torch.float32)

    for i, (ei, ea) in enumerate(zip(edge_indices, edge_attrs)):
        e = ei.shape[1]
        if e > 0:
            edge_index_pad[i, :, :e] = ei
            edge_attr_pad[i, :e, :]  = ea
            edge_valid[i, :e]        = 1.0

    return {
        "x"          : xs,
        "mask"       : masks,
        "edge_index" : edge_index_pad,
        "edge_attr"  : edge_attr_pad,
        "edge_valid" : edge_valid,
        "centers"    : centers,
        "y"          : ys,
        "graph_id"   : graph_ids,
    }


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
            f"      --repo_root artifacts/graph_repo_v2\n\n"
            f"  Kaggle : Upload artifacts/graph_repo_v2/ lên Kaggle dataset 'fer-graph-repo-v2',\n"
            f"           set graph_repo_path: '/kaggle/input/fer-graph-repo-v2/graph_repo_v2' trong env.yaml.\n"
        )
