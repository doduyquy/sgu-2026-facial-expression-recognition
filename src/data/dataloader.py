"""
Dataloader factory cho GNN FER-2013.

Hỗ trợ 2 chế độ (tự động detect từ file có trong graph_cache_dir):
    1. Vector cache  (*_vectors.pt)  → RAM nhỏ (~1 MB), dùng cho MLP Baseline
    2. Graph cache   (*_graphs.pt)   → RAM lớn (~24 GB), dùng cho GCN tương lai

Tương lai: thêm build_pyg_dataloader() cho GCN/GraphSAGE.
"""
import os
from torch.utils.data import DataLoader


def _detect_cache_mode(cache_dir: str) -> str:
    """
    Phát hiện loại cache trong thư mục.
    Ưu tiên vector cache (nhẹ hơn) nếu có cả 2.
    """
    has_vector = os.path.exists(os.path.join(cache_dir, "train_vectors.pt"))
    has_graph  = os.path.exists(os.path.join(cache_dir, "train_graphs.pt"))

    if has_vector:
        return "vector"
    elif has_graph:
        return "graph"
    else:
        raise FileNotFoundError(
            f"\n[ERROR] Không tìm thấy cache trong: {cache_dir}\n"
            f"  Kaggle : Add dataset .pt vào notebook, set graph_cache_path trong base.yaml.\n"
            f"  Local  : Chạy scripts/build_vector_cache.py (KHUYẾN NGHỊ)\n"
            f"           hoặc scripts/build_graph_cache.py  (tốn ~24GB RAM)\n"
        )


def build_dataloader(config: dict, graph_cache_dir: str):
    """
    Build DataLoader từ cache files trong graph_cache_dir.

    Args:
        config:          full config dict (từ load_config)
        graph_cache_dir: thư mục chứa *_vectors.pt hoặc *_graphs.pt

    Returns:
        train_loader, val_loader, test_loader, input_dim
    """
    mode = _detect_cache_mode(graph_cache_dir)
    print(f"--- Cache mode: [{mode}]  ←  {graph_cache_dir}")

    batch_size  = config["training"].get("batch_size", 128)
    num_workers = config.get("num_workers", config["data"].get("num_workers", 0))

    if mode == "vector":
        return _build_from_vector_cache(graph_cache_dir, batch_size, num_workers)
    else:
        return _build_from_graph_cache(graph_cache_dir, config, batch_size, num_workers)


# ────────────────────────────────────────────────────────────────────────────
#  Mode 1: Vector cache (khuyến nghị cho MLP Baseline)
# ────────────────────────────────────────────────────────────────────────────

def _build_from_vector_cache(cache_dir, batch_size, num_workers):
    from src.data.vector_cache_dataset import VectorCacheDataset

    train_ds = VectorCacheDataset(os.path.join(cache_dir, "train_vectors.pt"))
    val_ds   = VectorCacheDataset(os.path.join(cache_dir, "val_vectors.pt"))
    test_ds  = VectorCacheDataset(os.path.join(cache_dir, "test_vectors.pt"))

    input_dim = train_ds.get_input_dim()

    print(f"--- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"--- Input dim (graph vector): {input_dim}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, input_dim


# ────────────────────────────────────────────────────────────────────────────
#  Mode 2: Graph cache (dùng khi cần full PixelGraph cho GCN tương lai)
# ────────────────────────────────────────────────────────────────────────────

def _build_from_graph_cache(cache_dir, config, batch_size, num_workers):
    from src.data.graph_vector_dataset import GraphVectorDataset
    from src.features.graph_vectorizer import GraphVectorizer

    vectorizer = GraphVectorizer(use_mean=True, use_std=True, use_max=True)

    train_ds = GraphVectorDataset(os.path.join(cache_dir, "train_graphs.pt"), vectorizer)
    val_ds   = GraphVectorDataset(os.path.join(cache_dir, "val_graphs.pt"),   vectorizer)
    test_ds  = GraphVectorDataset(os.path.join(cache_dir, "test_graphs.pt"),  vectorizer)

    input_dim = train_ds.get_input_dim()

    print(f"--- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"--- Input dim (graph vector): {input_dim}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, input_dim
