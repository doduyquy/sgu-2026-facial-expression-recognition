"""
graph/io.py — Utility load/save graph cache files (.pt).
"""
import torch
from typing import List, Tuple
from src.graph.structures import PixelGraph


def save_graphs(graphs: List[PixelGraph], path: str) -> None:
    """Lưu danh sách PixelGraph vào file .pt."""
    torch.save(graphs, path)


def load_graphs(path: str) -> Tuple[List[PixelGraph], str]:
    """
    Load danh sách PixelGraph từ file .pt.

    Returns:
        graphs: list of PixelGraph
        load_mode: "torch_list" (duy nhất hiện tại)
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list):
        raise ValueError(f"File .pt phải chứa list[PixelGraph], nhận: {type(data)}")
    return data, "torch_list"
