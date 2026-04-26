"""Dataset for pixel-preserving motif-selected image-level samples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from data.chunked_graph_dataset import ChunkedGraphDataset


def remap_local_edges(node_indices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Keep full-graph edges whose endpoints are in node_indices and remap them to local ids."""
    valid_nodes = [int(v) for v in torch.as_tensor(node_indices).long().tolist() if int(v) >= 0]
    local = {node_id: local_id for local_id, node_id in enumerate(valid_nodes)}
    if not local or edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    src_out: list[int] = []
    dst_out: list[int] = []
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        src_i = int(src)
        dst_i = int(dst)
        if src_i in local and dst_i in local:
            src_out.append(local[src_i])
            dst_out.append(local[dst_i])
    if not src_out:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src_out, dst_out], dtype=torch.long)


def build_subgraph_tensor_from_node_indices(
    node_features: torch.Tensor,
    full_adj: torch.Tensor,
    node_indices: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded tensors for selected subgraphs from global node ids.

    Returns:
        sub_x:         [K, Nmax, F]
        sub_node_mask: [K, Nmax]
        sub_adj:       [K, Nmax, Nmax]
    """
    node_indices = torch.as_tensor(node_indices).long()
    K, Nmax = node_indices.shape
    F_dim = int(node_features.shape[1])
    if node_mask is None:
        node_mask = node_indices.ge(0)
    else:
        node_mask = torch.as_tensor(node_mask).bool() & node_indices.ge(0)

    sub_x = torch.zeros((K, Nmax, F_dim), dtype=node_features.dtype)
    sub_node_mask = torch.zeros((K, Nmax), dtype=torch.bool)
    sub_adj = torch.zeros((K, Nmax, Nmax), dtype=torch.float32)
    for k in range(K):
        valid = node_mask[k]
        if not bool(valid.any()):
            continue
        nodes = node_indices[k, valid].long()
        n = int(nodes.numel())
        sub_x[k, :n] = node_features[nodes]
        sub_node_mask[k, :n] = True
        sub_adj[k, :n, :n] = full_adj[nodes][:, nodes].to(dtype=torch.float32)
    return sub_x, sub_node_mask, sub_adj


def pad_selected_subgraphs(
    node_features: torch.Tensor,
    full_adj: torch.Tensor,
    node_indices: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for building padded selected-subgraph tensors."""
    return build_subgraph_tensor_from_node_indices(node_features, full_adj, node_indices, node_mask=node_mask)


class PixelMotifDataset(Dataset):
    """Load pixel-preserving motif dataset V2."""

    FILENAMES = {
        "train": "train_pixel_motif.pt",
        "val": "val_pixel_motif.pt",
        "test": "test_pixel_motif.pt",
    }
    _STATS_CACHE: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        normalize_x: bool = False,
        return_subgraph_tensors: bool = False,
        graph_repo_path: str | Path | None = None,
        graph_cache_chunks: int = 1,
    ) -> None:
        if split not in self.FILENAMES:
            raise ValueError(f"Unknown split {split!r}")
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize_x = bool(normalize_x)
        self.return_subgraph_tensors = bool(return_subgraph_tensors)
        self.path = self.data_dir / self.FILENAMES[split]
        if not self.path.exists():
            raise FileNotFoundError(
                f"[PixelMotifDataset] Missing file: {self.path}\n"
                f"  Run scripts/precompute_pixel_motif_dataset.py first."
            )
        try:
            self._samples: List[Dict] = torch.load(self.path, map_location="cpu", weights_only=False)
        except TypeError:
            self._samples = torch.load(self.path, map_location="cpu")
        if not self._samples:
            raise RuntimeError(f"Empty dataset: {self.path}")
        s0 = self._samples[0]
        self._num_subgraphs = int(s0["x"].shape[0])
        self._descriptor_dim = int(s0["x"].shape[1])
        if "node_indices" not in s0 and self.return_subgraph_tensors and "sub_x" not in s0:
            raise RuntimeError(
                "[PixelMotifDataset] This artifact does not contain node_indices. "
                "Rebuild it with scripts/precompute_pixel_motif_dataset.py after generating "
                "pixel candidates that save candidate_topologies/node_indices."
            )
        self._max_nodes = int(s0["node_indices"].shape[1]) if "node_indices" in s0 else int(s0["sub_x"].shape[1])
        self._graph_ds = None
        self._full_adj = None
        if self.return_subgraph_tensors and "sub_x" not in s0:
            if graph_repo_path is None:
                raise RuntimeError(
                    "[PixelMotifDataset] Hierarchical subgraph tensors require graph_repo_path "
                    "because the current pixel motif artifact stores node_indices but not sub_x/sub_adj."
                )
            self._graph_ds = ChunkedGraphDataset(
                repo_root=graph_repo_path,
                split=split,
                resolve=True,
                cache_chunks=graph_cache_chunks,
            )
            if len(self._graph_ds) != len(self._samples):
                raise RuntimeError(
                    f"[PixelMotifDataset] graph_repo split size ({len(self._graph_ds)}) does not match "
                    f"pixel motif split size ({len(self._samples)}) for split={split!r}."
                )
            shared = self._graph_ds.shared
            if shared is None:
                raise RuntimeError("[PixelMotifDataset] ChunkedGraphDataset(resolve=True) did not expose shared graph.")
            self._full_adj = torch.zeros((shared.num_nodes, shared.num_nodes), dtype=torch.float32)
            src = shared.edge_index[0].long()
            dst = shared.edge_index[1].long()
            self._full_adj[dst, src] = 1.0
        self._x_mean: torch.Tensor | None = None
        self._x_std: torch.Tensor | None = None
        if self.normalize_x:
            self._x_mean, self._x_std = self._load_or_compute_train_stats()
        print(
            f"[PixelMotifDataset] Loaded {len(self._samples)} samples from {self.path.name} "
            f"| K={self._num_subgraphs} | D={self._descriptor_dim} | max_nodes={self._max_nodes}"
            f" | normalize_x={self.normalize_x} | subgraph_tensors={self.return_subgraph_tensors}"
        )

    def _load_or_compute_train_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = str(self.data_dir.resolve())
        if cache_key in self._STATS_CACHE:
            return self._STATS_CACHE[cache_key]

        if self.split == "train":
            train_samples = self._samples
        else:
            train_path = self.data_dir / self.FILENAMES["train"]
            try:
                train_samples = torch.load(train_path, map_location="cpu", weights_only=False)
            except TypeError:
                train_samples = torch.load(train_path, map_location="cpu")

        total = torch.zeros(self._descriptor_dim, dtype=torch.float64)
        total_sq = torch.zeros(self._descriptor_dim, dtype=torch.float64)
        count = 0
        for sample in train_samples:
            x = torch.as_tensor(sample["x"]).float()
            mask = torch.as_tensor(sample.get("mask", torch.ones(x.shape[0], dtype=torch.bool))).bool()
            xv = x[mask]
            if xv.numel() == 0:
                continue
            total += xv.double().sum(dim=0)
            total_sq += xv.double().pow(2).sum(dim=0)
            count += int(xv.shape[0])
        if count <= 0:
            raise RuntimeError(f"Cannot compute x normalization stats from {self.data_dir}")
        mean = (total / count).float()
        var = (total_sq / count).float() - mean.pow(2)
        std = var.clamp_min(0.0).sqrt().clamp_min(1e-6)
        self._STATS_CACHE[cache_key] = (mean, std)
        return mean, std

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self._samples[idx]
        label = torch.tensor(int(s["label"]), dtype=torch.long)
        x = torch.as_tensor(s["x"]).float()
        if self.normalize_x:
            x = (x - self._x_mean) / self._x_std
        item = {
            "graph_id": int(s["graph_id"]),
            "x": x,
            "mask": torch.as_tensor(s["mask"]).bool(),
            "centers": torch.as_tensor(s["centers"]).float(),
            "bbox": torch.as_tensor(s["bbox"]).float(),
            "selected_indices": torch.as_tensor(s["selected_indices"]).long(),
            "node_indices": torch.as_tensor(s["node_indices"]).long(),
            "node_mask": torch.as_tensor(s["node_mask"]).bool(),
            "edge_index": torch.as_tensor(s["edge_index"]).long(),
            "edge_attr": torch.as_tensor(s["edge_attr"]).float(),
            "match_scores": torch.as_tensor(s["match_scores"]).float(),
            "matched_class": torch.as_tensor(s["matched_class"]).long(),
            "matched_motif_id": torch.as_tensor(s["matched_motif_id"]).long(),
            "matched_disc_score": torch.as_tensor(s["matched_disc_score"]).float(),
            "motif_score_vector": torch.as_tensor(s["motif_score_vector"]).float(),
            "coverage_cell": torch.as_tensor(s["coverage_cell"]).long(),
            "label": label,
            "y": label,
        }
        if self.return_subgraph_tensors:
            if "sub_x" in s and "sub_adj" in s:
                item["sub_x"] = torch.as_tensor(s["sub_x"]).float()
                item["sub_adj"] = torch.as_tensor(s["sub_adj"]).float()
                item["sub_node_mask"] = torch.as_tensor(s.get("sub_node_mask", s.get("node_mask"))).bool()
            else:
                if self._graph_ds is None or self._full_adj is None:
                    raise RuntimeError("[PixelMotifDataset] graph_repo is not initialized for subgraph tensors.")
                graph = self._graph_ds[idx]
                if int(graph.graph_id) != int(s["graph_id"]):
                    raise RuntimeError(
                        f"[PixelMotifDataset] graph_id mismatch at idx={idx}: "
                        f"pixel_motif={int(s['graph_id'])}, graph_repo={int(graph.graph_id)}"
                    )
                sub_x, sub_node_mask, sub_adj = build_subgraph_tensor_from_node_indices(
                    node_features=graph.node_features.float(),
                    full_adj=self._full_adj,
                    node_indices=item["node_indices"],
                    node_mask=item["node_mask"],
                )
                item["sub_x"] = sub_x.float()
                item["sub_node_mask"] = sub_node_mask.bool()
                item["sub_adj"] = sub_adj.float()
        return item

    @property
    def input_dim(self) -> int:
        return self._descriptor_dim

    @property
    def num_subgraphs(self) -> int:
        return self._num_subgraphs

    @property
    def max_nodes(self) -> int:
        return self._max_nodes
