import math
from typing import List, Tuple

import numpy as np

from configs.graph_config import GraphConfig
from graph.structures import PixelGraph


class ImageGraphBuilder:
    """
    Build graph từ ảnh FER-2013 48x48.

    Baseline node features:
        - intensity
        - x_norm
        - y_norm

    Thiết kế mở rộng:
        - gx
        - gy
        - grad_mag
        - contrast
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self.height = config.image_size
        self.width = config.image_size

        # edge_index cố định cho mọi ảnh cùng kích thước
        self._base_edge_index = self._build_edge_index(
            height=self.height,
            width=self.width,
            connectivity=self.config.connectivity,
        )

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if self.config.normalize_pixels:
            image = image / 255.0
        return image

    def _node_id(self, y: int, x: int) -> int:
        return y * self.width + x

    def _id_to_xy(self, node_id: int) -> Tuple[int, int]:
        y = node_id // self.width
        x = node_id % self.width
        return y, x

    def _get_neighbor_offsets(self, connectivity: int) -> List[Tuple[int, int]]:
        if connectivity == 4:
            return [
                (-1, 0), (1, 0), (0, -1), (0, 1),
            ]
        if connectivity == 8:
            return [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        raise ValueError("connectivity chỉ hỗ trợ 4 hoặc 8")

    def _build_edge_index(self, height: int, width: int, connectivity: int) -> np.ndarray:
        """
        Build edge có hướng 2 chiều.
        Nếu u và v là hàng xóm, lưu cả (u,v) và (v,u).
        """
        offsets = self._get_neighbor_offsets(connectivity)
        edges = []

        for y in range(height):
            for x in range(width):
                u = self._node_id(y, x)

                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        v = self._node_id(ny, nx)
                        edges.append((u, v))

        edge_index = np.array(edges, dtype=np.int64).T  # [2, M]
        return edge_index

    def _compute_gx_gy(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finite difference nhẹ.
        Dùng để mở rộng feature sau này.
        """
        gx = np.zeros_like(image, dtype=np.float32)
        gy = np.zeros_like(image, dtype=np.float32)

        gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
        gx[:, 0] = image[:, 1] - image[:, 0]
        gx[:, -1] = image[:, -1] - image[:, -2]

        gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
        gy[0, :] = image[1, :] - image[0, :]
        gy[-1, :] = image[-1, :] - image[-2, :]

        return gx, gy

    def _compute_grad_mag(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    def _compute_local_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        contrast = pixel - mean(3x3 neighborhood)
        """
        padded = np.pad(image, pad_width=1, mode="reflect")
        contrast = np.zeros_like(image, dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                patch = padded[y:y + 3, x:x + 3]
                local_mean = patch.mean()
                contrast[y, x] = image[y, x] - local_mean

        return contrast

    def _build_node_features(self, image: np.ndarray) -> np.ndarray:
        """
        Output:
            X shape [2304, d]
        theo đúng thứ tự self.config.node_features
        """
        H, W = image.shape
        if (H, W) != (self.height, self.width):
            raise ValueError(f"Image shape phải là {(self.height, self.width)}, nhận {image.shape}")

        gx = gy = grad_mag = contrast = None

        need_gx = "gx" in self.config.node_features
        need_gy = "gy" in self.config.node_features
        need_grad_mag = "grad_mag" in self.config.node_features
        need_contrast = "contrast" in self.config.node_features

        if need_gx or need_gy or need_grad_mag:
            gx, gy = self._compute_gx_gy(image)

        if need_grad_mag:
            grad_mag = self._compute_grad_mag(gx, gy)

        if need_contrast:
            contrast = self._compute_local_contrast(image)

        feats = []

        for y in range(H):
            for x in range(W):
                feat = []

                for name in self.config.node_features:
                    if name == "intensity":
                        feat.append(float(image[y, x]))
                    elif name == "x_norm":
                        feat.append(float(x / (W - 1)))
                    elif name == "y_norm":
                        feat.append(float(y / (H - 1)))
                    elif name == "gx":
                        feat.append(float(gx[y, x]))
                    elif name == "gy":
                        feat.append(float(gy[y, x]))
                    elif name == "grad_mag":
                        feat.append(float(grad_mag[y, x]))
                    elif name == "contrast":
                        feat.append(float(contrast[y, x]))
                    else:
                        raise ValueError(f"Node feature không hỗ trợ: {name}")

                feats.append(feat)

        return np.array(feats, dtype=np.float32)

    def _build_edge_attr(self, image: np.ndarray) -> np.ndarray:
        """
        Baseline edge features:
            - dx
            - dy
            - dist
            - delta_intensity
            - intensity_similarity

        similarity = exp(-alpha * abs(I_i - I_j))
        """
        edge_index = self._base_edge_index
        alpha = self.config.intensity_similarity_alpha
        attrs = []

        for k in range(edge_index.shape[1]):
            u = int(edge_index[0, k])
            v = int(edge_index[1, k])

            uy, ux = self._id_to_xy(u)
            vy, vx = self._id_to_xy(v)

            dx = float(vx - ux)
            dy = float(vy - uy)
            dist = float(math.sqrt(dx * dx + dy * dy))

            Iu = float(image[uy, ux])
            Iv = float(image[vy, vx])

            delta_intensity = abs(Iu - Iv)
            intensity_similarity = math.exp(-alpha * delta_intensity)

            feat_map = {
                "dx": dx,
                "dy": dy,
                "dist": dist,
                "delta_intensity": delta_intensity,
                "intensity_similarity": intensity_similarity,
            }

            attrs.append([feat_map[name] for name in self.config.edge_features])

        return np.array(attrs, dtype=np.float32)

    def build_graph(
        self,
        image: np.ndarray,
        label: int,
        image_id: int,
        split_name: str,
        usage: str = "",
    ) -> PixelGraph:
        """
        Convert 1 ảnh thành 1 PixelGraph.
        """
        image = self._normalize_image(image)

        if image.shape != (self.height, self.width):
            raise ValueError(
                f"Image shape phải là {(self.height, self.width)}, nhận {image.shape}"
            )

        node_features = self._build_node_features(image)
        edge_index = self._base_edge_index.copy()
        edge_attr = self._build_edge_attr(image)

        graph = PixelGraph(
            graph_id=int(image_id),
            label=int(label),
            split=split_name,
            usage=usage,
            height=self.height,
            width=self.width,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            image=image.astype(np.float32) if self.config.save_image_in_graph else None,
            metadata={
                "node_feature_names": list(self.config.node_features),
                "edge_feature_names": list(self.config.edge_features),
            },
        )
        return graph