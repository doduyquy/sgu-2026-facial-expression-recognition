# Graph Node Edge - Current Contract

File này mô tả graph representation hiện tại của project FER-2013 pixel motif. Đây là contract dữ liệu, không phải hướng phát triển cũ.

## 1. Vì sao dùng pixel graph

Ảnh FER-2013 là grayscale `48x48`. Project biểu diễn mỗi ảnh thành graph:

```text
1 pixel = 1 node
48 x 48 = 2304 nodes
8-neighbor connectivity
```

Mục tiêu là giữ chi tiết pixel-level đủ lâu để motif/subgraph selection có thể khai thác các pattern cục bộ của biểu cảm.

## 2. SharedGraphStructure

Topology dùng chung cho mọi ảnh:

```text
data/graph_types.py
class SharedGraphStructure
```

Chứa:

```text
height, width
connectivity
edge_index              [2, M]
edge_attr_static        [M, S]
static_feature_names
config_dict
```

Vì mọi ảnh đều là `48x48`, `edge_index` và static edge attributes giống nhau cho toàn dataset.

Static edge features hiện tại:

```text
dx
dy
dist
```

## 3. PixelGraphSample

Sample riêng từng ảnh:

```text
data/graph_types.py
class PixelGraphSample
```

Chứa:

```text
graph_id
label
split
usage
height, width
node_features           [2304, 7]
edge_attr_dynamic       [M, D]
node_feature_names
dynamic_feature_names
```

Node feature dim hiện tại là `7`.

## 4. Node features hiện tại

Trong `configs/base.yaml`:

```text
intensity
x_norm
y_norm
gx
gy
grad_mag
local_contrast
```

Ý nghĩa:

```text
intensity       grayscale normalized
x_norm, y_norm  vị trí pixel chuẩn hóa
gx, gy          gradient theo x/y
grad_mag        độ lớn gradient
local_contrast  tương phản cục bộ
```

Đây là feature set đang được dùng cho graph repo và hierarchical internal subgraph encoder.

## 5. Dynamic edge features

Trong `configs/base.yaml`:

```text
delta_intensity
intensity_similarity
```

Dynamic edge features phụ thuộc từng ảnh.

Khi resolve full graph:

```text
edge_attr = concat(edge_attr_static, edge_attr_dynamic)
```

File chịu trách nhiệm:

```text
data/graph_resolver.py
```

## 6. ResolvedPixelGraph

Được tạo từ:

```text
SharedGraphStructure + PixelGraphSample
```

thành:

```text
ResolvedPixelGraph
```

Chứa:

```text
node_features [2304, 7]
edge_index    [2, M]
edge_attr     [M, S + D]
```

Downstream code không nên tự ghép graph bằng tay; dùng resolver hoặc dataset đã chuẩn bị.

## 7. Vai trò trong Pixel Motif V2

Full pixel graph không được đưa trực tiếp vào GNN classifier chính. Nó được dùng để tạo:

```text
candidate subgraphs
subgraph descriptors 41D
node_indices trace
motif bank
selected motif subgraphs
```

`node_indices` giữ liên kết từ selected subgraph về pixel nodes thật. Đây là điểm quan trọng cho `HierarchicalMotifGNN`.

## 8. Internal subgraph tensors cho C

Với model C, dataset loader dựng:

```text
sub_x          [B, K, Nmax, 7]
sub_node_mask  [B, K, Nmax]
sub_adj        [B, K, Nmax, Nmax]
```

Nguồn:

```text
node_indices từ pixel_motif_dataset_v2
node_features từ graph_repo
shared adjacency từ SharedGraphStructure
```

`sub_adj` hiện là dense adjacency nhỏ, không dùng edge_attr nội bộ ở bản đầu.

## 9. Không nên nhầm với legacy

Các graph cache/vector cache cũ đã gom vào `legacy/`. Workflow chính hiện tại dựa trên:

```text
data/graph_repository.py
data/chunked_graph_dataset.py
data/graph_resolver.py
```

Không dùng graph cache cũ cho thí nghiệm B/C hiện tại.
