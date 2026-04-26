# Motif - Current Definition and Implementation

File này chốt cách hiểu motif hiện tại trong project.

## 1. Motif không phải frequent subgraph thuần túy

Với FER-2013 pixel graph, nếu hiểu motif là subgraph xuất hiện nhiều lần thì rất dễ học ra pattern sáng/tối nhiễu. Project hiện định nghĩa motif theo hướng thực dụng hơn:

```text
Motif = class-related prototype over local pixel-subgraph descriptors.
```

Motif được dùng để chọn các candidate subgraphs có khả năng mang tín hiệu cảm xúc.

## 2. Pipeline motif hiện tại

```text
graph_repo
-> candidate subgraphs
-> descriptor 41D cho từng candidate
-> motif bank theo emotion
-> greedy motif-guided selection
-> top-K selected subgraphs per image
```

Các script active:

```text
scripts/precompute_pixel_candidate_subgraphs.py
scripts/build_pixel_motif_bank.py
scripts/precompute_pixel_motif_dataset.py
```

Các module active:

```text
src/motif_v2/topology.py
src/motif_v2/matching.py
src/motif_v2/types.py
src/motif_v2/io.py
src/motif/motif_scoring.py
```

## 3. Candidate subgraphs

Candidate topologies được build một lần từ shared graph:

```text
src/motif_v2/topology.py
build_candidate_topologies(...)
```

Config mặc định hiện tại:

```text
seed_stride = 4
radii = [1, 2]
max_candidates = 128
coverage_grid = [4, 4]
```

Mỗi topology lưu:

```text
candidate_id
seed_node
radius
node_indices
edge_index_sub
edge_attr_indices
bbox
coverage_cell
num_nodes
num_edges
```

`node_indices` là trace quan trọng cho HierarchicalMotifGNN.

## 4. Descriptor 41D

Mỗi candidate được biến thành descriptor bằng:

```text
node mean/std/min/max
structural stats: num_nodes, num_edges, density
edge mean/std
```

Với node feature dim 7 và edge feature dim 5, descriptor hiện là 41D.

Descriptor 41D là input chính của baseline B.

## 5. Motif bank

Motif bank được build theo class emotion. Mục tiêu là có các prototype đại diện cho subgraph descriptors thường hữu ích trong từng emotion.

File:

```text
scripts/build_pixel_motif_bank.py
```

Output:

```text
artifacts/pixel_motif_bank_v2/pixel_motif_bank.pt
```

## 6. Motif-guided selection

Với mỗi ảnh, candidate descriptors được so khớp với motif bank rồi chọn top-K bằng logic có coverage/diversity.

File:

```text
src/motif_v2/matching.py
greedy_select_with_coverage(...)
```

Output trong pixel motif dataset:

```text
x                    [K, 41]
match_scores          [K]
matched_class         [K]
matched_motif_id      [K]
matched_disc_score    [K]
motif_score_vector    [7]
selected_indices      [K]
node_indices          [K, Nmax]
node_mask             [K, Nmax]
```

## 7. Motif-level graph

Sau selection, mỗi selected subgraph được xem như một node ở motif-level graph.

Motif-level edges hiện dùng directed KNN theo center:

```text
edge_index [2, E]
edge_attr  [E, 3] = dx, dy, dist
```

Trong baseline B và C đầu tiên:

```text
motif_use_edge_attr = false
```

Tức là edge_attr được lưu để inspect/ablation, nhưng model C đầu tiên không dùng rich edge.

## 8. Baseline B dùng motif như thế nào

Baseline B dùng:

```text
descriptor 41D + match_score + matched_class one-hot
```

rồi GraphSAGE ở motif-level.

## 9. Version C dùng motif như thế nào

Version C vẫn giữ toàn bộ motif pipeline của B, nhưng mỗi selected subgraph được encode thêm bằng internal pixel-subgraph GNN dựa trên `node_indices`.

```text
z_internal + descriptor 41D + motif metadata
```

Đây là thay đổi chính của C.
