# Xu Ly Data - Current Data Pipeline

File này mô tả lớp xử lý dữ liệu hiện tại. Phần lớn ý tưởng cũ đã được triển khai xong; bản này ghi lại trạng thái thực tế của repo.

Nếu cần tìm chính xác file/hàm/class để sửa, đọc:

```text
BanThietKeHeThong/ProjectMap.md
```

## 1. Input gốc

Workflow chính chỉ cần 3 file CSV FER-2013:

```text
train.csv
val.csv
test.csv
```

Notebook Kaggle chính sẽ scan `/kaggle/input` để tìm folder có đủ 3 file này.

## 2. Raw data layer

Files:

```text
data/raw_types.py
data/raw_fer_dataset.py
```

Vai trò:

```text
đọc CSV
parse pixels thành ảnh [48, 48]
đọc label
trả RawSample
```

Raw layer không graph hóa, không tạo motif.

## 3. Canonical graph layer

Files:

```text
data/graph_types.py
data/shared_graph_builder.py
data/canonical_graph_builder.py
data/graph_repository.py
data/graph_resolver.py
data/chunked_graph_dataset.py
```

Ba kiểu dữ liệu chính:

```text
SharedGraphStructure
PixelGraphSample
ResolvedPixelGraph
```

### SharedGraphStructure

Phần dùng chung cho mọi ảnh:

```text
height = 48
width = 48
connectivity = 8
edge_index
edge_attr_static = dx, dy, dist
```

### PixelGraphSample

Phần riêng từng ảnh:

```text
graph_id
label
split
usage
node_features [2304, 7]
edge_attr_dynamic
metadata
```

### ResolvedPixelGraph

Khi cần full graph:

```text
ResolvedPixelGraph = SharedGraphStructure + PixelGraphSample
edge_attr = concat(edge_attr_static, edge_attr_dynamic)
```

## 4. Graph repository

Output của stage graph_repo:

```text
artifacts/graph_repo/
  manifest.pt
  shared/shared_graph.pt
  train/chunk_*.pt
  val/chunk_*.pt
  test/chunk_*.pt
```

Script build:

```text
scripts/build_graph_repository.py
```

Trong workflow template, không gọi script này trực tiếp từ notebook. Nó được gọi bởi:

```text
src/pipeline/artifact_builder.py
```

## 5. Pixel motif data layer

Từ graph repo, pipeline build tiếp:

```text
pixel_candidate_subgraphs_v2
pixel_motif_bank_v2
pixel_motif_dataset_v2
```

Atomic scripts:

```text
scripts/precompute_pixel_candidate_subgraphs.py
scripts/build_pixel_motif_bank.py
scripts/precompute_pixel_motif_dataset.py
```

## 6. Pixel motif dataset contract

Output:

```text
artifacts/pixel_motif_dataset_v2/train_pixel_motif.pt
artifacts/pixel_motif_dataset_v2/val_pixel_motif.pt
artifacts/pixel_motif_dataset_v2/test_pixel_motif.pt
artifacts/pixel_motif_dataset_v2/meta.pt
```

Mỗi sample cần có:

```text
x                    [K, 41]
mask                 [K]
edge_index            [2, E]
edge_attr             [E, A]
match_scores          [K]
matched_class         [K]
matched_motif_id      [K]
matched_disc_score    [K]
motif_score_vector    [7]
node_indices          [K, Nmax]
node_mask             [K, Nmax]
label
```

`node_indices` là bắt buộc cho HierarchicalMotifGNN.

## 7. Dataloader hiện tại

Files:

```text
src/data/pixel_motif_dataset.py
src/data/dataloader.py
```

Với baseline B, dataloader trả descriptor/motif batch bình thường.

Với C, dataloader dựng thêm:

```text
sub_x          [B, K, Nmax, 7]
sub_node_mask  [B, K, Nmax]
sub_adj        [B, K, Nmax, Nmax]
```

Nguồn dựng:

```text
node_indices từ pixel motif dataset
node_features từ graph_repo
shared graph adjacency
```

## 8. Workflow chính

Tất cả data build/reuse chạy qua:

```bash
python -m scripts.run_experiment --config <experiment>
```

Ví dụ:

```bash
python -m scripts.run_experiment --config hierarchical_motif_gnn_c
```

Không dùng workflow publish dataset rời rồi train từ Kaggle input nữa.

## 9. Kiểm tra nhanh

Inspect pixel motif dataset:

```bash
python scripts/inspect_pixel_motif_dataset.py --data_dir artifacts/pixel_motif_dataset_v2
```

Debug C:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c \
  --out_root artifacts \
  --debug_only \
  --no_wandb
```

## 10. Tóm tắt

Data pipeline hiện tại đã chốt:

```text
CSV -> graph_repo -> candidates -> motif_bank -> pixel_motif_dataset -> model
```

Khi thêm model mới, không sửa data pipeline nếu model vẫn dùng cùng pixel motif dataset contract.
