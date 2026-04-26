# Pixel-preserving Motif V2 - Current Pipeline

File này mô tả pipeline data hiện tại của Pixel-preserving Motif V2.

## 1. Mục tiêu

Giữ nguồn thông tin pixel-level, nhưng không train trực tiếp full graph 2304 node. Thay vào đó:

```text
full pixel graph
-> local candidate subgraphs
-> motif-guided top-K selected subgraphs
-> image-level classifier
```

## 2. End-to-end stages

Pipeline data gồm 4 stage:

```text
1. graph_repo
2. candidates
3. motif_bank
4. motif_dataset
```

Trong workflow template, các stage này được gọi bởi:

```text
src/pipeline/artifact_builder.py
```

thông qua:

```text
scripts/run_experiment.py
```

## 3. Stage graph_repo

Input:

```text
train.csv
val.csv
test.csv
```

Script:

```text
scripts/build_graph_repository.py
```

Output:

```text
artifacts/graph_repo/shared/shared_graph.pt
artifacts/graph_repo/train/chunk_*.pt
artifacts/graph_repo/val/chunk_*.pt
artifacts/graph_repo/test/chunk_*.pt
artifacts/graph_repo/manifest.pt
```

## 4. Stage candidates

Script:

```text
scripts/precompute_pixel_candidate_subgraphs.py
```

Output:

```text
artifacts/pixel_candidate_subgraphs_v2/train_pixel_candidates.pt
artifacts/pixel_candidate_subgraphs_v2/val_pixel_candidates.pt
artifacts/pixel_candidate_subgraphs_v2/test_pixel_candidates.pt
artifacts/pixel_candidate_subgraphs_v2/meta.pt
```

Mỗi candidate có descriptor 41D, center, bbox, coverage cell. Candidate topology nằm trong `meta.pt` và chứa `node_indices`.

## 5. Stage motif_bank

Script:

```text
scripts/build_pixel_motif_bank.py
```

Output:

```text
artifacts/pixel_motif_bank_v2/pixel_motif_bank.pt
```

Motif bank là bộ prototype theo emotion class.

## 6. Stage motif_dataset

Script:

```text
scripts/precompute_pixel_motif_dataset.py
```

Output:

```text
artifacts/pixel_motif_dataset_v2/train_pixel_motif.pt
artifacts/pixel_motif_dataset_v2/val_pixel_motif.pt
artifacts/pixel_motif_dataset_v2/test_pixel_motif.pt
artifacts/pixel_motif_dataset_v2/meta.pt
```

Mỗi sample là top-K selected subgraphs cho một ảnh.

## 7. Data keys chính

```text
x                    [K, 41]
mask                 [K]
centers              [K, 2]
bbox                 [K, 4]
selected_indices      [K]
node_indices          [K, Nmax]
node_mask             [K, Nmax]
edge_index            [2, E]
edge_attr             [E, A]
match_scores          [K]
matched_class         [K]
matched_motif_id      [K]
matched_disc_score    [K]
motif_score_vector    [7]
coverage_cell         [K]
label
```

## 8. Vì sao pixel-preserving

Baseline B chỉ dùng descriptor 41D, nhưng artifact vẫn giữ `node_indices`. Nhờ vậy version C có thể quay lại pixel nodes thật trong từng selected subgraph:

```text
node_indices + graph_repo node_features -> sub_x
node_indices + shared adjacency -> sub_adj
```

Đây là điểm khác biệt quan trọng so với motif dataset chỉ có descriptor.

## 9. Rich edge status

Từng có thử nghiệm rich motif-level edge_attr 13D. Hiện nó nằm ở legacy/ablation, không phải workflow chính.

Experiment B và C hiện tại đều dùng:

```text
edge_attr_mode = spatial
motif_use_edge_attr = false
```

## 10. Workflow chính

Không chạy script stage bằng tay trong notebook. Notebook chỉ gọi:

```bash
python -m scripts.run_experiment --config <experiment>
```

`src/pipeline/artifact_builder.py` sẽ build hoặc reuse artifact theo config.
