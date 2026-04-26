# Trien Khai Ky Thuat Tu Baseline B Den C

File này ghi trạng thái triển khai kỹ thuật hiện tại từ baseline B sang version C.

## 1. Baseline B đã có

B hiện là:

```text
selected subgraph descriptor [41D]
+ match_score
+ matched_class one-hot
-> motif-level GraphSAGE
-> motif_attention pooling
-> motif_score_vector
-> classifier
```

File:

```text
src/models/motif_guided_gnn.py
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Kết quả tham chiếu:

```text
Accuracy ~45.11%
Macro F1 ~0.4196
```

## 2. Vấn đề của B

Descriptor 41D là thống kê thủ công. Nó hữu ích nhưng có thể mất topology nội bộ của selected subgraph.

Do đó hướng C là thêm một GNN nhỏ bên trong từng selected subgraph.

## 3. Version C đã triển khai

Files:

```text
src/models/internal_subgraph_encoder.py
src/models/hierarchical_motif_gnn.py
configs/hierarchical_motif_gnn.yaml
configs/experiments/hierarchical_motif_gnn_c.yaml
scripts/debug_hierarchical_batch.py
```

Registry:

```text
src/models/__init__.py
"hierarchical_motif_gnn" -> HierarchicalMotifGNN
```

## 4. Dataset support đã triển khai

File:

```text
src/data/pixel_motif_dataset.py
src/data/dataloader.py
```

Khi config/model yêu cầu hierarchical, dataloader trả thêm:

```text
sub_x
sub_node_mask
sub_adj
```

Các tensor này không lưu fake trong artifact. Chúng được dựng từ:

```text
node_indices trong pixel_motif_dataset_v2
graph_repo node_features
shared graph adjacency
```

## 5. InternalPixelSubgraphEncoder

Input:

```text
sub_x          [B, K, N, 7]
sub_adj        [B, K, N, N]
sub_node_mask  [B, K, N]
```

Output:

```text
z_internal [B, K, internal_out_dim]
```

Kiến trúc:

```text
Linear projection 7 -> hidden_dim
Dense GraphSAGE layers
LayerNorm + GELU + Dropout
masked mean/max/mean_max readout
```

Không dùng PyG, vì Nmax nhỏ và dense tensor dễ debug.

## 6. HierarchicalMotifGNN

Input motif node feature:

```text
z_internal
+ descriptor 41D
+ match_score
+ matched_disc_score
+ matched_class one-hot
```

Sau đó dùng motif-level GraphSAGE tương tự B.

C đầu tiên giữ:

```text
use_descriptor = true
motif_use_edge_attr = false
use_motif_score_vector = true
loss = weighted_ce
```

## 7. Kiểm chứng đã chạy

Debug command local:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c \
  --out_root artifacts \
  --debug_only \
  --no_wandb
```

Shape đã kiểm chứng:

```text
x              : (2, 32, 41)
sub_x          : (2, 32, 25, 7)
sub_adj        : (2, 32, 25, 25)
sub_node_mask  : (2, 32, 25)
logits         : (2, 7)
```

Checks:

```text
valid subgraphs have nodes: true
sub_adj finite: true
sub_x finite: true
logits finite: true
backward runs
```

## 8. Việc chưa làm

Chưa train full C để lấy kết quả test chính thức.

Không triển khai trong C đầu tiên:

```text
CNN
Graph Transformer
rich motif edge_attr
prototype contrastive loss
attention phức tạp trong internal encoder
end-to-end motif bank learning
```

## 9. Bước tiếp theo hợp lý

1. Chạy Kaggle full experiment C.
2. So sánh với B theo accuracy, macro F1, weighted F1, per-class F1, confusion matrix.
3. Nếu C yếu hơn, chạy ablation:

```text
internal_out_dim 64 vs 128
internal_num_layers 1 vs 2
internal_readout mean vs max vs mean_max
use_descriptor true vs false
```

Tất cả ablation nên thêm bằng config experiment mới, không sửa notebook/runner.
