# Pixel-preserving Motif V2 Baseline and Version C

## 1. Baseline B hiện tại

Baseline chính hiện tại:

```text
Pixel-preserving Motif V2 + MotifGuidedGNN
```

Experiment config:

```text
configs/experiments/pixel_motif_baseline_b.yaml
```

Training config:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Model:

```text
src/models/motif_guided_gnn.py
```

Kết quả tham chiếu:

```text
Accuracy    : khoảng 45.11%
Macro F1    : khoảng 0.4196
Weighted F1 : khoảng 0.4380
Best epoch  : khoảng 57
```

## 2. Input của baseline B

Mỗi ảnh được biểu diễn bằng K selected motif subgraphs:

```text
x                 [B, K, 41]
mask              [B, K]
match_scores       [B, K]
matched_class      [B, K]
motif_score_vector [B, 7]
edge_index         [B, 2, E]
```

Node ở motif-level graph là một selected subgraph.

## 3. Kiến trúc baseline B

```text
descriptor 41D
+ match_score
+ matched_class one-hot
-> node_encoder
-> MotifGraphSAGELayer x num_layers
-> motif_attention pooling
-> concat motif_score_vector
-> classifier
-> logits [B, 7]
```

Model B đã chứng minh motif-selected subgraphs + GNN tốt hơn MLP pooling đơn giản.

## 4. Điểm yếu của baseline B

Mỗi selected subgraph bị nén thành descriptor thống kê 41D. Điều này có thể mất:

```text
thứ tự không gian nội bộ
quan hệ node-node thật
topology cục bộ
chi tiết gradient/intensity theo pixel
```

Do đó câu hỏi nghiên cứu tiếp theo là: có nên encode graph nội bộ của từng selected subgraph không?

## 5. Version C: HierarchicalMotifGNN

Experiment config:

```text
configs/experiments/hierarchical_motif_gnn_c.yaml
```

Training config:

```text
configs/hierarchical_motif_gnn.yaml
```

Model files:

```text
src/models/internal_subgraph_encoder.py
src/models/hierarchical_motif_gnn.py
```

## 6. Input bổ sung của C

C dùng thêm:

```text
sub_x          [B, K, Nmax, 7]
sub_node_mask  [B, K, Nmax]
sub_adj        [B, K, Nmax, Nmax]
```

Các tensor này được dựng bởi `PixelMotifDataset` khi `return_subgraph_tensors=true`.

## 7. Kiến trúc C

```text
InternalPixelSubgraphEncoder:
    sub_x, sub_adj, sub_node_mask
    -> dense GraphSAGE nhỏ
    -> readout mean/max/mean_max
    -> z_internal [B, K, internal_out_dim]

Motif node feature:
    concat(
        z_internal,
        descriptor 41D,
        match_score,
        matched_disc_score,
        matched_class one-hot
    )

Motif-level GNN:
    GraphSAGE giống baseline B
    -> motif_attention pooling
    -> motif_score_vector
    -> classifier
```

## 8. Thiết lập công bằng cho B vs C

C đầu tiên giữ nguyên:

```text
data = pixel_motif_dataset_v2 spatial
loss = weighted_ce
use_descriptor = true
use_motif_score_vector = true
motif_use_edge_attr = false
no CNN
no prototype loss
no rich motif edge
```

Khác biệt chính:

```text
B: descriptor-only
C: descriptor + internal pixel-subgraph GNN
```

## 9. Lệnh chạy

Baseline B:

```bash
python -m scripts.run_experiment --config pixel_motif_baseline_b
```

Version C:

```bash
python -m scripts.run_experiment --config hierarchical_motif_gnn_c
```

Debug C:

```bash
python -m scripts.run_experiment --config hierarchical_motif_gnn_c --debug_only
```
