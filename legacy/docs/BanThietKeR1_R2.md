# Ban Thiet Ke R1 R2 - Current Interpretation

File cũ từng bàn nhiều hướng nghiên cứu motif/subgraph. Bản hiện tại chỉ giữ lại phần cần cho trạng thái project hiện nay.

## 1. R1: Baseline đã chốt

R1 hiện không còn là ý tưởng mở. Nó đã được hiện thực thành baseline B:

```text
Pixel-preserving Motif V2 + MotifGuidedGNN
```

Config:

```text
configs/experiments/pixel_motif_baseline_b.yaml
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Kết quả:

```text
Accuracy ~45.11%
Macro F1 ~0.4196
```

R1 chứng minh rằng motif-selected subgraphs + motif-level GNN có tín hiệu phân loại tốt hơn các baseline đơn giản trước đó.

## 2. R2: Cải tiến hiện tại

R2 hiện là version C:

```text
HierarchicalMotifGNN
```

Mục tiêu R2:

```text
Thay descriptor-only subgraph node bằng descriptor + internal pixel-subgraph embedding.
```

File:

```text
configs/experiments/hierarchical_motif_gnn_c.yaml
configs/hierarchical_motif_gnn.yaml
src/models/hierarchical_motif_gnn.py
src/models/internal_subgraph_encoder.py
```

## 3. Không còn ưu tiên các hướng cũ

Các hướng như:

```text
full graph GNN trực tiếp
CNN hybrid
rich edge ở motif-level
prototype contrastive loss
end-to-end motif bank learning
```

không phải R1/R2 hiện tại. Chúng có thể là future work sau khi B/C được so sánh rõ.

## 4. Nguyên tắc triển khai tiếp

Tất cả biến thể mới nên là experiment config mới:

```text
configs/experiments/<name>.yaml
```

Không sửa notebook/runner cho mỗi biến thể.

## 5. Handoff ngắn

Nếu cần nói R1/R2 trong báo cáo:

```text
R1 = xây dựng baseline motif-guided GNN từ selected subgraph descriptors.
R2 = hierarchical extension, encode pixel structure inside each selected subgraph.
```
