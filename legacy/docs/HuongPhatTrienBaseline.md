# Huong Phat Trien Baseline - Current Status

File này cập nhật lại tình trạng baseline. Các baseline MLP/subgraph/full-graph cũ đã chuyển vào legacy hoặc không còn là trọng tâm.

## 1. Baseline chính hiện tại

Baseline chính không còn là MLP graph vector. Baseline chính hiện tại là:

```text
B = pixel_motif_baseline_b
```

Dùng:

```text
configs/experiments/pixel_motif_baseline_b.yaml
configs/pixel_motif_guided_gnn_motif_norm.yaml
src/models/motif_guided_gnn.py
```

Kết quả:

```text
Accuracy ~45.11%
Macro F1 ~0.4196
```

## 2. Baseline cũ

Các baseline cũ như:

```text
mlp_baseline
subgraph_baseline
subgraph_gnn_baseline
motif_guided_mlp
rich_edges
```

đã được gom vào:

```text
legacy/configs/
legacy/scripts/
```

Chúng chỉ dùng để tham khảo lịch sử, không phải workflow chính.

## 3. Cải tiến đang ưu tiên

Cải tiến hiện tại là:

```text
C = hierarchical_motif_gnn_c
```

Mục tiêu là kiểm tra đóng góp của internal pixel-subgraph GNN.

## 4. Hướng phát triển hợp lý sau C

Chỉ sau khi có kết quả C, mới xét:

```text
ablation internal_out_dim
ablation internal_readout
ablation internal_num_layers
use_descriptor true/false
rich edge motif-level
prototype loss
```

Mỗi hướng nên là một experiment config mới.

## 5. Quy tắc tránh loạn project

Không thêm notebook mới cho từng model.
Không sửa runner cho từng model.
Không đưa file thử nghiệm mới vào root nếu chưa cần.

Thêm model mới theo template:

```text
src/models/<model>.py
src/models/__init__.py
configs/experiments/<experiment>.yaml
```
