# FER-2013 Pixel Motif GNN Architecture Context

File này là bản context ngắn gọn để gửi cho ChatGPT/Codex khi cần tiếp tục dự án. Mục tiêu là giúp người đọc hiểu ngay:

- dự án đang nghiên cứu gì
- baseline tốt nhất hiện tại là gì
- bản cải tiến đang thử là gì
- workflow template hiện tại chạy ra sao
- khi thêm model mới thì phải sửa những file nào

## 1. Mục tiêu nghiên cứu

Dự án phân loại cảm xúc FER-2013 bằng hướng graph/motif thay vì CNN.

Ảnh FER-2013 kích thước `48x48` được biểu diễn thành pixel graph:

```text
mỗi pixel = một node
8-neighbor connectivity = edges
node feature dim = 7
edge feature = static + dynamic
```

Node features mặc định:

```text
intensity
x_norm
y_norm
gx
gy
grad_mag
local_contrast
```

Luồng motif hiện tại không train trực tiếp full graph. Thay vào đó:

```text
CSV FER-2013
-> graph_repo
-> candidate pixel subgraphs
-> motif bank
-> selected motif subgraphs per image
-> image-level classifier
```

Mục tiêu nghiên cứu chính hiện tại:

```text
Kiểm tra xem cấu trúc pixel nội bộ của từng selected motif subgraph
có bổ sung thông tin hữu ích ngoài descriptor handcrafted 41D hay không.
```

## 2. Baseline B tốt nhất hiện tại

Baseline tốt nhất hiện tại là:

```text
B = Pixel-preserving Motif V2 + descriptor-only Motif Guided GNN
```

Config:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Model:

```text
src/models/motif_guided_gnn.py
class MotifGuidedGNN
```

Pipeline B:

```text
selected subgraph descriptor [41D]
+ match_score
+ matched_class one-hot
-> motif-level GraphSAGE giữa K selected subgraphs
-> motif_attention pooling
-> concat motif_score_vector nếu bật
-> classifier 7 emotion
```

Kết quả tốt nhất đã ghi nhận:

```text
Accuracy:    khoảng 45.11%
Macro F1:    khoảng 0.4196
Weighted F1: khoảng 0.4380
Best epoch:  khoảng 57
```

Baseline B là mốc so sánh chính. Không nên thay đổi behavior của model/config này khi thử model mới.

## 3. Bản cải tiến C hiện tại

Bản cải tiến hiện tại là:

```text
C = HierarchicalMotifGNN
```

Config model:

```text
configs/hierarchical_motif_gnn.yaml
```

Experiment config:

```text
configs/experiments/hierarchical_motif_gnn_c.yaml
```

Model files:

```text
src/models/internal_subgraph_encoder.py
src/models/hierarchical_motif_gnn.py
```

Ý tưởng:

```text
Thay vì chỉ dùng descriptor 41D cho mỗi selected subgraph,
lấy lại pixel nodes thật bên trong selected subgraph,
encode bằng một internal dense GraphSAGE nhỏ,
rồi concat embedding này với descriptor 41D và motif metadata.
```

Pipeline C:

```text
For each image:
    selected subgraphs K

    For each selected subgraph i:
        sub_x_i          [Nmax, 7]
        sub_adj_i        [Nmax, Nmax]
        sub_node_mask_i  [Nmax]
        -> InternalPixelSubgraphEncoder
        -> z_i [internal_out_dim]

    motif_node_i = concat(
        z_i,
        descriptor_i [41],
        match_score_i,
        matched_disc_score_i,
        matched_class_onehot_i
    )

    motif_node_features [K, D_new]
    -> motif-level GraphSAGE như baseline B
    -> motif_attention pooling
    -> concat motif_score_vector
    -> classifier
    -> logits [B, 7]
```

Quan trọng:

```text
Không dùng CNN.
Không bật rich motif-level edge_attr ở bản C đầu tiên.
Không dùng prototype/contrastive loss ở bản C đầu tiên.
Không bỏ descriptor 41D ở bản C đầu tiên.
Training loss vẫn là weighted CE giống baseline B.
```

Mục tiêu so sánh:

```text
B = descriptor-only motif GNN
C = descriptor + internal pixel-subgraph GNN
```

## 4. Dữ liệu đầu vào của model

Pixel motif dataset hiện trả các key chính:

```text
x                    [B, K, 41]
mask                 [B, K]
edge_index            [B, 2, E]
edge_attr             [B, E, A]
edge_valid            [B, E]
match_scores          [B, K]
matched_class         [B, K]
matched_motif_id      [B, K]
matched_disc_score    [B, K]
motif_score_vector    [B, 7]
label / y             [B]
```

Với `HierarchicalMotifGNN`, loader bổ sung:

```text
sub_x                 [B, K, Nmax, 7]
sub_node_mask          [B, K, Nmax]
sub_adj                [B, K, Nmax, Nmax]
```

Các tensor này được dựng từ:

```text
pixel_motif_dataset_v2: node_indices, node_mask
graph_repo: node_features thật + shared graph adjacency
```

Không được fake `node_indices`. Nếu artifact thiếu `node_indices`, phải rebuild pixel motif dataset.

## 5. Các file active quan trọng

### Workflow template

```text
kaggle_pixel_motif_end_to_end.ipynb
scripts/run_experiment.py
src/pipeline/artifact_builder.py
src/pipeline/experiment_runner.py
```

`scripts/run_experiment.py` là entrypoint duy nhất cho experiment. Notebook chỉ đổi tên experiment.

### Experiment configs

```text
configs/experiments/pixel_motif_baseline_b.yaml
configs/experiments/hierarchical_motif_gnn_c.yaml
```

### Model configs

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
configs/hierarchical_motif_gnn.yaml
```

### Data pipeline atomic scripts

Pipeline API gọi trực tiếp các atomic script này:

```text
scripts/build_graph_repository.py
scripts/precompute_pixel_candidate_subgraphs.py
scripts/build_pixel_motif_bank.py
scripts/precompute_pixel_motif_dataset.py
scripts/inspect_pixel_candidate_subgraphs.py
scripts/inspect_pixel_motif_bank.py
scripts/inspect_pixel_motif_dataset.py
scripts/audit_pixel_motif_dataset.py
```

### Train/eval

```text
scripts/train.py
scripts/debug_hierarchical_batch.py
src/data/dataloader.py
src/data/pixel_motif_dataset.py
src/training/trainer.py
src/evaluation/evaluator.py
```

### Model registry

```text
src/models/__init__.py
```

Model mới phải được register ở đây.

## 6. Workflow template hiện tại

Trên Kaggle, chỉ cần notebook:

```text
kaggle_pixel_motif_end_to_end.ipynb
```

và Kaggle input dataset chứa:

```text
train.csv
val.csv
test.csv
```

Notebook chạy:

```text
CSV
-> graph_repo
-> pixel_candidate_subgraphs_v2
-> pixel_motif_bank_v2
-> pixel_motif_dataset_v2
-> debug batch nếu cần
-> train
-> evaluate
-> zip outputs
```

Không cần upload sẵn `graph_repo`.
Không cần upload sẵn `pixel_motif_dataset_v2`.
Không còn dùng hai notebook build/train tách rời.

Trong notebook chỉ đổi:

```python
EXPERIMENT = "hierarchical_motif_gnn_c"
```

hoặc:

```python
EXPERIMENT = "pixel_motif_baseline_b"
```

## 7. Cách thêm model mới

Nếu muốn thử model D, không sửa notebook, không sửa runner, không sửa pipeline data.

Chỉ làm:

1. Thêm file model:

```text
src/models/<model_d>.py
```

2. Register model:

```text
src/models/__init__.py
```

3. Thêm model config nếu cần:

```text
configs/<model_d>.yaml
```

4. Thêm/copy experiment config:

```text
configs/experiments/<experiment_d>.yaml
```

5. Đổi trong notebook:

```python
EXPERIMENT = "<experiment_d>"
```

6. Chạy notebook.

Đây là nguyên tắc thiết kế template hiện tại.

## 8. Commands quan trọng

Debug C bằng artifact local có sẵn:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c \
  --out_root artifacts \
  --debug_only \
  --no_wandb
```

Build-only smoke local:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c \
  --csv_root data \
  --out_root artifacts \
  --build_only \
  --smoke \
  --no_wandb
```

Run C end-to-end trên Kaggle:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c
```

Run B end-to-end trên Kaggle:

```bash
python -m scripts.run_experiment \
  --config pixel_motif_baseline_b
```

## 9. Legacy

Các file cũ đã được gom vào:

```text
legacy/
```

Ví dụ:

```text
legacy/configs/
legacy/scripts/
legacy/deprecated_notebooks/
```

Chúng không bị xóa để còn tham khảo, nhưng không phải workflow chính.

Không nên dùng lại legacy trừ khi có lý do rõ ràng.

## 10. Tóm tắt cho ChatGPT/Codex

Nếu cần tiếp tục code, hãy hiểu rằng:

```text
Project hiện là template experiment-driven cho FER-2013 pixel motif GNN.
Baseline chính là B: motif_guided_gnn descriptor-only, macro F1 khoảng 0.4196.
Cải tiến hiện tại là C: hierarchical_motif_gnn, thêm internal pixel-subgraph GNN.
Workflow chính là kaggle_pixel_motif_end_to_end.ipynb + scripts/run_experiment.py.
Experiment config là nguồn sự thật.
Khi thêm model mới, chỉ thêm model + registry + experiment config.
Không sửa notebook/runner/data pipeline nếu chỉ thay model.
```
