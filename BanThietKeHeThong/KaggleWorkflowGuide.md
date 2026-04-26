# Kaggle Workflow Guide - Pixel Motif FER-2013

File này là bản hướng dẫn workflow hiện tại của project. Có thể gửi nguyên file này cho ChatGPT/Codex để nó hiểu cách chạy Kaggle đúng, tránh nhầm dataset/config/model.

Nếu cần biết file/hàm/class để sửa code, đọc thêm:

```text
BanThietKeHeThong/ProjectMap.md
```

## 1. Nguyên tắc hiện tại

Workflow Kaggle hiện tại là **single end-to-end notebook**:

```text
kaggle_pixel_motif_end_to_end.ipynb
```

Notebook này không nhận pixel motif dataset prebuilt. Nó chỉ cần CSV FER-2013:

```text
train.csv
val.csv
test.csv
```

Sau đó tự chạy:

```text
CSV
-> graph_repo
-> pixel_candidate_subgraphs_v2
-> pixel_motif_bank_v2
-> pixel_motif_dataset_v2
-> debug batch nếu config yêu cầu
-> train
-> evaluate
-> zip outputs
```

Lý do thiết kế như vậy:

```text
Trước đây có 2 notebook:
1. build dataset
2. train từ Kaggle Dataset đã publish

Cách đó dễ train nhầm dataset/config, nhất là khi có spatial/rich/hierarchical.
Hiện tại data vừa sinh trong cùng run sẽ được train ngay.
```

## 2. Active files

Notebook chính:

```text
kaggle_pixel_motif_end_to_end.ipynb
```

Entrypoint chính:

```text
scripts/run_experiment.py
```

Pipeline API:

```text
src/pipeline/artifact_builder.py
src/pipeline/experiment_runner.py
```

Experiment configs:

```text
configs/experiments/pixel_motif_baseline_b.yaml
configs/experiments/hierarchical_motif_gnn_c.yaml
```

Model configs:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
configs/hierarchical_motif_gnn.yaml
```

Atomic data scripts được pipeline API gọi trực tiếp:

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

Train/debug:

```text
scripts/train.py
scripts/debug_hierarchical_batch.py
```

## 3. Cách chạy trên Kaggle

### Bước 1: Push code lên GitHub

Notebook Kaggle clone repo từ GitHub. Vì vậy trước khi chạy Kaggle phải push code hiện tại lên branch notebook đang dùng.

Trong notebook hiện có các biến:

```python
GITHUB_USERNAME = "doduyquy"
GITHUB_REPO_NAME = "sgu-2026-facial-expression-recognition"
GITHUB_REPO_BRANCH = "Tri_GNN"
```

Nếu đổi branch thì sửa ở cell đầu.

### Bước 2: Add Kaggle input dataset CSV

Chỉ cần dataset có:

```text
train.csv
val.csv
test.csv
```

Không cần add:

```text
graph_repo
pixel_motif_dataset_v2
pixel_motif_dataset_v2_rich_edges
```

Vì workflow mới sẽ build trong `/kaggle/working/artifacts`.

### Bước 3: Chọn experiment

Trong notebook:

```python
EXPERIMENT = "hierarchical_motif_gnn_c"
```

hoặc baseline:

```python
EXPERIMENT = "pixel_motif_baseline_b"
```

Notebook sẽ gọi:

```bash
python -m scripts.run_experiment --config <EXPERIMENT>
```

### Bước 4: Run all

Luồng sẽ tự:

```text
scan /kaggle/input tìm CSV
build artifacts
debug nếu cần
train
evaluate
zip outputs
```

## 4. Experiment config là nguồn sự thật

Mỗi experiment nằm trong:

```text
configs/experiments/
```

Ví dụ `hierarchical_motif_gnn_c.yaml`:

```yaml
experiment:
  name: hierarchical_motif_gnn_c

data:
  recipe: pixel_motif_v2
  csv_root: auto
  artifact_root: /kaggle/working/artifacts
  stage: all
  skip_existing: true
  edge_attr_mode: spatial
  pixel_motif_dir: /kaggle/working/artifacts/pixel_motif_dataset_v2

training:
  config: hierarchical_motif_gnn
  epochs: 80
  debug_batch: true
```

Điều quan trọng:

```text
Notebook không map model/dataset thủ công.
Runner không hard-code model.
Experiment config quyết định data recipe và train config.
```

## 5. Baseline B

Baseline tốt nhất hiện tại:

```text
B = descriptor-only Motif Guided GNN
```

Experiment:

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
Accuracy:    khoảng 45.11%
Macro F1:    khoảng 0.4196
Weighted F1: khoảng 0.4380
```

Run baseline B:

```bash
python -m scripts.run_experiment \
  --config pixel_motif_baseline_b
```

## 6. Cải tiến C

Cải tiến hiện tại:

```text
C = HierarchicalMotifGNN
```

Experiment:

```text
configs/experiments/hierarchical_motif_gnn_c.yaml
```

Training config:

```text
configs/hierarchical_motif_gnn.yaml
```

Models:

```text
src/models/internal_subgraph_encoder.py
src/models/hierarchical_motif_gnn.py
```

Ý tưởng:

```text
selected subgraph pixel nodes thật
-> internal dense GraphSAGE
-> z_internal
-> concat descriptor 41D + motif metadata
-> motif-level GraphSAGE như baseline B
-> classifier
```

Bản C đầu tiên cố ý:

```text
edge_attr_mode = spatial
motif_use_edge_attr = false
use_descriptor = true
loss = weighted_ce
```

Mục tiêu là so sánh công bằng:

```text
B = descriptor-only motif GNN
C = descriptor + internal pixel-subgraph GNN
```

Run C:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c
```

## 7. Commands local để kiểm tra nhanh

Các lệnh này đã được dùng để kiểm chứng workflow trong môi trường:

```bash
conda activate fer-graph
```

Kiểm tra debug batch C từ artifact local có sẵn:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c \
  --out_root artifacts \
  --debug_only \
  --no_wandb
```

Kỳ vọng shape:

```text
x:             [2, 32, 41]
sub_x:         [2, 32, 25, 7]
sub_adj:       [2, 32, 25, 25]
sub_node_mask: [2, 32, 25]
logits:        [2, 7]
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

Help:

```bash
python -m scripts.run_experiment --help
```

## 8. Cách thêm model mới

Nguyên tắc template:

```text
Nếu chỉ thử model mới, không sửa notebook, không sửa runner, không sửa data pipeline.
```

Quy trình:

1. Thêm model:

```text
src/models/<new_model>.py
```

2. Register model:

```text
src/models/__init__.py
```

3. Thêm model config nếu cần:

```text
configs/<new_model>.yaml
```

4. Thêm experiment config:

```text
configs/experiments/<new_experiment>.yaml
```

5. Trong notebook đổi:

```python
EXPERIMENT = "<new_experiment>"
```

6. Run notebook.

Không sửa:

```text
kaggle_pixel_motif_end_to_end.ipynb
scripts/run_experiment.py
src/pipeline/artifact_builder.py
src/pipeline/experiment_runner.py
```

trừ khi thay đổi bản chất data pipeline.

## 9. Khi nào mới sửa data pipeline

Chỉ sửa `src/pipeline/artifact_builder.py` hoặc atomic data scripts nếu thay đổi:

```text
graph construction
node features
edge features
candidate topology
motif bank building
motif matching/selection
artifact format
```

Nếu chỉ thêm model classifier mới, không sửa data pipeline.

## 10. Output trên Kaggle

Artifacts trong run:

```text
/kaggle/working/artifacts/graph_repo
/kaggle/working/artifacts/pixel_candidate_subgraphs_v2
/kaggle/working/artifacts/pixel_motif_bank_v2
/kaggle/working/artifacts/pixel_motif_dataset_v2
```

Training outputs:

```text
/kaggle/working/sgu-2026-facial-expression-recognition/outputs/
```

Zip outputs:

```text
/kaggle/working/<experiment_name>_outputs.zip
```

## 11. Legacy

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

Hai notebook cũ:

```text
legacy/deprecated_notebooks/kaggle_build_pixel_motif_dataset_v2.ipynb
legacy/deprecated_notebooks/kaggle_train_pixel_motif_baseline.ipynb
```

Hai orchestration script cũ:

```text
legacy/scripts/run_pixel_motif_experiment.py
legacy/scripts/run_pixel_motif_v2_pipeline.py
```

Các file này giữ lại để tham khảo, không dùng cho workflow chính.

## 12. Những lỗi cần tránh

Không làm các việc sau:

```text
Train từ pixel motif dataset prebuilt không rõ version.
Scan /kaggle/input để chọn dataset motif đã publish.
Sửa notebook mỗi lần đổi model.
Sửa runner mỗi lần đổi model.
Train C với rich edge trong lần so sánh đầu.
Bỏ descriptor 41D khỏi C ở bản đầu.
Fake node_indices nếu artifact thiếu.
```

Nếu artifact thiếu `node_indices`:

```text
Không fake dữ liệu.
Rebuild pixel motif dataset từ pipeline.
```

## 13. Tóm tắt cho ChatGPT/Codex

Nếu gửi file này cho ChatGPT/Codex, cần nó hiểu:

```text
Project hiện dùng workflow template config-driven.
Notebook chính là kaggle_pixel_motif_end_to_end.ipynb.
Entrypoint chính là scripts/run_experiment.py.
Experiment config là nguồn sự thật.
Baseline B là pixel_motif_guided_gnn_motif_norm, macro F1 khoảng 0.4196.
Cải tiến C là hierarchical_motif_gnn, thêm internal pixel-subgraph encoder.
Muốn thêm model mới thì thêm model + registry + experiment config, rồi đổi EXPERIMENT trong notebook.
Không sửa notebook/runner/data pipeline nếu chỉ thay model.
```
