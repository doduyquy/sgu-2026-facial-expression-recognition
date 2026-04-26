# Kaggle Workflow Guide cho Pixel-preserving Motif V2

File này dùng để nhắc mọi lần code tiếp, debug, hoặc nhờ AI chỉnh sửa project:

```text
Local chỉ dùng để code/test nhỏ.
Kaggle là nơi build artifact nặng và train model chính.
```

Project hiện có 2 notebook Kaggle chính:

1. `kaggle_build_pixel_motif_dataset_v2.ipynb`
2. `kaggle_train_pixel_motif_baseline.ipynb`

Hai notebook này là entrypoint chính khi chạy trên Kaggle. Khi sửa code, cần hiểu notebook nào gọi đến file nào để tránh sửa sai nhánh.

## 0. Bài học bắt buộc khi tạo biến thể dataset/model mới

Lỗi đã gặp:

```text
Đã thêm bản rich edge trong code sinh dữ liệu,
nhưng ban đầu chưa đồng bộ tên output dataset, config train,
Notebook A và Notebook B.
Kết quả là dữ liệu có thể đã là rich nhưng vẫn nằm trong folder tên cũ,
dễ train nhầm config hoặc publish nhầm artifact.
```

Từ giờ, hễ tạo một biến thể mới như:

```text
spatial edge -> rich edge
top_k 32 -> top_k 48
radii 1 2 -> radii 1 2 3
dataset v2 -> dataset v3
```

thì phải kiểm tra đủ 6 điểm sau:

1. **Tên artifact cuối cùng**

   Ví dụ:

   ```text
   pixel_motif_dataset_v2              # baseline spatial/old
   pixel_motif_dataset_v2_rich_edges   # rich edge 13D
   ```

2. **Script build dataset**

   File cần kiểm tra:

   ```text
   scripts/run_pixel_motif_v2_pipeline.py
   scripts/precompute_pixel_motif_dataset.py
   ```

   Các option phải có và được truyền xuyên suốt:

   ```text
   --edge_attr_mode spatial|rich
   --pixel_motif_dir <output_dir>       # nếu cần override tên output
   ```

3. **Notebook A phải build đúng biến thể**

   File:

   ```text
   kaggle_build_pixel_motif_dataset_v2.ipynb
   ```

   Cần kiểm tra:

   ```python
   EDGE_ATTR_MODE = "rich"     # hoặc "spatial"
   ```

   Notebook A phải copy đúng thư mục final ra `/kaggle/working`:

   ```text
   rich    -> /kaggle/working/pixel_motif_dataset_v2_rich_edges
   spatial -> /kaggle/working/pixel_motif_dataset_v2
   ```

   Sau khi copy, Notebook A phải xóa artifact trung gian và repo source:

   ```text
   /kaggle/working/artifacts
   /kaggle/working/sgu-2026-facial-expression-recognition
   ```

   Lý do: Kaggle Dataset sẽ gom toàn bộ file còn lại trong `/kaggle/working`.

4. **Notebook B phải train đúng config tương ứng**

   File:

   ```text
   kaggle_train_pixel_motif_baseline.ipynb
   ```

   Cần kiểm tra:

   ```python
   DATASET_VARIANT = "rich"    # hoặc "spatial"
   ```

   Mapping phải đúng:

   ```text
   rich    -> configs/pixel_motif_guided_gnn_rich_edges.yaml
   spatial -> configs/pixel_motif_guided_gnn_motif_norm.yaml
   ```

5. **Config train phải trỏ đúng dataset và edge_attr_dim**

   Với rich edge:

   ```yaml
   data:
     pixel_motif_dataset_path: artifacts/pixel_motif_dataset_v2_rich_edges

   model:
     use_edge_attr: true
     edge_attr_dim: 13
   ```

   Với spatial baseline:

   ```yaml
   data:
     pixel_motif_dataset_path: artifacts/pixel_motif_dataset_v2

   model:
     edge_attr_dim: 3
   ```

6. **Inspect dataset trước khi train**

   Không tin vào tên folder. Phải kiểm tra nội dung:

   ```bash
   python scripts/inspect_pixel_motif_dataset.py --data_dir <dataset_path>
   ```

   Rich edge đúng phải thấy:

   ```text
   edge_attr: (128, 13)
   ```

   hoặc trong `meta.pt`:

   ```text
   edge_attr_mode: rich
   edge_attr_dim: 13
   ```

   Spatial baseline đúng phải thấy:

   ```text
   edge_attr: (128, 3)
   ```

Nguyên tắc chốt:

```text
Tên folder chỉ là nhãn.
meta.pt và shape edge_attr mới là sự thật.
Notebook A và Notebook B luôn phải được sửa theo cặp khi thêm biến thể dataset.
```

## 1. Notebook A: Build Pixel Motif Dataset

File:

```text
kaggle_build_pixel_motif_dataset_v2.ipynb
```

Mục tiêu:

```text
CSV FER-2013
-> graph_repo
-> pixel_candidate_subgraphs_v2
-> pixel_motif_bank_v2
-> pixel_motif_dataset_v2 hoặc pixel_motif_dataset_v2_rich_edges
```

Notebook này chỉ nên chạy khi thay đổi các phần:

- cách graph hóa ảnh
- node feature / edge feature
- candidate subgraph generation
- motif bank / clustering / scoring
- motif matching / selection
- top_k, knn_k, coverage/diversity

Notebook này không phải notebook train lặp lại hằng ngày.

### Input Kaggle của Notebook A

Cần add Kaggle Dataset chứa:

```text
train.csv
val.csv
test.csv
```

Notebook sẽ scan `/kaggle/input` để tìm folder chứa đủ 3 file này.

### Output Kaggle của Notebook A

Output sạch để tạo Kaggle Dataset mới:

```text
/kaggle/working/pixel_motif_dataset_v2/ hoặc
/kaggle/working/pixel_motif_dataset_v2_rich_edges/
  train_pixel_motif.pt
  val_pixel_motif.pt
  test_pixel_motif.pt
  meta.pt
  README_KAGGLE_DATASET.txt
```

Sau khi Notebook A chạy xong:

1. Save Version / Save & Run All.
2. Vào output files của notebook.
3. Tạo Kaggle Dataset mới từ folder final còn lại trong `/kaggle/working`.
4. Dataset này sẽ là input cho Notebook B.

### Notebook A gọi đến file nào?

Notebook A gọi script điều phối:

```text
scripts/run_pixel_motif_v2_pipeline.py
```

Script này gọi lần lượt:

```text
scripts/build_graph_repository.py
scripts/precompute_pixel_candidate_subgraphs.py
scripts/inspect_pixel_candidate_subgraphs.py
scripts/build_pixel_motif_bank.py
scripts/inspect_pixel_motif_bank.py
scripts/precompute_pixel_motif_dataset.py
scripts/inspect_pixel_motif_dataset.py
scripts/audit_pixel_motif_dataset.py
```

Các module code liên quan trực tiếp:

```text
data/raw_fer_dataset.py
data/shared_graph_builder.py
data/canonical_graph_builder.py
data/graph_repository.py
data/graph_resolver.py
src/graph/subgraph_descriptor.py
src/motif_v2/topology.py
src/motif_v2/types.py
src/motif_v2/io.py
src/motif_v2/matching.py
src/motif/motif_scoring.py
```

Artifact trung gian của Notebook A:

```text
/kaggle/working/artifacts/graph_repo/
/kaggle/working/artifacts/pixel_candidate_subgraphs_v2/
/kaggle/working/artifacts/pixel_motif_bank_v2/
/kaggle/working/artifacts/pixel_motif_dataset_v2/
/kaggle/working/artifacts/pixel_motif_dataset_v2_rich_edges/  # nếu EDGE_ATTR_MODE=rich
```

Artifact cuối cùng cần publish:

```text
/kaggle/working/pixel_motif_dataset_v2/              # spatial baseline
/kaggle/working/pixel_motif_dataset_v2_rich_edges/   # rich edge
```

Notebook A hiện có biến:

```python
EDGE_ATTR_MODE = "rich"  # hoặc "spatial"
```

Sau khi copy final dataset, Notebook A phải cleanup để Kaggle Dataset không gom artifact nặng:

```text
rm -rf /kaggle/working/artifacts
rm -rf /kaggle/working/sgu-2026-facial-expression-recognition
```

## 2. Notebook B: Train Pixel Motif Baseline

File:

```text
kaggle_train_pixel_motif_baseline.ipynb
```

Mục tiêu:

```text
pixel_motif_dataset_v2 hoặc pixel_motif_dataset_v2_rich_edges
-> inspect dataset
-> train MLP sanity nếu cần
-> train GNN baseline chính
-> evaluate test set
-> zip outputs
```

Notebook này nên chạy khi thay đổi:

- model
- loss
- optimizer / scheduler
- training config
- ablation config
- số epoch / seed / wandb
- evaluator / visualization output

Không cần chạy lại Notebook A nếu chỉ sửa model hoặc training.

### Input Kaggle của Notebook B

Cần add Kaggle Dataset được tạo từ Notebook A, chứa:

```text
train_pixel_motif.pt
val_pixel_motif.pt
test_pixel_motif.pt
meta.pt
```

Notebook sẽ scan `/kaggle/input` để tìm folder chứa đủ các file này.

Notebook B hiện có biến:

```python
DATASET_VARIANT = "rich"  # hoặc "spatial"
```

Mapping config:

```text
rich    -> configs/pixel_motif_guided_gnn_rich_edges.yaml
spatial -> configs/pixel_motif_guided_gnn_motif_norm.yaml
```

### Output Kaggle của Notebook B

Train output nằm ở:

```text
/kaggle/working/sgu-2026-facial-expression-recognition/outputs/
```

Notebook cũng zip output thành:

```text
/kaggle/working/pixel_motif_v2_train_outputs.zip
```

### Notebook B gọi đến file nào?

Notebook B gọi:

```text
scripts/inspect_pixel_motif_dataset.py
scripts/train.py
```

Config chính:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Config rich edge:

```text
configs/pixel_motif_guided_gnn_rich_edges.yaml
```

Config sanity MLP nếu bật:

```text
configs/pixel_motif_guided_mlp_clean.yaml
```

Các module train liên quan:

```text
src/data/pixel_motif_dataset.py
src/data/dataloader.py
src/models/__init__.py
src/models/motif_guided_gnn.py
src/models/motif_guided_mlp.py
src/training/losses.py
src/training/optimizer.py
src/training/trainer.py
src/evaluation/metrics.py
src/evaluation/evaluator.py
src/utils/config.py
src/utils/checkpoint.py
```

Model chính:

```text
src/models/motif_guided_gnn.py
```

Dataset loader chính:

```text
src/data/pixel_motif_dataset.py
```

Collate function chính:

```text
src/data/dataloader.py
collate_fn_pixel_motif
```

## 3. File config baseline chính

File:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Đây là config baseline tốt nhất hiện tại.

Các điểm quan trọng:

```yaml
data:
  mode: pixel_motif
  normalize_x: true

model:
  name: motif_guided_gnn
  use_motif_score_vector: true
  use_match_score_feature: true
  use_match_score_weighting: true
  pooling: motif_attention

loss:
  name: weighted_ce
  use_class_weights: true
  class_weight_power: 0.5
```

Không nên đổi nhầm sang:

```text
configs/pixel_motif_guided_gnn_clean.yaml
configs/pixel_motif_guided_gnn.yaml
```

Hai file đó dùng cho ablation / thử nghiệm, không phải baseline chính.

### Config rich edge

File:

```text
configs/pixel_motif_guided_gnn_rich_edges.yaml
```

Dùng cho dataset:

```text
pixel_motif_dataset_v2_rich_edges
```

Điểm bắt buộc:

```yaml
model:
  use_edge_attr: true
  edge_attr_dim: 13
```

Không train rich dataset bằng config spatial nếu `edge_attr` là 13 chiều.
Không train spatial dataset bằng config rich nếu `edge_attr` là 3 chiều.

## 4. Khi sửa gì thì chạy notebook nào?

### Chỉ sửa model, loss, optimizer, trainer

Ví dụ:

```text
src/models/motif_guided_gnn.py
src/models/motif_guided_mlp.py
src/training/losses.py
src/training/trainer.py
configs/*.yaml
```

Chỉ cần chạy:

```text
kaggle_train_pixel_motif_baseline.ipynb
```

Không cần build lại dataset.

### Sửa motif matching hoặc top-k selection

Ví dụ:

```text
src/motif_v2/matching.py
scripts/precompute_pixel_motif_dataset.py
src/motif_v2/topology.py  # nếu chỉ sửa hàm tạo edge giữa selected subgraphs
```

Cần chạy lại Notebook A từ stage:

```text
motif_dataset
```

Hoặc chạy full Notebook A nếu muốn chắc chắn artifact đồng bộ.

### Sửa motif bank / clustering / scoring

Ví dụ:

```text
scripts/build_pixel_motif_bank.py
src/motif_v2/types.py
src/motif_v2/io.py
src/motif/motif_scoring.py
```

Cần chạy lại Notebook A từ stage:

```text
motif_bank
motif_dataset
```

### Sửa candidate topology / descriptor

Ví dụ:

```text
src/motif_v2/topology.py
src/graph/subgraph_descriptor.py
scripts/precompute_pixel_candidate_subgraphs.py
```

Cần chạy lại Notebook A từ stage:

```text
candidates
motif_bank
motif_dataset
```

### Sửa node feature / edge feature / graph construction

Ví dụ:

```text
data/canonical_graph_builder.py
data/shared_graph_builder.py
configs/graph_config.py
scripts/build_graph_repository.py
```

Cần chạy lại Notebook A từ đầu:

```text
graph_repo
candidates
motif_bank
motif_dataset
```

## 5. Script điều phối stage

File:

```text
scripts/run_pixel_motif_v2_pipeline.py
```

Các stage:

```text
--stage graph_repo
--stage candidates
--stage motif_bank
--stage motif_dataset
--stage all
```

Các option quan trọng:

```text
--csv_root
--out_root
--skip_existing
--smoke
--smoke_samples
```

Ví dụ build full trên Kaggle:

```bash
python scripts/run_pixel_motif_v2_pipeline.py \
  --stage all \
  --csv_root /kaggle/input/fer13-split \
  --out_root /kaggle/working/artifacts \
  --edge_attr_mode rich \
  --skip_existing
```

Ví dụ smoke test:

```bash
python scripts/run_pixel_motif_v2_pipeline.py \
  --stage all \
  --csv_root /kaggle/input/fer13-split \
  --out_root /kaggle/working/artifacts_smoke \
  --edge_attr_mode rich \
  --smoke \
  --smoke_samples 100 \
  --skip_existing
```

Ví dụ chỉ build lại rich motif dataset từ candidates và motif bank đã có:

```bash
python scripts/run_pixel_motif_v2_pipeline.py \
  --stage motif_dataset \
  --out_root /kaggle/working/artifacts \
  --edge_attr_mode rich
```

Output mặc định:

```text
spatial -> /kaggle/working/artifacts/pixel_motif_dataset_v2
rich    -> /kaggle/working/artifacts/pixel_motif_dataset_v2_rich_edges
```

## 6. Lệnh train baseline chính

Trên Kaggle:

```bash
python -m scripts.train \
  --config pixel_motif_guided_gnn_motif_norm \
  --env kaggle \
  --pixel_motif_dataset_path /kaggle/input/<dataset-name>/pixel_motif_dataset_v2 \
  --epochs 100
```

Nếu dataset input không có thư mục con `pixel_motif_dataset_v2`, mà file `.pt` nằm trực tiếp trong dataset root, thì path là:

```bash
--pixel_motif_dataset_path /kaggle/input/<dataset-name>
```

Notebook B tự scan và dùng đúng folder tìm được, nên thường không cần sửa tay.

### Lệnh train rich edge

Trên Kaggle:

```bash
python -m scripts.train \
  --config pixel_motif_guided_gnn_rich_edges \
  --env kaggle \
  --pixel_motif_dataset_path /kaggle/input/<dataset-name>/pixel_motif_dataset_v2_rich_edges \
  --epochs 100
```

Nếu dataset input không có thư mục con `pixel_motif_dataset_v2_rich_edges`, mà file `.pt` nằm trực tiếp trong dataset root:

```bash
--pixel_motif_dataset_path /kaggle/input/<dataset-name>
```

## 7. Kết quả baseline hiện tại

Baseline chính:

```text
Pixel-preserving Motif V2 + Motif-guided GNN
```

Config:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Kết quả local tốt nhất:

```text
Best epoch: 57
Val Macro F1: 0.4100

Test Accuracy: 45.11%
Test Macro F1: 0.4196
Test Weighted F1: 0.4380
```

So với MLP clean:

```text
MLP clean:
Accuracy 39.20%
Macro F1 0.3174

GNN motif_norm:
Accuracy 45.11%
Macro F1 0.4196

Gain:
+5.91% accuracy
+0.1022 macro F1
```

## 8. Nhắc nhở khi yêu cầu AI chỉnh code

Khi yêu cầu AI sửa code, nên nói rõ:

```text
Tôi đang chạy project theo workflow Kaggle 2 notebook:
- kaggle_build_pixel_motif_dataset_v2.ipynb để build artifact
- kaggle_train_pixel_motif_baseline.ipynb để train

Baseline chính dùng:
- configs/pixel_motif_guided_gnn_motif_norm.yaml
- artifacts/pixel_motif_dataset_v2 hoặc Kaggle Dataset pixel_motif_dataset_v2

Nếu dùng rich edge thì phải nói rõ:
- configs/pixel_motif_guided_gnn_rich_edges.yaml
- pixel_motif_dataset_v2_rich_edges
- edge_attr_dim = 13
- Notebook A: EDGE_ATTR_MODE = "rich"
- Notebook B: DATASET_VARIANT = "rich"

Nếu chỉ sửa model/train thì không được yêu cầu build lại dataset.
Nếu sửa topology/matching/motif thì phải nói rõ cần chạy lại Notebook A từ stage nào.
```

Điều này giúp tránh việc AI đề xuất sai luồng, ví dụ:

- build artifact nặng ở local
- upload artifact thủ công lên Kaggle
- sửa nhầm motif V1
- train bằng config ablation thay vì baseline chính
- quên `normalize_x: true`
- tạo dataset mới nhưng không đổi tên output folder
- sửa Notebook A mà quên sửa Notebook B
- train rich dataset bằng config spatial hoặc ngược lại
- publish Kaggle Dataset kèm cả `/kaggle/working/artifacts` làm dataset phình lên nhiều GB
