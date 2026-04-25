# GNN / FER Project Architecture Notes

## 1. Mục tiêu của file này

File này dùng như một bản ghi chú nhanh để:

- nhận biết dự án đang chạy theo luồng nào
- biết mỗi thư mục / file đang làm gì
- phân biệt file đang dùng thật, file hỗ trợ, file cũ và file đang trống

## 2. Bức tranh tổng thể của project

Hiện tại project đi theo hướng biểu diễn ảnh FER-2013 thành **pixel graph** rồi huấn luyện các baseline đơn giản trước:

1. CSV FER-2013
2. parse thành `RawSample`
3. build `SharedGraphStructure` dùng chung cho mọi ảnh
4. build `PixelGraphSample` cho từng ảnh
5. lưu theo dạng **graph repository chia chunk**
6. đọc repository bằng `ChunkedGraphDataset`
7. chuyển sang một trong 3 mode:
   - `graph_vector`: graph-level vector -> `MLPBaseline`
   - `subgraph_descriptor`: bag of subgraph descriptors -> `SubgraphMLPBaseline`
   - `resolved`: full graph để dành cho GNN về sau
8. train / validate / evaluate

Nói ngắn gọn: **pipeline mới của project xoay quanh canonical graph repository**, không còn lấy CSV trực tiếp ở bước train chính.

## 3. Luồng chạy chính nên nhớ

### 3.1 Build dữ liệu graph

File chính:

- `scripts/build_graph_repository.py`

Luồng:

- đọc `train.csv`, `val.csv`, `test.csv`
- dùng `data/raw_fer_dataset.py` để parse dữ liệu thô
- dùng `data/shared_graph_builder.py` tạo graph structure dùng chung
- dùng `data/canonical_graph_builder.py` tạo graph sample cho từng ảnh
- dùng `data/graph_repository.py` để lưu thành repo dạng chunk

Kết quả:

- `artifacts/graph_repo/shared/shared_graph.pt`
- `artifacts/graph_repo/train/chunk_*.pt`
- `artifacts/graph_repo/val/chunk_*.pt`
- `artifacts/graph_repo/test/chunk_*.pt`
- `artifacts/graph_repo/manifest.pt`

### 3.2 Train model

File chính:

- `scripts/train.py`

Luồng:

- load config từ `src/utils/config.py`
- lấy `graph_repo_path` theo `configs/env.yaml`
- build dataloader từ `src/data/dataloader.py`
- tạo model từ `src/models/__init__.py`
- tạo loss / optimizer / scheduler
- train bằng `src/training/trainer.py`
- load best checkpoint
- evaluate bằng `src/evaluation/evaluator.py`

### 3.3 Evaluate / inspect

File đang dùng tốt:

- `src/evaluation/evaluator.py`: evaluate test set, in classification report, lưu confusion matrix
- `scripts/inspect_graph_repository.py`: kiểm tra graph repo, shape, feature names, sample resolve
- `scripts/build_shared_graph.py`: build riêng shared graph nếu cần debug topology

## 4. Phân loại nhanh các nhóm file

### 4.1 Nhóm đang là core pipeline mới

- `data/*.py` trong các file `raw_*`, `graph_*`, `chunked_graph_dataset.py`, `canonical_graph_builder.py`, `shared_graph_builder.py`
- `src/data/dataloader.py`
- `src/features/graph_vectorizer.py`
- `src/models/mlp_baseline.py`
- `src/models/subgraph_mlp_baseline.py`
- `src/training/*`
- `src/evaluation/*`
- `configs/base.yaml`, `configs/mlp_baseline.yaml`, `configs/subgraph_baseline.yaml`, `configs/env.yaml`
- `scripts/build_graph_repository.py`
- `scripts/train.py`
- `scripts/inspect_graph_repository.py`

### 4.2 Nhóm legacy / tương thích ngược

Đây là các file thuộc pipeline cũ kiểu `PixelGraph` đầy đủ hoặc cache `.pt` cũ:

- `src/graph/*`
- `src/data/fer_split_dataset.py`
- `src/data/graph_cache_dataset.py`
- `src/data/graph_vector_dataset.py`
- `src/data/vector_cache_dataset.py`
- `scripts/build_graph_cache.py`
- `scripts/build_vector_cache.py`
- `scripts/check_graph_cache.py`

Các file này vẫn có ích để tham khảo hoặc debug, nhưng **không phải trục chính của pipeline hiện tại**.

### 4.3 Nhóm tài liệu / note / notebook

- `README.md`: mô tả tổng quan, nhưng một phần cấu trúc trong đó đã cũ so với code hiện tại
- `Doanchat.md`: ghi chú trao đổi / phân tích
- `BanThietKeHeThong/*.md`: tài liệu thiết kế hệ thống, hướng phát triển, motif, baseline
- `notebooks/*.ipynb`: EDA, demo training, phân tích các backbone CNN
- `Kaggle_GNN_MLP_Baseline_sgu-2026-gnn-fer.ipynb`: notebook chạy trên Kaggle

### 4.4 Nhóm đang trống hoặc chưa hoàn thiện

- `GNN_kientruc.md`: trước khi bổ sung thì trống
- `scripts/evaluate.py`: đang trống
- `scripts/prepare_data.py`: đang trống
- `scripts/analyze_errors.py`: gần như trống
- `tests/test_dataset.py`: trống
- `tests/test_models.py`: trống
- `tests/test_trainer.py`: trống
- `tests/test_whatyouwant.py`: trống

## 5. Chức năng từng thư mục

## `configs/`

### `configs/base.yaml`

Config gốc của toàn bộ project:

- dữ liệu
- feature graph
- model mặc định
- training mặc định
- logging
- path local / kaggle cơ bản

### `configs/mlp_baseline.yaml`

Ghi đè config cho baseline:

- `model.name = mlp_baseline`
- tăng hidden dims
- training cho MLP baseline

### `configs/subgraph_baseline.yaml`

Config cho baseline bag-of-subgraphs:

- `data.mode = subgraph_descriptor`
- cấu hình số subgraph, radius, stride
- `model.name = subgraph_mlp_baseline`

### `configs/env.yaml`

Tách cấu hình môi trường:

- `local`: `graph_repo_path`, `root_path`, `num_workers`
- `kaggle`: path dataset và repo trên Kaggle

### `configs/graph_config.py`

Dataclass config chuẩn cho pipeline graph mới:

- kích thước ảnh
- connectivity
- node feature names
- static / dynamic edge feature names
- chunk size
- version của graph config

Đây là **source of truth** cho builder/repository mới.

## `data/`

Thư mục này là lớp dữ liệu chuẩn của pipeline mới.

### `data/raw_types.py`

Định nghĩa `RawSample`:

- sample thô đọc từ CSV
- chứa `image`, `label`, `split`, `usage`

### `data/raw_fer_dataset.py`

Dataset đọc file CSV FER-2013:

- parse cột `pixels`
- convert thành ảnh `48x48`
- trả về `RawSample`
- có hàm thống kê phân bố lớp

### `data/graph_types.py`

Định nghĩa 3 kiểu dữ liệu quan trọng:

- `SharedGraphStructure`: topology dùng chung
- `PixelGraphSample`: feature riêng từng ảnh
- `ResolvedPixelGraph`: graph đầy đủ sau khi merge shared + sample

Đây là file rất quan trọng vì nó định nghĩa contract dữ liệu toàn pipeline.

### `data/shared_graph_builder.py`

Build graph structure dùng chung cho mọi ảnh:

- build `edge_index`
- build static edge attrs như `dx`, `dy`, `dist`

### `data/canonical_graph_builder.py`

Build graph sample cho từng ảnh:

- normalize pixel
- build node features như `intensity`, `x_norm`, `y_norm`
- build dynamic edge attrs như `delta_intensity`, `intensity_similarity`

### `data/graph_repository.py`

Đọc / ghi canonical graph repository:

- `GraphRepositoryWriter`: ghi shared graph + chunk theo split
- `GraphRepositoryReader`: đọc repo, load chunk, duyệt split, đọc manifest

### `data/graph_resolver.py`

Merge:

- `SharedGraphStructure`
- `PixelGraphSample`

thành:

- `ResolvedPixelGraph`

File này là điểm nối duy nhất để reconstruct full graph cho downstream.

### `data/chunked_graph_dataset.py`

Dataset chính của pipeline mới:

- đọc graph repo theo chunk
- cache chunk kiểu lazy
- có thể trả về raw sample hoặc resolved graph

File này là nền của train / subgraph / future GNN.

### `data/__init__.py`

File package marker, không có logic đáng kể.

## `src/data/`

Trong `src/data/` hiện có cả file mới lẫn file legacy.

### `src/data/dataloader.py`

Factory build dataloader theo 3 mode:

- `graph_vector`
- `subgraph_descriptor`
- `resolved`

Ngoài ra còn có `GraphVectorDatasetFromRepo` để vectorize trực tiếp từ graph repository.

Đây là file bridge giữa repository mới và training.

### `src/data/subgraph_dataset.py`

Biến mỗi `ResolvedPixelGraph` thành:

- túi `K` subgraph descriptors
- `mask`
- `label`

Dùng cho baseline `SubgraphMLPBaseline`.

### `src/data/fer_split_dataset.py`

Dataset CSV kiểu cũ, dùng cho pipeline cũ trước canonical repository.

### `src/data/graph_cache_dataset.py`

Đọc file graph cache `.pt` kiểu cũ chứa list `PixelGraph`.

### `src/data/graph_vector_dataset.py`

Dataset cũ:

- đọc `GraphCacheDataset`
- vectorize graph thành tensor đầu vào cho MLP

### `src/data/vector_cache_dataset.py`

Dataset đọc vector cache `.pt` đã tính sẵn.

### `src/data/emotions_dict.py`

Ánh xạ id lớp cảm xúc -> tên cảm xúc.

### `src/data/__init__.py`

File package marker.

## `src/features/`

### `src/features/graph_vectorizer.py`

Chuyển node features của một graph thành graph-level vector cố định bằng:

- mean
- std
- max

Đây là feature extractor của `MLPBaseline`.

## `src/models/`

### `src/models/__init__.py`

Model registry:

- map tên model -> class khởi tạo
- hiện đang đăng ký:
  - `mlp_baseline`
  - `subgraph_mlp_baseline`

### `src/models/mlp_baseline.py`

Baseline đơn giản:

- input là vector graph-level
- backbone là MLP nhiều tầng
- output logits 7 lớp

### `src/models/subgraph_mlp_baseline.py`

Baseline cho túi subgraph descriptors:

- encode từng descriptor bằng shared MLP
- masked mean pooling theo số subgraph hợp lệ
- classifier ra logits

## `src/training/`

### `src/training/trainer.py`

Trainer chính:

- `train_one_epoch`
- `validate`
- `fit`
- early stopping theo `val_macro_f1`
- save best checkpoint
- log WandB nếu bật

### `src/training/losses.py`

Factory loss:

- chủ yếu đang dùng `CrossEntropyLoss`
- có helper `inception_loss`

### `src/training/optimizer.py`

Factory optimizer và scheduler:

- Adam / SGD
- `ReduceLROnPlateau`, `StepLR`, `CosineAnnealingLR`

### `src/training/__init__.py`

File package marker.

## `src/evaluation/`

### `src/evaluation/metrics.py`

Tính:

- accuracy
- macro F1
- weighted F1
- confusion matrix
- classification report

và có hàm vẽ confusion matrix.

### `src/evaluation/evaluator.py`

Chạy evaluate trên test loader:

- forward model
- gom nhãn thật / dự đoán
- in báo cáo
- lưu confusion matrix

### `src/evaluation/error_analysis.py`

Hiện đang trống, vai trò dự kiến là phân tích lỗi / trực quan hóa mẫu đúng sai.

### `src/evaluation/__init__.py`

File package marker.

## `src/utils/`

### `src/utils/config.py`

Load và merge config:

- `base.yaml`
- model yaml
- `env.yaml`

Lưu ý: file này vẫn còn hơi mang dấu vết pipeline cũ trong ví dụ minh họa, nhưng đang được `scripts/train.py` dùng thật.

### `src/utils/checkpoint.py`

Save / load checkpoint model + optimizer.

### `src/utils/logger_wandb.py`

Khởi tạo và log lên WandB:

- run config
- metrics
- image
- artifact model

### `src/utils/seed.py`

Set random seed cho Python / NumPy / Torch.

### `src/utils/visualization.py`

Các hàm vẽ:

- loss curves
- prediction grid

Phù hợp cho notebook hoặc error analysis.

### `src/utils/data_stats.py`

Thống kê phân bố lớp từ CSV.

## `src/graph/`

Đây là cụm **pipeline graph cũ / legacy**, vẫn hữu ích để tham khảo.

### `src/graph/graph_config.py`

Dataclass config cũ cho `ImageGraphBuilder`.

### `src/graph/structures.py`

Định nghĩa `PixelGraph` kiểu cũ:

- chứa full graph trong một object numpy

### `src/graph/image_to_graph.py`

Builder cũ:

- nhận ảnh
- build luôn full `PixelGraph`
- không tách shared / dynamic như pipeline mới

### `src/graph/subgraph_generator.py`

Sinh candidate subgraph từ `ResolvedPixelGraph`:

- chọn seed nodes
- BFS theo radius
- trích local subgraph

File này đang được pipeline `subgraph_descriptor` dùng thật.

### `src/graph/subgraph_descriptor.py`

Biến subgraph thành vector descriptor cố định:

- thống kê node features
- thống kê edge features
- số node, số edge, density

Đây là phần quan trọng của baseline subgraph.

### `src/graph/io.py`

Hàm save / load graph kiểu cũ.

### `src/graph/__init__.py`

Package marker.

## `scripts/`

### File đang là đường chạy chính

#### `scripts/build_graph_repository.py`

Script quan trọng nhất để build repo graph mới từ CSV.

#### `scripts/train.py`

Entry point huấn luyện hiện tại.

#### `scripts/inspect_graph_repository.py`

Script kiểm tra repo graph sau khi build.

#### `scripts/build_shared_graph.py`

Build riêng shared graph, tiện khi debug cấu trúc graph.

### File hỗ trợ / kiểm tra / legacy

#### `scripts/build_graph_cache.py`

Pipeline cũ:

- build full `PixelGraph`
- lưu `train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`

#### `scripts/build_vector_cache.py`

Pipeline cũ tối ưu RAM:

- build graph vector trực tiếp
- lưu vector cache `.pt`

#### `scripts/check_graph_cache.py`

Kiểm tra nhanh nội dung một graph cache `.pt` kiểu cũ.

#### `scripts/check_pipeline.py`

Script kiểm tra tổng hợp nhiều thành phần, nhưng hiện đang bám nhiều giả định cũ:

- kiểm tra `graph_cache_path`
- dùng `src.graph.*`
- không hoàn toàn khớp với pipeline canonical repository mới

Vì vậy file này nên xem là script kiểm thử tham khảo hơn là chuẩn chính thức.

### File đang trống / chưa dùng

#### `scripts/evaluate.py`

Đang trống.

#### `scripts/prepare_data.py`

Đang trống.

#### `scripts/analyze_errors.py`

Hầu như chưa có nội dung thực tế.

## `tests/`

Hiện tại các file test:

- `tests/test_dataset.py`
- `tests/test_models.py`
- `tests/test_trainer.py`
- `tests/test_whatyouwant.py`

đều đang trống hoặc chưa có nội dung kiểm thử hữu ích.

Có nghĩa là project hiện **chưa có unit test hoàn chỉnh** cho pipeline mới.

## `BanThietKeHeThong/`

Thư mục tài liệu thiết kế:

- `BanThietKeHeThong.md`: tổng quan thiết kế
- `Graph_Node_Edge.md`: ghi chú về node/edge
- `Motif.md`: ý tưởng motif
- `HuongPhatTrienBaseline.md`: hướng baseline
- `BanThietKeR1_R2.md`: kế hoạch theo phase
- `TrainWithMotif.md`: hướng train với motif
- `TrienKhaiKyThuat.md`, `TrienKhaiKyThuatVer2.md`: chi tiết kỹ thuật

Đây là phần tài liệu phục vụ hiểu định hướng hơn là code chạy trực tiếp.

## `notebooks/`

Chủ yếu phục vụ:

- EDA
- demo training
- phân tích backbone CNN
- xem kết quả đánh giá

Không phải nơi chứa pipeline chính để production/train chuẩn.

## 6. File nào cần nhớ nhất

Nếu chỉ cần nhớ các file quan trọng nhất để hiểu project, ưu tiên đọc theo thứ tự:

1. `scripts/train.py`
2. `src/data/dataloader.py`
3. `data/chunked_graph_dataset.py`
4. `data/graph_types.py`
5. `data/graph_repository.py`
6. `data/shared_graph_builder.py`
7. `data/canonical_graph_builder.py`
8. `data/graph_resolver.py`
9. `src/features/graph_vectorizer.py`
10. `src/models/__init__.py`
11. `src/models/mlp_baseline.py`
12. `src/models/subgraph_mlp_baseline.py`
13. `src/training/trainer.py`
14. `src/evaluation/evaluator.py`
15. `configs/base.yaml`
16. `configs/subgraph_baseline.yaml`
17. `configs/mlp_baseline.yaml`

## 7. Kết luận ngắn

Hiện trạng codebase có thể hiểu như sau:

- **trục chính hiện tại**: canonical graph repository + MLP baseline + subgraph baseline
- **trục cũ còn tồn tại**: full `PixelGraph` cache và vector cache
- **GNN full graph thực thụ**: chưa triển khai xong, mới dừng ở `resolved` mode để chuẩn bị cho bước sau
- **test và một số script phụ**: còn thiếu hoặc chưa hoàn thiện

Nếu sau này muốn tiếp tục cập nhật file này, nên giữ nguyên cách chia:

- luồng chính
- thư mục
- từng file
- file legacy / file trống

để nhìn vào là biết ngay project đang ở đâu.
