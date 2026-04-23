Cây thư mục hiện tại
project/
│
├── configs/
│   └── graph_config.py
│
├── data/
│   ├── fer_split_dataset.py
│   ├── graph_cache_dataset.py
│   └── graph_vector_dataset.py
│
├── features/
│   └── graph_vectorizer.py
│
├── graph/
│   ├── structures.py
│   └── image_to_graph.py
│
├── models/
│   └── mlp_baseline.py
│
├── training/
│   └── trainer_mlp_baseline.py
│
├── utils/
│   └── metrics.py
│
└── scripts/
    ├── build_graph_cache.py
    ├── check_graph_cache.py
    └── train_graph_baseline.py
Chức năng từng thư mục
configs/

Chứa các cấu hình cho bước graph hóa.

data/

Chứa các dataset class:

đọc CSV gốc FER split
đọc graph cache .pt
chuyển graph thành vector để train baseline
features/

Chứa các module trích đặc trưng ở mức graph.

graph/

Chứa phần biểu diễn graph và code graph hóa ảnh.

models/

Chứa mô hình baseline.

training/

Chứa train loop và eval loop.

utils/

Chứa metric hỗ trợ.

scripts/

Chứa các script chạy chính.

Giải thích từng file
configs/graph_config.py

Chức năng
Định nghĩa cấu hình graph hóa ảnh FER-2013.

Hiện tại đang quản lý

image_size = 48
connectivity = 8
normalize_pixels = True
node_features = ["intensity", "x_norm", "y_norm"]
edge_features = ["dx", "dy", "dist", "delta_intensity", "intensity_similarity"]

Ý nghĩa

Đây là nơi bạn chỉnh cấu hình baseline.
Sau này muốn mở rộng lên gx, gy, grad_mag, contrast thì sửa ở đây hoặc trong script build.
graph/structures.py

Chức năng
Định nghĩa dataclass PixelGraph.

Nó lưu

graph_id
label
split
usage
height, width
node_features
edge_index
edge_attr
image (optional)
metadata

Ý nghĩa

Đây là object chuẩn đại diện cho 1 ảnh đã graph hóa.
Toàn bộ pipeline sau này sẽ dựa trên object này.
data/fer_split_dataset.py

Chức năng
Đọc từng file:

train.csv
val.csv
test.csv

Nhiệm vụ

đọc emotion
parse pixels
reshape ảnh về 48x48
trả sample gồm:
id
image
label
usage
split

Ý nghĩa

Đây là cầu nối từ dữ liệu CSV gốc sang numpy image.
graph/image_to_graph.py

Chức năng
Chuyển một ảnh 48x48 thành PixelGraph.

Nhiệm vụ chính

normalize ảnh về 0..1
build node_features
build edge_index
build edge_attr
trả về PixelGraph

Hiện tại baseline node features

intensity
x_norm
y_norm

Hiện tại baseline edge features

dx
dy
dist
delta_intensity
intensity_similarity

Điểm quan trọng

Code đã viết theo kiểu mở rộng được.
Sau này thêm gx, gy, grad_mag, contrast không cần viết lại toàn bộ.
scripts/build_graph_cache.py

Chức năng
Script build graph cache cho toàn bộ dataset.

Input

train.csv
val.csv
test.csv

Output

train_graphs.pt
val_graphs.pt
test_graphs.pt

Ý nghĩa

Đây là bước chạy đầu tiên sau khi có CSV.
Nó biến dữ liệu ảnh gốc thành dữ liệu graph cache để tái sử dụng.
scripts/check_graph_cache.py

Chức năng
Kiểm tra nhanh 1 file graph cache.

Nó làm gì

load file .pt
in thông tin graph
in shape của:
node_features
edge_index
edge_attr
kiểm tra NaN
in vài node đầu
in vài edge đầu

Ý nghĩa

Dùng để debug graph hóa đã đúng chưa trước khi train.
data/graph_cache_dataset.py

Chức năng
Đọc file .pt graph cache.

Output

mỗi sample trả:
{"graph": PixelGraph}

Ý nghĩa

Đây là dataset tầng trung gian để load graph đã build.
features/graph_vectorizer.py

Chức năng
Biến 1 PixelGraph thành 1 vector graph-level cố định.

Baseline hiện tại

lấy mean(node_features)
lấy std(node_features)
lấy max(node_features)
rồi nối lại

Nếu node_features.shape = [2304, 3]
thì output sẽ là:

3 + 3 + 3 = 9 chiều

Ý nghĩa

Đây là baseline đơn giản nhất để test graph có tín hiệu không.
data/graph_vector_dataset.py

Chức năng
Kết hợp:

GraphCacheDataset
GraphVectorizer

để tạo dataset cuối cho baseline MLP.

Output mỗi sample

x: graph vector
y: label

Ý nghĩa

Đây là dataset mà model baseline sẽ ăn trực tiếp.
models/mlp_baseline.py

Chức năng
Mô hình baseline MLP cho graph-level vector.

Kiến trúc hiện tại

input: 9
hidden: 64
hidden: 32
output: 7

Ý nghĩa

Đây là baseline sanity check.
Nó chưa phải GNN.
Mục đích là xác nhận graph cache và feature pipeline chạy ổn.
utils/metrics.py

Chức năng
Tính metric phân loại.

Hiện tại có

accuracy
macro F1
weighted F1
confusion matrix

Ý nghĩa

Dùng cho train/val/test evaluation.
training/trainer_mlp_baseline.py

Chức năng
Chứa:

train_one_epoch
evaluate

Nhiệm vụ

forward
loss
backward
optimizer step
tính metric

Ý nghĩa

Đây là train loop tách riêng để code sạch hơn.
scripts/train_graph_baseline.py

Chức năng
Script train baseline MLP từ graph cache.

Pipeline

load train_graphs.pt, val_graphs.pt, test_graphs.pt
vectorize graph thành graph-level vector
train MLP
chọn best theo val macro F1
evaluate trên test
lưu best checkpoint
lưu test_metrics.txt

Ý nghĩa

Đây là script train baseline hiện tại của project.
Luồng chạy hiện tại của project

Thứ tự đúng nên là:

Bước 1 — Chuẩn bị dữ liệu CSV

Bạn cần có:

data/train.csv
data/val.csv
data/test.csv

Mỗi file có các cột tối thiểu:

emotion
pixels
Usage (nếu có)
Bước 2 — Build graph cache

Chạy:

python scripts/build_graph_cache.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --test_csv data/test.csv \
  --save_dir artifacts/graph_cache

Kết quả

artifacts/graph_cache/
  train_graphs.pt
  val_graphs.pt
  test_graphs.pt
Bước 3 — Kiểm tra graph cache

Ví dụ kiểm tra train:

python scripts/check_graph_cache.py \
  --graph_path artifacts/graph_cache/train_graphs.pt \
  --index 0

Bạn nên kiểm tra

node_features.shape == (2304, 3)
edge_index.shape[0] == 2
edge_attr.shape[0] == edge_index.shape[1]
không có NaN
label đúng
split đúng
Bước 4 — Train baseline MLP

Chạy:

python scripts/train_graph_baseline.py \
  --train_graphs artifacts/graph_cache/train_graphs.pt \
  --val_graphs artifacts/graph_cache/val_graphs.pt \
  --test_graphs artifacts/graph_cache/test_graphs.pt \
  --save_dir artifacts/baseline_mlp \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-3

Kết quả

artifacts/baseline_mlp/
  best_model.pt
  test_metrics.txt
Trạng thái hiện tại của pipeline

Hiện tại project của bạn đang có 2 phần hoàn chỉnh:

Phần 1 — Graph hóa ảnh FER-2013

Đã có:

đọc CSV split
parse ảnh 48x48
build PixelGraph
lưu graph cache
Phần 2 — Baseline classification rất nhẹ

Đã có:

load graph cache
vector hóa graph bằng mean/std/max
train MLP phân loại 7 emotion
Những gì chưa có ở thời điểm hiện tại

Hiện tại chưa triển khai các phần sau:

candidate subgraph generation
subgraph descriptor extraction
motif bank building
motif matching
top-k subgraph selection
bag-of-subgraphs classifier
GCN / GraphSAGE full graph baseline
prototype consistency loss

Tức là project hiện đang dừng ở:
Graph hóa + baseline MLP graph-level

Mục tiêu bước tiếp theo

Bước tiếp theo hợp lý nhất sau baseline hiện tại là một trong 2 hướng:

Hướng 1

Làm Baseline 2: GCN / GraphSAGE trên full graph

Hướng 2

Bắt đầu đúng hướng nghiên cứu chính:
Candidate Subgraph Generation

Với đề tài của bạn, mình khuyên nên đi:
Candidate Subgraph Generation trước, vì đó mới là bước mở đầu thật sự cho motif pipeline.

Tóm tắt cực ngắn

Hiện tại project có thể hiểu là:

configs/graph_config.py: cấu hình graph hóa
data/fer_split_dataset.py: đọc CSV FER split
graph/structures.py: dataclass PixelGraph
graph/image_to_graph.py: ảnh → graph
scripts/build_graph_cache.py: build .pt graph cache
scripts/check_graph_cache.py: kiểm tra graph cache
data/graph_cache_dataset.py: load graph cache
features/graph_vectorizer.py: graph → vector
data/graph_vector_dataset.py: dataset cho baseline
models/mlp_baseline.py: MLP classifier
training/trainer_mlp_baseline.py: train/eval loop
utils/metrics.py: metric
scripts/train_graph_baseline.py: train baseline hoàn chỉnh