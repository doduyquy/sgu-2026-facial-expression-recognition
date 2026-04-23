Baseline hợp lý nhất lúc này là:

Subgraph-descriptor baseline, không phải full-graph GNN, cũng không nên tiếp tục đầu tư vào MLP trên full graph.

Lý do rất thẳng:

full graph 2304 node đi ngược tinh thần đề tài và khá nặng
tài liệu của bạn đã chốt rằng hướng chính là graph → candidate subgraphs → motif → image-level learning, chứ không phải train trực tiếp full graph
MLP baseline trên vector toàn graph chỉ hợp làm sanity check, không còn đáng để phát triển tiếp như baseline chính
Tôi chốt cho bạn 3 mức baseline
Baseline 0 — Smoke test rất nhẹ

Mục tiêu: kiểm tra pipeline data/label/cache có sống không.

Làm:

load graph repository
lấy node_features
pooling mean/std/max
MLP classifier 7 lớp

Cái này không phải baseline nghiên cứu chính, chỉ để test nhanh pipeline. Tài liệu baseline ban đầu của bạn cũng đề xuất đúng kiểu này.

Baseline 1 — Baseline nên làm ngay bây giờ

Mục tiêu: đi đúng hướng đề tài nhưng vẫn đủ nhẹ để triển khai.

Pipeline:

image → pixel graph → candidate subgraphs → subgraph descriptor → aggregate ở mức ảnh → classifier

Cụ thể:

Từ mỗi ảnh graph hóa như bạn đã làm.
Sinh candidate subgraphs bằng radius-bounded local subgraphs.
Với mỗi subgraph, tính descriptor cố định.
Chọn cố định k subgraphs mỗi ảnh.
Encode descriptor bằng MLP nhỏ.
Mean pooling hoặc attention pooling nhẹ ở mức ảnh.
MLP head ra 7 lớp.

Đây là baseline tốt nhất hiện tại vì:

đã bỏ full graph
đã bắt đầu dùng subgraph, đúng tinh thần reduce-first
chưa cần motif bank hoàn chỉnh nên dễ debug hơn
vẫn mở đường rất đẹp sang motif sau đó
Baseline 2 — Baseline nghiên cứu thật sự sau đó

Pipeline:

image → pixel graph → candidate subgraphs → descriptor → motif bank → motif matching → top-k subgraphs → image-level classifier

Đây mới là baseline bám sát đề tài nhất. Nhưng tôi không khuyên nhảy thẳng vào đây nếu bạn chưa có subgraph pipeline chạy ổn. Vì lúc đó bug có thể đến từ:

subgraph extraction
descriptor
motif bank
matching
training

Tài liệu của bạn cũng khuyên đi theo kiểu offline precompute từng pha để dễ debug.

Tôi khuyên bạn nên chọn cái nào?

Nên làm Baseline 1.

Đây là điểm cân bằng đẹp nhất giữa:

đúng hướng nghiên cứu
đủ nhẹ
đủ nhanh để ra kết quả đầu tiên
dễ nâng cấp thành motif baseline thật
Baseline 1 nên chốt thế nào
Input

Dùng graph repository hiện tại:

node_features
edge_index
edge_attr nếu cần, nhưng bản đầu có thể chưa dùng sâu
Candidate subgraph

Dùng:

seed theo stride, ví dụ 2 hoặc 4 pixel lấy 1 seed
radius = 1 hoặc 2
giới hạn max_candidates_per_image

Tài liệu của bạn nghiêng rất rõ về radius-bounded local subgraphs.

Descriptor mỗi subgraph

Dùng đúng kiểu tài liệu đã gợi ý:

mean/std intensity
mean/std gradient magnitude
mean/std local contrast
node count
edge count
density
mean edge weight
std edge weight

Đây là descriptor rất hợp để làm baseline đầu tiên.

Chọn số subgraph mỗi ảnh

Để đơn giản:

lấy cố định top k theo một rule đơn giản, chưa cần motif
ví dụ chọn theo:
contrast mạnh
gradient magnitude mạnh
hoặc random + coverage đều

Bản đầu chưa cần motif bank thật, chỉ cần có bag-of-subgraphs cố định để train image-level classifier.

Encoder
MLP trên descriptor subgraph

Không dùng GNN ngay. Tài liệu của bạn cũng khuyên MLP trước rồi mới small GNN sau.

Aggregation
mean pooling trước
attention pooling để sau
Loss
CrossEntropy trước
chưa thêm motif consistency loss
Vì sao không nên làm full-graph GCN baseline?

Vì về mặt kỹ thuật nó vẫn được, nhưng về chiến lược thì không đáng:

nặng
không đúng hướng chính
sau này khó kể narrative đẹp trong báo cáo
không tận dụng phần motif mà bạn đã phân tích rất kỹ
Lộ trình triển khai thực tế tôi khuyên

Bước 1:
CandidateSubgraphGenerator

Hàm cần có:

sample_seed_nodes(...)
extract_radius_subgraph(...)
generate_candidate_subgraphs(...)

Bước 2:
SubgraphDescriptor

Hàm:

compute_subgraph_stats(...)
subgraph_to_descriptor(...)

Bước 3:
Subgraph dataset baseline

Mỗi ảnh trả:

descriptors: [k, D]
label

Bước 4:
Model baseline

subgraph MLP encoder
mean pooling
classifier 7 lớp

Bước 5:
Train thử

Sau khi Baseline 1 chạy ổn, mới làm:

motif bank builder
motif matching
top-k motif-filtered dataset
Nếu cần tôi chốt luôn một baseline cụ thể để bạn code

Tôi sẽ chốt thế này:

candidate subgraph: radius = 1
seed stride = 4
max candidates = 64 / ảnh
descriptor dimension: khoảng 10–20 chiều
chọn top-k = 16 subgraphs / ảnh
encoder: MLP D -> 64 -> 32
aggregation: mean pooling
classifier: 32 -> 7
loss: CrossEntropy

Đây là baseline đầu tiên đúng hướng, nhẹ, dễ train, dễ debug, và nâng cấp lên motif rất tự nhiên.