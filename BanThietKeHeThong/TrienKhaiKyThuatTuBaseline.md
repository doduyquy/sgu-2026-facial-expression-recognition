1. Đánh giá nhanh phân tích đó

Hiện tại model của bạn là:

selected subgraph descriptor [41D]
+ match score
+ matched class
→ GraphSAGE giữa 32 selected subgraph nodes
→ motif attention pooling
→ classifier

Trong file hiện tại cũng ghi rõ mỗi selected subgraph đang được xem như một node trong GNN, node feature là descriptor [41] + match_score [1] + matched_class one-hot [7], rồi qua GraphSAGE/message passing và motif-aware pooling để phân loại ảnh .

Điểm mạnh của bản này là đã chứng minh:

GNN giữa motif-selected subgraphs có ích hơn MLP pooling.

Nhưng điểm yếu cốt lõi vẫn là:

Mỗi subgraph bên trong vẫn bị nén thành descriptor thống kê 41D.

Chính tài liệu của bạn cũng đã chỉ ra descriptor 41D có thể làm mất thứ tự không gian, topology nội bộ, quan hệ node-node và cấu trúc pixel-level cụ thể của motif .

Vậy nên nhận định “muốn khai thác graph-only thì phải làm graph thật hơn” là chính xác.

2. Hướng nên chọn nhất: Two-level GNN

Tôi chốt hướng chính như sau:

Full pixel graph
→ motif-selected pixel subgraphs
→ Internal Subgraph GNN encode từng subgraph thật
→ Motif-level GNN học quan hệ giữa các subgraphs
→ Image-level classifier

Tên nên đặt:

Hierarchical Motif-guided Pixel Subgraph GNN

hoặc:

Two-level Pixel-preserving Motif GNN

Bản này giải quyết đúng vấn đề lớn nhất của baseline hiện tại:

V1/V2 hiện tại:
subgraph thật → descriptor 41D → motif-level GNN

Bản mới:
subgraph thật → internal pixel GNN → subgraph embedding → motif-level GNN

Tức là chuyển từ:

motif trên vector thống kê

sang gần hơn với:

motif trên cấu trúc pixel subgraph thật

Điều này cũng khớp với tài liệu thiết kế của bạn: sau motif filtering, subgraph đã nhỏ hơn full graph, nên đây mới là thời điểm hợp lý để dùng small GNN trên từng subgraph, thay vì train full graph 2304 node .

3. Kiến trúc đề xuất
Tầng 1 — Internal Pixel-Subgraph GNN

Mỗi selected subgraph giữ lại:

node_indices
local_node_features
local_edge_index
local_edge_attr
bbox
center
radius
matched_motif_id
match_score

Với mỗi selected subgraph:

node_features: [n_i, 7]
local_edge_index: [2, e_i]
local_edge_attr: [e_i, a]

Trong đó n_i có thể khoảng:

r = 1 → 3×3 ≈ 9 nodes
r = 2 → 5×5 ≈ 25 nodes
r = 3 → 7×7 ≈ 49 nodes

Sau đó dùng small GNN:

PixelSubgraphEncoder:
    1–2 layer GraphSAGE / GCN
    graph readout
    output z_i [128]

Luồng:

pixel nodes trong subgraph
→ internal message passing
→ readout
→ subgraph embedding z_i

Embedding này biểu diễn cấu trúc thật của subgraph, không chỉ là thống kê.

Tầng 2 — Motif-level GNN

Sau khi encode xong K selected subgraphs:

z_1, z_2, ..., z_K

mỗi z_i trở thành một node trong graph cấp cao.

Node feature cấp motif:

[
  z_i từ internal GNN,
  descriptor_41D,
  match_score,
  matched_disc_score,
  matched_class_onehot,
  center/bbox/region
]

Sau đó dùng motif-level GNN:

K motif/subgraph nodes
→ GraphSAGE / GAT / Graph Transformer
→ motif attention pooling
→ classifier

Tức là GNN cấp cao học:

miệng ↔ mắt
mắt ↔ lông mày
khóe miệng ↔ má
vùng upper face ↔ lower face

Đây là thứ MLP không học tốt được.

4. Vì sao two-level GNN là hướng đáng làm nhất?

Vì nó bám sát nhất 4 nguyên tắc gốc của đề tài:

Không bỏ pixel.
Không train full graph mù.
Không gán label cho từng subgraph.
Motif là cầu nối giữa pixel-level detail và image-level classification.

Tài liệu của bạn cũng nhấn mạnh full graph là nguồn thông tin gốc, motif là trung tâm, subgraph là đơn vị trung gian và nhãn vẫn là nhãn ảnh, không được biến từng subgraph thành sample riêng .

Two-level GNN giữ đúng điều đó:

Subgraph không bị gán nhãn riêng.
Subgraph chỉ được encode thành evidence.
Classifier vẫn dự đoán ở mức ảnh.
5. Thứ tự triển khai tôi khuyên

Tôi không khuyên nhảy thẳng vào Graph Transformer. Thứ tự nên là:

Step 1: Bật use_edge_attr hiện có.
Step 2: Làm giàu edge_attr giữa motif nodes.
Step 3: Làm two-level GNN.
Step 4: Thử Motif Graph Transformer.
Step 5: Thêm motif-prototype contrastive/consistency loss.

Lý do là Step 1 và Step 2 ít phá pipeline nhất. Step 3 mới là thay đổi lớn.

6. Step 1 — Bật use_edge_attr trước

Trong code hiện tại, MotifGraphSAGELayer đã có edge_mlp. Nếu use_edge_attr=True, edge attribute sẽ được đưa qua MLP và sigmoid để gate cạnh trước khi aggregate neighbor .

Hiện tại config đang tắt, nên đây là thử nghiệm rẻ nhất:

model:
  use_edge_attr: true
  edge_attr_dim: 3

Việc này giúp cạnh giữa selected subgraphs không còn chỉ là adjacency, mà có trọng số học được từ edge attributes.

Kỳ vọng:

Không chắc tăng mạnh,
nhưng là ablation bắt buộc vì code đã hỗ trợ sẵn.
7. Step 2 — Làm giàu edge_attr giữa motif nodes

Hiện edge_attr giữa selected subgraphs có thể mới chỉ là:

dx, dy, dist

Nên nâng thành:

dx
dy
dist
bbox_iou
center_distance
descriptor_cosine
same_matched_class
same_motif_id
match_score_diff
disc_score_pair
region_relation
left_right_symmetry

Lúc đó cạnh giữa hai subgraph biết thêm:

hai vùng gần hay xa?
có overlap không?
có cùng motif class không?
có cùng motif id không?
có tương tự descriptor không?
có quan hệ đối xứng trái-phải không?

Đây là graph-only nhưng giàu hơn nhiều.

8. Step 3 — Two-level GNN là bản chính

Dataset hiện tại cần nâng từ:

{
    "x": Tensor[32, 41],
    "edge_index": Tensor[2, E],
    "edge_attr": Tensor[E, 3],
    ...
}

sang:

{
    "descriptors": Tensor[K, 41],
    "subgraph_node_features": List[Tensor[n_i, 7]],
    "subgraph_edge_index": List[Tensor[2, e_i]],
    "subgraph_edge_attr": List[Tensor[e_i, a]],
    "subgraph_node_indices": List[Tensor[n_i]],
    "bbox": Tensor[K, 4],
    "centers": Tensor[K, 2],
    "match_scores": Tensor[K],
    "matched_class": Tensor[K],
    "matched_motif_id": Tensor[K],
    "matched_disc_score": Tensor[K],
    "motif_graph_edge_index": Tensor[2, E2],
    "motif_graph_edge_attr": Tensor[E2, A2],
    "label": int
}

Model:

for each image:
    for each selected subgraph:
        pixel node features + local edges
        → InternalSubgraphGNN
        → z_i

    z_i + descriptor + motif info
    → MotifLevelGNN
    → attention pooling
    → classifier

Đây là nâng cấp graph-only quan trọng nhất.

9. Step 4 — Motif Graph Transformer

Sau khi two-level GNN chạy ổn, mới thử Transformer.

Vì K chỉ khoảng:

32 / 48 / 64 selected subgraphs

nên attention giữa motif nodes là khả thi.

Graph Transformer nên dùng edge bias:

attention(i, j) =
Q_i K_j / sqrt(d)
+ edge_bias(i, j)

Edge bias lấy từ:

spatial distance
bbox relation
same/different motif class
descriptor similarity
region relation

Nhưng không nên làm ngay từ đầu, vì nếu two-level GNN chưa ổn mà đã thêm Transformer, debug sẽ rất khó.

10. Step 5 — Motif consistency loss bản graph-only

Loss cũ dựa trên motif_score_vector precomputed nên tác dụng có giới hạn.

Khi có internal subgraph embedding z_i, ta có thể làm motif consistency thật hơn.

Ý tưởng:

z_i nên gần prototype embedding của motif đúng
z_i nên xa prototype embedding của motif sai

Ví dụ:

L_proto =
- log exp(sim(z_i, p_pos) / τ)
  /
  Σ_m exp(sim(z_i, p_m) / τ)

Trong đó:

z_i = embedding từ InternalSubgraphGNN
p_m = prototype embedding của motif m

Nhưng nên làm sau khi two-level GNN đã chạy ổn.

11. Roadmap phiên bản

Tôi đề xuất chia version như sau.

V2.1 — Edge-aware Motif GNN
Descriptor 41D
+ current motif-level GNN
+ use_edge_attr=true

Mục tiêu:

test nhanh xem edge gating có gain không.
V2.2 — Rich Motif Edge GNN
Descriptor 41D
+ rich edge_attr giữa selected subgraphs
+ GraphSAGE/GAT

Mục tiêu:

làm graph cấp motif có nghĩa hơn.
V3.0 — Two-level GNN baseline
Internal pixel-subgraph GNN
+ motif-level GraphSAGE

Node motif-level:

internal_gnn_embedding
+ descriptor
+ match info

Đây là bản chính nên tập trung.

V3.1 — Two-level GNN + multi-scale
radii: 1, 2, 3
max_candidates: 256
top_k: 48 hoặc 64

Mục tiêu:

bắt được motif nhỏ như khóe miệng/mí mắt
và motif rộng như vùng miệng/mắt-lông mày.
V3.2 — Two-level GNN + prototype loss
internal embedding
→ align với motif prototypes

Mục tiêu:

motif bank không chỉ dùng để chọn offline,
mà còn định hình embedding khi train.
V4 — Motif Graph Transformer
internal subgraph GNN
+ graph transformer ở motif level
+ edge bias

Mục tiêu:

học quan hệ global giữa các motif-selected regions tốt hơn GraphSAGE.