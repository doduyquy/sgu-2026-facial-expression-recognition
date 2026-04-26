Bạn nói đúng. Sau khi lục lại các file thiết kế, điểm ban đầu không phải “bỏ pixel để giảm chi phí”, mà là:

Vì ảnh FER-2013 chỉ 48×48, grayscale, rất ít thông tin, nên phải giữ pixel-level graph làm nền, rồi dùng motif để nâng ngữ nghĩa và giảm đồ thị có kiểm soát, chứ không được bỏ mất tín hiệu pixel quá sớm.

Trong file thiết kế, động cơ đã ghi rất rõ: FER-2013 là ảnh nhỏ 48×48, grayscale, dữ liệu nhiễu, dễ mất chi tiết cục bộ nếu gom thô; nếu mỗi pixel là một node thì graph có 2304 node, lớn và thô, nên node/edge thiết kế kém sẽ làm graph “lớn nhưng nghèo thông tin”, còn motif cổ điển dễ ra pattern sáng-tối thay vì pattern biểu cảm.

Chốt lại: không nên hiểu là “bỏ full graph”

Hướng đúng ban đầu là:

Không bỏ full pixel graph.
Không train trực tiếp full pixel graph.
Mà dùng full pixel graph làm không gian gốc để sinh và kiểm chứng motif.

Tức là full graph 2304 node vẫn phải tồn tại ở vai trò:

1. nguồn thông tin gốc
2. nơi sinh candidate subgraphs
3. nơi truy vết motif/exemplar về pixel
4. nền để giải thích motif nằm ở đâu trên ảnh

Còn việc train không dùng toàn bộ 2304 node chỉ là để tránh chi phí và nhiễu, không phải vì pixel không quan trọng.

Tài liệu cũng chốt mục tiêu không phải đưa trực tiếp full graph vào GNN, mà là: biến ảnh thành pixel interaction graph, tìm motif đặc trưng từng emotion, dùng motif chọn subgraphs giàu thông tin cảm xúc, rồi học biểu diễn ảnh từ tập subgraphs đó.

Vấn đề bạn lo là hợp lý: top-k hiện tại đang hơi “cắt mạnh”

Pipeline hiện tại của bạn:

full pixel graph
→ 64 candidate descriptors
→ motif matching
→ top-32 selected subgraphs
→ GNN

Nó chạy được và có kết quả tốt hơn baseline, nhưng đúng là có rủi ro:

Full 2304 node
→ bị giảm còn 64 candidate
→ bị nén thành descriptor 41 chiều
→ bị chọn cứng top-32

Như vậy có thể mất các tín hiệu nhỏ nhưng quan trọng. Với ảnh 48×48, một vài pixel ở khóe miệng, mí mắt, lông mày có thể mang thông tin cảm xúc; nếu candidate/top-k bỏ qua thì mô hình không còn cơ hội học.

Trong file kỹ thuật, candidate subgraph được gọi là bước bắt buộc trước motif để giảm bài toán từ full graph sang cấu trúc con, nhưng bản đầu chỉ nên dùng radius-bounded local subgraphs, seed theo stride, giới hạn radius/max_nodes/max_candidates để tránh bùng nổ. Điều này nghĩa là candidate là xấp xỉ có ràng buộc, không phải thay thế hoàn toàn cho full graph.

Ý tưởng thầy của bạn nên được hiểu thế này

Tôi sẽ diễn giải lại cho đúng hơn:

Ảnh 48×48 ít thông tin
→ mỗi pixel là một node để không bỏ chi tiết sớm
→ từ full pixel graph tìm các substructure/motif có tính cảm xúc
→ gom motif theo từng emotion
→ dùng các motif đó tạo tập subgraphs đại diện cho ảnh
→ train ở mức ảnh

Vậy bản hiện tại mới đạt:

Descriptor-based motif-guided selection

Còn bản sát ý thầy hơn phải là:

Pixel-preserving motif discovery

Khác nhau ở chỗ:

Bản hiện tại:
motif chủ yếu sống trong không gian descriptor [41]

Bản sát ý thầy:
motif phải truy ngược được về tập pixel nodes, edges, vùng ảnh, exemplar subgraphs

Tài liệu motif cũng nói rất rõ: nếu vẫn giữ hướng pixel-level graph 2304 node, motif không thể là frequent subgraph thô; nó phải là class-discriminative prototype subgraph, tức là một subgraph cục bộ phổ biến trong emotion đó, khác với emotion khác, và được match mềm như prototype.

Vậy hiện tại nên đánh giá ra sao?

Tôi đánh giá lại công bằng hơn:

Pipeline hiện tại: đúng hướng MVP.
Nhưng chưa đủ pixel-preserving.

Điểm đúng:

giữ pixel graph ban đầu
không train full graph trực tiếp
có motif bank theo emotion
có motif matching
train image-level, không train subgraph-level

Điểm chưa đủ:

candidate generator có thể không phủ hết pixel evidence
descriptor làm mất cấu trúc pixel nội bộ
top-k hard selection có thể bỏ mất chi tiết yếu nhưng quan trọng
motif bank chưa lưu đủ exemplar/node indices/edge structure để chứng minh motif thật sự là pixel subgraph

Đặc biệt, file Graph_Node_Edge nhấn mạnh ở pixel graph, câu hỏi sống còn không phải “có graph được không”, mà là mỗi node đại diện cho điều gì ngoài một giá trị xám và edge đo loại quan hệ gì. Intensity đơn lẻ quá yếu, nên node cần intensity, position, gradient, gradient magnitude, local contrast để giữ raw appearance, geometry, local structure và robustness với lighting.

Không nên chuyển quá sớm sang CNN feature

Bạn cũng đúng khi lo “đi vào học đặc trưng” quá sớm có thể lệch mục tiêu.

Trong file Node/Edge, CNN micro-feature được ghi là hướng nâng cao: nó giúp node có chút ngữ nghĩa learned feature, nhưng nếu nhúng CNN quá mạnh từ đầu thì triết lý graph mining sẽ bị loãng, motif có thể thành motif của learned feature chứ không còn rõ là motif pixel-structure.

Vì vậy, trước mắt không nên nhảy ngay sang CNN feature. Nên sửa motif pipeline để giữ pixel evidence tốt hơn.

Hướng chỉnh đúng hơn: không bỏ top-k, nhưng top-k phải mềm hơn và có coverage

Top-k không sai. Tài liệu vẫn nói motif matching sẽ giữ top-k subgraphs có match tốt nhất. Nhưng top-k hiện tại cần nâng từ:

hard top-k theo score

thành:

motif set selection có coverage + diversity + pixel preservation

Cụ thể nên sửa theo 4 ý:

1. Candidate phủ rộng hơn:
   tăng hoặc đảm bảo mỗi vùng ảnh đều có candidate.
   Không để chỉ các vùng score cao ban đầu được đi tiếp.

2. Motif lưu lại pixel evidence:
   mỗi motif không chỉ có prototype vector,
   mà lưu exemplar graph_id, node_indices, center, bbox, edge_index_subgraph.

3. Selection không chỉ top score:
   chọn theo score + diversity + coverage.
   Tránh 32 subgraphs đều nằm quanh miệng hoặc một vùng sáng.

4. Dataset train giữ trace về full graph:
   motif_filtered sample nên có selected_indices/node_indices/bbox,
   để sau này visual và kiểm tra có bỏ mất mắt/miệng/lông mày không.

Như vậy bạn vẫn giảm graph để train được, nhưng không phản bội mục tiêu pixel-level.

Chốt hướng sửa narrative

Câu đúng để trình bày với thầy là:

Vì FER-2013 chỉ 48×48, em không bỏ pixel-level graph. Full graph 2304 node vẫn là biểu diễn gốc để giữ chi tiết cục bộ. Tuy nhiên, em không train trực tiếp trên toàn bộ graph vì chi phí lớn và dễ nhiễu. Em dùng motif như một cơ chế chọn lọc có giám sát: từ full pixel graph sinh các local subgraphs, học emotion-specific discriminative prototype motifs, sau đó chọn một tập subgraphs đại diện cho ảnh. Điểm cần cải tiến tiếp theo là làm selection có coverage/diversity và lưu lại node-level evidence để đảm bảo motif vẫn bám vào pixel structure, không chỉ là descriptor vector.

Đây là kết luận quan trọng:

Không bỏ pixel.
Không train full graph mù.
Motif phải là cầu nối giữa pixel-level detail và image-level classification.

Bản hiện tại đã chứng minh motif có ích. Nhưng để sát mục tiêu ban đầu hơn, bước tiếp theo nên là Pixel-preserving Motif V2: motif bank và motif-filtered dataset phải giữ lại cấu trúc pixel subgraph thật, không chỉ giữ descriptor.

## Bộ lệnh PowerShell chạy Pixel-preserving Motif V2

Dưới đây là bộ lệnh đúng cho PowerShell. Copy từng block chạy lần lượt.

### 1. Precompute Pixel Candidate Subgraphs

```powershell
conda run -n fer-graph python scripts/precompute_pixel_candidate_subgraphs.py `
  --repo_root artifacts/graph_repo `
  --out_dir artifacts/pixel_candidate_subgraphs_v2 `
  --max_candidates 128 `
  --seed_stride 4 `
  --radii 1 2 `
  --coverage_grid 4 4
```

```powershell
conda run -n fer-graph python scripts/inspect_pixel_candidate_subgraphs.py `
  --data_dir artifacts/pixel_candidate_subgraphs_v2
```

### 2. Build Pixel Motif Bank

```powershell
conda run -n fer-graph python scripts/build_pixel_motif_bank.py `
  --input_dir artifacts/pixel_candidate_subgraphs_v2 `
  --out_dir artifacts/pixel_motif_bank_v2 `
  --num_motifs_per_class 16 `
  --max_subgraphs_per_class 50000 `
  --alpha 0.5 `
  --seed 42 `
  --num_exemplars 5
```

```powershell
conda run -n fer-graph python scripts/inspect_pixel_motif_bank.py `
  --motif_bank_path artifacts/pixel_motif_bank_v2/pixel_motif_bank.pt
```

### 3. Precompute Pixel Motif Dataset

```powershell
conda run -n fer-graph python scripts/precompute_pixel_motif_dataset.py `
  --candidate_dir artifacts/pixel_candidate_subgraphs_v2 `
  --motif_bank_path artifacts/pixel_motif_bank_v2/pixel_motif_bank.pt `
  --out_dir artifacts/pixel_motif_dataset_v2 `
  --top_k 32 `
  --knn_k 4 `
  --beta 0.5 `
  --gamma 0.25 `
  --eta 0.05 `
  --diversity_sigma 0.12
```

```powershell
conda run -n fer-graph python scripts/inspect_pixel_motif_dataset.py `
  --data_dir artifacts/pixel_motif_dataset_v2
```

```powershell
conda run -n fer-graph python scripts/audit_pixel_motif_dataset.py `
  --data_dir artifacts/pixel_motif_dataset_v2 `
  --splits train val test
```

### 4. Visualize một sample

```powershell
conda run -n fer-graph python scripts/visualize_pixel_motif_evidence.py `
  --data_dir artifacts/pixel_motif_dataset_v2 `
  --split test `
  --index 0
```

### 5. Train local nhanh 1 epoch

```powershell
conda run -n fer-graph python -m scripts.train `
  --config pixel_motif_guided_gnn `
  --env local `
  --pixel_motif_dataset_path artifacts/pixel_motif_dataset_v2 `
  --epochs 1 `
  --no_wandb
```

### 6. Train local full

```powershell
conda run -n fer-graph python -m scripts.train `
  --config pixel_motif_guided_gnn `
  --env local `
  --pixel_motif_dataset_path artifacts/pixel_motif_dataset_v2
```

### MLP sanity nếu cần

```powershell
conda run -n fer-graph python -m scripts.train `
  --config pixel_motif_guided_mlp `
  --env local `
  --pixel_motif_dataset_path artifacts/pixel_motif_dataset_v2 `
  --epochs 1 `
  --no_wandb
```

Tôi sẽ phân tích theo hướng triển khai được ngay: kiến trúc, dữ liệu cần bổ sung, batching, loss, rủi ro, và thứ tự version. Trọng tâm là two-level GNN thuần graph, không dùng CNN.

Đã suy nghĩ trong 17s

Được. Ta sẽ phân tích sâu vào Two-level / Hierarchical GNN theo hướng bạn chuẩn bị triển khai.

Tôi chốt tên hướng này là:

Hierarchical Motif-guided Pixel Subgraph GNN

hoặc ngắn hơn:

Two-level Pixel-preserving Motif GNN

Bản chất của nó là:

Level 1: Pixel-subgraph GNN
Học cấu trúc thật bên trong từng selected subgraph.

Level 2: Motif-level GNN
Học quan hệ giữa các selected subgraphs ở mức ảnh.

Output:
Dự đoán emotion của toàn ảnh.

Điểm quan trọng: hướng này vẫn giữ đúng triết lý gốc của bạn: full graph là nguồn thông tin, motif là trung tâm, subgraph là đơn vị trung gian, và nhãn vẫn là nhãn ảnh chứ không phải nhãn subgraph. Tài liệu của bạn cũng nhấn mạnh không được biến mỗi subgraph thành sample độc lập mang nhãn emotion, vì FER-2013 chỉ có nhãn ở mức ảnh.

1. Vì sao cần Two-level GNN?

Hiện tại mô hình tốt nhất của bạn đang là:

selected subgraph descriptor [41D]
+ match_score
+ matched_class
→ motif-level GraphSAGE
→ motif attention pooling
→ classifier

Code hiện tại cũng đúng như vậy: MotifGuidedGNN nhận x [B, K, D], edge_index, edge_attr, mask, match_scores, matched_class, motif_score_vector; sau đó encode node, chạy các MotifGraphSAGELayer, rồi pooling bằng motif_attention nếu được cấu hình.

Nhưng điểm yếu là:

Mỗi selected subgraph bên trong vẫn chỉ là descriptor 41D.

Tức là subgraph thật ban đầu có:

pixel node features
local edge_index
local edge_attr
topology nội bộ
vị trí pixel
quan hệ sáng-tối
gradient/contrast

nhưng khi vào model lại bị nén thành:

descriptor vector [41]

Vì vậy hiện tại GNN mới học được:

quan hệ giữa các descriptor-subgraphs

chưa học được đầy đủ:

cấu trúc pixel thật bên trong từng subgraph

Two-level GNN giải quyết đúng lỗ hổng này.

2. Ý tưởng lõi của Two-level GNN

Thay vì:

subgraph thật
→ descriptor [41D]
→ motif-level GNN

ta đổi thành:

subgraph thật
→ small GNN encode pixel nodes bên trong subgraph
→ subgraph embedding z_i
→ motif-level GNN

Luồng đầy đủ:

Ảnh 48×48
→ full pixel graph 2304 nodes
→ motif-guided selected subgraphs
→ với mỗi selected subgraph:
       node_features [n_i, 7]
       local_edge_index [2, e_i]
       local_edge_attr [e_i, a]
       → Internal Pixel-Subgraph GNN
       → z_i [H]
→ K subgraph embeddings z_1...z_K
→ Motif-level GNN
→ attention pooling
→ classifier 7 emotion

Ở đây, K là số selected subgraphs, ví dụ:

K = 32 / 48 / 64

Còn n_i là số pixel nodes trong một selected subgraph, ví dụ:

radius 1 → khoảng 9 nodes
radius 2 → khoảng 25 nodes
radius 3 → khoảng 49 nodes

Tài liệu triển khai kỹ thuật của bạn cũng đã ghi rõ candidate subgraph nên lưu node indices gốc, node features con, edge con, metadata như center, size, radius; và sau motif filtering thì mới dùng các selected subgraphs để train.

3. Thiết kế dữ liệu cần đổi

Hiện tại sample train của bạn có thể đang gần dạng:

{
    "x": Tensor[K, 41],
    "edge_index": Tensor[2, E],
    "edge_attr": Tensor[E, A],
    "mask": Tensor[K],
    "match_scores": Tensor[K],
    "matched_class": Tensor[K],
    "motif_score_vector": Tensor[7],
    "label": int
}

Với Two-level GNN, sample nên nâng thành:

{
    "descriptors": Tensor[K, 41],
    "subgraph_node_features": List[Tensor[n_i, 7]],
    "subgraph_edge_index": List[Tensor[2, e_i]],
    "subgraph_edge_attr": List[Tensor[e_i, a]],
    "subgraph_node_indices": List[Tensor[n_i]],

    "bbox": Tensor[K, 4],
    "centers": Tensor[K, 2],
    "regions": Tensor[K],
    "radii": Tensor[K],

    "match_scores": Tensor[K],
    "matched_class": Tensor[K],
    "matched_motif_id": Tensor[K],
    "matched_disc_score": Tensor[K],
    "motif_score_vector": Tensor[7],

    "motif_graph_edge_index": Tensor[2, E2],
    "motif_graph_edge_attr": Tensor[E2, A2],
    "mask": Tensor[K],
    "label": int
}

Trong đó có hai loại graph:

1. Internal graph:
   graph nhỏ bên trong từng selected subgraph.

2. Motif-level graph:
   graph giữa các selected subgraphs của cùng một ảnh.

Đây là điểm phải phân biệt rất rõ khi code.

4. Level 1: Internal Pixel-Subgraph GNN
4.1. Input

Mỗi selected subgraph là một graph nhỏ:

X_i: [n_i, 7]
E_i: [2, e_i]
A_i: [e_i, edge_dim]

Node feature 7D vẫn là:

intensity
x_norm
y_norm
gx
gy
grad_mag
local_contrast

Thiết kế này bám đúng tài liệu node/edge của bạn: node không chỉ là intensity, mà cần position, gradient, gradient magnitude, local contrast để giữ appearance, geometry, local structure và robustness tương đối với lighting.

Edge bên trong subgraph nên lấy từ graph gốc:

8-neighbor local edges
+ dx, dy, dist
+ delta_intensity
+ intensity_similarity
4.2. Encoder nên đơn giản trước

Tôi khuyên bản đầu dùng:

2-layer GraphSAGE

Không nên dùng GAT ngay ở internal level, vì mỗi ảnh có K subgraphs, mỗi subgraph lại có graph riêng. Nếu dùng attention ngay từ đầu sẽ nặng và khó debug.

Kiến trúc:

node_features [n_i, 7]
→ Linear 7 → H
→ GraphSAGE layer 1
→ GraphSAGE layer 2
→ readout
→ z_i [H]

Readout có thể là:

mean pooling
max pooling
mean + max
attention pooling

Bản đầu nên dùng:

mean + max concat

Ví dụ:

z_i = MLP(concat(mean(h_nodes), max(h_nodes)))

Lý do: mean ổn định, max giữ tín hiệu biên/contrast mạnh.

5. Level 2: Motif-level GNN

Sau khi Level 1 encode xong, mỗi selected subgraph có embedding:

z_i [H]

Ta tạo node feature cấp motif:

u_i = concat(
    z_i,
    descriptor_i,
    match_score_i,
    matched_disc_score_i,
    one_hot(matched_class_i),
    center_i,
    bbox_i,
    region_embedding_i
)

Bản tối thiểu có thể là:

u_i = concat(
    z_i,
    descriptor_i,
    match_score_i,
    one_hot(matched_class_i)
)

Sau đó đưa vào motif-level GNN hiện tại.

Điều này tận dụng lại model bạn đã có:

MotifGuidedGNN

nhưng thay x [B, K, 41] thành:

x [B, K, D_new]

với:

D_new = internal_embedding_dim + descriptor_dim + motif_info_dim

Ví dụ:

internal_embedding_dim = 128
descriptor_dim = 41
match_score = 1
matched_disc_score = 1
matched_class_onehot = 7
center = 2
bbox = 4

D_new = 128 + 41 + 1 + 1 + 7 + 2 + 4 = 184
6. Edge giữa motif nodes nên thiết kế lại

Hiện tại code đã có edge_mlp để dùng edge_attr làm gate cạnh nếu use_edge_attr=True. Cụ thể, edge attribute đi qua MLP rồi sigmoid để nhân vào edge_valid trước khi aggregate neighbor.

Vậy bước đầu rất hợp lý là:

use_edge_attr: true

Nhưng với Two-level GNN, edge giữa motif nodes nên giàu hơn dx, dy, dist.

Tôi đề xuất edge_attr cấp motif:

dx_center
dy_center
center_distance
bbox_iou
descriptor_cosine
internal_embedding_cosine
same_matched_class
same_motif_id
match_score_abs_diff
disc_score_pair_mean
region_relation
symmetry_flag

Ý nghĩa:

dx, dy, dist:
    quan hệ không gian.

bbox_iou:
    hai subgraph có overlap hay không.

descriptor_cosine:
    giống nhau theo handcrafted descriptor.

internal_embedding_cosine:
    giống nhau theo learned graph embedding.

same_matched_class:
    hai subgraph cùng nghiêng về một emotion motif không.

same_motif_id:
    có đang match cùng prototype motif không.

region_relation:
    upper-upper, upper-lower, left-right...

symmetry_flag:
    hai vùng có đối xứng trái/phải không.

Tài liệu node/edge của bạn phân biệt rất rõ similarity và compatibility: similarity hỏi hai node có giống nhau không, còn compatibility hỏi hai thành phần có phối hợp tốt để tạo pattern biểu cảm không. Thiết kế edge_attr cấp motif chính là nơi đưa compatibility vào hợp lý hơn, thay vì đưa quá sớm ở full pixel graph.

7. Batching là phần khó nhất

Two-level GNN khó nhất không phải model, mà là batch dữ liệu.

Bạn có hai cách.

Cách A — Padding theo [B, K, Nmax, F]

Vì subgraph nhỏ, có thể pad mỗi subgraph về Nmax.

Ví dụ:

K = 32
Nmax = 49
F = 7

Batch:

sub_x:       [B, K, Nmax, 7]
sub_mask:    [B, K, Nmax]
sub_adj:     [B, K, Nmax, Nmax]
sub_eattr:   [B, K, Nmax, Nmax, A]

Ưu điểm:

dễ viết
dễ debug
không cần PyG
batch cố định

Nhược điểm:

tốn memory hơn
nhưng với Nmax <= 49 vẫn chấp nhận được

Với FER-2013, đây là cách tôi khuyên dùng bản đầu.

Cách B — Flatten toàn bộ subgraphs trong batch

Gộp tất cả selected subgraphs trong batch thành một danh sách:

B ảnh × K subgraphs = B*K small graphs

Sau đó encode bằng PyG Batch hoặc custom batch index.

Ưu điểm:

sạch về graph learning
không cần pad nhiều

Nhược điểm:

collate phức tạp hơn
debug khó hơn

Nếu bạn muốn chạy nhanh trong repo hiện tại, chọn Cách A trước.

8. Model forward nên chạy thế nào?

Logic forward:

def forward(batch):
    # 1. Encode internal pixel subgraphs
    sub_x = batch["sub_x"]              # [B, K, N, 7]
    sub_adj = batch["sub_adj"]          # [B, K, N, N]
    sub_node_mask = batch["sub_node_mask"]  # [B, K, N]

    z = internal_encoder(sub_x, sub_adj, sub_node_mask)
    # z: [B, K, H]

    # 2. Build motif-level node feature
    desc = batch["descriptors"]         # [B, K, 41]
    match = batch["match_scores"]       # [B, K]
    cls = batch["matched_class"]        # [B, K]

    motif_x = concat(z, desc, match, one_hot(cls), ...)
    # motif_x: [B, K, D_new]

    # 3. Motif-level GNN
    logits = motif_gnn(
        x=motif_x,
        edge_index=batch["motif_edge_index"],
        edge_attr=batch["motif_edge_attr"],
        mask=batch["mask"],
        ...
    )

    return logits

Điểm hay là bạn không cần bỏ model cũ. Bạn có thể tạo model mới:

HierarchicalMotifGNN

bên trong nó gọi lại hoặc tái sử dụng logic của MotifGuidedGNN.

9. Loss nên đi theo 3 mức
Mức 1 — Chỉ classification loss

Làm trước:

L = CrossEntropy(logits, y)

Có thể giữ class weight như bản motif_norm hiện tại.

Mục tiêu:

Kiểm tra two-level GNN có chạy ổn không.

Không thêm motif loss ngay, vì nếu kết quả lỗi bạn sẽ không biết lỗi do model hay loss.

Mức 2 — Motif consistency loss

Sau khi classification chạy ổn, thêm:

L_total = L_cls + λ_motif * L_motif

Ý tưởng:

subgraph embedding z_i nên gần prototype motif mà nó matched
và xa prototypes khác

Có hai cách.

Cách đơn giản

Mỗi motif prototype cũng có descriptor. Ta học một projection:

p_m = motif_projector(prototype_descriptor_m)
z_i = internal_subgraph_embedding

Sau đó dùng InfoNCE:

positive = matched_motif_id
negative = other motifs

Loss:

L_motif = CE(sim(z_i, p_m_all) / τ, matched_motif_id)

Nhưng cần cẩn thận: matched_motif_id là kết quả offline, không phải ground truth tuyệt đối. Vì vậy λ_motif nên nhỏ.

Gợi ý:

λ_motif = 0.05 hoặc 0.1
τ = 0.1 đến 0.2
Cách class-level mềm hơn

Không ép đúng motif id, chỉ ép đúng motif class:

z_i gần các prototypes thuộc label y hơn
z_i xa prototypes class khác

Cách này hợp hơn vì motif id offline có thể nhiễu.

Mức 3 — Diversity / redundancy regularization

Sau này thêm:

L_redundancy = Σ attention_i * attention_j * IoU(bbox_i, bbox_j)

Mục tiêu:

không để model dồn attention vào nhiều subgraphs trùng nhau.

Nhưng chỉ thêm sau khi model chính ổn.

10. Những rủi ro lớn khi triển khai
Rủi ro 1 — Overfit

Two-level GNN có nhiều tham số hơn. FER-2013 nhiễu và imbalance, nên dễ overfit.

Cách giảm:

dropout 0.3–0.5
weight_decay 1e-4
early stopping theo val macro F1
hidden_dim 64 hoặc 128, không lên quá lớn
internal GNN chỉ 1–2 layer
Rủi ro 2 — Oversmoothing trong subgraph nhỏ

Subgraph chỉ 9–49 nodes. Nếu dùng 3–4 layer GNN, node embedding dễ bị trộn quá mức.

Khuyến nghị:

internal_gnn_layers = 1 hoặc 2
motif_gnn_layers = 2
Rủi ro 3 — Batching sai node/edge

Đây là bug dễ nhất.

Cần test từng sample:

subgraph node_indices đúng không?
local_edge_index có chỉ số local 0..n_i-1 không?
edge_attr có cùng số cạnh không?
pad mask đúng không?
bbox/center khớp với node_indices không?
Rủi ro 4 — Two-level GNN không tăng ngay

Nếu không tăng, chưa chắc hướng sai. Có thể do:

internal subgraph quá nhỏ
edge_attr nghèo
motif selection top-k chưa tốt
learning rate quá cao
internal embedding bị descriptor lấn át

Nên phải chạy ablation rõ.

11. Version triển khai hợp lý
V3.0 — Internal GNN only, giữ motif-level cũ
internal GraphSAGE encode subgraph thật
concat descriptor + match info
motif-level GraphSAGE như cũ
CE loss

Mục tiêu:

Chứng minh learned pixel-subgraph embedding có ích hơn descriptor-only.

So sánh:

V2 motif_norm:
descriptor only

V3.0:
descriptor + internal_gnn_embedding
V3.1 — Internal GNN thay descriptor
motif node feature = internal_gnn_embedding + match info

Mục tiêu:

Kiểm tra internal GNN có tự thay descriptor được không.

Nếu V3.0 tốt hơn V3.1, nghĩa là descriptor vẫn có tín hiệu bổ sung.

V3.2 — Rich motif-level edge_attr
use_edge_attr=true
edge_attr = spatial + descriptor + motif relation + internal embedding similarity

Mục tiêu:

Làm graph cấp motif thật sự có nghĩa.
V3.3 — Multi-scale two-level GNN
radii = 1, 2, 3
K = 48 hoặc 64

Mục tiêu:

bắt cả motif nhỏ và motif rộng.
V3.4 — Prototype consistency loss
L = L_cls + λ L_proto

Mục tiêu:

motif bank không chỉ chọn offline,
mà còn định hướng embedding khi train.
12. Cấu hình khởi đầu tôi khuyên

Dùng cấu hình vừa đủ, tránh quá nặng:

model:
  name: hierarchical_motif_gnn

  # internal pixel-subgraph encoder
  internal_input_dim: 7
  internal_hidden_dim: 64
  internal_out_dim: 128
  internal_num_layers: 2
  internal_dropout: 0.25
  internal_readout: mean_max

  # motif-level node
  use_descriptor: true
  descriptor_dim: 41
  use_match_score_feature: true
  use_disc_score_feature: true
  use_matched_class_onehot: true
  use_bbox_feature: true

  # motif-level GNN
  motif_hidden_dim: 128
  motif_num_layers: 2
  motif_dropout: 0.3
  use_edge_attr: true
  edge_attr_dim: 10
  pooling: motif_attention
  use_motif_score_vector: true

training:
  loss: weighted_ce
  class_weight_power: 0.5
  lr: 0.0005
  weight_decay: 0.0001
  epochs: 80
  early_stop: 15
  monitor: val_macro_f1

Nếu bị overfit, giảm:

internal_hidden_dim: 32
internal_out_dim: 64
motif_hidden_dim: 96
dropout: 0.4
13. Kỳ vọng kết quả thực tế

Baseline hiện tại:

Accuracy 45.11%
Macro F1 0.4196

Kỳ vọng hợp lý:

V3.0 two-level GNN:
+2 đến +6 macro F1 điểm nếu dữ liệu subgraph đúng.

V3.2 rich edge:
có thể thêm +1 đến +3 điểm.

V3.3 multi-scale:
có thể giúp Fear/Sad/Angry hơn nếu selection tốt.

V3.4 prototype loss:
có thể ổn định motif embedding, nhưng cũng có rủi ro nếu motif offline nhiễu.

Mục tiêu gần:

Macro F1 0.45–0.50
Accuracy 48–52%

Mục tiêu graph-only tham vọng hơn:

Macro F1 0.55+
Accuracy 55–60%

Nhưng để lên cao hơn nữa mà không dùng learned visual feature sâu, sẽ khó.

14. Câu mô tả học thuật cho hướng này

Bạn có thể dùng đoạn này trong báo cáo:

Sau khi motif-guided GNN trên descriptor chứng minh rằng quan hệ giữa các motif-selected subgraphs có ích, bước phát triển tiếp theo là chuyển từ descriptor-based subgraph representation sang hierarchical pixel-preserving graph representation. Thay vì nén mỗi selected subgraph thành một vector thống kê 41 chiều, hệ thống giữ lại các pixel nodes và local edges thật của subgraph, sau đó dùng một small GNN để học embedding cấu trúc bên trong subgraph. Các embedding này tiếp tục được xem như nodes của motif-level graph, nơi một GNN cấp cao học quan hệ giữa các vùng biểu cảm đã được motif chọn. Thiết kế này giữ đúng nguyên tắc: full pixel graph là nguồn thông tin, motif là cơ chế chọn lọc, subgraph là đơn vị trung gian, và ảnh vẫn là đơn vị phân loại cuối cùng.
15. Chốt thiết kế nên triển khai

Hướng triển khai nên là:

V3.0 Hierarchical Motif GNN

Với pipeline:

Motif-filtered selected subgraphs
→ lấy lại pixel nodes thật qua node_indices
→ internal small GraphSAGE encode từng subgraph
→ concat descriptor + motif info
→ motif-level GraphSAGE/edge-aware GNN
→ motif attention pooling
→ image-level classifier

Thứ tự làm thực tế:

1. Sửa dataset để lưu node_indices/local_edge_index/local_edge_attr.
2. Viết collate padding cho [B, K, Nmax, F].
3. Viết InternalSubgraphGraphSAGE.
4. Bọc model thành HierarchicalMotifGNN.
5. Train chỉ CE loss.
6. So sánh V2 descriptor-only vs V3 descriptor+internalGNN.
7. Sau đó mới thêm rich edge_attr và prototype consistency loss.

Điểm sống còn:

Đừng bỏ descriptor ngay.
Bản đầu nên dùng descriptor + internal GNN embedding.

Vì descriptor đang là tín hiệu đã chứng minh có ích. Internal GNN là phần bổ sung để khôi phục cấu trúc pixel thật, không nên thay thế hoàn toàn ngay từ đầu.


tôi thêm edge vào thì giảm rồi 2026-04-26 13:08:30 ======================================================= 2026-04-26 13:08:30 --> Accuracy: 44.69% 2026-04-26 13:08:30 --> Macro F1: 0.4071 2026-04-26 13:08:30 --> Weighted F1: 0.4387 2026-04-26 13:08:30 2026-04-26 13:08:30 --> Classification Report: 2026-04-26 13:08:30 precision recall f1-score support 2026-04-26 13:08:30 0 0.316562 0.307536 0.311983 491.000000 2026-04-26 13:08:30 1 0.298246 0.309091 0.303571 55.000000 2026-04-26 13:08:30 2 0.292683 0.181818 0.224299 528.000000 2026-04-26 13:08:30 3 0.602345 0.642776 0.621904 879.000000 2026-04-26 13:08:30 4 0.368421 0.365320 0.366864 594.000000 2026-04-26 13:08:30 5 0.558753 0.560096 0.559424 416.000000 2026-04-26 13:08:30 6 0.415070 0.519169 0.461320 626.000000 2026-04-26 13:08:30 accuracy 0.446921 0.446921 0.446921 0.446921 2026-04-26 13:08:30 macro avg 0.407440 0.412258 0.407052 3589.000000 2026-04-26 13:08:30 weighted avg 0.436598 0.446921 0.438670 3589.000000 2026-04-26 13:08:31 Confusion matrix saved: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/motif_guided_gnn/motif_guided_gnn_26042026_0555/confusion_matrix.png 2026-04-26 13:08:32 2026-04-26 13:08:32 --> Figures saved to: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/motif_guided_gnn/motif_guided_gnn_26042026_0555 2026-04-26 13:08:33 --> [WandB] Send File motif_guided_gnn_26042026_0555_best.pth to cloud successfully! 2026-04-26 13:08:33 2026-04-26 13:08:33 DONE!
Đúng, bản thêm edge_attr bị giảm so với best trước.

So sánh trực tiếp:

Model	Accuracy	Macro F1	Weighted F1
GNN motif_norm best trước	45.11%	0.4196	0.4380
GNN + edge_attr	44.69%	0.4071	0.4387

Nhìn số thì:

Accuracy: 45.11 → 44.69   giảm -0.42%
Macro F1: 0.4196 → 0.4071 giảm -0.0125
Weighted F1: 0.4380 → 0.4387 tăng nhẹ +0.0007

Tức là edge_attr không làm sập model, nhưng không giúp macro F1, thậm chí làm yếu lớp nhỏ.

1. Điểm đáng chú ý nhất: Disgust giảm mạnh

Trước đó bạn báo:

Class 1 Disgust:
precision 0.4583
recall    0.4000
F1        0.4272

Bản thêm edge:

Class 1 Disgust:
precision 0.2982
recall    0.3091
F1        0.3036

Đây là giảm rất rõ:

0.4272 → 0.3036
giảm -0.1236 F1

Trong khi Fear lại hơi tăng:

Fear:
trước F1 ≈ 0.2060
sau F1 = 0.2243

Vậy edge_attr hiện tại không phải vô dụng hoàn toàn, nhưng nó đang đổi phân phối học giữa các lớp. Nó có thể giúp một số lớp yếu như Fear một chút, nhưng làm hỏng Disgust khá nặng.

2. Vì sao thêm edge lại giảm?

Khả năng cao không phải vì “edge là sai hướng”, mà vì edge_attr hiện tại chưa đủ nghĩa hoặc đang được đưa vào sai cách.

Trong code hiện tại, nếu use_edge_attr=True, edge_attr đi qua một edge_mlp, sau đó sigmoid thành gate:

edge_attr → edge_mlp → sigmoid → gate
ev = ev * gate

Tức là edge_attr không chỉ “bổ sung thông tin”, mà nó điều khiển độ mạnh của message passing.

Nếu gate học chưa tốt, nó có thể:

giảm message quan trọng
làm yếu kết nối giữa các motif nodes
khuếch đại nhiễu
làm model khó tối ưu hơn

Đặc biệt nếu edge_attr hiện tại chỉ là:

dx, dy, dist

thì nó khá nghèo. Vì graph cấp motif đã có edge_index, nghĩa là model đã biết subgraph nào nối với subgraph nào. Thêm dx, dy, dist đơn giản có thể không cung cấp nhiều thông tin mới, nhưng lại thêm một cơ chế gate làm nhiễu quá trình aggregate.

Nói ngắn gọn:

edge_attr hiện tại chưa đủ giàu để giúp,
nhưng đủ can thiệp vào message passing để làm giảm macro F1.
3. Kết luận cho thử nghiệm này

Không nên hiểu kết quả này là:

Edge không có ích.

Mà nên hiểu là:

Edge_attr hiện tại, với cách dùng hiện tại, chưa có ích.

Bản này nên được ghi thành một ablation:

Bật edge gating bằng edge_attr đơn giản dx/dy/dist không cải thiện kết quả. Macro F1 giảm từ 0.4196 xuống 0.4071, cho thấy edge attributes hiện tại chưa đủ biểu diễn quan hệ motif-level có ý nghĩa.

Đây là kết quả tốt về mặt nghiên cứu, vì nó nói rằng:

Không phải cứ thêm edge_attr là tốt.
Quan trọng là edge_attr phải mô tả đúng quan hệ giữa các motif-selected subgraphs.
4. Tôi khuyên xử lý thế nào?
Việc 1 — Quay lại best config làm baseline chính

Bản chính vẫn là:

GNN motif_norm
Accuracy 45.11%
Macro F1 0.4196
Weighted F1 0.4380

Bản edge_attr hiện tại chỉ giữ làm ablation phụ.

Không nên dùng bản edge làm main.

Việc 2 — Kiểm tra gate statistics

Bạn nên log thử:

gate.mean()
gate.std()
gate.min()
gate.max()

theo batch hoặc theo epoch.

Nếu gate mean quá thấp, ví dụ:

0.2–0.4

thì edge MLP đang làm yếu message passing.

Nếu gate gần như hằng số:

≈ 0.5 mọi cạnh

thì edge_attr không học được gì, chỉ làm nhiễu.

Nếu gate dao động mạnh:

gần 0 hoặc gần 1

thì model có thể đang cắt cạnh quá cực đoan.

Việc 3 — Không bật edge_attr nếu chỉ có dx/dy/dist

Nếu edge_attr chỉ là:

dx, dy, dist

thì tôi khuyên tạm thời không dùng.

Vì với motif-level graph, thông tin cần hơn là:

bbox overlap
center distance normalized
descriptor cosine
same matched class
same motif id
match score difference
disc score pair
region relation
left-right symmetry

Khi có các feature này, edge mới thực sự mang nghĩa:

hai vùng này có liên hệ biểu cảm không?
hai vùng này bổ sung nhau hay trùng nhau?
hai vùng này cùng match motif class nào?
5. Cách cải tiến edge_attr đúng hơn

Thay vì dùng edge_attr hiện tại, bạn nên nâng lên bản:

rich_motif_edge_attr

Gợi ý vector:

[
  dx_center,
  dy_center,
  euclidean_dist,
  bbox_iou,
  descriptor_cosine,
  same_matched_class,
  same_motif_id,
  abs_match_score_diff,
  mean_disc_score_pair,
  region_relation_id
]

Nếu có internal GNN embedding sau này thì thêm:

internal_embedding_cosine

Lúc đó edge không chỉ nói “hai node cách nhau bao xa”, mà còn nói:

hai motif-selected subgraphs này có giống nhau không,
có cùng motif không,
có trùng vùng không,
có quan hệ vùng mặt hợp lý không.
6. Cách dùng edge an toàn hơn

Hiện tại edge_attr đang dùng như gate trực tiếp lên message. Cách này hơi mạnh.

Có 3 cách an toàn hơn.

Cách A — Residual gate

Thay vì:

ev = ev * gate

dùng:

ev = ev * (0.5 + gate)

hoặc:

ev = ev * (1.0 + alpha * (gate - 0.5))

Với alpha = 0.2 hoặc 0.5.

Mục tiêu:

edge_attr chỉ điều chỉnh nhẹ message,
không bóp chết cạnh.
Cách B — Khởi tạo gate gần 1

Bias cuối của edge_mlp có thể init dương, ví dụ:

bias = 2.0
sigmoid(2.0) ≈ 0.88

Như vậy ban đầu model gần giống baseline không edge. Sau đó nó học điều chỉnh dần.

Hiện nếu gate ban đầu quanh 0.5, message passing bị giảm một nửa ngay từ đầu.

Cách C — Edge feature concat vào node-pair scorer, không gate message

Thay vì dùng edge_attr để nhân vào adjacency, dùng nó để tạo attention/bias ở tầng sau.

Ví dụ:

edge_attr → edge_bias
attention_score_ij += edge_bias_ij

Cách này hợp hơn với Graph Transformer/GAT sau này.

7. Hướng tiếp theo nên làm ngay

Tôi khuyên thứ tự:

1. Giữ edge_attr=false làm baseline chính.
2. Log gate statistics cho bản edge hiện tại.
3. Thử residual gate hoặc init gate gần 1.
4. Sau đó mới làm rich edge_attr.
5. Không mất thời gian tune dx/dy/dist quá nhiều.

Quan trọng nhất: đừng để kết quả này làm bạn bỏ hướng edge. Nó chỉ chứng minh rằng edge hiện tại chưa đủ tốt.

8. Ảnh hưởng đến kế hoạch Two-level GNN

Kết quả này càng củng cố rằng hướng tiếp theo không nên chỉ là “thêm edge vào motif-level GNN”.

Vì edge-level hiện tại chưa đủ giàu, còn descriptor 41D vẫn là bottleneck.

Hướng chính vẫn nên là:

Two-level GNN:
pixel nodes trong selected subgraph
→ internal subgraph GNN
→ motif-level GNN
→ classifier

Sau khi có internal embedding, edge_attr giữa motif nodes sẽ mạnh hơn vì có thể thêm:

internal_embedding_cosine
internal_embedding_distance

Khi đó edge mới có thêm thông tin học được từ cấu trúc pixel thật.

9. Câu ghi vào log/báo cáo

Bạn có thể ghi:

Khi bật edge_attr đơn giản cho motif-level GraphSAGE, mô hình đạt 44.69% accuracy, macro F1 0.4071 và weighted F1 0.4387. So với baseline motif_norm không dùng edge_attr, macro F1 giảm từ 0.4196 xuống 0.4071. Đặc biệt, F1 của lớp Disgust giảm từ 0.4272 xuống 0.3036. Kết quả này cho thấy edge attributes hiện tại chưa đủ giàu để cải thiện message passing, và việc dùng edge_attr như gate trực tiếp có thể làm suy yếu các kết nối quan trọng. Do đó, edge_attr đơn giản chỉ nên được xem là ablation, còn hướng phát triển tiếp theo là thiết kế rich motif-level edge attributes hoặc chuyển sang hierarchical GNN để học embedding cấu trúc bên trong từng selected subgraph.
10. Chốt quyết định

Quyết định hiện tại:

Không dùng bản edge_attr hiện tại làm best.
Giữ GNN motif_norm không edge làm baseline chính.
Tiếp tục two-level GNN.
Edge_attr sẽ quay lại sau, nhưng phải là rich edge_attr hoặc residual/attention edge, không phải gate dx/dy/dist đơn giản.

Nói ngắn gọn:

Thêm edge giảm không đáng lo.
Nó chỉ nói rằng edge hiện tại chưa đúng loại thông tin.
Bottleneck lớn hơn vẫn là descriptor 41D.
Bước đáng làm nhất vẫn là internal pixel-subgraph GNN.