Được. Nếu tạm thời giữ GNN-only, mình sẽ không đề xuất CNN nữa. Lộ trình hợp lý nhất không phải “làm C to hơn”, mà là phát triển theo hướng:

B/C hiện tại
→ audit selection
→ motif tốt hơn
→ selection học được
→ relation graph động hơn
→ loss phụ cho graph/prototype

Vì hiện tại B và C đã chứng minh được hai điều: B chứng minh motif-level GNN giữa selected subgraphs có ích, còn C chứng minh internal pixel-subgraph GNN bên trong selected subgraphs có thêm tín hiệu ngoài descriptor 41D. Kết quả C hiện tăng từ B macro F1 0.4196 lên 0.4414, và bản C hiện dùng z_internal + descriptor 41D + motif metadata trước khi đưa vào motif-level GraphSAGE.

1. Chốt tư duy phát triển GNN-only

Mục tiêu hiện tại không nên là:

Full pixel graph 2304 node → GNN thật sâu → classifier

Vì full graph quá nặng, dễ oversmoothing, và ảnh FER-2013 chỉ 48×48, nhiễu cao. Hướng hiện tại của bạn đúng hơn:

full pixel graph
→ local candidate subgraphs
→ motif/selection
→ motif-level graph
→ classifier

Tài liệu hiện tại cũng đã định nghĩa rõ: motif không phải frequent subgraph thuần túy, mà là class-related prototype over local pixel-subgraph descriptors, dùng để chọn candidate subgraphs có khả năng mang tín hiệu cảm xúc.

Vậy muốn bứt phá bằng GNN-only, phải nâng từ:

descriptor-based motif selection

lên:

learnable graph-aware motif selection

Lý do: C hiện học tốt hơn trên vùng đã được chọn, nhưng chưa học cách chọn vùng tốt hơn.

2. Lộ trình tổng quát

Mình đề xuất 6 giai đoạn:

Stage 0: Khóa baseline và audit
Stage 1: Multi-scale candidate subgraphs
Stage 2: Graph-aware motif prototype bank
Stage 3: Soft / learnable selection
Stage 4: Dynamic relation motif-level GNN
Stage 5: Prototype + contrastive auxiliary losses
Stage 6: End-to-end refinement trong giới hạn GNN-only

Đi theo thứ tự này vì mỗi stage trả lời một câu hỏi nghiên cứu riêng. Nếu làm nhảy cóc, bạn sẽ không biết tăng/giảm là do candidate, motif, selection, graph relation hay loss.

3. Stage 0 — Khóa baseline và audit selection
Mục tiêu

Trước khi phát triển model D, cần biết pipeline hiện tại đang chọn vùng có hợp lý không.

Hiện tại pixel motif dataset có:

x [32, 41]
bbox [32, 4]
centers [32, 2]
selected_indices [32]
node_indices [32, 25]
edge_index
match_scores
matched_class
matched_motif_id
matched_disc_score
motif_score_vector
coverage_cell

Các key này đủ để audit vùng nào được chọn, motif nào được match, class nào bị bias.

Cần làm

Viết script audit:

scripts/audit_selected_motifs.py

Nó nên xuất:

1. Histogram coverage_cell theo từng class thật.
2. Histogram matched_class theo từng class thật.
3. Top motif_id thường xuất hiện trong mỗi class.
4. Attention heatmap nếu model có attention.
5. Bbox overlay cho ảnh đúng/sai.
6. So sánh ảnh Fear/Angry/Sad bị sai: selected vùng nào?
Vì sao phải làm?

Vì nếu model đang chọn sai vùng, ví dụ Fear lại chọn nhiều vùng má/nền thay vì mắt-miệng, thì tăng hidden_dim hay thêm layer không giải quyết được.

Stage 0 trả lời câu hỏi:

Vấn đề nằm ở model hay nằm ở selection?
Kết quả mong muốn

Sau stage này, bạn phải biết:

Happy/Surprise đúng vì chọn vùng nào?
Fear/Angry/Sad sai vì thiếu vùng nào?
matched_class có bị lệch về Happy/Neutral không?
coverage có đang ép chọn vùng ít liên quan không?
4. Stage 1 — Multi-scale candidate subgraphs
Hiện tại

Candidate hiện tại:

seed_stride = 4
radii = [1, 2]
max_candidates = 128
coverage_grid = [4, 4]

Mỗi candidate lưu seed_node, radius, node_indices, edge_index_sub, bbox, coverage_cell, num_nodes, num_edges; descriptor 41D là input chính của baseline B.

Vấn đề

radii = [1, 2] rất cục bộ. Nó bắt được texture nhỏ, nhưng biểu cảm thường có nhiều scale:

scale nhỏ: khóe miệng, nếp nhăn mắt, biên lông mày
scale vừa: vùng mắt, vùng miệng
scale lớn hơn: quan hệ má-miệng, mắt-lông mày

Nếu candidate không chứa vùng đủ lớn, model không thể học shape biểu cảm.

Đề xuất

Tạo data recipe mới:

data_recipe_id: pixel_motif_v3_multiscale

candidate:
  seed_stride: 3 hoặc 4
  radii: [1, 2, 3]
  max_candidates: 192
  coverage_grid: [4, 4]

selection:
  top_k: 48
  nmax: 49

Hoặc bản nhẹ hơn:

radii: [1, 2, 3]
max_candidates: 160
top_k: 40
nmax: 49
Vì sao làm trước?

Vì nếu candidate set thiếu vùng tốt, motif bank và model phía sau đều bị giới hạn. Đây là “nguồn nguyên liệu”. Nguyên liệu nghèo thì model mạnh cũng không cứu được.

Lưu ý

Stage này là thay đổi data recipe, nên phải build artifact mới. Theo workflow hiện tại, thay candidate radii, max_candidates, top_k, Nmax, motif bank hoặc selection logic đều là thay đổi data, không thể dùng artifact cũ để kết luận công bằng.

Chạy gì?

Sau khi build artifact mới, chạy lại:

B trên artifact mới
C-mean trên artifact mới

Nếu chỉ multi-scale đã tăng, nghĩa là bottleneck lớn nằm ở candidate coverage.

5. Stage 2 — Graph-aware motif prototype bank
Hiện tại

Motif bank hiện tại là prototype trên descriptor 41D:

candidate subgraph
→ descriptor 41D
→ motif bank theo emotion
→ greedy motif-guided selection

Motif-guided selection hiện so khớp candidate descriptors với motif bank rồi chọn top-K bằng coverage/diversity.

Vấn đề

Descriptor 41D là thống kê thủ công:

node mean/std/min/max
num_nodes, num_edges, density
edge mean/std

Nó nhẹ, nhưng làm mất cấu trúc không gian bên trong subgraph.

Trong khi bản C đã học được z_internal từ pixel nodes thật:

sub_x_i [Nmax, 7]
sub_adj_i [Nmax, Nmax]
sub_node_mask_i [Nmax]
→ InternalPixelSubgraphEncoder
→ z_i

C hiện concat z_i với descriptor 41D và motif metadata.

Vậy motif bank mới nên tận dụng chính z_internal.

Đề xuất

Tạo:

MotifGraphPrototypeBank

Mỗi motif không chỉ lưu descriptor centroid, mà lưu:

MotifGraphPrototype = {
    "class_id": int,
    "motif_id": int,

    "descriptor_centroid": Tensor[41],
    "internal_embedding_centroid": Tensor[D],

    "exemplar_sub_x": Tensor[Nmax, 7],
    "exemplar_sub_adj": Tensor[Nmax, Nmax],
    "exemplar_sub_node_mask": Tensor[Nmax],

    "support_count": int,
    "coverage_cell_distribution": Tensor[16],
    "discriminative_score": float
}

Quan trọng: không làm 1 graph đại diện cho mỗi emotion. Làm nhiều local graph-aware prototypes cho mỗi emotion.

Vì sao?

Một class có nhiều biến thể:

Happy: khóe miệng, má, mắt híp
Angry: lông mày, mắt, miệng mím
Fear: mắt mở, miệng, lông mày
Sad: miệng trễ, mắt, lông mày

Một prototype/class là quá nghèo. ProtGNN cũng đi theo tinh thần prototype trong latent space để prediction có tính giải thích, thay vì chỉ post-hoc explain sau khi model đã dự đoán.

Cách triển khai thực tế

Không cần end-to-end ngay. Làm offline:

1. Train C-mean tốt nhất.
2. Freeze InternalPixelSubgraphEncoder.
3. Precompute z_internal cho tất cả candidate subgraphs.
4. Build motif bank mới trên concat([descriptor_41D, z_internal]).
5. Tính discriminative_score cho từng prototype.
6. Selection lại bằng hybrid score.

Hybrid score:

score(candidate, motif)
=
α * sim_descriptor(candidate, motif)
+ β * sim_z_internal(candidate, motif)
+ γ * discriminative_score(motif)
+ δ * coverage_bonus
- η * redundancy_penalty
Vì sao đây là bước rất đáng làm?

Vì nó đánh thẳng vào nút nghẽn hiện tại: selection vẫn descriptor-first. Stage 2 biến motif từ vector thống kê thành graph-aware prototype.

6. Stage 3 — Soft / learnable selection
Hiện tại

Selection đang hard top-K:

candidate descriptors
→ match motif bank
→ greedy_select_with_coverage
→ top-K selected subgraphs

Output là x [K,41], match_scores, matched_class, matched_motif_id, node_indices, node_mask.

Vấn đề

Hard top-K có lỗi lớn:

candidate bị loại là mất hoàn toàn

Nếu vùng quan trọng không vào top-K, model không thể học từ nó.

Đề xuất 3A — Soft top-M trước

Thay vì chọn cứng 32, giữ nhiều hơn:

candidate M = 64 hoặc 96

Sau đó model tự học attention:

candidate_i
→ h_i
→ score_i = MLP/GNN(h_i)
→ attention_i = softmax(score_i)
→ weighted pooling

Lúc đầu có thể dùng:

top_M = 64
pool_to_K = attention/gating

Không cần bỏ top-K ngay, có thể dùng hai tầng:

greedy prefilter 128 → 64
learnable attention 64 → image embedding
Đề xuất 3B — SAGPool-like selection

SAGPool đề xuất graph pooling bằng self-attention, trong đó score được học bằng graph convolution nên xét cả node features và graph topology.

Áp vào project:

candidate graph M nodes
node feature = descriptor + z_internal + motif metadata
edge = spatial + feature similarity
SAGPool-like scorer
→ chọn hoặc weight các candidate quan trọng
Vì sao stage này quan trọng?

Vì nó biến lựa chọn subgraph từ:

tiền xử lý heuristic

thành:

một phần được học của GNN

Đây là điều cần thiết nếu muốn GNN-only đi xa hơn.

7. Stage 4 — Dynamic multi-relation motif-level graph
Hiện tại

Motif-level graph hiện dùng directed KNN theo center:

edge_index [2, E]
edge_attr [E, 3] = dx, dy, dist

Nhưng B và C đầu tiên không dùng edge_attr rich; edge_attr được lưu để inspect/ablation.

Bạn cũng đã thử dùng edge_attr dx, dy, dist, kết quả giảm, nên việc tạm dừng edge_attr là hợp lý.

Vấn đề

KNN theo khoảng cách không gian chỉ nói:

hai vùng này gần nhau

nhưng FER cần:

hai vùng này phối hợp biểu cảm với nhau

Ví dụ mắt và miệng có thể xa nhau nhưng cùng quyết định Surprise/Fear.

Face2Nodes xây dựng dynamic graph bằng dilated k-nearest neighbors và relation-aware graph convolution để học tương quan giữa các vùng mặt, thay vì chỉ dựa vào grid cố định.

Đề xuất

Tạo multi-relation graph:

Relation 1: spatial edge
- gần nhau theo center/bbox

Relation 2: feature-similarity edge
- cosine(z_internal_i, z_internal_j)

Relation 3: motif-prototype edge
- cùng matched motif/class hoặc motif group

Relation 4: long-range facial relation
- top attention relation learned từ h_i, h_j

Forward:

h_spatial = GNN_spatial(h, edge_spatial)
h_feature = GNN_feature(h, edge_feature)
h_motif   = GNN_motif(h, edge_motif)

h = fuse([h_spatial, h_feature, h_motif])

Fuse có thể là:

concat + Linear
attention over relation types
gated sum
Vì sao không dùng lại dx/dy/dist gate cũ?

Vì dx, dy, dist quá nghèo. Nó chỉ là hình học thô. Nếu dùng như gate trực tiếp, nó có thể làm yếu thông tin. Thay vào đó, hãy dùng nhiều loại edge và để model học loại relation nào hữu ích.

8. Stage 5 — Prototype + contrastive losses
Hiện tại

B và C dùng weighted CE, class_weight_power = 0.5, không dùng prototype loss hoặc contrastive loss trong C đầu tiên.

Weighted CE là đúng cho baseline, nhưng nếu muốn kéo class khó như Fear/Angry/Sad, cần loss phụ.

5.1. Supervised contrastive loss cấp ảnh

Mục tiêu:

embedding ảnh cùng class gần nhau
embedding ảnh khác class xa nhau

Nó giúp phân tách các lớp dễ nhầm:

Fear vs Sad
Sad vs Neutral
Angry vs Neutral
Fear vs Surprise
5.2. Prototype consistency loss cấp motif

Vì bạn có motif prototype, ép selected subgraph gần prototype đúng class hơn:

L_proto =
- log exp(sim(z_i, p_y)/τ)
  / Σ_c exp(sim(z_i, p_c)/τ)

Không áp quá mạnh ngay. Dùng weight nhỏ:

loss:
  ce_weight: 1.0
  proto_weight: 0.05
  supcon_weight: 0.05
5.3. Diversity loss cho selected subgraphs

Tránh chọn 32 vùng quá giống nhau:

L_div = mean cosine_similarity(h_i, h_j)

Minimize nhẹ để giữ đa dạng.

5.4. Coverage regularization mềm

Không ép cứng phủ đều 4×4, mà regularize nhẹ để tránh collapse vào 1 vùng.

9. Stage 6 — End-to-end refinement GNN-only

Sau khi có các bước trên, mới nghĩ đến end-to-end hơn.

Pipeline cuối GNN-only có thể là:

CSV
→ pixel graph
→ multi-scale candidate subgraphs
→ candidate internal GNN encoder
→ graph-aware motif prototypes
→ candidate graph
→ learnable graph pooling
→ dynamic motif-level GNN
→ prototype-aware classifier

Tên model có thể đặt:

GraphAwareMotifGNN

hoặc:

LearnableMotifPoolingGNN
10. Lộ trình triển khai cụ thể theo version
Version C-final — Chốt bản hiện tại

Mục tiêu: khóa baseline hiện tại.

Làm:

1. Chạy B trên cùng artifact mới.
2. Chốt C-mean là current best.
3. Lưu kết quả B/C/C-light/C-no-desc.
4. Viết audit script.

Lý do: không có mốc sạch thì các bản sau không biết có thật sự tốt hơn không.

Version D0 — AuditMotifSelection

Không đổi model.

Thêm:

scripts/audit_selected_motifs.py
scripts/visualize_selected_motifs.py

Output:

audit/coverage_by_class.csv
audit/matched_class_by_true_class.csv
audit/top_motifs_by_class.csv
audit/wrong_cases_overlay/
audit/attention_overlay/

Lý do: xác định selection sai ở đâu trước khi sửa.

Version D1 — MultiScale-C

Đổi data, giữ model C-mean.

Config:

candidate:
  radii: [1, 2, 3]
  max_candidates: 192
selection:
  top_k: 48
  nmax: 49
model:
  internal_readout: mean

Lý do: tăng vùng ứng viên và scale trước khi đổi thuật toán chọn.

Kỳ vọng: nếu macro F1 tăng, chứng minh candidate set cũ thiếu vùng/scale.

Version D2 — GraphAwareMotifBank

Đổi motif bank/selection, model vẫn gần C.

Pipeline:

Train C-mean
→ freeze internal encoder
→ precompute z_internal for candidates
→ build motif bank trên [descriptor, z_internal]
→ hybrid selection
→ train C-mean lại

Lý do: đây là bước quan trọng nhất để vượt bottleneck descriptor-only motif.

Version D3 — SoftMotifSelectionGNN

Đổi model nhận nhiều candidate hơn.

Thay:

hard top-K 32

bằng:

top-M 64 + learnable attention/gating

Model:

candidate features
→ CandidateGNN
→ attention scorer
→ weighted pooling hoặc soft selected motif nodes
→ classifier

Lý do: giảm mất thông tin do hard selection.

Version D4 — SAGPoolMotifGNN

Selection học bằng graph topology.

Dựa trên tinh thần SAGPool: score node bằng graph convolution/self-attention để xét cả feature và topology.

Pipeline:

candidate graph M nodes
→ GNN
→ SAGPool-like top-K
→ motif-level GNN
→ classifier

Lý do: selection trở thành một phần của GNN, không còn chỉ là pre-processing.

Version D5 — ASAP-like Motif Coarsening

ASAP học sparse soft cluster assignment để pool subgraphs thành pooled graph, nhằm giữ local substructure tốt hơn so với chỉ chọn node độc lập.

Áp dụng:

candidate nodes
→ local clusters/substructures
→ pooled motif graph
→ classifier

Lý do: FER có pattern vùng, không chỉ từng candidate độc lập. ASAP-like hợp với ý tưởng “motif là substructure”.

Version D6 — PrototypeAwareMotifGNN

Thêm prototype classifier/loss.

Model prediction không chỉ từ classifier MLP, mà thêm prototype logits:

logit_class_c = max / avg similarity(image_or_motif_embedding, prototypes_c)

ProtGNN cho thấy prototype learning có thể tạo mô hình tự giải thích, prediction dựa trên so sánh input với learned prototypes trong latent space.

Lý do: rất hợp với motif bank của bạn, vì motif vốn đã là prototype. Chỉ cần nâng prototype từ offline descriptor bank thành learned prototype latent bank.

11. Thứ tự ưu tiên nếu thời gian có hạn

Nếu bạn chỉ có thời gian làm 3 bản tiếp theo, làm theo thứ tự này:

1. D0 AuditMotifSelection
2. D2 GraphAwareMotifBank
3. D3 SoftMotifSelectionGNN

Không chọn D1 đầu tiên nếu build data quá lâu. Nhưng nếu bạn có thể build artifact mới, D1 cũng rất đáng.

Nếu bạn có thời gian vừa phải:

1. D0
2. D1
3. D2
4. D3

Nếu muốn hướng nghiên cứu mạnh nhất:

D0 → D1 → D2 → D4 → D6
12. Vì sao không nên nhảy thẳng D6?

Vì prototype loss/prototype classifier chỉ hiệu quả khi candidate và motif đã tương đối tốt. Nếu selected subgraphs vẫn sai, prototype sẽ học prototype của nhiễu.

Thứ tự đúng là:

chọn vùng tốt hơn
→ học quan hệ tốt hơn
→ mới ép prototype/contrastive mạnh hơn

Không nên:

selection còn nhiễu
→ thêm loss mạnh
→ model tự tin hơn trên vùng sai
13. Kết luận chiến lược

Trong GNN-only, điểm bứt phá của bạn không nằm ở việc tăng số layer. Nó nằm ở ba câu hỏi:

1. Candidate có chứa đúng vùng biểu cảm không?
2. Motif có đại diện được local graph pattern không?
3. Selection có được học bởi GNN không?

Vì vậy lộ trình đúng là:

C hiện tại
→ audit
→ multi-scale candidate
→ graph-aware motif bank
→ soft/learnable selection
→ dynamic relation graph
→ prototype/contrastive losses

Câu chốt:

Muốn GNN-only đi xa, hãy biến motif selection từ bước tiền xử lý thủ công thành một tầng học được của mô hình.

Bản C đã chứng minh internal subgraph GNN có ích. Bản D nên chứng minh điều tiếp theo: GNN không chỉ học trên motif đã chọn, mà còn học cách chọn và tổ chức motif tốt hơn.