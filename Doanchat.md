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
