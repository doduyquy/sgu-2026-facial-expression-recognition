Giai đoạn 0 — Chốt phạm vi bản đầu tiên
Trước khi code, phải chốt rõ bạn đang làm bản MVP nghiên cứu chứ chưa phải bản full cuối.
Bản đầu tiên nên là:
•	ảnh 48x48 → pixel graph 
•	sinh candidate subgraphs cục bộ 
•	xây motif bank đơn giản 
•	chọn top-k subgraphs theo motif matching 
•	encode subgraphs 
•	aggregate ở mức ảnh 
•	classifier 7 lớp 
Tức là chưa cần làm ngay:
•	learned compatibility quá phức tạp 
•	prototype learning end-to-end quá nặng 
•	exact motif mining kiểu graph isomorphism 
•	attention quá cầu kỳ trên full graph 
________________________________________
Giai đoạn 1 — Chuẩn bị dữ liệu và cấu trúc project
Bạn cần dựng bộ khung code trước.
Việc phải làm
1.	Loader cho FER-2013 hoặc bản split bạn đang dùng 
2.	Chuẩn hóa ảnh 
3.	Tạo folder/module rõ ràng 
Ví dụ cấu trúc:
project/
│
├── configs/
│   └── base.yaml
│
├── data/
│   └── fer2013_loader.py
│
├── graph/
│   ├── image_to_graph.py
│   ├── subgraph_generator.py
│   ├── subgraph_features.py
│   └── motif_matching.py
│
├── motif/
│   ├── motif_bank_builder.py
│   └── motif_bank_utils.py
│
├── models/
│   ├── subgraph_encoder.py
│   ├── image_aggregator.py
│   └── motif_classifier.py
│
├── training/
│   ├── losses.py
│   ├── trainer.py
│   └── evaluator.py
│
├── scripts/
│   ├── build_graph_cache.py
│   ├── build_motif_bank.py
│   └── train.py
│
└── utils/
    ├── seed.py
    ├── io.py
    └── metrics.py
Kết quả đầu ra của giai đoạn này
Bạn phải đọc được dữ liệu và lấy được:
•	image 
•	label 
•	image_id 
________________________________________
Giai đoạn 2 — Image to Graph
Đây là bước biến ảnh thành pixel graph. Đây là phần nền quan trọng nhất. 
Bạn phải code gì
1.	Node feature extractor 
o	intensity 
o	normalized position (x, y) 
o	gradient (gx, gy) 
o	gradient magnitude 
o	local contrast 
2.	Edge builder 
o	8-neighbor grid 
o	edge weight theo similarity nhẹ 
Dữ liệu đầu ra cho mỗi ảnh
Ví dụ:
graph = {
    "node_features": X,      # shape [2304, d]
    "edge_index": E,         # shape [2, num_edges]
    "edge_attr": W,          # shape [num_edges, e_dim]
    "label": y,
    "image_id": id_
}
Hàm cần có
•	build_node_features(image) 
•	build_edge_index(height, width, connectivity=8) 
•	build_edge_attr(node_features, edge_index) 
•	image_to_graph(image) 
Mục tiêu kiểm tra
•	1 ảnh tạo đúng 2304 node 
•	edge đúng số lượng 
•	feature không NaN 
•	shape chuẩn 
________________________________________
Giai đoạn 3 — Sinh candidate subgraphs
Đây là bước giảm bài toán từ full graph sang các cấu trúc con. Đây là bước bắt buộc trước motif. 
Bản đầu tiên nên dùng gì
Dùng radius-bounded local subgraphs.
Ví dụ:
•	lấy mỗi node seed 
•	BFS trong bán kính r=1 hoặc r=2 
•	thu được một subgraph nhỏ 
Hoặc đơn giản hơn:
•	chỉ lấy seed theo stride 
•	ví dụ mỗi 2 hoặc 4 pixel lấy 1 seed để giảm số lượng 
Bạn phải code gì
•	sample_seed_nodes(...) 
•	extract_radius_subgraph(graph, seed, radius) 
•	generate_candidate_subgraphs(graph, max_candidates, radius) 
Output
Mỗi ảnh sẽ có:
candidate_subgraphs = [g1, g2, g3, ...]
Mỗi subgraph gồm:
•	node indices gốc 
•	node features con 
•	edge con 
•	metadata: center, size, radius 
Điều nên làm ngay
Giới hạn:
•	radius 
•	max_nodes 
•	max_candidates_per_image 
Không thì số subgraph sẽ bùng nổ.
________________________________________
Giai đoạn 4 — Biểu diễn subgraph
Bạn chưa nên nhảy ngay vào GNN. Trước tiên cần một representation đơn giản cho subgraph để xây motif bank. 
Bản đầu tiên nên làm
Mỗi subgraph → 1 vector descriptor.
Descriptor nên gồm
•	mean/std intensity 
•	mean/std gradient magnitude 
•	mean/std local contrast 
•	node count 
•	edge count 
•	density 
•	mean edge weight 
•	std edge weight 
Ví dụ:
subgraph_descriptor.shape = [D]
Bạn phải code gì
•	compute_subgraph_stats(subgraph) 
•	subgraph_to_descriptor(subgraph) 
Mục tiêu
Có thể biến mọi subgraph thành vector cố định để:
•	clustering 
•	scoring 
•	prototype selection 
________________________________________
Giai đoạn 5 — Xây motif bank theo emotion
Đây là trái tim của phần motif. Nhưng ở bản đầu tiên, đừng làm quá nặng. 
Bản đầu tiên nên làm thế nào
Với mỗi emotion:
1.	gom descriptor của các subgraphs từ ảnh thuộc lớp đó 
2.	cluster chúng 
3.	lấy centroid hoặc exemplar làm motif prototype 
Tức là motif bank ban đầu có thể làm theo kiểu:
•	KMeans 
•	MiniBatchKMeans 
•	hoặc chọn top representative subgraphs 
Bạn phải code gì
•	collect_class_subgraphs(dataset, class_id) 
•	build_motif_bank_for_class(descriptors, num_motifs) 
•	build_all_motif_banks(...) 
Output
motif_bank = {
    0: [m0_1, m0_2, ...],   # angry
    1: [m1_1, m1_2, ...],   # disgust
    ...
    6: [m6_1, m6_2, ...],   # neutral/surprise tùy mapping
}
Mỗi motif có thể gồm:
•	prototype vector 
•	class label 
•	optional exemplar subgraph id 
Lưu ý rất quan trọng
Ở bản đầu tiên, motif bank chưa cần “inter-class discrimination” cực mạnh.
Chỉ cần:
•	intra-class representative tốt 
•	số motif vừa phải 
•	dễ match 
Sau đó mới nâng cấp thêm discrimination score.
________________________________________
Giai đoạn 6 — Motif matching
Khi đã có motif bank, bạn cần cho mỗi ảnh match các candidate subgraphs với motif. Đây là bước chọn subgraph quan trọng. 
Bản đầu tiên nên match thế nào
Dùng soft score trên descriptor vector trước.
Ví dụ:
•	cosine similarity 
•	euclidean distance âm 
•	weighted similarity 
score(subgraph, motif) = cosine(desc_s, proto_m)
Bạn phải code gì
•	match_subgraph_to_motif(subgraph_desc, motif_proto) 
•	match_subgraph_to_bank(subgraph_desc, motif_bank) 
•	select_topk_subgraphs(candidate_subgraphs, motif_bank, k) 
Output cho mỗi ảnh
selected = {
    "subgraphs": [...],         # top-k subgraphs
    "scores": [...],
    "matched_classes": [...],
    "matched_motifs": [...]
}
Đây là thứ bạn dùng để train
Không train trên full graph nữa, mà train trên top-k matched subgraphs. 
________________________________________
Giai đoạn 7 — Xây dataset mới cho training
Sau motif filtering, bạn nên tạo một dataset train mới.
Mỗi sample nên gồm
•	selected_subgraphs 
•	selected_descriptors 
•	match_scores 
•	image_label 
Ví dụ:
sample = {
    "subgraphs": [sg1, sg2, ..., sgk],
    "descriptors": [d1, d2, ..., dk],
    "scores": [s1, s2, ..., sk],
    "label": y
}
Bạn phải code gì
•	MotifFilteredDataset 
•	collate function cho batch có số subgraph cố định hoặc padding 
________________________________________
Giai đoạn 8 — Subgraph encoder
Đây là chỗ bắt đầu model learning thực sự. Theo tài liệu của bạn, lựa chọn hợp lý nhất là small GNN trên subgraph nhỏ. 
Bản đầu tiên nên chọn gì
Chọn 1 trong 2 cách:
Cách A — MLP baseline
Lấy descriptor vector của subgraph → MLP.
z_i = MLP(descriptor_i)
Ưu điểm:
•	cực dễ train 
•	ít bug 
•	phù hợp baseline 
Cách B — Small GNN
Mỗi subgraph là graph nhỏ:
•	1–2 layer GCN hoặc GraphSAGE 
•	graph readout để thành embedding z_i 
z_i = SubgraphEncoder(subgraph_i)
Khuyên bạn
Làm A trước, rồi lên B sau.
Vì nếu bạn làm GNN ngay từ đầu, bug sẽ đến từ:
•	graph construction 
•	subgraph extraction 
•	batching graph 
•	model
rất khó debug. 
________________________________________
Giai đoạn 9 — Image-level aggregation
Theo hướng bạn đã chốt, ảnh vẫn là đơn vị dự đoán cuối cùng. Vậy cần gộp embedding của top-k subgraphs thành embedding ảnh. 
Bản đầu tiên nên làm
1.	Mean pooling baseline 
2.	Sau đó mới attention pooling 
Công thức
Nếu có k subgraphs:
h_img = mean(z_1, z_2, ..., z_k)
Nâng cấp:
alpha_i = softmax(score_i)
h_img = sum(alpha_i * z_i)
Bạn phải code gì
•	MeanAggregator 
•	AttentionAggregator 
________________________________________
Giai đoạn 10 — Classifier
Rất đơn giản:
logits = MLP(h_img)
Output:
•	7 lớp emotion 
________________________________________
Giai đoạn 11 — Loss
Theo thiết kế của bạn, bản đầu tiên nên chia 2 mức. 
Mức 1 — chỉ classification loss
Làm trước:
L = CrossEntropy(logits, label)
Mức 2 — thêm motif consistency loss
Sau khi pipeline chạy ổn, thêm:
•	subgraph gần motif đúng 
•	xa motif sai 
Kiểu đơn giản:
L_total = L_cls + lambda_mc * L_motif
Khuyên bạn
Đừng thêm ngay từ đầu.
Hãy train được với classification loss trước.
________________________________________
Giai đoạn 12 — Training pipeline
Bạn cần một script train rõ ràng.
Luồng đầy đủ
1.	load dataset ảnh 
2.	convert ảnh → graph 
3.	generate candidate subgraphs 
4.	build motif bank từ train set 
5.	motif matching + select top-k 
6.	build motif-filtered train/val dataset 
7.	train model image-level 
Thực tế tốt nhất
Tách thành 2 pha offline + 1 pha train:
Pha A — Precompute graph cache
•	ảnh → graph 
•	lưu cache 
Pha B — Precompute motif cache
•	graph → candidate subgraphs 
•	subgraph → descriptor 
•	build motif bank 
•	select top-k 
•	lưu cache 
Pha C — Train
•	đọc motif-filtered dataset 
•	train model 
Cách này debug dễ hơn rất nhiều.
________________________________________
Giai đoạn 13 — Evaluation
Bạn cần đánh giá không chỉ accuracy.
Nên có
•	accuracy 
•	macro F1 
•	per-class F1 
•	confusion matrix 
Riêng cho motif
•	mỗi emotion hay match motif nào nhiều nhất 
•	số subgraph được chọn mỗi ảnh 
•	visualization top matched subgraphs 
Phần này rất quan trọng vì bài của bạn có tính giải thích.
________________________________________
Thứ tự code thực tế mình khuyên
Đây là thứ tự an toàn nhất:
Bước 1
Code image_to_graph.py
Bước 2
Code subgraph_generator.py
Bước 3
Code subgraph_to_descriptor
Bước 4
Code motif_bank_builder.py
Bước 5
Code motif_matching.py
Bước 6
Code dataset mới sau motif filtering
Bước 7
Code baseline model:
•	subgraph descriptor → MLP 
•	mean pooling 
•	classifier 
Bước 8
Train baseline end-to-end
Bước 9
Nâng cấp:
•	small GNN encoder 
•	attention pooling 
•	motif consistency loss 
________________________________________
Bản MVP nên chốt thế này
Để ra được kết quả đầu tiên, bạn nên làm bản tối thiểu sau:
•	Node feature: intensity + position + gradient magnitude + contrast 
•	Edge: 8-neighbor + similarity weight 
•	Candidate subgraph: radius-bounded 
•	Subgraph representation: handcrafted descriptor 
•	Motif bank: KMeans per class 
•	Matching: cosine similarity 
•	Selected top-k subgraphs 
•	Encoder: MLP trên descriptor 
•	Aggregation: mean pooling 
•	Loss: cross entropy 
Bản này chưa “đẹp nhất”, nhưng đủ để chạy, debug, ablation, rồi nâng cấp.
________________________________________
Những gì để sau
Sau khi MVP chạy ổn, mới làm tiếp:
•	thêm gx, gy đầy đủ 
•	thêm descriptor topology tốt hơn 
•	motif bank có discrimination score 
•	prototype consistency loss 
•	small GNN encoder 
•	attention pooling 
•	compatibility score tốt hơn thay vì chỉ cosine 
________________________________________
Kết luận chốt
Nếu viết thành checklist triển khai, thì các giai đoạn bạn phải làm là:
1.	Chuẩn bị data + project skeleton 
2.	Image to Graph 
3.	Candidate Subgraph Generation 
4.	Subgraph Descriptor Extraction 
5.	Motif Bank Building per Emotion 
6.	Motif Matching + Top-k Subgraph Selection 
7.	Motif-filtered Dataset Construction 
8.	Subgraph Encoder 
9.	Image-level Aggregation 
10.	Classifier 
11.	Loss + Training 
12.	Evaluation + Visualization

