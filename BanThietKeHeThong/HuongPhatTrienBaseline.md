Baseline nên chọn lúc này
Baseline 1 — Graph-level cực đơn giản, dễ chạy nhất
Không dùng GNN ngay.
Làm như sau:
1.	lấy node_features của graph 
2.	pooling toàn bộ node: 
o	mean pooling 
o	std pooling 
o	max pooling 
3.	ghép thành vector graph-level 
4.	đưa qua MLP classifier 
Tức là:
PixelGraph -> global node statistics -> MLP -> 7 emotion
Vì sao nên làm cái này trước
Vì nó giúp bạn kiểm tra:
•	graph cache có đúng không 
•	node feature có tín hiệu không 
•	label/split/train loop có đúng không 
Đây là baseline “sanity check” rất quan trọng.
________________________________________
Baseline 2 — GNN nhẹ trên full graph
Sau baseline 1, bạn làm tiếp:
PixelGraph -> 1-2 layer GCN/GraphSAGE -> graph pooling -> MLP -> 7 emotion
Vì sao chưa nên dùng GAT
Vì:
•	graph đã lớn 
•	edge đã nhiều 
•	GAT nặng hơn 
•	debug khó hơn 
Khuyên:
•	GraphSAGE hoặc GCN trước 
•	2 layer là đủ 
________________________________________
Mình khuyên thứ tự thế này
Bước kế tiếp ngay bây giờ
Làm Baseline 1: Node-statistics MLP
Sau khi chạy được, mới sang Baseline 2: GNN nhẹ
________________________________________
Bạn phải làm gì để có Baseline 1
Pipeline:
train_graphs.pt / val_graphs.pt / test_graphs.pt
-> GraphDataset
-> graph vectorizer
-> MLP classifier
-> train/eval
________________________________________
Giai đoạn 3A — Xây baseline graph vector
Bạn cần một hàm biến mỗi graph thành 1 vector cố định.
Cách đơn giản nhất
Từ node_features [N, d], lấy:
•	mean theo node 
•	std theo node 
•	max theo node 
Nếu d = 3, thì:
•	mean: 3 chiều 
•	std: 3 chiều 
•	max: 3 chiều 
Tổng cộng:
•	3 + 3 + 3 = 9 chiều 
Tức là:
graph_vec = concat(
    mean(node_features, axis=0),
    std(node_features, axis=0),
    max(node_features, axis=0)
)
Ý nghĩa
Bạn đang hỏi:
•	toàn graph sáng tối trung bình thế nào 
•	phân bố vị trí feature ra sao 
•	giá trị cực đại thế nào 
Nó thô, nhưng đủ để kiểm tra pipeline.
________________________________________
Giai đoạn 3B — Dataset cho baseline
Bạn cần tạo dataset đọc từ graph cache.
Mỗi sample trả:
•	graph_vector 
•	label 
Ví dụ:
{
    "x": graph_vec,   # shape [9]
    "y": label
}
________________________________________
Giai đoạn 3C — Model baseline
MLP rất đơn giản:
input_dim -> hidden -> hidden -> 7
Ví dụ:
•	input = 9 
•	hidden = 64 
•	hidden = 32 
•	output = 7 
Activation:
•	ReLU 
Loss:
•	CrossEntropyLoss 
Optimizer:
•	Adam 
________________________________________
Giai đoạn 3D — Training baseline
Train chuẩn:
•	train set 
•	val set 
•	chọn best val acc hoặc macro F1 
•	test cuối cùng 
Metric nên có:
•	accuracy 
•	macro F1 
•	confusion matrix 
________________________________________
Sau khi baseline 1 chạy xong, bạn làm gì tiếp
Baseline 2 — GCN/GraphSAGE
Lúc này bạn mới kiểm tra graph learning thật sự.
Pipeline:
PixelGraph
-> GCN/GraphSAGE (2 layer)
-> global mean pooling
-> MLP
-> 7 lớp
Input dùng gì
•	node_features 
•	edge_index 
•	có thể bỏ qua edge_attr ở baseline GCN đầu tiên 
Vì sao có thể bỏ edge_attr tạm thời
Vì baseline đầu tiên chỉ cần kiểm tra:
•	topology có ích không 
•	GNN train được không 
Sau đó mới nâng cấp sang edge-aware model.
________________________________________
Thứ tự kỹ thuật mình khuyên
Ngay bây giờ làm 4 file này
1. data/graph_cache_dataset.py
Đọc train_graphs.pt, val_graphs.pt, test_graphs.pt
2. features/graph_vectorizer.py
Biến PixelGraph thành vector graph-level
3. models/mlp_baseline.py
MLP classifier
4. scripts/train_graph_baseline.py
Train/eval baseline 1
________________________________________
Baseline 1 cụ thể nên chốt thế nào
Input
Từ mỗi graph:
•	node_features.shape = [2304, 3] 
Vectorizer
Dùng:
•	mean 
•	std 
•	max 
Model
MLP:
•	9 -> 64 -> 32 -> 7 
Loss
CrossEntropy
Optimizer
Adam, lr = 1e-3
Epoch
20–30 epoch là đủ test pipeline
Batch size
64 hoặc 128 vì input rất nhẹ
________________________________________
Vì sao đây là bước đúng
Vì nó trả lời ngay 3 câu hỏi:
Câu 1
Graph cache bạn build có dùng được để train không?
Câu 2
Ba feature intensity, x_norm, y_norm có mang tín hiệu phân loại gì không?
Câu 3
Toàn bộ train/val/test pipeline có chạy sạch không?
Nếu baseline 1 còn chưa chạy ổn, thì triển motif sau sẽ rất mệt.
________________________________________
Sau Baseline 1, lộ trình đúng là
Baseline 2
Full graph GCN/GraphSAGE
Baseline 3
Candidate subgraph extraction + subgraph descriptor baseline
Baseline 4
Motif bank + top-k motif-matched subgraphs
Baseline 5
Bag-of-subgraphs classifier
Tức là đi từ dễ → khó, chứ không nhảy cóc.

