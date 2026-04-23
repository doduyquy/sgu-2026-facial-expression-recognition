1. Vấn đề 3 thực chất là gì?
Sau Vấn đề 2, bạn đã có:
	một pixel graph cho mỗi ảnh, 
	một motif bank cho từng emotion, 
	và cơ chế motif-guided subgraph selection. 
Lúc này câu hỏi không còn là “graph hóa thế nào” hay “motif là gì”, mà là:
biểu diễn cuối cùng của ảnh sẽ được xây từ những subgraph đã match motif như thế nào, và mô hình sẽ học trên biểu diễn đó ra sao?
Tức là Vấn đề 3 là bài toán learning on motif-filtered substructures.
Điểm rất quan trọng là: với FER-2013, nhãn nằm ở mức ảnh, không phải mức subgraph. FER-2013 cũng có các khó khăn riêng như ảnh grayscale 48×48, dữ liệu nhiễu, class imbalance, và label noise; các khảo sát và phân tích gần đây đều xem low resolution, class imbalance, lighting variation và noisy labels là những thách thức cốt lõi của FER/FER-2013. 
Vì vậy, nếu ở giai đoạn 6 bạn chuyển sang “mỗi subgraph là một sample độc lập mang nhãn emotion”, thì về mặt học thuật nó sẽ yếu hơn, vì bạn đang ép nhãn ảnh xuống subgraph quá cứng.
________________________________________
2. Các hướng mà literature hiện nay đang làm cho kiểu bài toán này
Mình chia thành 4 hướng chính.
Hướng A — Subgraph features / motif-score rồi classification
Đây là hướng cổ điển và rất vững về logic: tìm ra các discriminative subgraphs, sau đó biểu diễn mỗi graph bằng mức xuất hiện hoặc mức match với các subgraphs đó, rồi mới phân loại. DS-Span 2025 đi theo tư tưởng này khi xem các subgraphs được chọn như một efficient, interpretable basis cho downstream graph embedding và classification. 
Ưu điểm
	rất hợp với luận điểm “motif là nền biểu diễn mới”, 
	dễ giải thích, 
	rất sạch về mặt học thuật. 
Nhược điểm
	dễ bị hơi “symbolic”, 
	có thể bỏ mất thông tin tinh hơn trong tương tác giữa nhiều subgraphs. 
Mức hợp với FER-2013
	hợp để làm baseline motif-learning đầu tiên, 
	đặc biệt tốt nếu bạn muốn chứng minh rằng motif bank thực sự có giá trị phân biệt. 
________________________________________
Hướng B — Motif-guided key subgraph selection rồi image-level learning
MOSGSL 2024 cho graph classification nhấn mạnh hai thứ: key subgraph selection và motif-driven structure guidance. Ý tưởng ở đây là mô hình không học từ toàn bộ graph một cách mù, mà học từ những subgraphs quan trọng đã được motif dẫn hướng. 
Ưu điểm
	cực sát với hướng của thầy bạn, 
	giữ được motif như một cơ chế chọn lọc cấu trúc, 
	vẫn học ở mức graph/image-level, rất hợp với FER. 
Nhược điểm
	phức tạp hơn A, 
	cần thiết kế thêm bước encode và aggregate các subgraphs. 
Mức hợp với FER-2013
	theo mình đây là hướng hợp nhất với kế hoạch của bạn, 
	vì nó không phá tinh thần Vấn đề 1 và 2. 
________________________________________
Hướng C — Pattern/prototype-based graph representation
PXGL-GNN 2025 xem representation của graph là tổ hợp có trọng số của nhiều graph patterns/substructures. Còn các hướng prototype-based self-explainable GNN trong 2025 cũng đi theo tinh thần dùng prototype structures để vừa dự đoán vừa giải thích. 
Ưu điểm
	rất đẹp về mặt nghiên cứu, 
	kết nối tốt giữa motif, representation và explanation, 
	hợp với ý tưởng “mỗi emotion có một motif bank như prototype bank”. 
Nhược điểm
	dễ nặng nếu làm ngay, 
	đòi hỏi không gian embedding subgraph/motif rõ ràng. 
Mức hợp với FER-2013
	rất hợp để làm phiên bản nâng cao sau baseline. 
________________________________________
Hướng D — Class-level sparse subgraphs / class-level explanations
GraphOracle 2025 nhấn mạnh việc học structured, sparse subgraphs discriminative for each class cùng với classifier. Điều này rất gần khái niệm “motif cho từng emotion” của bạn. 
Ưu điểm
	cho một narrative rất mạnh: 
	happy có các subgraphs điển hình, 
	sad có các subgraphs điển hình, 
	vừa dự đoán vừa giải thích. 
Nhược điểm
	khó hơn hướng A, 
	cần cân đối giữa sparsity, discrimination và classification. 
Mức hợp với FER-2013
	hợp về mặt ý tưởng, 
	nhưng nếu triển khai ngay sẽ nặng. 
________________________________________
3. Đánh giá hướng nào hợp nhất với FER-2013
Đây là phần quan trọng nhất: không phải cái gì hay trong graph learning cũng hợp với FER-2013.
3.1. Những ràng buộc của FER-2013
FER-2013 có:
	ảnh 48×48 grayscale, 
	dữ liệu nhiễu, 
	class imbalance, 
	label noise, 
	và mỗi ảnh chỉ là static expression chứ không có chuỗi thời gian. 
Ngoài ra, survey graph-FER 2024 cho thấy phần lớn graph-based FER hiện nay dùng landmark, region, temporal graph, hoặc multi-stream graph, chứ không trực tiếp train end-to-end trên full pixel graph. 
Điều đó dẫn đến 3 hệ quả:
Hệ quả 1
Bạn không nên chọn full end-to-end GAT/GNN trên toàn bộ 2304-node graph sau khi đã mất công xây motif. Làm như vậy sẽ phá tinh thần Vấn đề 2.
Hệ quả 2
Bạn cũng không nên gán nhãn emotion cứng cho từng subgraph rồi train subgraph classifier như bài toán chính. Điều đó không thật sự khớp với dữ liệu.
Hệ quả 3
Hướng hợp lý nhất phải là:
	subgraph là đơn vị trung gian, 
	ảnh vẫn là đơn vị dự đoán cuối cùng. 
Đó chính là logic của image-level learning from motif-selected subgraphs.
________________________________________
4. Chốt định hướng cho Vấn đề 3
Nếu ghép Vấn đề 1 + 2 + 3 lại cho nhất quán, mình chốt:
Vấn đề 3 nên đi theo hướng motif-guided image-level learning: dùng motif bank để chọn ra các subgraphs giàu thông tin cảm xúc, rồi học biểu diễn của toàn ảnh từ tập subgraphs đó, thay vì từ full pixel graph.
Đây là hướng hợp nhất vì:
	giữ nguyên graph nền từ Vấn đề 1, 
	tận dụng motif bank từ Vấn đề 2, 
	phù hợp với nhãn ảnh của FER-2013, 
	tránh train trực tiếp trên graph 2304 node, 
	vẫn cho ra một representation giải thích được. 
________________________________________
5. Thiết kế cụ thể cho Vấn đề 3
Mình đề xuất hướng triển khai như sau.
5.1. Input của giai đoạn 3 là gì?
Từ Vấn đề 2, với mỗi ảnh I, bạn đã có:
	graph G, 
	tập candidate subgraphs \mathcal{S}(G), 
	motif bank theo emotion \mathcal{M}_{angry},\ldots,\mathcal{M}_{neutral}, 
	và điểm match giữa từng candidate subgraph với motif bank. 
Từ đó bạn chọn ra top-k subgraphs:
\mathcal{S}^\ast(G)={S_1^\ast,S_2^\ast,\ldots,S_k^\ast}\bigm
Đây mới là đầu vào thực sự của giai đoạn train.
________________________________________
5.2. 3 kiểu biểu diễn có thể dùng
Kiểu 1 — Motif-score vector
Mỗi ảnh được biến thành vector:
	match với happy motifs bao nhiêu, 
	match với sad motifs bao nhiêu, 
	... 
	match với neutral motifs bao nhiêu. 
Tức là:
z(G)=[s_1,\ldots,s_m]

Khi nào nên dùng
	làm baseline, 
	kiểm chứng motif bank có ích hay không, 
	muốn một pipeline rất giải thích được. 
Nhược điểm
	hơi thô, 
	khó tận dụng quan hệ giữa các subgraphs. 
________________________________________
Kiểu 2 — Bag of matched subgraphs
Mỗi ảnh được biểu diễn bởi tập \mathcal{S}^\ast(G).
Mỗi subgraph được encode thành một embedding:
h_i=f(S_i^\ast)\bigm
Sau đó aggregate:
h_G=\mathrm{Pool}(h_1,\ldots,h_k)\bigm
rồi classifier:
\hat{y}=g(h_G)\bigm
Đây là hướng mình khuyên chọn.
Vì:
	nhãn vẫn ở mức ảnh, 
	subgraphs là đơn vị trung gian hợp lý, 
	rất sát với MOSGSL-style thinking về key subgraph selection. 
________________________________________
Kiểu 3 — Prototype alignment
Mỗi subgraph vừa được encode, vừa được so gần/xa với motif prototypes của các emotion, rồi ảnh được dự đoán dựa trên toàn bộ pattern alignment.
Đây là hướng đẹp nhất về research, nhưng nên xem là bản nâng cao.
________________________________________
6. Subgraph encoder nên là gì?
Vì bạn đang ở FER-2013, ảnh nhỏ, motif đã giúp giảm graph, nên subgraph encoder không cần quá lớn.
Có 3 lựa chọn:
A. MLP trên descriptor của subgraph
Nếu mỗi subgraph đã có vector descriptor tốt, chỉ cần MLP nhỏ.
Ưu
	nhẹ, 
	dễ giải thích, 
	hợp baseline. 
Nhược
	không khai thác hết cấu trúc graph bên trong subgraph. 
________________________________________
B. Small GNN trên subgraph
Mỗi subgraph vốn là graph nhỏ, nên dùng 1–2 layer GCN/GAT/GraphSAGE để encode.
Ưu
	khai thác topology thật sự, 
	rất hợp logic bài của bạn. 
Nhược
	phức tạp hơn MLP. 
Mình đánh giá đây là lựa chọn hợp lý nhất.
Lý do:
	full graph 2304 node thì nặng, 
	nhưng subgraph đã được motif-filtered thì nhỏ hơn rất nhiều, 
	lúc này mới là thời điểm hợp để dùng graph encoder. 
________________________________________
C. Hybrid descriptor + small GNN
Nối:
	handcrafted subgraph descriptors, 
	với learned embedding từ small GNN. 
Đây là bản mạnh hơn, nhưng nên để sau.
________________________________________
7. Readout / aggregation ở mức ảnh nên làm thế nào?
Đây là phần rất quan trọng của Vấn đề 3.
Survey graph classification gần đây và các hướng subgraph readout mới như SUBRead 2026 cho thấy việc aggregate subgraph representations với trọng số/attention có thể cải thiện graph classification. SUBRead đề xuất attention-based weighting để gộp subgraph representations thay vì readout đồ thị kiểu thô. 
Với bài của bạn, có 3 kiểu:
7.1. Mean pooling
h_G=\frac{1}{k}\sum_{i} h_i\bigm
Đơn giản nhưng hơi mờ.
7.2. Max pooling
Giữ tín hiệu mạnh nhất, nhưng dễ mất tính ổn định.
7.3. Attention pooling
h_G=\sum_{i}\alpha_ih_i\bigm
với \alpha_ihọc được.
Đây là hướng mình khuyên dùng.
Vì:
	không phải mọi subgraph match motif đều quan trọng như nhau, 
	attention ở đây áp trên số ít subgraphs đã lọc, không phải full 2304 nodes, 
	nên hợp lý hơn rất nhiều so với self-attention trên full pixel graph. Điều này cũng khớp với file vấn đề 1 của bạn: attention nên đến sau bước giảm graph, không phải trước. 
________________________________________
8. Loss của giai đoạn 3 nên thiết kế thế nào?
Đây là chỗ cần chốt sạch.
8.1. Classification loss
Chuẩn:
\mathcal{L}_{cls}\bigm
dự đoán đúng 7 emotion.
Vì FER-2013 có imbalance và noisy labels, việc dùng class balancing / noise-aware strategies vẫn đáng cân nhắc trong loss chính. Các phân tích/dataset papers gần đây về FER13 và noisy FER cũng nhấn mạnh các vấn đề này. 
8.2. Motif-consistency loss
Subgraph lấy từ ảnh lớp cnên:
	gần motif bank của c, 
	xa motif banks của lớp khác. 
Tinh thần:
\mathcal{L}_{motif}\bigm
là một contrastive / prototype-alignment loss.
Đây là phần làm cho motif thực sự tham gia vào học, thay vì chỉ tham gia chọn lọc ở bước trước.
8.3. Sparsity / diversity regularization
Hai regularization nên nghĩ tới:
	sparsity: không giữ quá nhiều subgraphs; 
	diversity: các subgraphs được chọn hoặc các motifs trong bank không nên quá trùng nhau. 
Điều này gần với tinh thần class-level sparse/discriminative structures của GraphOracle. 
Tổng loss
\mathcal{L}=\mathcal{L}_{cls}+\lambda_1\mathcal{L}_{motif}+\lambda_2\mathcal{L}_{reg}\bigm
Đây là form đẹp nhất cho Vấn đề 3.
________________________________________
9. Hướng nào không nên chọn cho bài của bạn?
Để chốt rõ hơn, mình nói thẳng những hướng không nên:
Không nên 1
Train trực tiếp full graph 2304 node bằng GCN/GAT rồi bỏ qua motif.
Làm vậy thì toàn bộ công sức Vấn đề 2 mất ý nghĩa.
Không nên 2
Gán nhãn ảnh cho mọi subgraph rồi coi subgraph là sample chính.
Điều đó quá cứng, không hợp với nature của FER-2013.
Không nên 3
Exact subgraph matching cứng trong toàn bộ pipeline train.
FER-2013 quá nhiễu, exact matching sẽ giòn và khó khái quát.
Không nên 4
Global attention trên toàn bộ pixel graph.
Quá nặng và không cần thiết; literature và chính file của bạn đều đang hướng tới “reduce first, attend later”. 
________________________________________
10. Hướng phù hợp nhất cho Vấn đề 3
Mình chốt một dòng duy nhất:
Motif-guided bag-of-subgraphs image-level classification with prototype-consistency learning
Nghĩa là:
	ảnh → graph, 
	graph → candidate subgraphs, 
	subgraphs → match với motif banks, 
	chọn top-k subgraphs, 
	encode từng subgraph bằng small GNN, 
	attention pooling ở mức ảnh, 
	classifier dự đoán emotion, 
	thêm motif-consistency loss để giữ logic motif. 
Đây là hướng hợp nhất với:
	Vấn đề 1: graph gốc được thiết kế rõ, 
	Vấn đề 2: motif là discriminative prototype subgraphs, 
	Vấn đề 3: học ở mức ảnh nhưng dựa trên subgraphs đã được motif dẫn hướng. 

