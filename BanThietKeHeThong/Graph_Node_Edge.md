1. Bài toán của bạn khác graph thường ở chỗ nào
Ở đây:
	ảnh chỉ có 48x48 grayscale 
	mỗi pixel là 1 node 
	tức 2304 node / graph 
	dữ liệu FER-2013 vốn đã nhiễu, lệch sáng, lệch mặt, khác người 
	mà node lại ở mức pixel, rất thô 
Nên nếu chọn feature node và edge không chuẩn, thì:
	graph sẽ rất lớn nhưng nghèo thông tin 
	motif mining sẽ ra các pattern nhiễu 
	edge weight sẽ chỉ phản ánh “ảnh sáng tối”, chưa phản ánh “biểu cảm” 
Cho nên ở mức pixel graph, câu hỏi quan trọng không phải “có graph được không”, mà là:
mỗi node đại diện cho điều gì ngoài 1 giá trị xám đơn thuần?
và
mỗi edge đang đo loại quan hệ gì?
Đây là phần sống còn.
________________________________________
2. Node: nếu chỉ lưu intensity thì chưa đủ
Nếu node chỉ lưu:
x_i=[I_i]

với I_ilà grayscale intensity của pixel thứ i, thì đúng là đơn giản nhất, nhưng rất yếu.
Vì:
	intensity đơn lẻ rất nhạy với ánh sáng 
	cùng một biểu cảm nhưng khuôn mặt khác nhau có intensity khác 
	motif nếu dựa quá mạnh vào intensity raw sẽ dễ học ra “pattern sáng tối cục bộ”, không phải “pattern cảm xúc” 
Tức là intensity cần có, nhưng không nên là thứ duy nhất.
________________________________________
3. Node nên lưu gì trong FER-2013?
Mình chia thành 3 tầng:
Tầng A — bắt buộc có
Đây là thứ gần như nên có trong mọi thiết kế node.
(1) Intensity đã chuẩn hóa
I_i\bigm
Đây là giá trị xám của pixel, nhưng nên là:
	normalized về 01hoặc \left[-1,1\right]
	không để raw 0–255 
Lợi ích
	giữ thông tin thị giác gốc 
	là nền cho mọi quan hệ cường độ 
	motif có thể dùng distribution sáng/tối 
Ý nghĩa
Node biết “mức độ sáng” của chính nó.
________________________________________
(2) Tọa độ không gian
xiyi

hoặc dạng chuẩn hóa:
xi47yi47

Lợi ích
Cực kỳ quan trọng.
Vì nếu không có vị trí:
	pixel sáng ở miệng và pixel sáng ở trán có thể bị coi giống nhau 
	graph mất sense về hình học khuôn mặt 
Ý nghĩa
Node không chỉ biết “sáng bao nhiêu”, mà còn biết “nó nằm ở đâu trên mặt”.
Đây là điểm rất quan trọng cho FER, vì:
	mắt ở trên 
	miệng ở dưới 
	lông mày ở trên mắt 
	vị trí mang nghĩa ngữ nghĩa mạnh 
Kết luận
Nếu hỏi node tối thiểu nên có gì, thì mình chốt:
x_i=[I_i,x_i,y_i]

Đây là baseline tối thiểu hợp lý.
________________________________________
Tầng B — rất nên có
Đây là phần giúp node bớt “thô pixel”, chuyển từ raw pixel sang pixel có ngữ cảnh local.
(3) Gradient theo trục x, y
∇xIi∇yIi

Có thể tính bằng:
	Sobel 
	Scharr 
	hoặc finite difference đơn giản 
Lợi ích
Gradient cho biết:
	pixel nằm trong vùng biên hay không 
	thay đổi intensity theo hướng nào 
	cấu trúc local quanh pixel 
Trong FER, rất có giá trị vì:
	khóe miệng 
	nếp giữa lông mày 
	viền mắt 
	đường mũi
đều liên quan tới biên và thay đổi cường độ, không chỉ cường độ tuyệt đối. 
Ý nghĩa
Node biết “xung quanh tôi đang thay đổi thế nào”.
________________________________________
(4) Gradient magnitude
\mid\nabla I_i\mid=\sqrt{\left(\nabla_xI_i)^2+(\nabla_yI_i)^2\right.}\bigm
Lợi ích
Cho biết pixel có nằm ở vùng chuyển mạnh hay không:
	biên mắt 
	biên môi 
	nếp mặt 
Ý nghĩa
Node có khả năng biểu diễn “độ sắc nét cấu trúc” tại vị trí đó.
________________________________________
(5) Local contrast / local variance
Ví dụ trong cửa sổ 3x3 hoặc 5x5 quanh pixel:
	mean local 
	std local 
	contrast với hàng xóm 
Ví dụ:
c_i=I_i-\mu_{\mathcal{N}(i)}\bigm
Lợi ích
FER-2013 dễ bị ảnh hưởng bởi:
	ánh sáng tổng thể 
	ảnh tối / sáng khác nhau 
Local contrast giúp node biết:
	nó nổi bật so với lân cận không 
	thay vì chỉ nhìn giá trị tuyệt đối 
Ý nghĩa
Node biết “tôi sáng hơn hay tối hơn vùng xung quanh”.
________________________________________
Tầng C — nâng cao, có thể cân nhắc
Đây là phần tốt về mặt nghiên cứu nhưng cần tiết chế.
(6) Local texture code
Ví dụ:
	LBP-like code 
	tiny texture descriptor 
Lợi ích
Biểu cảm khuôn mặt là local texture + local deformation rất mạnh.
LBP historically rất hợp với FER.
Vấn đề
Nếu bạn gắn full LBP descriptor vào từng pixel node:
	dimension tăng 
	graph nặng 
	motif mining khó hơn 
Kết luận
Không nên nhét descriptor lớn vào node ngay từ đầu. Nếu dùng, chỉ nên dùng bản rất nhẹ.
________________________________________
(7) CNN micro-feature
Cho ảnh đi qua 1 stem CNN rất nông, rồi lấy feature map cùng kích thước hoặc gần cùng kích thước, ánh xạ lại về node.
Lợi ích
Node không chỉ là pixel nữa, mà có chút ngữ nghĩa learned feature.
Vấn đề
Hướng của thầy bạn là khá “graph mining trước”, nên nếu nhúng CNN quá mạnh từ đầu:
	triết lý motif thuần graph sẽ bị loãng 
	motif có thể thành motif của learned feature chứ không còn rõ là motif pixel-structure 
Kết luận
Đây là hướng nâng cao, chưa phải chỗ đầu tiên.
________________________________________
4. Chốt lại: node feature nào hợp nhất cho FER-2013?
Nếu cần một cấu hình thực chiến, gọn mà vẫn có ý nghĩa, mình đề xuất 3 mức:
Mức 1 — tối thiểu nhưng đúng
x_i=[I_i,x_i,y_i]

Dùng khi:
	bạn muốn pipeline rõ ràng 
	dễ giải thích 
	dễ motif mining hơn 
Mức 2 — mình đánh giá hợp lý nhất
x_i=[I_i,x_i,y_i,\nabla_xI_i,\nabla_yI_i,\mid\nabla I_i\mid,c_i]

Trong đó:
	I_i: intensity 
	x_i,y_i: position 
	\nabla_xI_i,\nabla_yI_i: edge orientation 
	\mid\nabla I_i\mid: edge strength 
	c_i: local contrast 
Đây là cấu hình mình đánh giá đẹp nhất cho FER-2013 pixel graph.
Vì nó cân bằng:
	raw appearance 
	geometry 
	local structure 
	robustness tương đối với lighting 
Mức 3 — nghiên cứu nâng cao
Thêm:
	local variance 
	tiny texture code 
	learned micro-feature 
Nhưng chỉ nên dùng sau khi baseline rõ.
________________________________________
5. Cách lưu node như thế nào?
Mỗi node nên là một vector feature.
Ví dụ với 2304 pixel:
	số node: N=2304
	mỗi node feature chiều d
Ta có ma trận:
X\in\mathbb{R}^{2304\times d}\bigm
Ví dụ nếu dùng mức 2 ở trên:
	d=7
Mỗi hàng là 1 node:
X_i=[I_i,x_i,y_i,g_i^x,g_i^y,g_i,c_i]

Ý nghĩa lưu kiểu này
	dễ dùng cho graph library 
	dễ tính node similarity 
	dễ feeding vào motif mining / graph matching / GNN sau này 
________________________________________
6. Một cảnh báo rất quan trọng về node feature
Bạn không nên nhồi quá nhiều thứ vào node ngay từ đầu.
Vì khi node feature quá giàu:
	motif mining khó hiểu hơn 
	subgraph matching tốn hơn 
	motif tìm được có thể khó giải thích 
Trong đề tài của bạn, mục tiêu không chỉ accuracy, mà còn phải:
	giải thích được motif 
	nói được motif đó đại diện cho emotion nào 
Cho nên node feature nên đủ giàu để có nghĩa, nhưng không quá học sâu / quá phức tạp.
________________________________________
7. Sang edge: edge đang biểu diễn cái gì?
Đây là phần còn quan trọng hơn node.
Với pixel graph, edge không chỉ là “có nối hay không”, mà là:
mối quan hệ giữa hai pixel đó là gì?
Có mấy loại quan hệ khả dĩ:
(A) Spatial adjacency
Hai pixel cạnh nhau trong không gian.
(B) Similarity
Hai pixel giống nhau về intensity / feature.
(C) Compatibility
Hai pixel không nhất thiết giống nhau, nhưng “hợp nhau” để tạo nên một pattern cấu trúc.
(D) Learned attention
Quan hệ được model học tự động.
________________________________________
8. Similarity và compatibility khác nhau thế nào?
Đây là chỗ rất nhiều người nhầm.
Similarity
Hỏi rằng:
hai node có giống nhau không?
Ví dụ:
	intensity gần nhau 
	gradient gần nhau 
	feature cosine cao 
Công thức kiểu:
w_{ij}^{sim}=\exp\funcapply(-\parallel x_i-x_j\parallel^2/\sigma^2)\bigm
hoặc cosine similarity.
Ý nghĩa
	hai pixel có appearance giống nhau 
	phù hợp để tạo cụm vùng đồng nhất 
Ưu điểm
	dễ tính 
	ổn định 
	dễ giải thích 
	hợp cho graph construction ban đầu 
Nhược điểm
Biểu cảm khuôn mặt không chỉ dựa vào “giống nhau”.
Ví dụ:
	biên miệng là nơi sáng–tối tương phản 
	vùng mắt và lông mày không nhất thiết “giống nhau”, nhưng kết hợp lại mới tạo fear/angry 
	motif cảm xúc thường là quan hệ bổ sung, không chỉ là similarity 
________________________________________
Compatibility
Hỏi rằng:
hai node có “phối hợp tốt” để tạo thành một pattern không?
Hai node có thể:
	khác nhau 
	nhưng đi cùng nhau hợp lý trong một cấu trúc cảm xúc 
Ví dụ:
	một pixel biên tối cạnh một pixel sáng tạo thành edge miệng 
	vùng mắt mở lớn + vùng dưới mắt + chân mày nâng lên là pattern surprise 
	tức là không giống nhau, nhưng “compatible” về cấu trúc 
Ưu điểm
	gần với bản chất emotion hơn 
	biểu diễn được quan hệ cấu trúc phức tạp 
	có giá trị hơn similarity trong motif discovery 
Nhược điểm
	khó định nghĩa thủ công 
	dễ mơ hồ 
	nếu learned hoàn toàn thì chi phí tăng 
________________________________________
9. Với FER-2013, hướng nào ổn hơn: similarity hay compatibility?
Mình chốt thế này:
Ở bước graph nền ban đầu:
Similarity ổn hơn
Vì:
	cần một graph construction cơ bản, ổn định 
	pixel-level graph đã rất lớn 
	nếu ngay từ đầu dùng compatibility learned phức tạp, bạn sẽ khó kiểm soát 
Nhưng ở mức nghiên cứu motif:
Compatibility mới là thứ đáng quan tâm hơn
Vì emotion là pattern cấu trúc:
	không phải vùng nào giống nhau là đủ 
	mà là vùng nào đi cùng nhau tạo ra biểu cảm 
Nên câu trả lời đúng không phải chọn 1 trong 2 tuyệt đối, mà là:
Thiết kế 2 tầng edge
Tầng 1: graph construction bằng spatial + similarity
Tầng 2: motif scoring / matching dùng compatibility
Đây là hướng mình đánh giá đẹp nhất.
________________________________________
10. Edge ban đầu nên lưu gì?
Mỗi edge có thể có:
	loại cạnh 
	trọng số 
	quan hệ hình học 
Ví dụ edge giữa node ivà jnên lưu:
e_{ij}=[\Delta x_{ij},\Delta y_{ij},\parallel p_i-p_j\parallel,\Delta I_{ij},s_{ij}]

Trong đó:
	\Delta x_{ij},\Delta y_{ij}: chênh vị trí 
	\parallel p_i-p_j\parallel: khoảng cách 
	\Delta I_{ij}=I_i-I_j: chênh intensity 
	s_{ij}: similarity score 
Ý nghĩa
Edge không còn chỉ là “nối”, mà còn biết:
	hai pixel lệch nhau thế nào 
	khác cường độ ra sao 
	quan hệ hình học gì 
Đây là rất tốt cho motif.
________________________________________
11. Relational weights nên định nghĩa thế nào?
Mình đề xuất 3 mức.
Mức 1 — an toàn nhất
Spatial + intensity similarity
Chỉ nối pixel hàng xóm 4-neighbor hoặc 8-neighbor, rồi gán:
w_{ij}=\exp\funcapply(-\alpha\mid I_i-I_j\mid)\bigm
hoặc với feature node đầy đủ:
w_{ij}=\exp\funcapply(-\alpha\parallel x_i-x_j\parallel^2)\bigm
Lợi ích
	dễ 
	nhẹ 
	phù hợp graph nền 
Hạn chế
	quan hệ còn nông 
________________________________________
Mức 2 — tốt hơn cho FER
Spatial + feature similarity + contrast cue
Trọng số có thể là kết hợp:
w_{ij}=\lambda_1s_{ij}^{feat}+\lambda_2s_{ij}^{grad}+\lambda_3s_{ij}^{contrast}\bigm
Ở đây:
	s^{feat}: similarity của node features 
	s^{grad}: đồng hướng / tương quan gradient 
	s^{contrast}: quan hệ sáng–tối bổ sung 
Lợi ích
	bắt đầu chạm vào compatibility nhẹ 
	giữ được local structure tốt hơn 
Mình đánh giá đây là mức đáng thử nhất.
________________________________________
Mức 3 — learned compatibility
Dùng một hàm học được:
w_{ij}=\phi(x_i,x_j,\Delta p_{ij})\bigm
với \philà:
	MLP nhỏ 
	bilinear score 
	attention score 
Lợi ích
	linh hoạt 
	có thể học được “giống không quan trọng bằng hợp” 
Vấn đề
	với 2304 node thì cực dễ nặng 
	nếu học trên quá nhiều cặp node, chi phí lớn 
	dễ khó giải thích hơn 
________________________________________
12. Neural Attention có khả quan không?
Có, nhưng phải nói đúng kiểu.
Nếu bạn nói self-attention full trên 2304 node
Thì rất nặng.
Vì attention pairwise giữa mọi cặp node là:
{2304}^2\approx5.3\mathrm{\ triệu cặp

cho 1 graph.
Đó mới chỉ là 1 layer, chưa tính batch, head, epoch.
Nên:
	full global attention ở pixel graph là không hợp lý cho giai đoạn đầu 
________________________________________
Nếu dùng local neural attention
Thì khả quan hơn nhiều.
Ví dụ:
	chỉ attention trong neighborhood 4-neighbor / 8-neighbor / radius nhỏ 
	hoặc top-k neighbor 
	hoặc attention sau khi đã qua motif filtering 
Lợi ích
	học được compatibility tốt hơn 
	không bùng nổ như full attention 
	hợp với bản chất local facial structure 
Kết luận
Neural attention không phải không khả thi, nhưng chỉ nên dùng local hoặc sparse.
________________________________________
13. Nên dùng similarity, compatibility hay neural attention?
Mình chốt rất rõ theo từng giai đoạn:
Giai đoạn 1 — xây graph gốc
Dùng:
	spatial adjacency 
	weight theo similarity nhẹ 
Vì mục tiêu là có graph ổn định, dễ kiểm soát.
Giai đoạn 2 — motif mining
Bổ sung:
	compatibility score 
	tức đánh giá subgraph không chỉ giống nhau, mà có cấu trúc phối hợp hợp lý 
Giai đoạn 3 — model train nâng cao
Khi đã có motif/subgraph nhỏ hơn, mới cân nhắc:
	local neural attention 
	graph attention trên subgraph đã được chọn 
Tức là:
Attention nên đến sau bước giảm graph, không nên đến quá sớm ở full 2304-node graph.
________________________________________
14. Nếu là mình thiết kế bản đầu tiên cho bạn
Mình sẽ chọn như sau.
Node feature
x_i=[I_i,x_i,y_i,\nabla_xI_i,\nabla_yI_i,\mid\nabla I_i\mid,c_i]

Đây là cấu hình đẹp nhất hiện tại cho bạn:
	đủ giàu 
	không quá nặng 
	có ý nghĩa rõ 
Edge topology
	8-neighbor grid 
	self-loop nếu cần cho downstream learning 
Edge weight
Ban đầu:
w_{ij}=\lambda_1\exp\funcapply(-\mid I_i-I_j\mid)+\lambda_2\cos\funcapply(g_i,g_j)\bigm
trong đó:
	phần 1 đo similarity intensity 
	phần 2 đo quan hệ gradient 
Đây chưa phải attention, nhưng đã khá hơn raw similarity.
Compatibility
Không nhét learned compatibility vào graph full ngay.
Thay vào đó, dùng compatibility ở bước:
	motif scoring 
	subgraph matching 
	prototype matching 
Attention
Chỉ dùng sau khi:
	đã chọn được subgraph theo motif 
	hoặc đã giảm node đáng kể 
________________________________________
15. Ý nghĩa học thuật của thiết kế này
Thiết kế trên có một narrative rất mạnh:
	Intensity giữ thông tin thị giác gốc 
	Position giữ cấu trúc hình học khuôn mặt 
	Gradient và contrast giúp node mang thông tin biến đổi cục bộ, vốn quan trọng cho biểu cảm 
	Edge similarity giúp xây dựng liên kết ổn định ban đầu 
	Compatibility được dùng ở tầng motif để phát hiện các cấu trúc con phối hợp tạo thành cảm xúc 
	Attention chỉ áp dụng ở tầng sau khi graph đã được rút gọn, nhằm tránh chi phí tính toán quá lớn trên đồ thị 2304 nút 
Đây là cách giải thích rất sạch cho báo cáo.
________________________________________
16. Kết luận chốt
Với FER-2013 pixel graph:
Node không nên chỉ lưu intensity
Nên lưu tối thiểu:
	intensity 
	tọa độ 
Và tốt nhất nên thêm:
	gradient x/y 
	gradient magnitude 
	local contrast 
Edge ban đầu nên ưu tiên similarity hơn compatibility
Vì:
	ổn định 
	nhẹ 
	dễ kiểm soát 
Nhưng ở mức motif, compatibility quan trọng hơn
Vì cảm xúc là pattern phối hợp, không chỉ là giống nhau.
Neural attention có khả quan, nhưng không nên áp lên full 2304-node graph
Hợp lý nhất là:
	local attention 
	hoặc attention sau motif filtering / subgraph reduction

