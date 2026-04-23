1. Phát biểu lại bài toán motif cho đúng
Trong bài của bạn, motif không nên được hiểu đơn giản là “một subgraph xuất hiện nhiều lần”. Với FER-2013, nếu định nghĩa như vậy thì rất dễ thu được các pattern sáng-tối cục bộ, pattern nhiễu, hoặc pattern phổ thông của khuôn mặt, chứ chưa chắc là pattern biểu cảm. Điều này cũng phù hợp với xu hướng chung trong graph-FER hiện nay: đa số công trình chọn landmark graph hoặc region graph để tăng tính ngữ nghĩa và giảm complexity, thay vì đi thẳng vào full pixel graph. Survey 2024 về graph deep representation learning cho FER, cùng các công trình gần đây như Exp-Graph và GLaRE, đều cho thấy cộng đồng đang thiên về graph hóa ở mức landmark/region thay vì pixel-level. 
Vì vậy, nếu bạn vẫn đi theo hướng “1 pixel = 1 node”, thì motif của bạn phải được định nghĩa chặt hơn:
Motif là một class-discriminative prototype subgraph: một subgraph cục bộ vừa đủ phổ biến trong một emotion, vừa đủ khác với các emotion còn lại, và được so khớp theo kiểu mềm như một prototype, chứ không phải exact subgraph template cứng.
Đây là hướng phù hợp với các ý tưởng mới trong graph mining/classification: discriminative subgraphs như basis cho embedding và classification trong DS-Span, motif-guided key subgraph selection trong MOSGSL, pattern-based representation trong PXGL-GNN, và class-level discriminative sparse subgraphs trong GraphOracle. 
2. Literature hiện nay đã làm motif/subgraph theo những hướng nào
Có ba dòng ý tưởng mà bạn nên lấy làm nền.
2.1. Discriminative subgraph mining
DS-Span 2025 không chỉ tìm subgraph lặp lại, mà nhấn mạnh việc chọn các subgraphs có class-separating ability, dùng information-gain-guided selection và coverage-capped exploration để giảm redundancy. Điểm hay của dòng này là xem tập subgraph thu được như một interpretable basis cho graph embedding và classification. Đây rất gần với nhu cầu của bạn: motif không chỉ để giải thích, mà còn để tạo biểu diễn mới cho ảnh/graph. 
2.2. Motif-guided key subgraph selection
MOSGSL 2024 đi theo tư tưởng: graph-level tasks cần chọn important subgraphs, và motif có thể đóng vai trò “structure guidance” để dẫn hướng quá trình đó. Nói theo ngôn ngữ bài của bạn: motif không chỉ là thứ được khai quật ra sau cùng, mà là một ngân hàng mẫu cấu trúc dùng để nói phần nào của 2304-node graph là đáng giữ lại. Điều này cực sát với ý bạn: “chọn Subgraphs, sẽ chọn những match với motif”. 
2.3. Pattern-based graph representation
PXGL-GNN 2025 xem graph representation như tổ hợp có trọng số của nhiều graph patterns/substructures đã được lấy mẫu. Đây là một bước nối rất đẹp từ motif sang representation learning: thay vì full graph, ảnh/graph có thể được biểu diễn bởi một tập patterns quan trọng và trọng số đóng góp của chúng. Nếu áp vào FER-2013 của bạn, điều này gợi ý rằng biểu diễn cuối cùng của ảnh nên được xây từ bag of matched motifs/subgraphs, không phải từ toàn bộ 2304 nodes. 
2.4. Class-level sparse subgraphs / prototypes
GraphOracle 2025 cho thấy một hướng rất đáng chú ý: jointly học classifier cùng với một tập structured, sparse, class-discriminative subgraphs cho từng class. Dù bài này không phải FER, nó rất hợp tư duy “motif cho từng emotion”: happy có bộ subgraphs riêng, sad có bộ riêng, surprise có bộ riêng. Đây chính là cách biến “motif của emotion” thành một đối tượng nghiên cứu rõ ràng. 
3. Hướng nào phù hợp với FER-2013 nhất
Nếu xét riêng cho FER-2013 và ràng buộc bạn đã chọn là pixel-level graph, thì mình chốt như sau:
Không phù hợp nhất
Frequent motif thuần túy, exact matching cứng, hoặc subgraph mining tự do trên toàn bộ 2304-node graph là không hợp. Search space quá lớn, exact matching quá giòn, và pixel graph quá dễ bị nhiễu ảnh chi phối. Trong khi đó, literature graph-FER hiện hành lại chủ yếu tránh vùng khó này bằng cách dùng landmark/region graph hoặc hierarchical coarsening. 
Phù hợp nhất
Một hướng lai giữa:
	discriminative subgraphs của DS-Span, 
	motif-guided selection của MOSGSL, 
	pattern-based representation của PXGL-GNN, 
	class-specific sparse subgraphs của GraphOracle 
là hợp nhất cho bài của bạn. 
Nói gọn lại, hướng hợp nhất cho FER-2013 pixel graph là:
emotion-specific discriminative prototype motifs + motif-guided subgraph selection + image-level learning
4. Motif của FER-2013 nên chứa những gì
Dựa trên file node/edge của bạn, motif không thể chỉ là topology. Nó nên đồng thời mang ba tầng nội dung.
4.1. Cấu trúc cục bộ
Motif phải là một subgraph nhỏ, giới hạn về kích thước, đủ cục bộ để không bùng nổ chi phí và đủ giàu để chứa một pattern biểu cảm nhỏ. Với FER-2013, motif hợp lý nên nằm ở cỡ:
	9 đến 25 node nếu muốn rất local, 
	25 đến 49 node nếu muốn giàu cấu trúc hơn. 
Lý do không nên lớn hơn sớm là vì graph gốc đã 2304 node; motif quá lớn thì vừa khó mining vừa khó giải thích.
4.2. Appearance cục bộ
Motif phải mang:
	intensity pattern, 
	gradient pattern, 
	contrast pattern, 
	và gián tiếp là local shape cue. 
Nếu không, motif sẽ chỉ là “một cụm node nối nhau”, chưa đủ để đại diện cho biểu cảm.
4.3. Compatibility
File của bạn đã chốt rất đúng: edge nền ưu tiên similarity, còn compatibility mới là thứ đáng dùng ở tầng motif. Điều đó có nghĩa là khi so khớp một candidate subgraph với motif, không nên chỉ hỏi “chúng có giống nhau không”, mà nên hỏi “chúng có phối hợp với nhau thành cùng một cấu trúc biểu cảm không”. Đây là một bước nâng rất quan trọng về mặt nghiên cứu. 
Vì vậy, định nghĩa chặt của motif trong bài này là:
Một motif là một subgraph cục bộ có topology bị chặn, mang appearance cục bộ và được đánh giá theo mức compatibility cấu trúc, sao cho nó vừa phổ biến trong một emotion vừa phân biệt được emotion đó với các lớp khác.
5. Thiết kế hướng triển khai cụ thể
Mình đề xuất cho bạn một hướng hoàn chỉnh có thể gọi là:
Emotion-Specific Discriminative Prototype Motif Learning on Pixel Graphs
Giai đoạn A — Xây pixel interaction graph
Đây là phần bạn đã giải quyết trong file, nên mình dùng nó làm input:
	mỗi ảnh 48×48 → 2304-node graph, 
	node features gồm intensity, position, gradient, contrast theo mức bạn đã chốt, 
	edge nền là 8-neighbor grid với trọng số similarity ổn định. 
Kết quả là mỗi ảnh thành một graph G=(V,E,X).
Giai đoạn B — Sinh candidate subgraphs
Không mine motif trên toàn bộ không gian subgraph tự do. Thay vào đó, với mỗi graph G, sinh một tập ứng viên \mathcal{S}(G)={S_1,\ldots,S_n}theo một generator có ràng buộc.
Cách phù hợp nhất ở đây là local bounded subgraphs:
	lấy các subgraphs trong bán kính nhỏ quanh một node seed, 
	hoặc sliding local neighborhoods, 
	hoặc random-walk subgraphs ngắn. 
Nếu muốn chỉnh chu mà vẫn giữ logic motif, mình khuyên ưu tiên:
	radius-bounded local subgraphs trên graph, 
	hơn là window ảnh thuần túy. 
Lý do là như vậy bạn vẫn giữ đúng ngôn ngữ “subgraph”, không bị trôi về patch CNN thông thường.
Mỗi candidate subgraph nên bị chặn bởi:
	số node tối đa m, 
	bán kính tối đa r, 
	số lượng ứng viên mỗi ảnh n,
để search space không nổ tung. 
Giai đoạn C — Biểu diễn subgraph
Mỗi candidate subgraph S_icần được biểu diễn thành một vector mô tả để:
	so sánh với motif prototypes, 
	gom nhóm, 
	chấm điểm phân biệt. 
Representation của subgraph nên gồm ba phần:
	node statistics: mean/std của intensity, gradient, contrast; 
	edge statistics: phân bố similarity, degree, gradient alignment; 
	shape/topology descriptors: số node, số edge, density, local centrality hoặc spectrum rất nhẹ. 
Không nhất thiết phải quá nặng. Mục tiêu ở đây là biến subgraph thành một đối tượng có thể so khớp mềm.
Giai đoạn D — Xây motif bank cho từng emotion
Với mỗi emotion c\in{\mathrm{angry,\ disgust,\ fear,\ happy,\ sad,\ surprise,\ neutral}}, gom toàn bộ candidate subgraphs từ các ảnh thuộc lớp c. Từ đó xây motif bank \mathcal{M}_c={M_1^c,\ldots,M_K^c}.
Mỗi motif prototype được chọn theo ba tiêu chí:
1. Intra-class consistency
Subgraph kiểu này có xuất hiện đủ thường xuyên trong emotion ckhông?
2. Inter-class discrimination
Nó có hiếm hơn đáng kể ở các lớp khác không?
3. Compactness / interpretability
Nó có đủ gọn để giải thích không, hay chỉ là một mảng cồng kềnh và nhiễu?
Tức là score của một motif ứng viên Mcho emotion cnên mang dạng tinh thần như:
\mathrm{MotifScore}(M,c)=\alpha\cdot\mathrm{IntraFreq}(M,c)-\beta\cdot\mathrm{InterFreq}(M,\lnot c)-\gamma\cdot\mathrm{Redundancy}(M)\bigm
Đây không phải công thức paper-ready cuối cùng, nhưng là khung tư duy rất đúng: phổ biến trong lớp, hiếm ngoài lớp, ít dư thừa.
Giai đoạn E — Motif matching cho từng ảnh
Với một graph ảnh G:
	sinh candidate subgraphs \mathcal{S}(G), 
	với mỗi S_i, tính độ tương hợp với các motif banks, 
	giữ lại top-k subgraphs có match tốt nhất. 
Độ tương hợp không nên là exact isomorphism, mà nên là soft compatibility score:
\mathrm{Match}(S_i,M_j^c)=\lambda_1\cdot\mathrm{TopoSim}+\lambda_2\cdot\mathrm{FeatSim}+\lambda_3\cdot\mathrm{CompatSim}\bigm
Trong đó:
	TopoSim đo mức giống về cấu trúc, 
	FeatSim đo mức giống về intensity/gradient/contrast descriptors, 
	CompatSim đo mức “phối hợp biểu cảm” theo relational cues. 
Đây chính là chỗ bạn biến file vấn đề 1 thành động lực cho vấn đề 2: similarity ở graph nền, compatibility ở motif matching. 
Giai đoạn F — Tạo biểu diễn ảnh
Có hai đường.
Đường 1: motif-score vector
Mỗi ảnh được biểu diễn thành vector:
z(G)=[s(G,\mathcal{M}_{angry}),\ldots,s(G,\mathcal{M}_{neutral})]

hoặc chi tiết hơn là score với từng motif trong mỗi bank.
Đây là cách dễ giải thích nhất.
Đường 2: bag of matched subgraphs
Giữ lại top-k subgraphs match tốt nhất, encode từng cái, rồi pooling/attention để ra image-level embedding.
Đây là cách đẹp hơn về học thuật vì ảnh vẫn là đơn vị dự đoán cuối cùng, và mô hình học từ những subgraphs đã được motif xác nhận.
Nếu bạn muốn làm chỉnh chu, mình nghiêng về đường 2.
6. Giai đoạn train nên thiết kế thế nào
Bạn nói trước đó giai đoạn 6 còn mơ hồ, nên mình chốt giúp luôn.
Mục tiêu
Không train trên full 2304-node graph nữa, mà train trên:
	subgraphs đã match motif, hoặc 
	biểu diễn ảnh được tạo từ chúng. 
Kiểu học nên dùng
Đây là hướng phù hợp nhất:
image-level classification with motif-guided subgraph selection
Tức là:
	subgraph là đơn vị trung gian, 
	nhãn vẫn ở mức ảnh, 
	motif là bộ lọc và bộ chuẩn hóa cấu trúc. 
Loss nên gồm gì
1. Classification loss
Ảnh dự đoán đúng emotion.
2. Motif consistency loss
Subgraphs từ ảnh lớp cnên gần motif bank \mathcal{M}_c, và xa motif banks của lớp khác.
Tư duy này học từ class-discriminative subgraphs của DS-Span và GraphOracle, nhưng chuyển sang bài toán FER pixel-graph của bạn. 
3. Diversity / sparsity regularization
Trong cùng một motif bank, các motifs không nên trùng lặp quá mức; quá nhiều motif gần nhau sẽ làm giảm tính giải thích.
Vì sao không nên train subgraph-level label trực tiếp
Có thể làm, nhưng không đẹp bằng image-level learning. Lý do là nhãn gốc là của toàn ảnh; không có gì đảm bảo mọi subgraph đều tự mang đầy đủ thông tin cảm xúc. Literature kiểu MOSGSL cũng nghiêng về graph-level tasks với key subgraph selection, chứ không chuyển nhãn cứng xuống mọi subgraph. 
7. Hướng triển khai này “học” gì từ các paper, và “mới” ở đâu
Nó học từ literature ở bốn điểm:
	Từ DS-Span: motif phải có tính phân biệt, không chỉ tần suất. 
	Từ MOSGSL: motif nên dùng để dẫn hướng chọn subgraph quan trọng. 
	Từ PXGL-GNN: biểu diễn graph/ảnh có thể xây từ patterns/substructures. 
	Từ GraphOracle: motif của từng emotion có thể được nhìn như class-level sparse/discriminative subgraphs. 
Điểm riêng của bài bạn là:
	không dùng landmark/region graph như đa số graph-FER hiện tại, 
	mà giữ pixel-level graph, 
	rồi dùng motif như cơ chế “nâng ngữ nghĩa” và “giảm đồ thị” trước khi học. 
Đó là điểm khác biệt rõ ràng với Exp-Graph, GLaRE và xu hướng graph-FER hiện nay.

