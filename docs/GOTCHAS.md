NHỮNG RỦI RO KỸ THUẬT & CÁCH PHÒNG THỦ (THE GOTCHAS)
===============================================

Tài liệu ngắn này tóm tắt các rủi ro kỹ thuật quan trọng khi chạy kiến trúc "Semantic Compositional Facial Program Network" (dynamic routing / hyperedge composition), và cách phòng thủ thực tiễn.

1) Rủi ro: "Routing Collapse" (Sụp đổ định tuyến)
- Triệu chứng: Dynamic router học phương án tĩnh — mọi mẫu đều được gán trọng số cao cho 1 hyperedge (Dead Experts cho phần còn lại). Validation accuracy dừng tăng.
- Nguyên nhân phổ biến: hàm phạt sparsity dạng L1 / L0 đơn giản ép mạng chọn 1 luồng duy nhất.

Phòng thủ:
- Giám sát chặt validation accuracy và phân bố `routing_weights` theo epoch. Nếu thấy collapse, tạm thời chuyển từ L1 sang Loss tối đa hóa Entropy trên trung bình batch (Load-balancing loss) để khuyến khích phân phối đồng đều. Ví dụ entropy loss trung bình trên batch: H = -sum(p*log p) / log(N).
- Hoặc thêm một regularizer nhỏ đẩy entropy lên (maximize entropy) thay vì ép L1 xuống 0.

2) Rủi ro: Gradient vanishing ở `torch.sigmoid`
- Triệu chứng: Một hyperedge bị phạt mạnh → tham số incidence weight trở nên rất âm (vd -10) → sigmoid ≈ 0 và gradient ≈ 0 → expert chết vĩnh viễn.

Phòng thủ:
- Thử thêm temperature scaling vào sigmoid lúc train ban đầu: `torch.sigmoid(self.hyperedge_incidence / temp)` với `temp` khoảng 0.3–1.0 để gradient mượt hơn.
- Hoặc thay biểu diễn phân phối nhị phân bằng một cơ chế kích hoạt khác (vd LeakyReLU + clamp) hoặc sử dụng Gumbel-Sigmoid / concrete distribution nếu muốn rời rạc hóa có gradient.
- Giữ learning rate vừa phải, monitor histogram của `hyperedge_incidence` trong vài epoch đầu.

3) Rủi ro: Bùng nổ tham số do nhiều loss term
- Triệu chứng: Khó tune, một số loss cấu trúc áp đảo CE chính dẫn đến suy giảm accuracy.

Phòng thủ / Lời khuyên cấu hình ban đầu:
- Đặt `loss_ce` (CrossEntropy) = 1.0 (cơ bản).
- Các loss phụ quan trọng (Contrastive, Disentanglement, Semantic consistency) ~ 0.1.
- Các loss cấu trúc (Topology alignment, Sparsity, Coordination) đặt rất nhỏ lúc khởi đầu: ~0.01 hoặc 0.001 để không áp đảo CE.
- Theo dõi các loss riêng lẻ (logging, tensorboard/wandb) và scale adaptively nếu một loss quá lớn.

Ví dụ cấu hình khuyến nghị (sớm):
- `loss_ce: 1.0`
- `region_contrastive_weight` / `semantic_consistency_weight`: 0.1
- `semantic_disentanglement_weight`: 0.1
- `topology_alignment_weight`, `region_coordination_weight`, `program_sparsity_weight`: 0.01

Gợi ý thao tác khi phát hiện vấn đề trong training:
- Bước 1: Kiểm tra phân bố `routing_weights` và histogram `hyperedge_incidence`.
- Bước 2: Nếu collapse → đổi sparsity L1 → entropy-maximization loss hoặc giảm weight sparsity xuống 10×.
- Bước 3: Nếu expert chết (sigmoid → 0 nhanh) → thêm temperature cho sigmoid / thử Gumbel-Softmax hoặc concrete relaxations.
- Bước 4: Giảm learning rate cho phần incidence/topology parameters (grouped optimizer) hoặc tăng weight decay nhẹ cho chúng.

Snippet thay L1 sparsity (ý tưởng):

```py
# thay vì L1: loss = routing_weights.abs().mean()
def load_balance_entropy_loss(routing_weights):
    # routing_weights: (B, num_hyperedges)
    p = routing_weights.clamp_min(1e-6)
    ent = -(p * p.log()).sum(dim=-1)  # entropy per sample
    denom = math.log(p.size(-1))
    return -(ent / denom).mean()  # maximize entropy => minimize negative entropy
```

Snippet temperature cho sigmoid:

```py
# trong HyperedgeComposer.forward
temp = config.get("hyperedge_sigmoid_temp", 1.0)
incidence = self.hyperedge_incidence / float(temp)
incidence = incidence.clamp(-50, 50)
incidence_weights = torch.sigmoid(incidence)
```

Kết luận ngắn: kiến trúc mạnh nhưng cần monitor chặt training, track per-loss logging, và có sẵn plan để chuyển sparsity → entropy hoặc thêm temperature khi cần.

---
File này do agent tự động tạo theo yêu cầu; nếu muốn mình có thể:
- a) Patch trực tiếp loss function `semantic_program_sparsity_loss` để đổi sang negative-entropy khi config bật flag `use_entropy_sparsity: true`.
- b) Thêm `hyperedge_sigmoid_temp` vào `configs/semantic_roi_graph.yaml` và dùng giá trị này trong module tương ứng.

Bạn muốn mình thực hiện (a), (b), hay chỉ lưu tài liệu này thôi?
