# Pixel-preserving Motif V2 Baseline

## 1. Chốt baseline hiện tại

Baseline chính hiện tại là:

```text
Pixel-preserving Motif V2 + Motif-guided GNN
```

Config dùng để báo cáo:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Checkpoint tốt nhất hiện tại:

```text
outputs/checkpoints/motif_guided_gnn/motif_guided_gnn_25042026_2257_best.pth
```

Kết quả test tốt nhất hiện tại:

```text
Accuracy    : 45.11%
Macro F1    : 0.4196
Weighted F1 : 0.4380
Best epoch  : 57
Val Macro F1: 0.4100
```

## 2. Ý tưởng chính

FER-2013 là ảnh grayscale 48x48, có rất ít thông tin và dễ mất chi tiết cục bộ nếu gom thô quá sớm. Vì vậy baseline này không bỏ pixel-level graph.

Full pixel graph vẫn giữ vai trò:

1. nguồn thông tin gốc của ảnh
2. nơi sinh candidate subgraphs
3. nơi truy vết motif về pixel nodes
4. nền để visualize evidence

Tuy nhiên, model không train trực tiếp trên full graph 2304 node vì chi phí lớn và dễ nhiễu. Thay vào đó, pipeline dùng motif để chọn các subgraphs giàu tín hiệu cảm xúc, rồi train GNN ở mức subgraph-level.

## 3. Pipeline

Luồng tổng thể:

```text
FER image 48x48
-> full pixel graph 2304 nodes
-> pixel candidate subgraphs
-> descriptor + pixel trace
-> emotion-specific motif bank
-> motif matching + coverage/diversity selection
-> top-32 motif-selected subgraphs
-> subgraph-level GNN
-> emotion classification
```

Mỗi ảnh sau khi precompute có dạng:

```text
x                  : [32, 41]
mask               : [32]
centers            : [32, 2]
bbox               : [32, 4]
selected_indices   : [32]
node_indices       : [32, 25]
node_mask          : [32, 25]
edge_index         : [2, 128]
edge_attr          : [128, 3]
match_scores       : [32]
matched_class      : [32]
matched_motif_id   : [32]
motif_score_vector : [7]
label              : int
```

Trong đó:

- `x` là descriptor 41 chiều của selected subgraphs.
- `node_indices` giữ trace về pixel nodes trong full image.
- `edge_index` là graph KNN giữa các selected subgraphs.
- `motif_score_vector` là điểm match tổng hợp với motif bank theo 7 emotion.

## 4. Các bước tạo artifact

### 4.1 Precompute pixel candidate subgraphs

```powershell
conda run -n fer-graph --no-capture-output python scripts/precompute_pixel_candidate_subgraphs.py `
  --repo_root artifacts/graph_repo `
  --out_dir artifacts/pixel_candidate_subgraphs_v2 `
  --max_candidates 128 `
  --seed_stride 4 `
  --radii 1 2 `
  --coverage_grid 4 4 `
  --log_every 100
```

Output:

```text
artifacts/pixel_candidate_subgraphs_v2/
  meta.pt
  train_pixel_candidates.pt
  val_pixel_candidates.pt
  test_pixel_candidates.pt
```

### 4.2 Build pixel motif bank

```powershell
conda run -n fer-graph --no-capture-output python scripts/build_pixel_motif_bank.py `
  --input_dir artifacts/pixel_candidate_subgraphs_v2 `
  --out_dir artifacts/pixel_motif_bank_v2 `
  --num_motifs_per_class 16 `
  --max_subgraphs_per_class 50000 `
  --alpha 0.5 `
  --seed 42 `
  --num_exemplars 5
```

Output:

```text
artifacts/pixel_motif_bank_v2/
  pixel_motif_bank.pt
```

Motif bank có:

```text
7 classes
16 motifs/class
112 motifs total
descriptor_dim = 41
```

### 4.3 Precompute pixel motif dataset

```powershell
conda run -n fer-graph --no-capture-output python scripts/precompute_pixel_motif_dataset.py `
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

Output:

```text
artifacts/pixel_motif_dataset_v2/
  meta.pt
  train_pixel_motif.pt
  val_pixel_motif.pt
  test_pixel_motif.pt
```

## 5. Model baseline

Model chính:

```text
MotifGuidedGNN
```

Config:

```yaml
data:
  mode: pixel_motif
  pixel_motif_dataset_path: artifacts/pixel_motif_dataset_v2
  normalize_x: true

model:
  name: motif_guided_gnn
  hidden_dim: 128
  gnn_hidden_dim: 128
  num_layers: 2
  use_motif_score_vector: true
  use_match_score_feature: true
  use_match_score_weighting: true
  pooling: motif_attention

loss:
  name: weighted_ce
  use_class_weights: true
  class_weight_power: 0.5
```

Điểm quan trọng:

- `normalize_x: true` là bắt buộc trong baseline tốt nhất.
- `weighted_ce` tốt hơn `weighted_ce_motif` trong baseline này vì `motif_score_vector` là precomputed, motif loss không tạo gradient hữu ích cho model.
- GNN dùng `motif_score_vector`, `match_scores`, `matched_class` như feature phụ, nhưng không dùng motif-consistency loss.

## 6. Lệnh train baseline chính

```powershell
conda run -n fer-graph --no-capture-output python -m scripts.train `
  --config pixel_motif_guided_gnn_motif_norm `
  --env local `
  --pixel_motif_dataset_path artifacts/pixel_motif_dataset_v2 `
  --epochs 60 `
  --no_wandb
```

## 7. Kết quả thực nghiệm

### 7.1 MLP clean baseline

Config:

```text
configs/pixel_motif_guided_mlp_clean.yaml
```

Kết quả test:

```text
Accuracy    : 39.20%
Macro F1    : 0.3174
Weighted F1 : 0.3753
```

### 7.2 GNN baseline chính

Config:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Kết quả test:

```text
Accuracy    : 45.11%
Macro F1    : 0.4196
Weighted F1 : 0.4380
```

So với MLP clean:

```text
+5.91% accuracy
+0.1022 macro F1
```

Điều này cho thấy quan hệ giữa các motif-selected subgraphs qua GNN có đóng góp thực sự, không chỉ là pooling descriptor.

## 8. Classification report tốt nhất

```text
              precision    recall  f1-score   support
0 Angry         0.3174    0.2566    0.2838       491
1 Disgust       0.4583    0.4000    0.4272        55
2 Fear          0.2986    0.1572    0.2060       528
3 Happy         0.6213    0.6758    0.6474       879
4 Sad           0.3421    0.4158    0.3754       594
5 Surprise      0.4943    0.6274    0.5530       416
6 Neutral       0.4333    0.4569    0.4448       626

accuracy                           0.4511      3589
macro avg      0.4236    0.4271    0.4196      3589
weighted avg   0.4360    0.4511    0.4380      3589
```

Class yếu nhất hiện tại là `Fear`, còn `Disgust` đã cải thiện rõ dù số mẫu rất ít.

## 9. Nhận xét

Baseline này chưa cạnh tranh với SOTA CNN khoảng 0.73-0.76 accuracy, nhưng có giá trị nghiên cứu riêng:

- giữ pixel-level evidence
- chọn motif có coverage/diversity
- trace được selected subgraphs về pixel nodes
- GNN vượt MLP trên cùng dữ liệu motif-selected subgraphs

Khoảng cách với SOTA chủ yếu đến từ việc node feature hiện vẫn là handcrafted descriptor 41 chiều, chưa có learned visual feature mạnh như CNN.

## 10. Hướng phát triển tiếp theo

Hướng nâng cấp hợp lý nhất:

```text
CNN-augmented Pixel-preserving Motif GNN
```

Tức là giữ pipeline motif hiện tại, nhưng node feature của mỗi selected subgraph sẽ gồm:

```text
descriptor_41D
+ CNN patch/region embedding
+ match score
+ motif class one-hot
+ center/bbox
```

Khi đó motif vẫn giữ vai trò chọn vùng và giải thích pixel evidence, còn CNN cung cấp feature thị giác mạnh hơn để thu hẹp khoảng cách với SOTA.
