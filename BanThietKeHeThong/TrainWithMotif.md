# Train With Motif - Current Training Contract

File này mô tả cách training hiện tại hoạt động với pixel motif dataset.

## 1. Training không đọc CSV trực tiếp

Training nhận artifact đã được build trong cùng experiment run:

```text
pixel_motif_dataset_path = <artifact_root>/pixel_motif_dataset_v2
graph_repo_path          = <artifact_root>/graph_repo
```

Trong Kaggle workflow chính, hai path này nằm trong:

```text
/kaggle/working/artifacts/
```

## 2. Entrypoint training chuẩn

Không gọi `scripts/train.py` trực tiếp từ notebook trong workflow chính. Notebook gọi:

```bash
python -m scripts.run_experiment --config <experiment>
```

Sau đó `src/pipeline/experiment_runner.py` gọi `scripts/train.py` với đúng path artifact vừa build.

## 3. Trainer contract

File:

```text
src/training/trainer.py
```

Trainer hỗ trợ batch dict và forward theo model:

```text
motif batch -> model(batch)
plain x/mask -> model(x, mask)
```

Với B và C, batch có:

```text
motif_score_vector
match_scores
matched_class
```

nên trainer gọi:

```python
model(batch)
```

## 4. Loss hiện tại

Baseline B và C dùng:

```text
weighted_ce
class_weight_power = 0.5
```

File:

```text
src/training/losses.py
```

Không dùng prototype loss hoặc contrastive loss trong C đầu tiên.

## 5. Metrics

Evaluation tính:

```text
accuracy
macro F1
weighted F1
per-class report
confusion matrix
```

File:

```text
src/evaluation/evaluator.py
src/evaluation/metrics.py
```

## 6. Baseline B training

Experiment:

```text
configs/experiments/pixel_motif_baseline_b.yaml
```

Model config:

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
```

Command:

```bash
python -m scripts.run_experiment --config pixel_motif_baseline_b
```

## 7. Version C training

Experiment:

```text
configs/experiments/hierarchical_motif_gnn_c.yaml
```

Model config:

```text
configs/hierarchical_motif_gnn.yaml
```

Command:

```bash
python -m scripts.run_experiment --config hierarchical_motif_gnn_c
```

C có `debug_batch: true`, nên runner sẽ chạy batch forward/backward sanity trước khi train.

## 8. Debug batch expected shape

```text
x:             [B, 32, 41]
sub_x:         [B, 32, 25, 7]
sub_adj:       [B, 32, 25, 25]
sub_node_mask: [B, 32, 25]
mask:          [B, 32]
logits:        [B, 7]
```

Nếu debug batch fail, không train full.
