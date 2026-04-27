# Ke hoach thi nghiem D3.1

## D3.1 - Stabilized Learnable Slot Candidate Motif GNN

Muc tieu: giu data pipeline D3-Full, khong dung CNN, khong quay lai hard top-K motif selection. D3.1 them global candidate context de on dinh bieu dien anh, dong thoi giam do manh class weight de tranh keo model qua Angry/Disgust.

### Diem chung cua D3.1

- Model: `learnable_slot_candidate_motif_gnn`
- Pooling chinh: `class_conditioned_slot_attention`
- Global candidate pooling: bat
- Global pooling type mac dinh: `mean_max`
- Candidate slots: `32`
- Max candidates: `128`
- Epochs: `120`
- Early stopping patience: `20`
- Expected debug shape:
  - `logits`: `[B, 7]`
  - `candidate_attention`: `[B, 32, 128]`
  - `class_slot_attention`: `[B, 7, 32]`

Classifier input dim D3.1:
- D3-Full cu: `class_logit` nhan `z_class [B, 7, 128]`.
- D3.1 `mean_max`: global context = masked mean `[B, 128]` + masked max `[B, 128]` = `[B, 256]`.
- Class-conditioned branch: concat thanh `z_class_aug [B, 7, 384]`, sau do `class_logit: Linear(384, 1)`.
- Neu doi sang `global_pooling_type: mean`: input se la `128 + 128 = 256`.

## 4 ban config can theo doi

| Ban | File | Vai tro | Khac biet chinh | Lenh/chuc nang |
| --- | --- | --- | --- | --- |
| D3.1 model w035 | `configs/learnable_slot_candidate_motif_gnn_d3_1.yaml` | Config train model truc tiep | `class_weight_power=0.35`, global pooling `mean_max` | Dung boi experiment `w035` |
| D3.1 model w025 | `configs/learnable_slot_candidate_motif_gnn_d3_1_w025.yaml` | Config train model truc tiep | `class_weight_power=0.25`, global pooling `mean_max` | Dung boi experiment `w025` |
| D3.1 experiment w035 | `configs/experiments/learnable_slot_candidate_motif_gnn_d3_1_w035.yaml` | Wrapper de build/train bang runner | Goi config model `learnable_slot_candidate_motif_gnn_d3_1`, epochs `120` | `python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w035 --mode train_only` |
| D3.1 experiment w025 | `configs/experiments/learnable_slot_candidate_motif_gnn_d3_1_w025.yaml` | Wrapper de build/train bang runner | Goi config model `learnable_slot_candidate_motif_gnn_d3_1_w025`, epochs `120` | `python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w025 --mode train_only` |

## Lenh chay

Train tu artifact da co:

```bash
python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w035 --mode train_only
python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w025 --mode train_only
```

Build artifact roi train:

```bash
python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w035 --mode build_and_train
python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w025 --mode build_and_train
```

Debug batch:

```bash
python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w035 --mode debug_only
python scripts/run_experiment.py --config learnable_slot_candidate_motif_gnn_d3_1_w025 --mode debug_only
```

## Log ket qua

### D3.1 w035

- Ngay chay:
- Commit/run id:
- Artifact:
- Epoch best:
- Val accuracy:
- Val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Confusion matrix note:
- Nhan xet Happy/Sad/Neutral:
- Nhan xet Angry/Disgust:
- Ket luan:

### D3.1 w025

- Ngay chay:
- Commit/run id:
- Artifact:
- Epoch best:
- Val accuracy:
- Val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Confusion matrix note:
- Nhan xet Happy/Sad/Neutral:
- Nhan xet Angry/Disgust:
- Ket luan:

## Baseline de so sanh

### D3-Full

- Accuracy: `45.75%`
- Macro F1: `0.4434`
- Weighted F1: `0.4570`
- Ghi chu: D3 gan C nhung chua vuot C-mean. Angry/Disgust recall tot hon, Happy/Sad/Neutral giam.

### C-mean

- Accuracy:
- Macro F1:
- Weighted F1:
- Ghi chu:
