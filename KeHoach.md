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

5027.9s	14426	Evaluate on TEST SET with best checkpoint
5027.9s	14427	=======================================================
5027.9s	14428	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_27042026_1217_best.pth
5027.9s	14429	--> Restored ep=84  best_val_macro_f1=0.4944
5028.2s	14430	
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:15,  3.54it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.93it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:15,  3.54it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.93it/s]
5028.4s	14431	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:15,  3.54it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.93it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.13it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.44it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.13it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.44it/s]
5028.6s	14432	
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.13it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.44it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 15.33it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 16.23it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 15.33it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 16.23it/s]
5028.8s	14433	
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 15.33it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 16.23it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 17.19it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 17.42it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 17.19it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 17.42it/s]
5029.1s	14434	
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 17.19it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 17.42it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 17.68it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.81it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 17.68it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.81it/s]
5029.3s	14435	
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 17.68it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.81it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.99it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 18.47it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.99it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 18.47it/s]
5029.5s	14436	
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.99it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 18.47it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.37it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.48it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.37it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.48it/s]
5029.7s	14437	
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.37it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.48it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.35it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.79it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.35it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.79it/s]
5029.9s	14438	
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.35it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.79it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.86it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 18.54it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.86it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 18.54it/s]
5030.1s	14439	
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.86it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 18.54it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 17.99it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.84it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 17.99it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.84it/s]
5030.4s	14440	
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 17.99it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.84it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 18.41it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 18.49it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 18.41it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 18.49it/s]
5030.6s	14441	
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 18.41it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 18.49it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 18.80it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 18.83it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 18.80it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 18.83it/s]
5031.1s	14442	
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 18.80it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 18.83it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 18.18it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:02<00:00, 18.19it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 18.18it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.90it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.93it/s]
5031.5s	14443	
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 18.18it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:02<00:00, 18.19it/s]
5031.5s	14444	
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 18.18it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:02<00:00, 18.19it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 18.18it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.90it/s]
5031.5s	14445	
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 18.18it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.90it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.93it/s]
5031.5s	14446	
5031.5s	14447	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.93it/s]
5036.4s	14448	[1;34mwandb[0m: 
5036.4s	14449	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_27042026_1217[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/numg8hqh[0m
5036.4s	14450	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260427_121743-numg8hqh/logs[0m
5036.5s	14451	
5036.5s	14452	=======================================================
5036.5s	14453	TEST SET EVALUATION
5036.5s	14454	=======================================================
5036.5s	14455	--> Accuracy:    51.02%
5036.5s	14456	--> Macro F1:    0.5000
5036.5s	14457	--> Weighted F1: 0.5074
5036.5s	14458	
5036.5s	14459	--> Classification Report:
5036.5s	14460	              precision    recall  f1-score     support
5036.5s	14461	0              0.363462  0.384929  0.373887   491.00000
5036.5s	14462	1              0.492308  0.581818  0.533333    55.00000
5036.5s	14463	2              0.413420  0.361742  0.385859   528.00000
5036.5s	14464	3              0.689113  0.698521  0.693785   879.00000
5036.5s	14465	4              0.396040  0.336700  0.363967   594.00000
5036.5s	14466	5              0.655963  0.687500  0.671362   416.00000
5036.5s	14467	6              0.449296  0.509585  0.477545   626.00000
5036.5s	14468	accuracy       0.510170  0.510170  0.510170     0.51017
5036.5s	14469	macro avg      0.494229  0.508685  0.499963  3589.00000
5036.5s	14470	weighted avg   0.506810  0.510170  0.507358  3589.00000
5036.5s	14471	Confusion matrix saved: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_27042026_1217/confusion_matrix.png
5036.5s	14472	
5036.5s	14473	--> Figures saved to: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_27042026_1217
5036.5s	14474		--> [WandB] Send File `learnable_slot_candidate_motif_gnn_27042026_1217_best.pth` to cloud successfully!
5036.5s	14475	
5036.5s	14476			DONE!


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
4751.2s	13487	Evaluate on TEST SET with best checkpoint
4751.2s	13488	=======================================================
4751.2s	13489	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_27042026_1218_best.pth
4751.2s	13490	--> Restored ep=76  best_val_macro_f1=0.5104
4751.5s	13491	
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:17,  3.19it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.07it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:17,  3.19it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.07it/s]
4751.7s	13492	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:17,  3.19it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.07it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.17it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.64it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.17it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.64it/s]
4751.9s	13493	
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.17it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.64it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.80it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.37it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.80it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.37it/s]
4752.2s	13494	
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.80it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.37it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:01<00:02, 15.87it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.29it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:01<00:02, 15.87it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.29it/s]
4752.4s	13495	
Evaluating test set:  23%|██▎       | 13/57 [00:01<00:02, 15.87it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.29it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.99it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.25it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.99it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.25it/s]
4752.6s	13496	
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.99it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.25it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.65it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 17.88it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.65it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 17.88it/s]
4752.8s	13497	
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.65it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 17.88it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.17it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.35it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.17it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.35it/s]
4753.0s	13498	
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.17it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.35it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.10it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.43it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.10it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.43it/s]
4753.3s	13499	
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.10it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.43it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.78it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 19.12it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.78it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 19.12it/s]
4753.5s	13500	
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.78it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 19.12it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:00, 19.37it/s]
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 19.04it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:00, 19.37it/s]
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 19.04it/s]
4753.7s	13501	
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:00, 19.37it/s]
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 19.04it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:02<00:00, 19.07it/s]
Evaluating test set:  77%|███████▋  | 44/57 [00:02<00:00, 19.02it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:02<00:00, 19.07it/s]
Evaluating test set:  77%|███████▋  | 44/57 [00:02<00:00, 19.02it/s]
4754.0s	13502	
Evaluating test set:  74%|███████▎  | 42/57 [00:02<00:00, 19.07it/s]
Evaluating test set:  77%|███████▋  | 44/57 [00:02<00:00, 19.02it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 19.26it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 19.20it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 19.26it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 19.20it/s]
4754.2s	13503	
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 19.26it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 19.20it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 18.93it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 19.20it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 18.93it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 19.20it/s]
4754.3s	13504	
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 18.93it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 19.20it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 21.15it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.99it/s]
4754.4s	13505	
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 21.15it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.99it/s]
4754.4s	13506	
4754.4s	13507	
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 21.15it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.99it/s]
4760.2s	13508	[1;34mwandb[0m: 
4760.2s	13509	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_27042026_1218[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/pnmnv319[0m
4760.2s	13510	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260427_121807-pnmnv319/logs[0m
4760.3s	13511	
4760.3s	13512	=======================================================
4760.3s	13513	TEST SET EVALUATION
4760.3s	13514	=======================================================
4760.3s	13515	--> Accuracy:    52.83%
4760.3s	13516	--> Macro F1:    0.5096
4760.3s	13517	--> Weighted F1: 0.5231
4760.3s	13518	
4760.3s	13519	--> Classification Report:
4760.3s	13520	              precision    recall  f1-score      support
4760.3s	13521	0              0.413580  0.409369  0.411464   491.000000
4760.3s	13522	1              0.438356  0.581818  0.500000    55.000000
4760.3s	13523	2              0.405830  0.342803  0.371663   528.000000
4760.3s	13524	3              0.645854  0.753129  0.695378   879.000000
4760.3s	13525	4              0.424184  0.372054  0.396413   594.000000
4760.3s	13526	5              0.735043  0.620192  0.672751   416.000000
4760.3s	13527	6              0.496361  0.544728  0.519421   626.000000
4760.3s	13528	accuracy       0.528281  0.528281  0.528281     0.528281
4760.3s	13529	macro avg      0.508458  0.517728  0.509584  3589.000000
4760.3s	13530	weighted avg   0.523161  0.528281  0.523125  3589.000000
4760.3s	13531	Confusion matrix saved: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_27042026_1218/confusion_matrix.png
4760.3s	13532	
4760.3s	13533	--> Figures saved to: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_27042026_1218
4760.3s	13534		--> [WandB] Send File `learnable_slot_candidate_motif_gnn_27042026_1218_best.pth` to cloud successfully!
4760.3s	13535	
4760.3s	13536			DONE!

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
