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

## 6 kich ban D3.1 co ban sau current best w025

Muc tieu: giu pipeline graph-only D3.1, khong dung CNN, khong them loss attention moi, khong quay lai hard top-K. Chi kiem tra scaling descriptor, LR, class weight power, va do on dinh repeat.

Current best truoc batch nay:
- Config goc: `learnable_slot_candidate_motif_gnn_d3_1_w025`
- Accuracy: `52.83%`
- Macro F1: `0.5096`
- Weighted F1: `0.5231`

Luu y scaling:
- `normalize_candidate_x=true`: loader standardize `candidate_x` bang mean/std tinh tu valid candidates cua split dang load.
- `normalize_candidate_x=false`: khong standardize descriptor 41D trong loader.
- Geometry van normalized: `candidate_centers`, `candidate_bbox` o toa do `[0,1]`; `candidate_radius` la integer topology.
- `edge_attr`: `dx/dy/dist` tinh tu normalized centers, `edge_type` khong scale.
- Neu artifact meta bao `candidate_x` da scaled san, can rebuild rawdesc/candidate attention artifact truoc khi tin ket qua no-scale.

### Bang tong hop 6 run

| Kich ban | Experiment config | Model config | normalize_candidate_x | LR | class_weight_power | Test Acc | Macro F1 | Weighted F1 | Ghi chu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline repeat | `configs/experiments/d3_1_w025_repeat.yaml` | `learnable_slot_candidate_motif_gnn_d3_1_w025` | `true` | `0.001` | `0.25` |  |  |  |  |
| No descriptor scaling | `configs/experiments/d3_1_w025_no_desc_scale.yaml` | `learnable_slot_candidate_motif_gnn_d3_1_no_desc_scale` | `false` | `0.001` | `0.25` |  |  |  |  |
| LR 7e-4 | `configs/experiments/d3_1_w025_lr7e4.yaml` | `learnable_slot_candidate_motif_gnn_d3_1_lr7e4` | `true` | `0.0007` | `0.25` |  |  |  |  |
| LR 5e-4 | `configs/experiments/d3_1_w025_lr5e4.yaml` | `learnable_slot_candidate_motif_gnn_d3_1_lr5e4` | `true` | `0.0005` | `0.25` |  |  |  |  |
| class_weight_power 0.20 | `configs/experiments/d3_1_w020.yaml` | `learnable_slot_candidate_motif_gnn_d3_1_w020` | `true` | `0.001` | `0.20` |  |  |  |  |
| No desc scale + LR 7e-4 | `configs/experiments/d3_1_w025_no_desc_scale_lr7e4.yaml` | `learnable_slot_candidate_motif_gnn_d3_1_no_desc_scale_lr7e4` | `false` | `0.0007` | `0.25` |  |  |  |  |

### Lenh chay 6 kich ban

```bash
python -m scripts.run_experiment --config d3_1_w025_repeat --mode train_only
python -m scripts.run_experiment --config d3_1_w025_no_desc_scale --mode train_only
python -m scripts.run_experiment --config d3_1_w025_lr7e4 --mode train_only
python -m scripts.run_experiment --config d3_1_w025_lr5e4 --mode train_only
python -m scripts.run_experiment --config d3_1_w020 --mode train_only
python -m scripts.run_experiment --config d3_1_w025_no_desc_scale_lr7e4 --mode train_only
```

### Mau ghi ket qua tung run

#### d3_1_w025_repeat

- Ngay chay:
- Commit/run id:
- Artifact:
- Descriptor storage:
- normalize_candidate_x:
- Candidate_x batch stats:
- Epoch best:
- Best val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Per-class note:
- Ket luan:

Evaluate on TEST SET with best checkpoint
3438.7s	10146	=======================================================
3438.7s	10147	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_28042026_0526_best.pth
3438.7s	10148	--> Restored ep=53  best_val_macro_f1=0.4709
3438.8s	10149	
Val:  89%|████████▉ | 51/57 [00:02<00:00, 17.98it/s]
Val:  95%|█████████▍| 54/57 [00:03<00:00, 19.37it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
3438.9s	10150	Epoch  73/120 | loss: 0.8025  acc: 0.6894  macro_f1: 0.7007 | val_loss: 1.9674  val_acc: 0.4826  val_macro_f1: 0.4575
3438.9s	10151	          val pred_count: [521, 83, 498, 852, 584, 447, 604]
3438.9s	10152		-!- No improvement: 20/20
3438.9s	10153		-_- Early stopping at ep=73
3438.9s	10154	
3438.9s	10155	=======================================================
3438.9s	10156	Evaluate on TEST SET with best checkpoint
3438.9s	10157	=======================================================
3438.9s	10158	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_28042026_0526_best.pth
3438.9s	10159	--> Restored ep=53  best_val_macro_f1=0.4709
3439.2s	10160	
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:14,  3.77it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:05,  9.32it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:14,  3.77it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:05,  9.32it/s]
3439.4s	10161	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:14,  3.77it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:05,  9.32it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.30it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.25it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.30it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.25it/s]
3439.6s	10162	
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.30it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.25it/s]
Evaluating test set:  18%|█▊        | 10/57 [00:00<00:02, 16.34it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:02, 16.73it/s]
Evaluating test set:  18%|█▊        | 10/57 [00:00<00:02, 16.34it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:02, 16.73it/s]
3439.9s	10163	
Evaluating test set:  18%|█▊        | 10/57 [00:00<00:02, 16.34it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:02, 16.73it/s]
Evaluating test set:  25%|██▍       | 14/57 [00:00<00:02, 17.49it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:01<00:02, 18.03it/s]
Evaluating test set:  25%|██▍       | 14/57 [00:00<00:02, 17.49it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:01<00:02, 18.03it/s]
3440.1s	10164	
Evaluating test set:  25%|██▍       | 14/57 [00:00<00:02, 17.49it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:01<00:02, 18.03it/s]
Evaluating test set:  32%|███▏      | 18/57 [00:01<00:02, 18.41it/s]
Evaluating test set:  35%|███▌      | 20/57 [00:01<00:02, 18.38it/s]
Evaluating test set:  32%|███▏      | 18/57 [00:01<00:02, 18.41it/s]
Evaluating test set:  35%|███▌      | 20/57 [00:01<00:02, 18.38it/s]
3440.3s	10165	
Evaluating test set:  32%|███▏      | 18/57 [00:01<00:02, 18.41it/s]
Evaluating test set:  35%|███▌      | 20/57 [00:01<00:02, 18.38it/s]
Evaluating test set:  39%|███▊      | 22/57 [00:01<00:01, 18.43it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.67it/s]
Evaluating test set:  39%|███▊      | 22/57 [00:01<00:01, 18.43it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.67it/s]
3440.5s	10166	
Evaluating test set:  39%|███▊      | 22/57 [00:01<00:01, 18.43it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.67it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.62it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.90it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.62it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.90it/s]
3440.7s	10167	
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.62it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.90it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 19.15it/s]
Evaluating test set:  60%|█████▉    | 34/57 [00:01<00:01, 19.73it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 19.15it/s]
Evaluating test set:  60%|█████▉    | 34/57 [00:01<00:01, 19.73it/s]
3441.0s	10168	
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 19.15it/s]
Evaluating test set:  60%|█████▉    | 34/57 [00:01<00:01, 19.73it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:02<00:01, 19.33it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:00, 19.51it/s]
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 19.61it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:02<00:01, 19.33it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:00, 19.51it/s]
3441.2s	10169	
Evaluating test set:  63%|██████▎   | 36/57 [00:02<00:01, 19.33it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:00, 19.51it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:02<00:00, 19.06it/s]
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 19.61it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:02<00:00, 19.06it/s]
3441.4s	10170	
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 19.61it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:02<00:00, 19.06it/s]
Evaluating test set:  77%|███████▋  | 44/57 [00:02<00:00, 18.63it/s]
Evaluating test set:  81%|████████  | 46/57 [00:02<00:00, 17.32it/s]
Evaluating test set:  77%|███████▋  | 44/57 [00:02<00:00, 18.63it/s]
Evaluating test set:  81%|████████  | 46/57 [00:02<00:00, 17.32it/s]
3441.7s	10171	
Evaluating test set:  77%|███████▋  | 44/57 [00:02<00:00, 18.63it/s]
Evaluating test set:  81%|████████  | 46/57 [00:02<00:00, 17.32it/s]
Evaluating test set:  84%|████████▍ | 48/57 [00:02<00:00, 17.05it/s]
Evaluating test set:  88%|████████▊ | 50/57 [00:02<00:00, 17.37it/s]
Evaluating test set:  84%|████████▍ | 48/57 [00:02<00:00, 17.05it/s]
Evaluating test set:  88%|████████▊ | 50/57 [00:02<00:00, 17.37it/s]
3441.9s	10172	
Evaluating test set:  84%|████████▍ | 48/57 [00:02<00:00, 17.05it/s]
Evaluating test set:  88%|████████▊ | 50/57 [00:02<00:00, 17.37it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 18.43it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 20.55it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 18.43it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 20.55it/s]
3442.0s	10173	
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 18.43it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 20.55it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 17.38it/s]
3442.2s	10174	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 17.38it/s]
3442.2s	10175	
3442.2s	10176	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 17.38it/s]
3466.0s	10177	[1;34mwandb[0m: 
3466.0s	10178	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_28042026_0526[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/0pzkpahc[0m
3466.0s	10179	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260428_052700-0pzkpahc/logs[0m
3466.1s	10180	
3466.1s	10181	=======================================================
3466.1s	10182	TEST SET EVALUATION
3466.1s	10183	=======================================================
3466.1s	10184	--> Accuracy:    51.07%
3466.1s	10185	--> Macro F1:    0.4844
3466.1s	10186	--> Weighted F1: 0.5036
3466.1s	10187	
3466.1s	10188	--> Classification Report:
3466.1s	10189	              precision    recall  f1-score      support
3466.1s	10190	0              0.332155  0.382892  0.355724   491.000000
3466.1s	10191	1              0.419355  0.472727  0.444444    55.000000
3466.1s	10192	2              0.422222  0.287879  0.342342   528.000000
3466.1s	10193	3              0.640594  0.736064  0.685019   879.000000
3466.1s	10194	4              0.424710  0.370370  0.395683   594.000000
3466.1s	10195	5              0.653670  0.685096  0.669014   416.000000
3466.1s	10196	6              0.494505  0.503195  0.498812   626.000000
3466.1s	10197	accuracy       0.510727  0.510727  0.510727     0.510727
3466.1s	10198	macro avg      0.483887  0.491175  0.484434  3589.000000
3466.1s	10199	weighted avg   0.503186  0.510727  0.503649  3589.000000

#### d3_1_w025_no_desc_scale

- Ngay chay:
- Commit/run id:
- Artifact:
- Descriptor storage:
- normalize_candidate_x:
- Candidate_x batch stats:
- Epoch best:
- Best val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Per-class note:
- Ket luan:

=======================================================
1129.4s	3152	Evaluate on TEST SET with best checkpoint
1129.4s	3153	=======================================================
1129.4s	3154	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_28042026_0526_best.pth
1129.4s	3155	--> Restored ep=1  best_val_macro_f1=0.0570
1129.4s	3156	
Val:  96%|█████████▋| 55/57 [00:03<00:00, 20.68it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
1129.6s	3157	
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:14,  3.92it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:05,  9.27it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:14,  3.92it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:05,  9.27it/s]
1129.8s	3158	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:14,  3.92it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:05,  9.27it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.57it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.82it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.57it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.82it/s]
1130.1s	3159	
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 12.57it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 14.82it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 15.86it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 16.82it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 15.86it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 16.82it/s]
1130.3s	3160	
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 15.86it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 16.82it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 17.25it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 17.97it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 17.25it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 17.97it/s]
1130.5s	3161	
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 17.25it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 17.97it/s]
Evaluating test set:  32%|███▏      | 18/57 [00:01<00:02, 18.66it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:01, 19.31it/s]
Evaluating test set:  32%|███▏      | 18/57 [00:01<00:02, 18.66it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:01, 19.31it/s]
1130.8s	3162	
Evaluating test set:  32%|███▏      | 18/57 [00:01<00:02, 18.66it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:01, 19.31it/s]
Evaluating test set:  42%|████▏     | 24/57 [00:01<00:01, 19.46it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:01<00:01, 19.43it/s]
Evaluating test set:  42%|████▏     | 24/57 [00:01<00:01, 19.46it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:01<00:01, 19.43it/s]
1131.0s	3163	
Evaluating test set:  42%|████▏     | 24/57 [00:01<00:01, 19.46it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:01<00:01, 19.43it/s]
Evaluating test set:  49%|████▉     | 28/57 [00:01<00:01, 19.36it/s]
Evaluating test set:  53%|█████▎    | 30/57 [00:01<00:01, 19.40it/s]
Evaluating test set:  49%|████▉     | 28/57 [00:01<00:01, 19.36it/s]
Evaluating test set:  53%|█████▎    | 30/57 [00:01<00:01, 19.40it/s]
1131.3s	3164	
Evaluating test set:  49%|████▉     | 28/57 [00:01<00:01, 19.36it/s]
Evaluating test set:  53%|█████▎    | 30/57 [00:01<00:01, 19.40it/s]
Evaluating test set:  56%|█████▌    | 32/57 [00:01<00:01, 18.67it/s]
Evaluating test set:  60%|█████▉    | 34/57 [00:02<00:01, 17.51it/s]
Evaluating test set:  56%|█████▌    | 32/57 [00:01<00:01, 18.67it/s]
Evaluating test set:  60%|█████▉    | 34/57 [00:02<00:01, 17.51it/s]
1131.5s	3165	
Evaluating test set:  56%|█████▌    | 32/57 [00:01<00:01, 18.67it/s]
Evaluating test set:  60%|█████▉    | 34/57 [00:02<00:01, 17.51it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:02<00:01, 17.34it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:01, 17.79it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:02<00:01, 17.34it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:01, 17.79it/s]
1131.7s	3166	
Evaluating test set:  63%|██████▎   | 36/57 [00:02<00:01, 17.34it/s]
Evaluating test set:  67%|██████▋   | 38/57 [00:02<00:01, 17.79it/s]
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 17.38it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 18.44it/s]
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 17.38it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 18.44it/s]
1132.0s	3167	
Evaluating test set:  70%|███████   | 40/57 [00:02<00:00, 17.38it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 18.44it/s]
Evaluating test set:  81%|████████  | 46/57 [00:02<00:00, 19.48it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 20.07it/s]
Evaluating test set:  81%|████████  | 46/57 [00:02<00:00, 19.48it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 20.07it/s]
1132.3s	3168	
Evaluating test set:  81%|████████  | 46/57 [00:02<00:00, 19.48it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 20.07it/s]
Evaluating test set:  91%|█████████ | 52/57 [00:02<00:00, 20.28it/s]
Evaluating test set:  96%|█████████▋| 55/57 [00:03<00:00, 21.17it/s]
Evaluating test set:  91%|█████████ | 52/57 [00:02<00:00, 20.28it/s]
Evaluating test set:  96%|█████████▋| 55/57 [00:03<00:00, 21.17it/s]
1132.4s	3169	
Evaluating test set:  91%|█████████ | 52/57 [00:02<00:00, 20.28it/s]
Evaluating test set:  96%|█████████▋| 55/57 [00:03<00:00, 21.17it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 17.70it/s]
1132.6s	3170	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 17.70it/s]
1132.6s	3171	
1132.6s	3172	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 17.70it/s]
1138.9s	3173	[1;34mwandb[0m: 
1138.9s	3174	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_28042026_0526[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/fq6hbk5q[0m
1138.9s	3175	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260428_052648-fq6hbk5q/logs[0m
1138.9s	3176	
1138.9s	3177	=======================================================
1138.9s	3178	TEST SET EVALUATION
1138.9s	3179	=======================================================
1138.9s	3180	--> Accuracy:    24.49%
1138.9s	3181	--> Macro F1:    0.0562
1138.9s	3182	--> Weighted F1: 0.0964
1138.9s	3183	
1138.9s	3184	--> Classification Report:
1138.9s	3185	              precision    recall  f1-score      support
1138.9s	3186	0              0.000000  0.000000  0.000000   491.000000
1138.9s	3187	1              0.000000  0.000000  0.000000    55.000000
1138.9s	3188	2              0.000000  0.000000  0.000000   528.000000
1138.9s	3189	3              0.244915  1.000000  0.393465   879.000000
1138.9s	3190	4              0.000000  0.000000  0.000000   594.000000
1138.9s	3191	5              0.000000  0.000000  0.000000   416.000000
1138.9s	3192	6              0.000000  0.000000  0.000000   626.000000
1138.9s	3193	accuracy       0.244915  0.244915  0.244915     0.244915
1138.9s	3194	macro avg      0.034988  0.142857  0.056209  3589.000000
1138.9s	3195	weighted avg   0.059983  0.244915  0.096365  3589.000000



#### d3_1_w025_lr7e4

- Ngay chay:
- Commit/run id:
- Artifact:
- Descriptor storage:
- normalize_candidate_x:
- Candidate_x batch stats:
- Epoch best:
- Best val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Per-class note:
- Ket luan:
Evaluate on TEST SET with best checkpoint
5402.3s	16399	=======================================================
5402.3s	16400	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_28042026_0527_best.pth
5402.3s	16401	--> Restored ep=119  best_val_macro_f1=0.4986
5413.8s	16402	[1;34mwandb[0m: 
5413.8s	16403	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_28042026_0527[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/x1fq8kcx[0m
5413.8s	16404	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260428_052712-x1fq8kcx/logs[0m
5413.9s	16405	
5413.9s	16406	=======================================================
5413.9s	16407	TEST SET EVALUATION
5413.9s	16408	=======================================================
5413.9s	16409	--> Accuracy:    51.88%
5413.9s	16410	--> Macro F1:    0.4990
5413.9s	16411	--> Weighted F1: 0.5168
5413.9s	16412	
5413.9s	16413	--> Classification Report:
5413.9s	16414	              precision    recall  f1-score      support
5413.9s	16415	0              0.403766  0.393075  0.398349   491.000000
5413.9s	16416	1              0.430769  0.509091  0.466667    55.000000
5413.9s	16417	2              0.378698  0.363636  0.371014   528.000000
5413.9s	16418	3              0.694196  0.707622  0.700845   879.000000
5413.9s	16419	4              0.403285  0.372054  0.387040   594.000000
5413.9s	16420	5              0.658140  0.680288  0.669031   416.000000
5413.9s	16421	6              0.485714  0.515974  0.500387   626.000000
5413.9s	16422	accuracy       0.518807  0.518807  0.518807     0.518807
5413.9s	16423	macro avg      0.493510  0.505963  0.499048  3589.000000
5413.9s	16424	weighted avg   0.515321  0.518807  0.516761  3589.000000


#### d3_1_w025_lr5e4

- Ngay chay:
- Commit/run id:
- Artifact:
- Descriptor storage:
- normalize_candidate_x:
- Candidate_x batch stats:
- Epoch best:
- Best val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Per-class note:
- Ket luan:
3000.2s	8668	Evaluate on TEST SET with best checkpoint
3000.2s	8669	=======================================================
3000.2s	8670	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_28042026_0529_best.pth
3000.2s	8671	--> Restored ep=39  best_val_macro_f1=0.4880
3000.5s	8672	
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:17,  3.27it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.12it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:17,  3.27it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.12it/s]
3000.8s	8673	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:17,  3.27it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.12it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.26it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.42it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.26it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.42it/s]
3001.0s	8674	
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.26it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.42it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.29it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:03, 14.44it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.29it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:03, 14.44it/s]
3001.2s	8675	
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.29it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:03, 14.44it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:01<00:02, 15.39it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 15.33it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:01<00:02, 15.39it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 15.33it/s]
3001.5s	8676	
Evaluating test set:  23%|██▎       | 13/57 [00:01<00:02, 15.39it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 15.33it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 15.76it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 16.15it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 15.76it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 16.15it/s]
3001.7s	8677	
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 15.76it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 16.15it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 16.51it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:02, 16.74it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 16.51it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:02, 16.74it/s]
3002.0s	8678	
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 16.51it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:02, 16.74it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 16.68it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 17.30it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 16.68it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 17.30it/s]
3002.2s	8679	
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 16.68it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 17.30it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 17.69it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:02<00:01, 17.04it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 17.69it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:02<00:01, 17.04it/s]
3002.4s	8680	
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 17.69it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:02<00:01, 17.04it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 16.58it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 16.44it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 16.58it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 16.44it/s]
3002.7s	8681	
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 16.58it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 16.44it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 16.64it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 16.92it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 16.64it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 16.92it/s]
3002.9s	8682	
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 16.64it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 16.92it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.63it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.37it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.63it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.37it/s]
3003.1s	8683	
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.63it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.37it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.14it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:03<00:00, 17.27it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.14it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:03<00:00, 17.27it/s]
3003.4s	8684	
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.14it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:03<00:00, 17.27it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:03<00:00, 16.99it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 17.07it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:03<00:00, 16.99it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 17.07it/s]
3003.6s	8685	
Evaluating test set:  86%|████████▌ | 49/57 [00:03<00:00, 16.99it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 17.07it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.49it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.80it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.49it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.80it/s]
3003.6s	8686	
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.49it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.80it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 15.82it/s]
3003.8s	8687	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 15.82it/s]
3003.8s	8688	
3003.8s	8689	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 15.82it/s]
3030.6s	8690	[1;34mwandb[0m: 
3030.6s	8691	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_28042026_0529[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/w8qdg3lc[0m
3030.6s	8692	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260428_052907-w8qdg3lc/logs[0m
3030.6s	8693	
3030.6s	8694	=======================================================
3030.6s	8695	TEST SET EVALUATION
3030.6s	8696	=======================================================
3030.6s	8697	--> Accuracy:    50.77%
3030.6s	8698	--> Macro F1:    0.4852
3030.6s	8699	--> Weighted F1: 0.5057
3030.6s	8700	
3030.6s	8701	--> Classification Report:
3030.6s	8702	              precision    recall  f1-score      support
3030.6s	8703	0              0.375000  0.415479  0.394203   491.000000
3030.6s	8704	1              0.363636  0.581818  0.447552    55.000000
3030.6s	8705	2              0.399527  0.320076  0.355415   528.000000
3030.6s	8706	3              0.711111  0.691695  0.701269   879.000000
3030.6s	8707	4              0.394604  0.393939  0.394271   594.000000
3030.6s	8708	5              0.579798  0.689904  0.630077   416.000000
3030.6s	8709	6              0.487310  0.460064  0.473295   626.000000
3030.6s	8710	accuracy       0.507662  0.507662  0.507662     0.507662
3030.6s	8711	macro avg      0.472998  0.507568  0.485155  3589.000000
3030.6s	8712	weighted avg   0.507325  0.507662  0.505666  3589.000000


#### d3_1_w020

- Ngay chay:
- Commit/run id:
- Artifact:
- Descriptor storage:
- normalize_candidate_x:
- Candidate_x batch stats:
- Epoch best:
- Best val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Per-class note:
- Ket luan:

Evaluate on TEST SET with best checkpoint
3637.2s	10687	=======================================================
3637.2s	10688	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_28042026_0529_best.pth
3637.2s	10689	--> Restored ep=53  best_val_macro_f1=0.4931
3637.5s	10690	
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:15,  3.56it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.57it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:15,  3.56it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.57it/s]
3637.7s	10691	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:15,  3.56it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.57it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.91it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.56it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.91it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.56it/s]
3637.9s	10692	
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.91it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.56it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.60it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.52it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.60it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.52it/s]
3638.2s	10693	
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.60it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.52it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 16.03it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.52it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 16.03it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.52it/s]
3638.4s	10694	
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 16.03it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.52it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.99it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.57it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.99it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.57it/s]
3638.6s	10695	
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.99it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 17.57it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.91it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 18.38it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.91it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 18.38it/s]
3638.8s	10696	
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 17.91it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:01, 18.38it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.47it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.70it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.47it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.70it/s]
3639.0s	10697	
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 18.47it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 18.70it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.52it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.62it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.52it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.62it/s]
3639.3s	10698	
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 18.52it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:01<00:01, 18.62it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.08it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 17.64it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.08it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 17.64it/s]
3639.5s	10699	
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 18.08it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 17.64it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 18.04it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.74it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 18.04it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.74it/s]
3639.7s	10700	
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 18.04it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.74it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.64it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.83it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.64it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.83it/s]
3639.9s	10701	
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.64it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.83it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.56it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 17.49it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.56it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 17.49it/s]
3640.2s	10702	
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.56it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 17.49it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 16.88it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 16.94it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 16.88it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 16.94it/s]
3640.4s	10703	
Evaluating test set:  86%|████████▌ | 49/57 [00:02<00:00, 16.88it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 16.94it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.33it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.53it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.33it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.53it/s]
3640.5s	10704	
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.33it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.53it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.47it/s]
3640.7s	10705	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.47it/s]
3640.7s	10706	
3640.7s	10707	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.47it/s]
3673.1s	10708	[1;34mwandb[0m: 
3673.1s	10709	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_28042026_0529[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/2141dfb2[0m
3673.1s	10710	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260428_052914-2141dfb2/logs[0m
3673.2s	10711	
3673.2s	10712	=======================================================
3673.2s	10713	TEST SET EVALUATION
3673.2s	10714	=======================================================
3673.2s	10715	--> Accuracy:    51.55%
3673.2s	10716	--> Macro F1:    0.4832
3673.2s	10717	--> Weighted F1: 0.5071
3673.2s	10718	
3673.2s	10719	--> Classification Report:
3673.2s	10720	              precision    recall  f1-score      support
3673.2s	10721	0              0.368952  0.372709  0.370821   491.000000
3673.2s	10722	1              0.433962  0.418182  0.425926    55.000000
3673.2s	10723	2              0.387534  0.270833  0.318841   528.000000
3673.2s	10724	3              0.672996  0.725825  0.698413   879.000000
3673.2s	10725	4              0.417293  0.373737  0.394316   594.000000
3673.2s	10726	5              0.610063  0.699519  0.651736   416.000000
3673.2s	10727	6              0.490196  0.559105  0.522388   626.000000
3673.2s	10728	accuracy       0.515464  0.515464  0.515464     0.515464
3673.2s	10729	macro avg      0.482999  0.488559  0.483206  3589.000000
3673.2s	10730	weighted avg   0.504242  0.515464  0.507136  3589.000000

#### d3_1_w025_no_desc_scale_lr7e4

- Ngay chay:
- Commit/run id:
- Artifact:
- Descriptor storage:
- normalize_candidate_x:
- Candidate_x batch stats:
- Epoch best:
- Best val macro F1:
- Test accuracy:
- Test macro F1:
- Test weighted F1:
- Per-class note:
- Ket luan:
	Evaluate on TEST SET with best checkpoint
5244.1s	19521	=======================================================
5244.1s	19522	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/learnable_slot_candidate_motif_gnn/learnable_slot_candidate_motif_gnn_28042026_0529_best.pth
5244.1s	19523	--> Restored ep=81  best_val_macro_f1=0.4948
5244.1s	19524	
Val:  96%|█████████▋| 55/57 [00:03<00:00, 19.04it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
5244.4s	19525	
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:16,  3.37it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.48it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:16,  3.37it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.48it/s]
5244.6s	19526	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:16,  3.37it/s]
Evaluating test set:   5%|▌         | 3/57 [00:00<00:06,  8.48it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.52it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.21it/s]
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.52it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.21it/s]
5244.8s	19527	
Evaluating test set:   9%|▉         | 5/57 [00:00<00:04, 11.52it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:03, 13.21it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.86it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.85it/s]
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.86it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.85it/s]
5245.1s	19528	
Evaluating test set:  16%|█▌        | 9/57 [00:00<00:03, 14.86it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:02, 15.85it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 16.55it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.86it/s]
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 16.55it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.86it/s]
5245.3s	19529	
Evaluating test set:  23%|██▎       | 13/57 [00:00<00:02, 16.55it/s]
Evaluating test set:  26%|██▋       | 15/57 [00:01<00:02, 16.86it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.44it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 16.50it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.44it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 16.50it/s]
5245.5s	19530	
Evaluating test set:  30%|██▉       | 17/57 [00:01<00:02, 16.44it/s]
Evaluating test set:  33%|███▎      | 19/57 [00:01<00:02, 16.50it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 16.38it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:02, 16.81it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 16.38it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:02, 16.81it/s]
5245.8s	19531	
Evaluating test set:  37%|███▋      | 21/57 [00:01<00:02, 16.38it/s]
Evaluating test set:  40%|████      | 23/57 [00:01<00:02, 16.81it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 17.03it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 17.13it/s]
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 17.03it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 17.13it/s]
5246.0s	19532	
Evaluating test set:  44%|████▍     | 25/57 [00:01<00:01, 17.03it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:01<00:01, 17.13it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 16.93it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:02<00:01, 16.48it/s]
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 16.93it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:02<00:01, 16.48it/s]
5246.3s	19533	
Evaluating test set:  51%|█████     | 29/57 [00:01<00:01, 16.93it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:02<00:01, 16.48it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 16.59it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 16.70it/s]
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 16.59it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 16.70it/s]
5246.5s	19534	
Evaluating test set:  58%|█████▊    | 33/57 [00:02<00:01, 16.59it/s]
Evaluating test set:  61%|██████▏   | 35/57 [00:02<00:01, 16.70it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 17.08it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.27it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 17.08it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.27it/s]
5246.7s	19535	
Evaluating test set:  65%|██████▍   | 37/57 [00:02<00:01, 17.08it/s]
Evaluating test set:  68%|██████▊   | 39/57 [00:02<00:01, 17.27it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.49it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.59it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.49it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.59it/s]
5246.9s	19536	
Evaluating test set:  72%|███████▏  | 41/57 [00:02<00:00, 17.49it/s]
Evaluating test set:  75%|███████▌  | 43/57 [00:02<00:00, 17.59it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.85it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 17.97it/s]
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.85it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 17.97it/s]
5247.2s	19537	
Evaluating test set:  79%|███████▉  | 45/57 [00:02<00:00, 17.85it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:02<00:00, 17.97it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:03<00:00, 17.54it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 17.16it/s]
Evaluating test set:  86%|████████▌ | 49/57 [00:03<00:00, 17.54it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 17.16it/s]
5247.4s	19538	
Evaluating test set:  86%|████████▌ | 49/57 [00:03<00:00, 17.54it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:03<00:00, 17.16it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.14it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.22it/s]
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.14it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.22it/s]
5247.4s	19539	
Evaluating test set:  93%|█████████▎| 53/57 [00:03<00:00, 17.14it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:03<00:00, 19.22it/s]
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.02it/s]
5247.6s	19540	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.02it/s]
5247.6s	19541	
5247.6s	19542	
Evaluating test set: 100%|██████████| 57/57 [00:03<00:00, 16.02it/s]
5256.1s	19543	[1;34mwandb[0m: 
5256.1s	19544	[1;34mwandb[0m: 🚀 View run [33mlearnable_slot_candidate_motif_gnn_28042026_0529[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/jq864a32[0m
5256.1s	19545	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260428_052955-jq864a32/logs[0m
5256.2s	19546	
5256.2s	19547	=======================================================
5256.2s	19548	TEST SET EVALUATION
5256.2s	19549	=======================================================
5256.2s	19550	--> Accuracy:    51.91%
5256.2s	19551	--> Macro F1:    0.5034
5256.2s	19552	--> Weighted F1: 0.5211
5256.2s	19553	
5256.2s	19554	--> Classification Report:
5256.2s	19555	              precision    recall  f1-score      support
5256.2s	19556	0              0.395393  0.419552  0.407115   491.000000
5256.2s	19557	1              0.456140  0.472727  0.464286    55.000000
5256.2s	19558	2              0.372137  0.369318  0.370722   528.000000
5256.2s	19559	3              0.732759  0.676906  0.703726   879.000000
5256.2s	19560	4              0.417563  0.392256  0.404514   594.000000
5256.2s	19561	5              0.690073  0.685096  0.687575   416.000000
5256.2s	19562	6              0.458807  0.515974  0.485714   626.000000
5256.2s	19563	accuracy       0.519086  0.519086  0.519086     0.519086
5256.2s	19564	macro avg      0.503267  0.504547  0.503379  3589.000000
5256.2s	19565	weighted avg   0.524415  0.519086  0.521069  3589.000000

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
