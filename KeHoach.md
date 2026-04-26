1.
Ablation can chay truoc
C without descriptor

Config:
- Train config: `configs/hierarchical_motif_gnn_c_no_descriptor.yaml`
- Experiment config: `configs/experiments/hierarchical_motif_gnn_c_no_descriptor.yaml`

Thay doi chinh:
```yaml
use_descriptor: false
```

Muc tieu: kiem tra internal GNN tu hoc duoc bao nhieu.

Lenh goi y:
```bash
python -m scripts.run_experiment --config hierarchical_motif_gnn_c_no_descriptor --mode train_only
```
1450.7s	5816	Evaluate on TEST SET with best checkpoint
1450.7s	5817	=======================================================
1450.7s	5818	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1118_best.pth
1450.7s	5819	--> Restored ep=79  best_val_macro_f1=0.3901
1450.9s	5820	
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:12,  4.39it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:02, 23.42it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:12,  4.39it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:02, 23.42it/s]
1451.1s	5821	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:12,  4.39it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:02, 23.42it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:01, 30.71it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:00<00:01, 33.96it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:01, 30.71it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:00<00:01, 33.96it/s]
1451.4s	5822	
Evaluating test set:  21%|██        | 12/57 [00:00<00:01, 30.71it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:00<00:01, 33.96it/s]
Evaluating test set:  39%|███▊      | 22/57 [00:00<00:00, 37.17it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 34.89it/s]
Evaluating test set:  39%|███▊      | 22/57 [00:00<00:00, 37.17it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 34.89it/s]
1451.6s	5823	
Evaluating test set:  39%|███▊      | 22/57 [00:00<00:00, 37.17it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 34.89it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 37.47it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:01<00:00, 39.36it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 37.47it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:01<00:00, 39.36it/s]
1451.8s	5824	
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 37.47it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:01<00:00, 39.36it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 40.75it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 41.50it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 40.75it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 41.50it/s]
1452.1s	5825	
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 40.75it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 41.50it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 42.00it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:01<00:00, 43.99it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 42.00it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:01<00:00, 43.99it/s]
1452.1s	5826	
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 42.00it/s]
Evaluating test set:  98%|█████████▊| 56/57 [00:01<00:00, 43.99it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 34.91it/s]
1452.3s	5827	
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 34.91it/s]
1452.3s	5828	
1452.3s	5829	
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 34.91it/s]
1456.7s	5830	[1;34mwandb[0m: 
1456.7s	5831	[1;34mwandb[0m: 🚀 View run [33mhierarchical_motif_gnn_26042026_1118[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/umuud2yt[0m
1456.7s	5832	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260426_111824-umuud2yt/logs[0m
1456.8s	5833	
1456.8s	5834	=======================================================
1456.8s	5835	TEST SET EVALUATION
1456.8s	5836	=======================================================
1456.8s	5837	--> Accuracy:    46.14%
1456.8s	5838	--> Macro F1:    0.4051
1456.8s	5839	--> Weighted F1: 0.4438
1456.8s	5840	
1456.8s	5841	--> Classification Report:
1456.8s	5842	              precision    recall  f1-score     support
1456.8s	5843	0              0.353383  0.191446  0.248349   491.00000
1456.8s	5844	1              0.194030  0.472727  0.275132    55.00000
1456.8s	5845	2              0.390000  0.147727  0.214286   528.00000
1456.8s	5846	3              0.601533  0.714448  0.653146   879.00000
1456.8s	5847	4              0.332587  0.500000  0.399462   594.00000
1456.8s	5848	5              0.593137  0.581731  0.587379   416.00000
1456.8s	5849	6              0.451863  0.464856  0.458268   626.00000
1456.8s	5850	accuracy       0.461410  0.461410  0.461410     0.46141
1456.8s	5851	macro avg      0.416648  0.438991  0.405146  3589.00000
1456.8s	5852	weighted avg   0.458629  0.461410  0.443810  3589.00000
1456.8s	5853	Confusion matrix saved: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1118/confusion_matrix.png
1456.8s	5854	
1456.8s	5855	--> Figures saved to: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1118
1456.8s	5856		--> [WandB] Send File `hierarchical_motif_gnn_26042026_1118_best.pth` to cloud successfully!
1456.8s	5857	
1456.8s	5858			DONE!

2.
C-light

Config:
- Train config: `configs/hierarchical_motif_gnn_c_light.yaml`
- Experiment config: `configs/experiments/hierarchical_motif_gnn_c_light.yaml`

Thay doi chinh:
```yaml
internal_num_layers: 1
internal_out_dim: 64
```

Muc tieu: xem gain co giu duoc voi model nhe hon khong.

Lenh goi y:
```bash
python -m scripts.run_experiment --config hierarchical_motif_gnn_c_light --mode train_only
```
1315.1s	5255	Evaluate on TEST SET with best checkpoint
1315.1s	5256	=======================================================
1315.1s	5257	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123_best.pth
1315.1s	5258	--> Restored ep=63  best_val_macro_f1=0.4225
1315.3s	5259	
Val:  89%|████████▉ | 51/57 [00:01<00:00, 44.66it/s]
Val:  98%|█████████▊| 56/57 [00:01<00:00, 45.85it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
1315.3s	5260	Epoch  78/80 | loss: 1.3387  acc: 0.4880  macro_f1: 0.4616 | val_loss: 1.4653  val_acc: 0.4675  val_macro_f1: 0.4187
1315.3s	5261	          val pred_count: [382, 70, 211, 1016, 640, 505, 765]
1315.3s	5262		-!- No improvement: 15/15
1315.3s	5263		-_- Early stopping at ep=78
1315.3s	5264	
1315.3s	5265	=======================================================
1315.3s	5266	Evaluate on TEST SET with best checkpoint
1315.3s	5267	=======================================================
1315.3s	5268	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123_best.pth
1315.3s	5269	--> Restored ep=63  best_val_macro_f1=0.4225
1315.6s	5270	
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:11,  4.69it/s]
Evaluating test set:  11%|█         | 6/57 [00:00<00:02, 22.14it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:11,  4.69it/s]
Evaluating test set:  11%|█         | 6/57 [00:00<00:02, 22.14it/s]
1315.8s	5271	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:11,  4.69it/s]
Evaluating test set:  11%|█         | 6/57 [00:00<00:02, 22.14it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:01, 31.41it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:00<00:01, 36.53it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:01, 31.41it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:00<00:01, 36.53it/s]
1316.0s	5272	
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:01, 31.41it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:00<00:01, 36.53it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:00<00:00, 39.55it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 40.57it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:00<00:00, 39.55it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 40.57it/s]
1316.2s	5273	
Evaluating test set:  37%|███▋      | 21/57 [00:00<00:00, 39.55it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 40.57it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 41.88it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:00<00:00, 43.13it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 41.88it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:00<00:00, 43.13it/s]
1316.4s	5274	
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 41.88it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:00<00:00, 43.13it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 44.93it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 44.93it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 44.93it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 44.93it/s]
1316.7s	5275	
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 44.93it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 44.93it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 44.35it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 48.42it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 37.69it/s]
1316.7s	5276	
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 44.35it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 48.42it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 37.69it/s]
1316.7s	5277	
1316.7s	5278	
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 44.35it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 48.42it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 37.69it/s]
1321.3s	5279	[1;34mwandb[0m: 
1321.3s	5280	[1;34mwandb[0m: 🚀 View run [33mhierarchical_motif_gnn_26042026_1123[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/38bg643i[0m
1321.3s	5281	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260426_112341-38bg643i/logs[0m
1321.3s	5282	
1321.3s	5283	=======================================================
1321.3s	5284	TEST SET EVALUATION
1321.3s	5285	=======================================================
1321.3s	5286	--> Accuracy:    47.56%
1321.3s	5287	--> Macro F1:    0.4402
1321.3s	5288	--> Weighted F1: 0.4594
1321.3s	5289	
1321.3s	5290	--> Classification Report:
1321.3s	5291	              precision    recall  f1-score     support
1321.3s	5292	0              0.342163  0.315682  0.328390   491.00000
1321.3s	5293	1              0.450980  0.418182  0.433962    55.00000
1321.3s	5294	2              0.363636  0.143939  0.206242   528.00000
1321.3s	5295	3              0.608358  0.712173  0.656184   879.00000
1321.3s	5296	4              0.378086  0.412458  0.394525   594.00000
1321.3s	5297	5              0.572104  0.581731  0.576877   416.00000
1321.3s	5298	6              0.438144  0.543131  0.485021   626.00000
1321.3s	5299	accuracy       0.475620  0.475620  0.475620     0.47562
1321.3s	5300	macro avg      0.450496  0.446757  0.440172  3589.00000
1321.3s	5301	weighted avg   0.461524  0.475620  0.459387  3589.00000
1321.3s	5302	Confusion matrix saved: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123/confusion_matrix.png
1321.3s	5303	
1321.3s	5304	--> Figures saved to: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123
1321.3s	5305		--> [WandB] Send File `hierarchical_motif_gnn_26042026_1123_best.pth` to cloud successfully!
1321.3s	5306	
1321.3s	5307			DONE!



3.
C mean vs mean_max readout

Config mean:
- Train config: `configs/hierarchical_motif_gnn_c_readout_mean.yaml`
- Experiment config: `configs/experiments/hierarchical_motif_gnn_c_readout_mean.yaml`

Config mean_max:
- Train config: `configs/hierarchical_motif_gnn_c_readout_mean_max.yaml`
- Experiment config: `configs/experiments/hierarchical_motif_gnn_c_readout_mean_max.yaml`

Thay doi chinh:
```yaml
internal_readout: mean
internal_readout: mean_max
```

Muc tieu: kiem tra max pooling co giup bat edge/contrast manh khong.

Lenh goi y:
```bash
python -m scripts.run_experiment --config hierarchical_motif_gnn_c_readout_mean --mode train_only
1315.2s	5467	Evaluate on TEST SET with best checkpoint
1315.2s	5468	=======================================================
1315.2s	5469	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123_best.pth
1315.2s	5470	--> Restored ep=74  best_val_macro_f1=0.4245
1315.2s	5471	
Val:  89%|████████▉ | 51/57 [00:01<00:00, 46.11it/s]
Val: 100%|██████████| 57/57 [00:01<00:00, 49.76it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:10,  5.15it/s]
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:10,  5.15it/s]
1315.5s	5472	
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:10,  5.15it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:01, 26.34it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:01, 34.40it/s]
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:01, 26.34it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:01, 34.40it/s]
1315.7s	5473	
Evaluating test set:  12%|█▏        | 7/57 [00:00<00:01, 26.34it/s]
Evaluating test set:  21%|██        | 12/57 [00:00<00:01, 34.40it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:00<00:01, 39.07it/s]
Evaluating test set:  39%|███▊      | 22/57 [00:00<00:00, 41.71it/s]
Evaluating test set:  30%|██▉       | 17/57 [00:00<00:01, 39.07it/s]
Evaluating test set:  39%|███▊      | 22/57 [00:00<00:00, 41.71it/s]
1315.9s	5474	
Evaluating test set:  30%|██▉       | 17/57 [00:00<00:01, 39.07it/s]
Evaluating test set:  39%|███▊      | 22/57 [00:00<00:00, 41.71it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:00<00:00, 43.38it/s]
Evaluating test set:  56%|█████▌    | 32/57 [00:00<00:00, 44.58it/s]
Evaluating test set:  47%|████▋     | 27/57 [00:00<00:00, 43.38it/s]
Evaluating test set:  56%|█████▌    | 32/57 [00:00<00:00, 44.58it/s]
1316.1s	5475	
Evaluating test set:  47%|████▋     | 27/57 [00:00<00:00, 43.38it/s]
Evaluating test set:  56%|█████▌    | 32/57 [00:00<00:00, 44.58it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:00<00:00, 44.07it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:01<00:00, 45.44it/s]
Evaluating test set:  65%|██████▍   | 37/57 [00:00<00:00, 44.07it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:01<00:00, 45.44it/s]
1316.3s	5476	
Evaluating test set:  65%|██████▍   | 37/57 [00:00<00:00, 44.07it/s]
Evaluating test set:  74%|███████▎  | 42/57 [00:01<00:00, 45.44it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:01<00:00, 46.42it/s]
Evaluating test set:  91%|█████████ | 52/57 [00:01<00:00, 46.65it/s]
Evaluating test set:  82%|████████▏ | 47/57 [00:01<00:00, 46.42it/s]
Evaluating test set:  91%|█████████ | 52/57 [00:01<00:00, 46.65it/s]
1316.4s	5477	
Evaluating test set:  82%|████████▏ | 47/57 [00:01<00:00, 46.42it/s]
Evaluating test set:  91%|█████████ | 52/57 [00:01<00:00, 46.65it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 39.76it/s]
1316.6s	5478	
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 39.76it/s]
1316.6s	5479	
1316.6s	5480	
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 39.76it/s]
1320.5s	5481	[1;34mwandb[0m: 
1320.5s	5482	[1;34mwandb[0m: 🚀 View run [33mhierarchical_motif_gnn_26042026_1123[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/muall1gx[0m
1320.5s	5483	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260426_112313-muall1gx/logs[0m
1320.6s	5484	
1320.6s	5485	=======================================================
1320.6s	5486	TEST SET EVALUATION
1320.6s	5487	=======================================================
1320.6s	5488	--> Accuracy:    47.87%
1320.6s	5489	--> Macro F1:    0.4473
1320.6s	5490	--> Weighted F1: 0.4666
1320.6s	5491	
1320.6s	5492	--> Classification Report:
1320.6s	5493	              precision    recall  f1-score      support
1320.6s	5494	0              0.346793  0.297352  0.320175   491.000000
1320.6s	5495	1              0.436364  0.436364  0.436364    55.000000
1320.6s	5496	2              0.339683  0.202652  0.253855   528.000000
1320.6s	5497	3              0.637605  0.690557  0.663026   879.000000
1320.6s	5498	4              0.380471  0.380471  0.380471   594.000000
1320.6s	5499	5              0.521336  0.675481  0.588482   416.000000
1320.6s	5500	6              0.458626  0.522364  0.488424   626.000000
1320.6s	5501	accuracy       0.478685  0.478685  0.478685     0.478685
1320.6s	5502	macro avg      0.445840  0.457892  0.447257  3589.000000
1320.6s	5503	weighted avg   0.463655  0.478685  0.466593  3589.000000
1320.6s	5504	Confusion matrix saved: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123/confusion_matrix.png
1320.6s	5505	
1320.6s	5506	--> Figures saved to: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123
1320.6s	5507		--> [WandB] Send File `hierarchical_motif_gnn_26042026_1123_best.pth` to cloud successfully!
1320.6s	5508	
1320.6s	5509			DONE!





python -m scripts.run_experiment --config hierarchical_motif_gnn_c_readout_mean_max --mode train_only
```

1321.2s	5555	=======================================================
1321.2s	5556	Evaluate on TEST SET with best checkpoint
1321.2s	5557	=======================================================
1321.2s	5558	--> Loading checkpoint: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/checkpoints/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123_best.pth
1321.2s	5559	--> Restored ep=77  best_val_macro_f1=0.4259
1321.2s	5560	
Val:  91%|█████████ | 52/57 [00:01<00:00, 46.14it/s]
                                                    

Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
1321.8s	5561	
Evaluating test set:   0%|          | 0/57 [00:00<?, ?it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:11,  4.90it/s]
Evaluating test set:  11%|█         | 6/57 [00:00<00:02, 23.36it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:01, 32.56it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:00<00:01, 36.97it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:00<00:00, 38.86it/s]
Evaluating test set:   2%|▏         | 1/57 [00:00<00:11,  4.90it/s]
Evaluating test set:  11%|█         | 6/57 [00:00<00:02, 23.36it/s]
1321.8s	5562	
Evaluating test set:   2%|▏         | 1/57 [00:00<00:11,  4.90it/s]
Evaluating test set:  11%|█         | 6/57 [00:00<00:02, 23.36it/s]
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:01, 32.56it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:00<00:01, 36.97it/s]
1321.9s	5563	
Evaluating test set:  19%|█▉        | 11/57 [00:00<00:01, 32.56it/s]
Evaluating test set:  28%|██▊       | 16/57 [00:00<00:01, 36.97it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 39.33it/s]
Evaluating test set:  37%|███▋      | 21/57 [00:00<00:00, 38.86it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 39.33it/s]
1322.1s	5564	
Evaluating test set:  37%|███▋      | 21/57 [00:00<00:00, 38.86it/s]
Evaluating test set:  46%|████▌     | 26/57 [00:00<00:00, 39.33it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 40.64it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:00<00:00, 43.10it/s]
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 40.64it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:00<00:00, 43.10it/s]
1322.3s	5565	
Evaluating test set:  54%|█████▍    | 31/57 [00:00<00:00, 40.64it/s]
Evaluating test set:  63%|██████▎   | 36/57 [00:00<00:00, 43.10it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 44.73it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 45.96it/s]
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 44.73it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 45.96it/s]
1322.5s	5566	
Evaluating test set:  72%|███████▏  | 41/57 [00:01<00:00, 44.73it/s]
Evaluating test set:  81%|████████  | 46/57 [00:01<00:00, 45.96it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 46.47it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 49.76it/s]
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 46.47it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 49.76it/s]
1322.5s	5567	
Evaluating test set:  89%|████████▉ | 51/57 [00:01<00:00, 46.47it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 49.76it/s]
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 37.61it/s]
1322.7s	5568	
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 37.61it/s]
1322.7s	5569	
1322.7s	5570	
Evaluating test set: 100%|██████████| 57/57 [00:01<00:00, 37.61it/s]
1326.6s	5571	[1;34mwandb[0m: 
1326.6s	5572	[1;34mwandb[0m: 🚀 View run [33mhierarchical_motif_gnn_26042026_1123[0m at: [34mhttps://wandb.ai/phucga15062005/FER2013/runs/l9z3q2vt[0m
1326.6s	5573	[1;34mwandb[0m: Find logs at: [1;35mwandb/run-20260426_112341-l9z3q2vt/logs[0m
1326.6s	5574	
1326.6s	5575	=======================================================
1326.6s	5576	TEST SET EVALUATION
1326.6s	5577	=======================================================
1326.6s	5578	--> Accuracy:    46.78%
1326.6s	5579	--> Macro F1:    0.4414
1326.6s	5580	--> Weighted F1: 0.4620
1326.6s	5581	
1326.6s	5582	--> Classification Report:
1326.6s	5583	              precision    recall  f1-score      support
1326.6s	5584	0              0.340426  0.293279  0.315098   491.000000
1326.6s	5585	1              0.356164  0.472727  0.406250    55.000000
1326.6s	5586	2              0.340237  0.217803  0.265589   528.000000
1326.6s	5587	3              0.645857  0.647327  0.646591   879.000000
1326.6s	5588	4              0.347518  0.412458  0.377213   594.000000
1326.6s	5589	5              0.588235  0.625000  0.606061   416.000000
1326.6s	5590	6              0.440165  0.511182  0.473023   626.000000
1326.6s	5591	accuracy       0.467818  0.467818  0.467818     0.467818
1326.6s	5592	macro avg      0.436943  0.454254  0.441404  3589.000000
1326.6s	5593	weighted avg   0.462738  0.467818  0.461950  3589.000000
1326.6s	5594	Confusion matrix saved: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123/confusion_matrix.png
1326.6s	5595	
1326.6s	5596	--> Figures saved to: /kaggle/working/sgu-2026-facial-expression-recognition/outputs/figures/hierarchical_motif_gnn/hierarchical_motif_gnn_26042026_1123
1326.6s	5597		--> [WandB] Send File `hierarchical_motif_gnn_26042026_1123_best.pth` to cloud successfully!
1326.6s	5598	
1326.6s	5599			DONE!

4.
B chay lai tren artifact moi

Config:
- Experiment config: `configs/experiments/pixel_motif_baseline_b_v3_artifact.yaml`
- Training config van dung: `configs/pixel_motif_guided_gnn_motif_norm.yaml`

Thay doi chinh:
```yaml
pixel_motif_dir: /kaggle/working/artifacts/pixel_motif_dataset_v3_hierarchical
```

Dam bao B va C so sanh cung artifact. Neu B moi van quanh 0.419-0.42 macro F1, thi gain cua C chac chan.

Lenh goi y:
```bash
python -m scripts.run_experiment --config pixel_motif_baseline_b_v3_artifact --mode train_only
```
