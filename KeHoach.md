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
python -m scripts.run_experiment --config hierarchical_motif_gnn_c_readout_mean_max --mode train_only
```

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
