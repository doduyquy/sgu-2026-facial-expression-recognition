# Project Map - Where To Edit

File này là bản đồ codebase để AI hoặc người mới biết cần mở file nào, class/hàm nào khi triển khai. Đây là file nên đọc đầu tiên khi muốn sửa code.

## 1. Workflow chính

```text
kaggle_pixel_motif_end_to_end.ipynb
-> scripts/run_experiment.py
-> src/pipeline/experiment_runner.py
-> src/pipeline/artifact_builder.py
-> scripts/train.py
```

Notebook chỉ đổi:

```python
EXPERIMENT = "hierarchical_motif_gnn_c"
```

hoặc:

```python
EXPERIMENT = "pixel_motif_baseline_b"
```

## 2. Khi muốn thêm model mới

Sửa/thêm:

```text
src/models/<new_model>.py
src/models/__init__.py
configs/<new_model>.yaml              # nếu cần model config riêng
configs/experiments/<experiment>.yaml
```

Không sửa:

```text
kaggle_pixel_motif_end_to_end.ipynb
scripts/run_experiment.py
src/pipeline/artifact_builder.py
src/pipeline/experiment_runner.py
```

trừ khi thay đổi data pipeline.

## 3. Experiment runner

### `scripts/run_experiment.py`

Thin CLI entrypoint.

Hàm chính:

```text
main()
```

Nhiệm vụ:

```text
parse CLI args
call src.pipeline.run_experiment(...)
```

### `src/pipeline/experiment_runner.py`

Điều phối một experiment từ config.

Hàm quan trọng:

```text
load_experiment_config(config_name_or_path)
run_experiment(...)
debug_hierarchical_batch(train_cfg, paths)
train_model(train_cfg, paths, epochs, no_wandb)
zip_outputs(outputs_cfg, experiment_name)
```

Khi cần đổi logic cấp experiment, sửa file này.

Ví dụ:

```text
thêm phase validate-only
đổi cách zip output
thêm pre-train sanity check chung
```

## 4. Artifact/data builder

### `src/pipeline/artifact_builder.py`

API build/reuse artifacts. Đây là nơi thay thế các orchestration script cũ.

Hàm quan trọng:

```text
ensure_pixel_motif_artifacts(data_cfg, csv_root, out_root)
resolve_artifact_paths(data_cfg, out_root_override)
normalize_data_config(data_cfg)
find_csv_root(search_root)
resolve_csv_root(value)
build_graph_repo(...)
build_candidates(...)
build_motif_bank(...)
build_pixel_motif_dataset(...)
```

Chỉ sửa file này nếu thay đổi data recipe/stage.

## 5. Atomic data scripts

Pipeline API gọi trực tiếp các script này.

### `scripts/build_graph_repository.py`

Build canonical graph repo từ CSV.

Liên quan:

```text
data/raw_fer_dataset.py
data/shared_graph_builder.py
data/canonical_graph_builder.py
data/graph_repository.py
```

### `scripts/precompute_pixel_candidate_subgraphs.py`

Build candidate subgraphs và descriptors.

Hàm chính:

```text
_process_split(...)
main()
```

Dùng:

```text
src/motif_v2/topology.build_candidate_topologies
src/motif_v2/topology.descriptor_from_topology
```

### `scripts/build_pixel_motif_bank.py`

Build motif bank theo emotion class.

Hàm quan trọng:

```text
_collect_sampled_descriptors(...)
_cluster_descriptors(...)
_make_exemplars(...)
main()
```

### `scripts/precompute_pixel_motif_dataset.py`

Chọn top-K motif subgraphs per image.

Hàm quan trọng:

```text
_pad_selected_nodes(...)
_process_split(...)
main()
```

Dùng:

```text
src/motif_v2.matching.greedy_select_with_coverage
src/motif_v2.topology.build_directed_knn_edges
src/motif_v2.topology.build_directed_knn_rich_edges
```

## 6. Data types và graph repo

### `data/graph_types.py`

Dataclasses chính:

```text
SharedGraphStructure
PixelGraphSample
ResolvedPixelGraph
```

### `data/graph_repository.py`

Classes:

```text
GraphRepositoryWriter
GraphRepositoryReader
```

### `data/graph_resolver.py`

Class:

```text
GraphResolver
```

Hàm/method quan trọng:

```text
GraphResolver.resolve(sample)
```

### `data/chunked_graph_dataset.py`

Dataset đọc graph repo theo chunk.

Class:

```text
ChunkedGraphDataset
```

## 7. Pixel motif dataset loader

### `src/data/pixel_motif_dataset.py`

Class:

```text
PixelMotifDataset
```

Hàm quan trọng:

```text
remap_local_edges(...)
build_subgraph_tensor_from_node_indices(...)
pad_selected_subgraphs(...)
```

Vai trò:

```text
load train/val/test_pixel_motif.pt
normalize descriptor x nếu bật
trả batch keys cũ cho baseline B
dựng sub_x/sub_adj/sub_node_mask cho C nếu return_subgraph_tensors=true
```

### `src/data/dataloader.py`

Hàm quan trọng:

```text
build_dataloader(config, graph_repo_path)
_build_pixel_motif_loaders(...)
collate_fn_pixel_motif(batch)
```

Khi thêm key mới vào dataset batch, thường cần sửa:

```text
PixelMotifDataset.__getitem__
collate_fn_pixel_motif
```

## 8. Models

### Registry: `src/models/__init__.py`

Biến:

```text
MODEL_REGISTRY
```

Hàm:

```text
get_model(name, config, **kwargs)
```

Khi thêm model mới, bắt buộc register ở đây.

### Baseline B: `src/models/motif_guided_gnn.py`

Classes:

```text
MotifGraphSAGELayer
MotifGuidedGNN
```

Forward expects batch keys:

```text
x
edge_index
edge_attr
edge_valid
mask
match_scores
matched_class
motif_score_vector
```

### Version C internal encoder: `src/models/internal_subgraph_encoder.py`

Classes:

```text
DenseGraphSAGELayer
InternalPixelSubgraphEncoder
```

Input:

```text
sub_x          [B, K, N, 7]
sub_adj        [B, K, N, N]
sub_node_mask  [B, K, N]
```

Output:

```text
z_internal [B, K, internal_out_dim]
```

### Version C model: `src/models/hierarchical_motif_gnn.py`

Class:

```text
HierarchicalMotifGNN
```

Forward expects:

```text
x
sub_x
sub_adj
sub_node_mask
mask
match_scores
matched_class
matched_disc_score
motif_score_vector
edge_index
edge_valid
edge_attr
```

## 9. Training

### `scripts/train.py`

Entry train/evaluate script.

Hàm quan trọng:

```text
resolve_device()
main()
```

Runner gọi file này với đúng:

```text
--config
--env
--pixel_motif_dataset_path
--graph_repo_path
--epochs
```

### `src/training/trainer.py`

Class:

```text
Trainer
```

Methods:

```text
train_one_epoch()
validate()
fit()
_move_batch_to_device(batch)
_forward_batch(batch, x)
```

### `src/training/losses.py`

Hàm/classes:

```text
build_loss(config)
WeightedCrossEntropy
compute_class_weights(...)
```

### `src/training/optimizer.py`

Hàm:

```text
build_optimizer(model, config)
build_scheduler(optimizer, config)
```

## 10. Evaluation

### `src/evaluation/evaluator.py`

Hàm:

```text
evaluate_and_show(model, test_loader, device, save_dir, config)
```

### `src/evaluation/metrics.py`

Hàm:

```text
compute_classification_metrics(...)
plot_confusion_matrix(...)
```

## 11. Debug/sanity

### `scripts/debug_hierarchical_batch.py`

Hàm:

```text
main()
```

Lệnh:

```bash
python -m scripts.run_experiment \
  --config hierarchical_motif_gnn_c \
  --out_root artifacts \
  --debug_only \
  --no_wandb
```

Expected:

```text
x              [2, 32, 41]
sub_x          [2, 32, 25, 7]
sub_adj        [2, 32, 25, 25]
sub_node_mask  [2, 32, 25]
logits         [2, 7]
```

## 12. Config files

### Experiment configs

```text
configs/experiments/pixel_motif_baseline_b.yaml
configs/experiments/hierarchical_motif_gnn_c.yaml
```

### Model configs

```text
configs/pixel_motif_guided_gnn_motif_norm.yaml
configs/hierarchical_motif_gnn.yaml
```

### Shared configs

```text
configs/base.yaml
configs/env.yaml
configs/graph_config.py
```

## 13. Legacy

Legacy files are in:

```text
legacy/
```

Do not edit legacy for the current workflow unless deliberately restoring an older experiment.

## 14. Common implementation tasks

### Add a new model

Edit:

```text
src/models/<new_model>.py
src/models/__init__.py
configs/experiments/<new_experiment>.yaml
```

Maybe add:

```text
configs/<new_model>.yaml
```

### Add new batch tensor

Edit:

```text
src/data/pixel_motif_dataset.py
src/data/dataloader.py
src/models/<model>.py
scripts/debug_hierarchical_batch.py
```

### Change motif selection

Edit:

```text
src/motif_v2/matching.py
scripts/precompute_pixel_motif_dataset.py
```

### Change candidate topology/descriptor

Edit:

```text
src/motif_v2/topology.py
scripts/precompute_pixel_candidate_subgraphs.py
```

### Change node/edge features

Edit:

```text
configs/base.yaml
configs/graph_config.py
data/canonical_graph_builder.py
data/shared_graph_builder.py
```

Then rebuild artifacts from graph_repo stage.
