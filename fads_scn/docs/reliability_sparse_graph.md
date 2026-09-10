# Reliability-Gated Sparse Dynamic Region Graph

This is an experimental graph variant for the existing 48x48, one-channel
ConvNeXt + spatial-attention + SCN model. It does not use landmarks, facial
action-unit labels, a second model, or an ensemble.

For regional tokens `h_i`, the graph keeps an explicit self-loop and the top-k
non-self neighbours ranked by the existing semantic-plus-geometric score. The
row-wise softmax is applied only on those retained edges. This prevents the
dense graph from mixing all eight regions for every image.

The SCN importance head predicts a *pre-fusion* reliability `alpha` from the
global visual feature. The graph representation is then scaled as:

`g = alpha * sigmoid(MLP([global feature, mean regional token, alpha]))`

`f_graph' = g * f_graph`

`alpha` is detached inside the graph gate: classification gradients cannot
learn the trivial solution of increasing reliability. It remains optimized by
the existing SCN rank objective. The model returns `graph_gain` for diagnosis.

## Required experiments

Use `scn_convnext.yaml` as the dense baseline and keep every training setting
identical. Train each variant with seeds 42, 123, and 3407; select epoch and
any TTA/bias choices on validation only, then evaluate each selected checkpoint
once on the locked test split.

Minimum ablation table:

| Variant | Dense/top-k | Reliability gate | Expected purpose |
| --- | --- | --- | --- |
| ConvNeXt + SCN | no graph | no | visual baseline |
| Dense graph + SCN | dense | no | current full model |
| Sparse graph + SCN | top-k | no | isolate sparse structure |
| Proposed graph + SCN | top-k | yes | isolate reliability conditioning |

The present configuration implements the final row. For the sparse-only row,
use `convnext_48_sparse_graph.yaml` (`graph_mode: sparse`); do not describe a
gain until the multi-seed result includes mean, standard deviation, macro-F1,
and per-class recall.
