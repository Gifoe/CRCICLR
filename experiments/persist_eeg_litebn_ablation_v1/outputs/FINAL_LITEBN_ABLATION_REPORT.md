# Independent LiteBN ablation report

Seed 0, five folds. All ablation cells used the exact frozen loaders, train-only normalizers, losses, and subject-equal inner-validation checkpoint metric.

User-approved acceleration: maximum 60 epochs, minimum checkpoint epoch 10, and early stopping after 8 epochs without strict eligible inner-validation BA improvement.

Execution-order deviation: the user explicitly paused TFFormer after its saved checkpoint state and requested LiteBN next at full GPU allocation. This does not change either independent model protocol.

WBCIC deviation: all checkpoints were frozen from source/development data before direct comparison on the already-accessed true-outer future-session S2 ten-subject set. This is an exposed ablation benchmark, not untouched confirmation.

| Variant | OpenBMI-MI BA | OpenBMI-ERP BA | OpenBMI-SSVEP BA | WBCIC-MI BA | Mean BA | Mean delta vs Full (pp) |
|---|---:|---:|---:|---:|---:|---:|
| B0_FULL | 0.7449 | 0.8465 | 0.9070 | 0.8093 | 0.8269 | +0.000 |
| B1_SINGLE_SCALE_TEMPORAL | 0.7271 | 0.8501 | 0.8973 | 0.8071 | 0.8204 | -0.653 |
| B2_SPATIAL_D1 | 0.6954 | 0.8534 | 0.9021 | 0.8047 | 0.8139 | -1.300 |
| B3_ONE_STAGE_BACKEND | 0.7377 | 0.8381 | 0.8731 | 0.7890 | 0.8095 | -1.743 |
| B4_BN_TO_GN | 0.7010 | 0.8581 | 0.8929 | 0.8087 | 0.8152 | -1.177 |

## Per-task paired subject analysis

### B1_SINGLE_SCALE_TEMPORAL

| Task | Ablation BA | Delta vs Full (pp) | Paired 95% bootstrap CI (pp) |
|---|---:|---:|---:|
| OpenBMI-MI | 0.7271 | -1.771 | [-4.529, +0.757] |
| OpenBMI-ERP | 0.8501 | +0.352 | [-0.190, +0.937] |
| OpenBMI-SSVEP | 0.8973 | -0.971 | [-4.486, +2.157] |
| WBCIC-MI | 0.8071 | -0.220 | [-1.250, +0.910] |

### B2_SPATIAL_D1

| Task | Ablation BA | Delta vs Full (pp) | Paired 95% bootstrap CI (pp) |
|---|---:|---:|---:|
| OpenBMI-MI | 0.6954 | -4.943 | [-7.086, -2.843] |
| OpenBMI-ERP | 0.8534 | +0.688 | [+0.176, +1.223] |
| OpenBMI-SSVEP | 0.9021 | -0.486 | [-1.671, +0.629] |
| WBCIC-MI | 0.8047 | -0.460 | [-1.650, +0.630] |

### B3_ONE_STAGE_BACKEND

| Task | Ablation BA | Delta vs Full (pp) | Paired 95% bootstrap CI (pp) |
|---|---:|---:|---:|
| OpenBMI-MI | 0.7377 | -0.714 | [-2.514, +1.100] |
| OpenBMI-ERP | 0.8381 | -0.844 | [-1.130, -0.577] |
| OpenBMI-SSVEP | 0.8731 | -3.386 | [-5.471, -1.514] |
| WBCIC-MI | 0.7890 | -2.030 | [-3.640, -0.540] |

### B4_BN_TO_GN

| Task | Ablation BA | Delta vs Full (pp) | Paired 95% bootstrap CI (pp) |
|---|---:|---:|---:|
| OpenBMI-MI | 0.7010 | -4.386 | [-6.243, -2.543] |
| OpenBMI-ERP | 0.8581 | +1.153 | [+0.581, +1.749] |
| OpenBMI-SSVEP | 0.8929 | -1.414 | [-2.715, -0.314] |
| WBCIC-MI | 0.8087 | -0.060 | [-0.660, +0.450] |

The results support conclusions only about the multi-scale temporal stem, filter-specific spatial richness, hierarchical temporal backend, and normalization choice.
