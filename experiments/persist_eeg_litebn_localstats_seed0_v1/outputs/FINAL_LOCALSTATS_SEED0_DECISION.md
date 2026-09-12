# LocalStats seed0 decision

LOCALSTATS_SEED0_NO_BROAD_SIGNAL

1. Historical LiteBN: shared architecture/initialization, exact manifests or original deterministic task batches, normalizer/loss/optimizer/selection retained. Original baseline metrics replayed within 1e-10. Windows runtime is not the historical Linux build.
2. Parameter counts (baseline -> LS):
   OpenBMI_MI: 47978 -> 54634; +6656 (13.873%).
   OpenBMI_ERP: 47978 -> 54634; +6656 (13.873%).
   OpenBMI_SSVEP: 48108 -> 54764; +6656 (13.836%).
   WBCIC_MI: 47786 -> 54442; +6656 (13.929%).
3. Initial function equivalence: 20/20 PASS, including eval/train and FP32/AMP; zero prediction mismatches.

| Task | LiteBN BA | LS BA | Outer delta pp | Delta F1 pp | Positive folds | Internal heldout delta pp |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 0.791500 | 0.769500 | -2.200 | -2.511 | 0/5 | -0.671 |
| OpenBMI_ERP | 0.832553 | 0.830341 | -0.221 | -0.785 | 1/5 | -0.048 |
| OpenBMI_SSVEP | 0.892000 | 0.893500 | +0.150 | -0.044 | 3/5 | -0.214 |
| WBCIC_MI | 0.787299 | 0.783491 | -0.381 | -0.297 | 2/5 | -0.110 |

4-5. Outer and current internal heldout deltas: table above. Heldout is previously exposed diagnostic, never untouched final test.
6. Positive tasks: 1/4; positive folds 6/20. Four-of-four: False.
7. Worst task -2.200 pp; equal-task mean -0.663 pp; median -0.301 pp.
8. Readout contribution (trial-weighted outer, diagnostics not used for selection):
OpenBMI_MI: base_norm=2.229670, stats_norm=0.545737, stats_base_ratio=0.252616, mu_contribution_norm=0.464842, logvar_contribution_norm=0.607021
OpenBMI_ERP: base_norm=10.406153, stats_norm=4.143719, stats_base_ratio=0.449475, mu_contribution_norm=2.074969, logvar_contribution_norm=2.805867
OpenBMI_SSVEP: base_norm=1.778023, stats_norm=0.305417, stats_base_ratio=0.226968, mu_contribution_norm=0.159104, logvar_contribution_norm=0.208773
WBCIC_MI: base_norm=1.738290, stats_norm=0.480503, stats_base_ratio=0.300098, mu_contribution_norm=0.260729, logvar_contribution_norm=0.353773
9. No evidence for broad-upgrade continuation; do not expand seeds/tasks.
10. No new sealed test accessed. No seed1/2 or ablation was run.

```text
SEED = 0
ARCHITECTURE = EXACT_LITEBN_PLUS_SHALLOW_LOCAL_STATS
LOCAL_STATS_LOCATION = BEFORE_FIRST_TEMPORAL_POOLING
LOCAL_STATS = 8BIN_MEAN_PLUS_LOGVAR
LOW_RANK = 8
AUXILIARY_LOSS = NONE
NEW_BN = NONE
NEW_CLASSIFIER = NO
ENSEMBLE = NO
ROUTING = NO
TEST_TIME_ADAPTATION = NO
NEW_SEALED_FINAL_DATA_ACCESSED = NO
```
