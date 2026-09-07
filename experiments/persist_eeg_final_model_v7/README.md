# PERSIST-EEG V7: future-utility meta-adaptation

V7 is a one-seed, adaptive exploratory study. It is not the final unified
PERSIST method claimed in the original research target. The terminal state is
`V7_SCIENTIFIC_EXHAUSTION`, the secondary state is
`V7_PERSIST_UTILITY_SIGNAL_FOUND`, and `READY_FOR_OUTER_FREEZE=false`.
The WBCIC outer cohort was never opened, enumerated, featurized, or scored
(`OUTER_TEST_USED=false`).

## Direct result

The best legal method on both benchmarks is a **generic**, fixed blend of the
V6 anchor and a compact EEG-Conformer history-fitted head. It is not
PERSIST-Meta.

| Benchmark | Strongest fair V7 method | BA | Delta vs V6 strong anchor (paired subject-bootstrap 95% CI) | Requested matched target |
|---|---|---:|---:|---:|
| OpenBMI S1 -> S2, 54 subjects | `ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD` | 83.778% | +0.574 pp `[-0.630, +1.685]` | 88.204% |
| WBCIC S1/S2 -> S3 development, 41 subjects | `ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD` | 82.497% | +0.415 pp `[-0.195, +1.037]` | 87.082% |

Neither generic gain is statistically resolved. OpenBMI exceeds the old
EEGNet reference by 8.481 pp, but it does not exceed the strongest matched
anchor by 5 pp. WBCIC exceeds the old EEGNet reference by only 3.061 pp. The
dual-benchmark target therefore failed.

Within the capacity-matched initial controller family, adding P/U/D/G/R gives
small positive point estimates over META-GENERIC:

| Benchmark | PERSIST minus matched META-GENERIC | 95% CI | Interpretation |
|---|---:|---:|---|
| OpenBMI | +0.093 pp | `[-0.130, +0.333]` | unresolved |
| WBCIC development | +0.233 pp | `[-0.316, +0.843]` | unresolved |

PERSIST did not consistently reduce harmful adaptations or improve the worst
subject. It also remained below the strongest generic Conformer blend.

## What the utility result does and does not show

P/U/D/G/R improved mean cross-fitted utility prediction relative to the
capacity-matched generic context in this development analysis. Across the 15
matched fold/controller configurations per benchmark, PERSIST improved R2 in
13/15 OpenBMI configurations and 14/15 WBCIC configurations. The averaged
metrics were:

| Benchmark | Context | R2 | Pearson | Sign accuracy | Outcome Pearson | Outcome sign accuracy |
|---|---|---:|---:|---:|---:|---:|
| OpenBMI | generic | -0.0265 | 0.1718 | 0.5781 | -0.1797 | 0.4728 |
| OpenBMI | P/U/D/G/R | 0.0280 | 0.2191 | 0.5883 | -0.0551 | 0.5370 |
| WBCIC development | generic | 0.0576 | 0.2957 | 0.6013 | 0.4919 | 0.6192 |
| WBCIC development | P/U/D/G/R | 0.1118 | 0.3394 | 0.6115 | 0.5346 | 0.6396 |

These configurations reuse the same development subjects and are not 15
independent replications. OpenBMI transfer to held-out outcomes is particularly
weak. The correct conclusion is a development-only mechanistic signal, not a
confirmed predictive or performance result.

## Legal protocol

- OpenBMI uses the frozen five subject folds: Session 1 labels form legal target
  history and Session 2 labels are scoring-only for outcome subjects.
- WBCIC uses 41 authorized development subjects: Sessions 1/2 labels form legal
  target history and Session 3 labels are scoring-only for outcome subjects.
- Within each fold, model-fit/discovery subjects provide legal
  history-to-future meta episodes. Outcome-subject future labels do not fit or
  select the within-run controller, threshold, scale, adapter, or backbone.
- Development outcomes were observed between major structural iterations.
  Consequently, every V7 result is adaptive/exploratory and must not be
  reported as confirmatory.
- Only seed `20260820` was run. No multi-seed robustness claim is supported.

## Structural search retained in the audit

Nine major families were evaluated: coarse Meta-SGD/projected updates;
calibration, ridge, LDA, and prototype actions; history Euclidean alignment;
filter-bank log-variance; compact EEG-Conformer; class-conditional session
alignment; a low-rank history hypernetwork; risk-aware generic/PERSIST utility
controllers; and multi-backbone mixture/headroom analysis.

History EA, class-conditional alignment, FBC, and the low-rank hypernetwork
were negative. The compact Conformer added modest generic diversity. The
outcome-only subject oracle over all new experts reached only 85.259% on
OpenBMI and 83.461% on WBCIC, leaving +2.056 pp and +1.379 pp over the anchors.
This is below the +5 pp target even before requiring a prospective router.

The synthetic positive control achieves 100% adaptable sensitivity, protected
specificity, and harmful rejection. It validates wiring only; it is explicitly
not real EEG evidence.

## Reproduction

The recorded server environment used Python 3.11.15, PyTorch 2.11.0+cu128,
scikit-learn 1.9.0, NumPy 2.4.4, pandas 3.0.5, and an RTX 5090. Run commands
from this experiment directory in the order recorded below:

```powershell
python code/protocol/bootstrap_protocol.py
python code/backbones/extract_openbmi_episode_cache.py
python code/meta_learning/run_initial_meta.py
python code/backbones/build_raw_ea_cache.py
python code/backbones/train_ea_eegnet.py --benchmark openbmi --variant identity --epochs <locked_epochs>
python code/backbones/train_ea_eegnet.py --benchmark openbmi --variant ea --epochs <locked_epochs>
python code/backbones/train_structural_backbone.py --benchmark <openbmi|wbcic> --architecture <locked_architecture> --epochs <locked_epochs>
python code/meta_learning/run_hypernetwork_meta.py
python code/persist/run_positive_control.py
python code/evaluation/audit_new_expert_headroom.py
python code/evaluation/finalize_v7.py
```

The repository intentionally excludes raw EEG, `outputs/cache/`, checkpoints,
and large trial-prediction CSVs. The runtime paths can be overridden with
`PERSIST_V6_RUNTIME`, `PERSIST_STAGE0_REPO`, `PERSIST_WBCIC_SOURCE_REPO`, and
`PERSIST_V5_OUTPUTS`; see `code/common.py`. Exact epoch counts and training
histories are retained in `outputs/diagnostics/*_TRAINING.csv`.

Authoritative outputs are `outputs/FINAL_DECISION.json`,
`outputs/SCIENTIFIC_REPORT.md`, and
`outputs/final_candidate/DEVELOPMENT_RESULTS.json`. The hypothesis history and
failed alternatives are retained under `outputs/research_log/`.
