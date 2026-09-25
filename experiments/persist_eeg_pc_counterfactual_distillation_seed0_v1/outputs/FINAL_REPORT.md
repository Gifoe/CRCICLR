# Counterfactual distillation, seed 0: completed development experiment

## Scope and provenance

Canonical EEGNet, OpenBMI MI and SSVEP, folds 0–4. All five continuations start from the exact frozen checkpoint, use the same 10 epochs, batch order, AdamW settings, and frozen BN running state. Student inference is one native EEGNet forward, with no P/C module. Final heldout and outer dev arrays were never read. Oracle labels are privileged and are not deployable.

The source checkpoint and V3 geometry were previously refit on non-final source-session subjects, including the discovery subjects. Future-session discovery is therefore a development cross-session check, not a fully subject-independent test. No new source checkpoint or geometry was fitted here.

## Primary future-session subject-equal BA

| Task | Original | CE-only continuation | Top3 distill | Margin-Upper distill | CE-Upper logit | CE-Upper full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenBMI_MI | 0.7497 | 0.7513 | 0.7512 | 0.7501 | 0.7497 | 0.7501 |
| OpenBMI_SSVEP | 0.9343 | 0.9324 | 0.9340 | 0.9333 | 0.9340 | 0.9348 |

### Secondary discovery metrics

| Task | Student | Future-session macro-F1 | Future-session NLL | Worst-session BA |
| --- | --- | ---: | ---: | ---: |
| OpenBMI_MI | ORIGINAL | 0.7477 | 0.5373 | 0.7497 |
| OpenBMI_MI | CE_ONLY_CONTINUATION | 0.7482 | 0.5224 | 0.7485 |
| OpenBMI_MI | TOP3_DISTILL | 0.7485 | 0.5279 | 0.7467 |
| OpenBMI_MI | MARGIN_UPPER_DISTILL | 0.7473 | 0.5434 | 0.7461 |
| OpenBMI_MI | CE_UPPER_LOGIT_ONLY | 0.7469 | 0.5438 | 0.7462 |
| OpenBMI_MI | CE_UPPER_FULL | 0.7473 | 0.5434 | 0.7461 |
| OpenBMI_SSVEP | ORIGINAL | 0.9332 | 0.2220 | 0.9338 |
| OpenBMI_SSVEP | CE_ONLY_CONTINUATION | 0.9310 | 0.2223 | 0.9274 |
| OpenBMI_SSVEP | TOP3_DISTILL | 0.9327 | 0.2272 | 0.9300 |
| OpenBMI_SSVEP | MARGIN_UPPER_DISTILL | 0.9320 | 0.2326 | 0.9283 |
| OpenBMI_SSVEP | CE_UPPER_LOGIT_ONLY | 0.9329 | 0.2327 | 0.9290 |
| OpenBMI_SSVEP | CE_UPPER_FULL | 0.9337 | 0.2338 | 0.9298 |

All four distillation arms have higher (worse) future-session NLL than CE-only on both tasks. The SSVEP BA gains therefore come with worse probabilistic predictions under this fixed protocol.

## Paired biological-subject BA contrasts and recovered headroom

| Task | Method | ΔBA vs CE-only | 95% paired bootstrap CI | Matched future-session oracle headroom | Recovered fraction |
| --- | --- | ---: | --- | ---: | ---: |
| OpenBMI_MI | TOP3_DISTILL | -0.0001 | [-0.0063, +0.0067] | +0.0617 | -0.001 |
| OpenBMI_MI | MARGIN_UPPER_DISTILL | -0.0013 | [-0.0083, +0.0058] | +0.1487 | -0.008 |
| OpenBMI_MI | CE_UPPER_LOGIT_ONLY | -0.0017 | [-0.0085, +0.0050] | +0.1487 | -0.011 |
| OpenBMI_MI | CE_UPPER_FULL | -0.0013 | [-0.0083, +0.0058] | +0.1487 | -0.008 |
| OpenBMI_SSVEP | TOP3_DISTILL | +0.0016 | [+0.0003, +0.0031] | +0.0197 | +0.080 |
| OpenBMI_SSVEP | MARGIN_UPPER_DISTILL | +0.0009 | [-0.0008, +0.0027] | +0.0398 | +0.023 |
| OpenBMI_SSVEP | CE_UPPER_LOGIT_ONLY | +0.0016 | [+0.0000, +0.0036] | +0.0411 | +0.039 |
| OpenBMI_SSVEP | CE_UPPER_FULL | +0.0024 | [+0.0007, +0.0045] | +0.0411 | +0.059 |

The denominator uses matched discovery future-session oracle BA minus matched original BA. It is distinct from the earlier pooled source/future-session headroom. All other paired metrics, including F1, NLL, and worst-session BA, appear in `PAIRED_DISCOVERY_CONTRASTS.csv`.

## Teacher strength

| Task | Teacher | Train accepted % | Mean ΔCE | Mean P movement | Teacher oracle future-session BA |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenBMI_MI | TOP3 | 99.8% | +0.0972 | 0.2821 | 0.8114 |
| OpenBMI_MI | MARGIN_UPPER | 100.0% | +0.2212 | 0.7593 | 0.8983 |
| OpenBMI_MI | CE_UPPER | 100.0% | +0.2212 | 0.7592 | 0.8983 |
| OpenBMI_SSVEP | TOP3 | 67.9% | +0.0475 | 0.3000 | 0.9541 |
| OpenBMI_SSVEP | MARGIN_UPPER | 65.4% | +0.0778 | 0.5047 | 0.9742 |
| OpenBMI_SSVEP | CE_UPPER | 75.0% | +0.0809 | 0.5092 | 0.9754 |

Teacher generation and student training used inner-train labels only. For each trial, KD and P supervision were disabled unless teacher CE improved native CE by more than 1e-4. Old Top3 and margin Upper discovery subject/session BA were reproduced exactly (maximum absolute error 0). CE-Upper was evaluated only after every student had completed native discovery inference.

OpenBMI MI has two classes. For two-class softmax, cross-entropy is strictly decreasing in the true-class margin, so CE-Upper and margin Upper select the same direction-wise alpha except for numerical ties. This comparison does not provide an independent teacher objective on MI.

`KL(q_T || onehot(y))` is mathematically infinite for a softmax teacher with nonzero off-class mass. `TEACHER_TARGET_ANALYSIS.csv` records that infinity and reports finite reverse KL, entropy, true-class probability, and KL to native at T=2. A nearly one-hot CE-Upper teacher should be interpreted as extra hard-label pressure, not independent class structure.

At T=2, the observed target sharpness is:

| Task | Target | Mean true-class probability | Mean entropy (nats) |
| --- | --- | ---: | ---: |
| OpenBMI_MI | NATIVE | 0.6779 | 0.5443 |
| OpenBMI_MI | CE_UPPER | 0.7716 | 0.4753 |
| OpenBMI_SSVEP | NATIVE | 0.8917 | 0.3469 |
| OpenBMI_SSVEP | CE_UPPER | 0.9322 | 0.2736 |

## Post-hoc P alignment and incremental value

| Task | Student | Standardized P-MSE to CE-Upper oracle | Movement direction cosine |
| --- | --- | ---: | ---: |
| OpenBMI_MI | CE_ONLY_CONTINUATION | 0.2616 | +0.0282 |
| OpenBMI_MI | CE_UPPER_LOGIT_ONLY | 0.2499 | +0.0902 |
| OpenBMI_MI | CE_UPPER_FULL | 0.2460 | +0.0933 |
| OpenBMI_SSVEP | CE_ONLY_CONTINUATION | 0.6397 | -0.0020 |
| OpenBMI_SSVEP | CE_UPPER_LOGIT_ONLY | 0.6502 | -0.0174 |
| OpenBMI_SSVEP | CE_UPPER_FULL | 0.6609 | -0.0064 |

This diagnostic does not enter student inference or model selection. On MI, full training slightly lowers P-MSE relative to logit-only but does not improve BA reliably. On SSVEP, full training raises P-MSE relative to logit-only; its small BA point gain therefore does not establish that matching the oracle P representation caused the gain. The full-minus-logit paired BA and NLL intervals are in `PAIRED_DISCOVERY_CONTRASTS.csv`.

## Interpretation

- OpenBMI_MI: PC_ORACLE_DISTILLATION_SUPPORTED not established
- OpenBMI_MI: LOGIT_DISTILLATION_SUFFICIENT is compatible with the observed difference; equivalence is not proven
- OpenBMI_MI: ORACLE_HEADROOM_NOT_DISTILLABLE under this fixed protocol
- OpenBMI_MI: CONSERVATIVE_ORACLE_MORE_DISTILLABLE (point estimates only)
- OpenBMI_SSVEP: PC_ORACLE_DISTILLATION_SUPPORTED
- OpenBMI_SSVEP: LOGIT_DISTILLATION_SUFFICIENT is compatible with the observed difference; equivalence is not proven
- OpenBMI_SSVEP: LARGE_ORACLE_MORE_DISTILLABLE (point estimates only)

Representation distances and movement agreement are post-hoc discovery diagnostics only. The P target is not used during inference. Fixed epoch 10 was used for every arm without discovery selection. Checkpoint binaries and trial teacher caches remain in server runtime and are excluded from publication.
