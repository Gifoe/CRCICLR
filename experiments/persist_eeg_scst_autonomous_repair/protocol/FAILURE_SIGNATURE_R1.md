# Failure signature: Repair-R1 task-protected Bures

Repair-R1 was evaluated on all 30 source-only units (OpenBMI and WBCIC,
5 folds x 3 seeds). No future, outer, sealed, or S3 resource was opened.

## Observed signature

* Target transport was numerically realized.  Target-distance 95% CI lower
  bounds were `0.2157636062` (OpenBMI) and `0.3231461537` (WBCIC); target-NLL
  CI lower bounds were `3.2090889804` and `4.5135538690`.
* Transport magnitude was not negligible: median displacement/local-radius was
  `0.4523761906` (OpenBMI) and `0.6031140089` (WBCIC).
* Candidate validity was nontrivial (`0.4161611671` mean survival), but usable
  coverage was below the preregistered 0.50 threshold (`0.4036666667` and
  `0.4095274475`).
* Class fidelity remained far below the 0.90 threshold (`0.3129895833` and
  `0.3263749983`). The protected one-dimensional projection therefore did not
  remove task-semantic contamination robustly.
* The pooled non-negative biological-subject fraction was `0.4057971014`,
  below the 0.60 threshold.
* All six preregistered recipes failed the utility checks.  The best OpenBMI
  primary delta was `-0.0018253968`; the best WBCIC delta was `-0.0016288905`.
  Every recipe had non-positive paired CI lower bounds versus ERM and matched
  random, and mean delta versus Mixup was negative.

## Diagnosis

The R1 projection reached target affinity but remained globally conditioned and
class-unsafe.  The failure is therefore not weak transport (F1/F2); it is a
combination of task-semantic contamination and a distribution-model mismatch:
the global subject-style Bures map produces heterogeneous local candidates that
are rejected or class-unsafe.  R1 also remained indistinguishable from its
matched random control in utility.

This signature motivates exactly one next hypothesis: replace the global
Gaussian/Bures operator with a source-only local, low-rank target-conditional
operator while retaining the orthogonal residual.  No thresholds, folds,
seeds, or target claims are changed.

Terminal: `R1_SOURCE_GATE_FAILED`.
