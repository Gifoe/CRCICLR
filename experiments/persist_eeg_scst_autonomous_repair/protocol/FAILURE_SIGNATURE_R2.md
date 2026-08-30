# Failure signature: Repair-R2 low-rank local OT

Repair-R2 was evaluated on all 30 source-only units (OpenBMI and WBCIC,
5 folds x 3 seeds). No future, outer, sealed, or S3 resource was opened.

## Observed signature

* Target affinity remained positive.  Target-distance 95% CI lower bounds were
  `0.2631459153` (OpenBMI) and `0.3594969741` (WBCIC); target-NLL CI lower
  bounds were `1.9318574132` and `2.8794350959`.
* Transport magnitude remained non-negligible: median displacement/local-radius
  was `0.4772807620` (OpenBMI) and `0.5798990726` (WBCIC).
* Local candidate survival was `0.3486281890`; usable coverage remained below
  the `0.50` gate (`0.3369861111` and `0.3599998638`).
* Class fidelity was lower than R1, `0.1960920139` and `0.2244585904`, far
  below the preregistered `0.90` gate.
* All six frozen recipes failed the utility checks.  The structured and matched
  random BA values are identical at the reported precision for every recipe;
  the local OT direction therefore did not separate from its random control.

## Diagnosis

Replacing the global Bures map with a local low-rank OT map improved target
affinity but did not recover semantic validity or utility.  The failure is not
weak transport; it is persistent task-semantic contamination/coverage loss and
lack of structure-specific utility.  R3 therefore tests exactly one final
hypothesis: remove the source-only task-centroid component from the frozen R2
displacement, with all other settings unchanged.

Terminal: `R2_SOURCE_GATE_FAILED`.
