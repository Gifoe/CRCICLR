# Failure signature: Repair-R3 task-protected local OT

Repair-R3 was evaluated on all 30 source-only units (OpenBMI and WBCIC,
5 folds x 3 seeds).  The final constructive round completed with no runtime
failure.  No future, outer, sealed, or S3 resource was opened.

## Observed signature

* Target transport remained numerically realized.  Target-distance 95% CI
  lower bounds were `0.2256534470` (OpenBMI) and `0.3198364798` (WBCIC);
  target-NLL CI lower bounds were `1.9748029844` and `3.0321355833`.
* Transport magnitude was non-negligible: median displacement/local-radius
  was `0.4590097740` (OpenBMI) and `0.5692414045` (WBCIC).
* Candidate survival was `0.3541724052`, while usable coverage remained below
  the frozen `0.50` gate (`0.3403819444` and `0.3675384374`).
* Class fidelity remained far below the frozen `0.90` gate (`0.1969835069`
  and `0.2328908431`).  Removing the source task-centroid component from the
  local displacement therefore did not remove task-semantic contamination.
* The six recipe-level non-negative subject fractions were only
  `0.3623188406`, `0.3913043478`, or `0.4492753623`, all below `0.60`.
* Every preregistered recipe failed the utility gate.  Primary deltas versus
  ERM were negative for both datasets in every recipe (OpenBMI range
  `-0.0029166667` to `-0.0022420635`; WBCIC range `-0.0035866798` to
  `-0.0015475897`).  All paired CI lower bounds versus ERM and Mixup were
  non-positive.  The matched-random differences were small and had no
  positive CI lower bound (OpenBMI `-0.0003670635` to `+0.0003075397`;
  WBCIC `-0.0012708385` to `+0.0007682512`).

## Diagnosis

The final source-only repair preserved target affinity and meaningful
displacement but did not recover class fidelity, coverage, or utility.  The
rank-one task-protected projection of the frozen R2 local OT displacement is
therefore insufficient.  Across R1, R2, and R3, the evidence is consistent
with persistent task-semantic contamination and a distribution/operator
mismatch; no tested construction establishes structure-specific utility.

The preregistered three-round constructive budget is exhausted.  No further
constructive method search or threshold/recipe changes are authorized in this
space.

Terminal: `SCST_CONSTRUCTIVE_SEARCH_EXHAUSTED`.
