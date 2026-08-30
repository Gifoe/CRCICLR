# Failure signature: Bures-SCST V3

V3 stopped before model-specific confirmation with terminal
`BURES_SCST_TRANSPORT_NOT_REALIZED`.  The stop is preserved; it is not
rewritten as a positive result.

| axis | evidence | diagnosis |
|---|---|---|
| transport realization | OpenBMI target-distance CI lower 0.3051 and target-NLL CI lower 3.4033; WBCIC target-distance CI lower 0.2834 and target-NLL CI lower 4.7707 | the operator moves candidates toward the target affinity cloud, so a completely absent target signal is not the primary failure |
| transport strength | median displacement/local-radius 0.5209 (OpenBMI) and 0.5751 (WBCIC) | displacement is not negligibly small; weak transport is not the primary failure |
| task-semantic contamination | class fidelity 0.2967 (OpenBMI), 0.4425 (WBCIC); OpenBMI coverage 0.3592; median margin-drop -0.00305/-0.00090 | many target-affine candidates do not preserve task semantics; this is the primary R1 hypothesis |
| structure specificity | V2 nearest/top-3/top-5 self-neighbor rate 1.0; HardRandom whitened mismatch 6.43e-8; prediction disagreement 1.64e-4; head distance 5.10e-4 | prior structured transport was effectively indistinguishable from its matched random control |
| representation capacity | V3 was not authorized for model confirmation, so capacity is not isolated by this result | defer capacity redesign until semantic contamination is tested |
| distribution mismatch | positive global target affinity but low semantic validity | possible secondary issue; do not select before R1 |
| utility saturation | no authorized V3 utility result | unknown |

R1 therefore tests exactly one change: project the subject displacement
orthogonally to a source-only two-class centroid-difference direction.  The
protected component is removed only from the transport operator, never from
clean representations.
