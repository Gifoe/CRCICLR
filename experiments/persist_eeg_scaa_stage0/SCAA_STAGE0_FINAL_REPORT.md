# SCAA Stage-0 final report

## Frozen experiment

1. Only the 41 WBCIC development subjects were used: **yes**.
2. The outer 10 remained untouched and unenumerated: **yes**.
3. Every target used its outcome-fold anchor, which had never seen that target: **yes**.
4. Frozen adaptation: classifier-head-only supervised AdamW, encoder and normalization frozen, LR `0.001`, weight decay `0.0001`, maximum `50` epochs.
5. It was selected because it passed the source/S1-only competence gate without a last-block repair.
6. Selection and protocol locking occurred before S2/S3 utility inspection: **yes**.
7. S1-only competence was nontrivial: mean BA delta `+1.52` pp, prediction-change rate `0.072`, catastrophic fraction `0.000`.

## Prospective utility transfer

8. EEGNet Spearman: `0.1862`.
9. EEGNet 95% subject-bootstrap CI: `[-0.1722, 0.5058]`.
10. EEGConformer Spearman: `0.3107`.
11. EEGConformer 95% CI: `[-0.0163, 0.5914]`.
12. Pooled within-subject Spearman: `0.3150`; CI `[-0.0183, 0.5986]`.
13. Pooled sign concordance: `0.537` (`22/41`).
14. Exact two-sided binomial p versus 0.5: `0.755229`; exact CI `[0.374, 0.693]`.
15. Always-Adapt S3 negative-transfer rate: `0.390`.
16. S2-positive-certified S3 negative-transfer rate: `0.350`.
17. Relative harm reduction: `0.103`.
18. Certificate coverage: `0.488` (`20/41`).
19. Mean pooled S3 Anchor BA: `0.7870`.
20. Mean pooled S3 Always-Adapt BA: `0.7899`.
21. Mean pooled S3 S2-Gated BA: `0.7897`.
22. Subjects whose gated policy exceeds Anchor: `14`.
23. S2-positive subjects reversing negative on S3: `7`.
24. Backbone consistency: EEGNet/EEGConformer Spearman signs are `compatible`.
25. Target-history utility exceeds the absolute pooled Spearman of each simple S1 proxy: **true** (descriptive, not a gate).
26. Secondary 90% LCB coverage/harm: `0.195` / `0.25`; it does not rescue the primary analysis.

## Decision

27. Strong Support gates:
- A_utility_transfer: `FAIL`
- B_sign_persistence: `FAIL`
- C_reduced_future_harm: `FAIL`
- D_nontrivial_coverage: `PASS`
- E_policy_usefulness: `PASS`
28. Authorization: `SCAA_DEVELOPMENT_NOT_AUTHORIZED`.
29. Strongest justified claim: Same-target historical utility shows favorable but insufficient prospective evidence under the frozen WBCIC analysis.
30. Not justified: This Stage-0 does not establish that SCAA improves generalization or formally controls negative transfer.
31. Terminal: `TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL`.
