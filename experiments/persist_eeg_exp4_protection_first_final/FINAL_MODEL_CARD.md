# Final model card — negative Exp4 development decision

## Model selected for reporting

The report uses V1, the simplest hard projected PERSISTGuard, because it is the prospective starting formulation and has the highest Guard−Generic development mean among the three tested variants. This is a reporting selection, not a claim that V1 passed.

* Dataset: WBCIC/Yang2025/NEMAR `nm000348`; 41 development subjects only.
* Anchor: EEGNet, 58 channels × 1000 samples, S1 (`ses-0`) only, 32-dimensional embedding, dropout 0.25, 30 epochs, AdamW 3e-4 / 5e-4, batch 64.
* Update: zero-initialized linear residual `A(h)=hW+b` on S2 (`ses-1`); frozen EEGNet anchor and classifier; 25 epochs, AdamW 1e-3 / 5e-4, batch 256 update chunks.
* Protected basis: rank-four `P01_04` in the fold-specific S1/S2 persistence basis; no held-out S3 information.
* Guard equation: `h_guard=h_anchor+(I-U_P U_P^T)A(h_anchor)`.
* Controls: Frozen, Generic, three RandomGuard draws, PCAGuard, PersistenceGuard (`P05_08`), and IdentityGuard.
* Inference unit: subject; paired S3 balanced accuracy.

## Safety status

The generic adapter is useful but not reliably safer: it improves mean S3 BA while harming six subjects. V1 removes protected-coordinate drift by construction, but its negative-transfer rate is higher than Generic and the decision-response drift is not eliminated. The development gate failed, so this model is not authorized for sealed outer evaluation and should not be described as a validated final method.
