# Final-layer projector amendment (authorized 2026-09-23)

This amendment resolves an inconsistency between the mechanism-closure prompt's universal orthogonal formula and the frozen upstream meaning of final Protected. The user delegated the choice after being shown the conflict. The original prompt and upstream protocol remain unchanged and are retained as evidence.

## Diagnostic evidence

On capped TRAIN representations, the orthogonal projector onto the final Protected raw-vector span, `Q_final Q_final.T`, differs materially from the frozen canonical-coordinate projector `L_final R_final`. In OpenBMI MI/fold0, mean relative P-vector discrepancy was 0.805 for EEGNet and 1.005 for EEGConformer. The canonical operator was idempotent to numerical tolerance but asymmetric (oblique). Exact audit records are `gate/FINAL_PROJECTOR_CONSISTENCY_*.json`; no final-heldout data were used.

## Locked interpretation

- At intermediate audited stages, use the prompt's TRAIN-only orthogonal pathway mapping: `P_l = (a_l - mu_l) Q_l Q_l.T`, `C_l = a_l - mu_l - P_l`. Preserve the previously verified mapping hashes and per-trial reconstruction.
- At `classifier_input`, retain the frozen upstream canonical Protected coordinates: `L_final = (basis / scale) @ directions[:, protected_dims]`; `R_final = raw_base(spec)[protected_dims]`; `P_final = (h - mu_final) L_final R_final`; `C_final = h - mu_final - P_final`. Verify projector idempotence, upstream basis/checkpoint hashes, exact agreement with the prior canonical decomposition, and per-trial reconstruction. Do not substitute the orthogonal span projector.
- Random final subsets use the same canonical-coordinate construction with equal rank. Intermediate random pathways remain TRAIN-only mappings to those frozen random final coordinates.
- Adjacent transitions ending at `classifier_input` and all long-range-to-final mediation use this exact canonical final projector. Earlier transitions use the orthogonal intermediate projectors. Directional final P axes are the frozen canonical raw vectors, with TRAIN SD scaling; they are not described as mutually orthogonal. Intermediate ordered P axes remain the mapping's singular-vector order.
- Report projector mode per source and destination stage. Cross-layer representation displacement is a descriptive norm under each stage's declared decomposition; it is not a conserved P mass or literal percentage transferred. Logit mediation effects and nonlinear residual are compared within a fixed frozen network and stage pair. This does not establish biological causality.

This is a separately provenanced Phase 2 protocol clarification, not a rerun or change to Phase 1. Prior orthogonal-final engineering probes remain in `probe/` as implementation-path checks only and must not be used as scientific results under this amended definition. Full 20-cell analysis remains gated until an amended single-cell probe verifies exact canonical final P and the adjacent/long-range paths for both architectures.
