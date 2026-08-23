# D mechanism provenance

| quantity | Exp3 definition | Exp4-V1 definition | same? | source path | function | normalization / aggregation | intervention | output scale |
|---|---|---|---|---|---|---|---|---|
| D_finite | RMS class-centered logit displacement | not used | NO | `experiments/persist_eeg_dda_v1/code/persist_dda_v1.py` | `centered_logit_sq`, `subject_decision_metrics` | center across class logits; RMS over subject trials | finite block/subspace erasure | continuous logit units |
| D_jac | projected binary-margin Jacobian energy | not used | NO | same | `jacobian_margin` | squared projected gradient divided by `2*rank`; subject mean | local subspace sensitivity | continuous energy |
| D_flip | decision argmax change fraction | protected decision-flip coupling | YES (control only) | same | `subject_decision_metrics` | subject trial mean | finite erasure | [0,1] discrete rate |

**Did Exp4-V1 use the same D as Exp3? NO.** Exp4-V1 used only whether the
classification crossed the decision boundary after erasure. Exp3's successful
quantity retained the continuous magnitude of all centered-logit changes,
including changes that did not flip a label.
