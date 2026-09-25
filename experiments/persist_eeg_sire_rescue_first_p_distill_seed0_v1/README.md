# Native SIRE rescue-first P distillation, seed 0

This OpenBMI MI five-fold pilot asks whether the existing SIRE `depth1` and
`point1` block can learn a P correction defined by **actual prediction rescue**.
It starts from the same canonical `selected_best.pt` checkpoints as
`persist_eeg_sire_local_p_construction_seed0_v1`. Only inner-train trials build
the teacher or train the student. Discovery subjects are used after the fixed
epoch-20 checkpoint is frozen.

The code imports the prior experiment's canonical stage, split, normalization,
geometry, decomposition, state-audit and evaluation helpers. It rebuilds the
inner-train P/C geometry and requires exact historical projector and complement
basis hashes. For a native-wrong trial, the frozen teacher searches one source
complement direction and alpha in `{0, 0.25, 0.5, 0.75, 1.25, 1.5, 2}`. It
preserves native successor C, accepts only candidates that correct the actual
prediction, and picks the smallest standardized P movement. Native-correct and
unrescuable trials retain native P.

The student receives group-normalized rescue and preservation P losses plus C
preservation. There is no CE, KD, margin, or classifier loss. Only
`depth1.weight` and `point1.weight` train. The full model stays in `eval()` to
freeze BN running state and disable dropout. Batches distribute rescue samples
without repetition or oversampling; every trial appears once per epoch.

From the repository root, with the canonical source assets present:

```bash
python experiments/persist_eeg_sire_rescue_first_p_distill_seed0_v1/code/run.py preflight
python experiments/persist_eeg_sire_rescue_first_p_distill_seed0_v1/code/run.py smoke
for fold in 0 1 2 3 4; do
  python experiments/persist_eeg_sire_rescue_first_p_distill_seed0_v1/code/run.py cell --fold "$fold"
done
python experiments/persist_eeg_sire_rescue_first_p_distill_seed0_v1/code/run.py aggregate
```

Set `SIRE_RESCUE_P_RUNTIME` to change the external runtime directory. Dense
teacher caches, train-only geometry arrays and student checkpoints stay outside
Git; SHA256 receipts are in the protocol and compact audits. Completed cells
are verified and reused. `outputs/FINAL_REPORT.md` answers the six locked
questions. The run reads zero outer-dev or final-heldout EEG arrays; historical
diagnostic exposure of reused canonical checkpoints is disclosed in the audit.
