# Native SIRE local P construction, seed 0

This five-fold OpenBMI MI pilot tests whether SIRE's existing `depth1` and
`point1` weights can learn a privileged, locally beneficial P correction during
training. The primary comparison is `LOCAL_CONSTRAINED_P` against an otherwise
identical `CE_ONLY_CONTINUATION`, both initialized from the same canonical
`selected_best.pt` checkpoint. The original checkpoint is a secondary baseline.

The experiment imports canonical SIRE stage, split, normalization, geometry,
and counterfactual functions from
`persist_eeg_sire_cp_actionability_observability_seed0_v1`. Each fold rebuilds
the train-only P/C projectors and complement basis and verifies their hashes
against that experiment's audits. The teacher searches one source C direction
over alpha values 0.5, 0.75, 1.25, and 1.5, keeps native successor C fixed,
and selects the smallest standardized successor P movement among candidates
that improve training CE without damaging a correct prediction.

Both continuation arms train for exactly 20 epochs with identical batch order,
AdamW settings, and optimizer step count. Only `depth1.weight` and
`point1.weight` receive gradients. The entire model remains in eval mode,
fixing BatchNorm state and disabling dropout. Local-P adds weighted standardized
P target loss and standardized C preservation loss with both coefficients 1.
Discovery subjects are evaluated only after both endpoint checkpoints are
frozen. The diagnostic discovery teacher cannot change training or selection.

Run from the repository root, after making the source experiment and its
canonical checkpoint/data files available:

```bash
python experiments/persist_eeg_sire_local_p_construction_seed0_v1/code/run.py preflight
python experiments/persist_eeg_sire_local_p_construction_seed0_v1/code/run.py smoke
for fold in 0 1 2 3 4; do
  python experiments/persist_eeg_sire_local_p_construction_seed0_v1/code/run.py cell --fold "$fold"
done
python experiments/persist_eeg_sire_local_p_construction_seed0_v1/code/run.py aggregate
```

`SIRE_LOCAL_P_RUNTIME` may point to a runtime directory outside Git. Dense
teacher caches, geometry arrays, and continuation checkpoints remain there;
their SHA256 hashes are recorded in compact audits. Completed cells are
hash-checked and reused. The protocol lock and smoke audit are committed with
the source and compact results. `outputs/FINAL_REPORT.md` answers the five
predefined experimental questions.

This run reads only inner-train and discovery EEG. Outer-development and
final-heldout EEG reads are zero. The reused canonical checkpoints carry the
historical final-heldout diagnostic exposure disclosed in the source record.
