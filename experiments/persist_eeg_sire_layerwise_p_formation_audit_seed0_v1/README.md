# Frozen SIRE layerwise Protected-formation audit, seed 0

This five-fold OpenBMI MI diagnostic tests whether the previously trained
`depth1 + point1` scope spans the native path on which final Protected
coordinates form. It trains no parameters or classifier. It reuses canonical
SIRE checkpoints, normalizers, roles, PERSIST final coordinates and PathFit
geometry from `persist_eeg_sire_cp_actionability_observability_seed0_v1`.

The native forward is captured at three parallel temporal branches,
`H_CONCAT`, `H_SHARED1`, `H_SHARED2`, and `EMBEDDING`. The final target is the
same frozen train-only PERSIST/PEEH coordinate set in every stage. PathFit is
trained on capped inner-train subject-session-class centroids. The final
embedding uses the exact canonical PERSIST linear map. The audit separately
measures subject-grouped train OOF and subject-disjoint discovery readout,
cross-session persistence, stage-specific Protected erasure against four
equal-rank random controls, branch contribution, equal-energy adjacent P/C
transfer and nonlinear interaction. It never treats linear readout alone as
evidence of causal formation.

Run from the repository root on a machine with the canonical source assets:

```bash
python experiments/persist_eeg_sire_layerwise_p_formation_audit_seed0_v1/code/run.py preflight
python experiments/persist_eeg_sire_layerwise_p_formation_audit_seed0_v1/code/run.py smoke
for fold in 0 1 2 3 4; do
  python experiments/persist_eeg_sire_layerwise_p_formation_audit_seed0_v1/code/run.py cell --fold "$fold"
done
python experiments/persist_eeg_sire_layerwise_p_formation_audit_seed0_v1/code/run.py aggregate
```

`SIRE_LAYERWISE_RUNTIME` can point to external runtime storage. Stage
activations stay in memory. Fitted Q bases, centering vectors, PathFit maps
and final PERSIST spectrum are saved in per-fold runtime archives; SHA256
receipts, ranks, code, protocol and compact CSV/JSON outputs are committed.
Completed cells are hash-checked and reused. The final report applies the
locked three-of-four transition rule to the five proposed trainable scopes;
it does not train any of them.

This run reads only inner-train and discovery EEG. Outer-dev and final-heldout
EEG reads are zero. Historical final-heldout diagnostic exposure of the reused
checkpoints is disclosed in the audit.
