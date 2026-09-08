# Matched homogeneous-ensemble control

This frozen, post-hoc control tests whether the fixed 50/50 EEGNet + LiteBN
fusion has value beyond ordinary same-backbone ensembling.  It reuses only
the selected checkpoints, source-only normalizers, subject memberships and
future-session caches of the completed MI and OpenBMI task-generality studies.

The prespecified units are all five frozen folds and seed pairs `(0,1)`,
`(0,2)`, and `(1,2)`.  For each pair, the two heterogeneous orientations are
evaluated separately and averaged as outcomes; they are never made into a
four-model ensemble.  No training, calibration, checkpoint selection, or
fusion-weight learning is performed here.

Run on the provenance server:

```bash
python code/run_matched_ensemble_control.py --preflight
python code/run_matched_ensemble_control.py --evaluate
```

`--preflight` is signal/metadata/checkpoint-only.  `--evaluate` is the single
fixed label-opening and inference pass.  Compact results only are committed;
checkpoints, tensors, caches, and per-trial logits remain outside Git.
