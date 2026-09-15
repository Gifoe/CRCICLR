# LiteBN / TFFormer CSGD v1

Frozen post-hoc cross-session robustness analysis for the final LiteBN and
TFFormer checkpoints. The analysis performs no training, tuning, recalibration,
normalizer recomputation, BN update, or target adaptation.

Run on the authoritative Linux runtime:

```bash
python experiments/persist_eeg_litebn_tfformer_csgd_v1/code/run_csgd.py
```

The implementation is resumable at the model/task/fold/seed level through
`outputs/csgd_v1/SESSION_SUBJECT_RESULTS.csv`.
