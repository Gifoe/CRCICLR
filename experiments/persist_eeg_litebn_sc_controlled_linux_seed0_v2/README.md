# Controlled LiteBN-SC v2

This experiment is the controlled Linux seed-0 evaluation of `LiteBN-SC`, an
exact historical LiteBN trained with an R-Drop-style symmetric prediction
consistency objective. The inference architecture is unchanged.

The completed WBCIC_MI seed-0 run contains C0 and C1 on folds 0--4. The exact
frozen B0 Linux checkpoints are replayed for comparison. C0 is a matched DualCE
compute/sampling control and is not substituted by an earlier SC run.

Run from the repository root:

```bash
python -u experiments/persist_eeg_litebn_sc_controlled_linux_seed0_v2/code/run_c1_wbcic.py
python -u experiments/persist_eeg_litebn_sc_controlled_linux_seed0_v2/code/run_c0_wbcic.py
```

Runtime checkpoints are stored outside Git in
`/root/rivermind-data/litebn_sc_controlled_v2_runtime`. Outputs are written
atomically under this experiment directory.
