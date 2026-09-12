# LiteBN-SBTR seed0 repair ledger

- The experiment reused the server's existing `Gifoe/CRCICLR` worktree, combined cache, frozen five-fold split, and historical LiteBN seed0 checkpoints. No raw EEG was copied and no baseline was retrained.
- The first launch failed before training because the runner called `build_bundle` without its task argument. The call was corrected and the unit checks were rerun successfully.
- The exact historical LiteBN baseline was audited before training: all five OpenBMI MI seed0 checkpoints existed and had strict state-key and tensor-shape matches.
- Only `OpenBMI_MI`, seed 0, folds 0--4 were run. Seed 1/2, WBCIC, ERP, and SSVEP were not started.
- The current internal heldout cohort was evaluated only after all five checkpoints were fixed; it is an internal diagnostic, not an untouched final test.
- The final decision is `STOP_SBTR` because SBTR degraded both outer and internal-heldout BA and increased fold dispersion.
- `TRAINING_HISTORY.csv` is the flushed checkpoint-progress history from the completed run (epochs 1, 5, selection events, and every fifth epoch); fold-level checkpoint and exposure audits are complete.
