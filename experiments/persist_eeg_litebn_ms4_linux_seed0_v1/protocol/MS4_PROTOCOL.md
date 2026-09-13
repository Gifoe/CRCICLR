# LiteBN-MS4 protocol lock

- Candidate: exact historical LiteBN plus mean supervised CE over K=4 stochastic forwards.
- Task/seed/folds: WBCIC-MI, seed 0, canonical folds 0 through 4.
- K is fixed at 4; no search or rescue run is allowed.
- A uses the historical main Torch RNG stream. B, C, and D sequentially consume the persistent secondary stream.
- A executes on live buffers. B, C, and D use three distinct pre-A buffer clones through `torch.func.functional_call` while sharing live Parameter objects.
- Persistent BatchNorm buffers update once per original batch.
- Loss is exactly `mean(CE_A, CE_B, CE_C, CE_D)`; no KL or other auxiliary loss.
- One AdamW step per original batch, lr 3e-4, weight decay 5e-4, clip 5, historical AMP/GradScaler, 60 epochs.
- Selection uses only canonical subject-equal inner-validation BA, eligible epochs 10 through 60, earliest strict improvement.
- All five checkpoints are frozen before outer evaluation; outer is materialized before the already-open internal-heldout diagnostic.
- Evaluation is one deterministic eval-mode forward. No averaging, ensemble, TTA, recalibration, or reselection.
- Internal heldout status: `DEVELOPMENT_MODEL_SELECTION_DATA`. New sealed test access: no.
- No OpenBMI task and no seed 1 or 2 is in scope.
