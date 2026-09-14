# Linux W0R0 screen protocol

- Seed: 0.
- Tasks: OpenBMI MI, ERP, SSVEP, and WBCIC MI.
- Splits: canonical frozen five-fold SEARCH/development-outer split.
- Optimizer: AdamW, learning rate 3e-4, weight decay 5e-4.
- Training: 60 epochs, gradient clipping 5.0, one stochastic forward, CE.
- ERP: historical source-count-weighted CE.
- Checkpoint: highest subject-equal inner-validation BA at epochs 10 through
  60, earliest epoch on ties.
- Batch: 64 for ERP/SSVEP; historical 128-trial subject-disjoint MI episodes.
- Baseline: existing matched Linux historical 64-feature LiteBN seed-0
  checkpoint, strict-loaded and replayed with the same audited normalizer.
- Candidate: W0R0 only; no W0R1 or W1R0 is trained.
- Internal heldout status: `DEVELOPMENT_MODEL_SELECTION_DATA`.
- New sealed test accessed: no.
- Final test accessed: no.

The internal development-heldout cohort is evaluated only after all 20 W0R0
checkpoints are frozen. It is used for structure screening only and never for
checkpoint selection, normalization, or adaptation.
