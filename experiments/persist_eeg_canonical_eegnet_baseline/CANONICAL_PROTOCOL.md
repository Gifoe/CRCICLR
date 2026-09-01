# Canonical protocol

## CANONICAL BASELINE STATUS

This is a pre-registered vanilla EEGNet baseline; no outcome label is read until the final scoring call.

### Architecture

- F1=8, depth multiplier D=2, F2=16; temporal kernel 64; ELU; average pooling 4 then 8; dropout=0.25; Linear(496,64)+ELU+LayerNorm(64); Linear(64,2).
- Input is the frozen 1000-sample MI epoch; channel count is 62 for OpenBMI and 58 for WBCIC.

### Training

- AdamW, learning rate 3e-4, weight decay 5e-4, batch size 64, cross entropy, max 60 epochs, minimum 10 epochs, patience 8.
- Seeds are exactly 0, 1, and 2. Each dataset/fold/seed has one initial fit and one deterministic refit.
- Initial fit uses model-fit subjects and S1+S2. Epoch is selected only on disjoint discovery subjects by mean biological-subject balanced accuracy, then lower NLL, then earlier epoch.
- Refit uses model-fit+discovery subjects and S1+S2 for exactly the selected epoch count. Outcome subjects are never in training or epoch selection.

### Dataset roles

- OpenBMI uses all 54 Stage-0-frozen Lee2019 MI subjects; model-fit = frozen train subjects, discovery = frozen validation subjects, outcome = frozen outer-test subjects. Physical sessions are 1/2 (S1/S2).
- WBCIC uses only the 41 `DEVELOPMENT_SCOPE_LOCK.json` allowed subjects; model-fit/discovery/outcome roles are the frozen audit roles. Physical sessions 0/1/2 are S1/S2/S3. The sealed outer 10 are not enumerated or opened.

### Statistics

- Primary summary averages the three seed probabilities per trial, then computes subject BA. CI is a 10,000-draw biological-subject bootstrap.
- Secondary robustness statistic is the mean of the three single-seed aggregate subject BAs. Fold and seed tables retain both views.
