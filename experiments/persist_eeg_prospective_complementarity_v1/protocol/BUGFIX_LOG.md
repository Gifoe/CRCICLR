# Bugfix log

1. The first analysis invocation used the wrong runtime root for existing EEGNet/LiteBN checkpoints; corrected it to the frozen carrier runtime. No predictor or outcome row was produced by the failed invocation.
2. The first analysis invocation deleted the shared search bundle inside the fold loop, causing a real `UnboundLocalError`; changed only the variable lifetime so all five folds use the same loaded SEARCH bundle.
3. Partner-ranking correctness was initially initialized to false; replaced it with the predeclared sign comparison using the fixed 1e-8 tie tolerance. This changes only an output field, not any predictor or outcome.

No final holdout data were accessed.
