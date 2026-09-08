# Bugfix log

## Pre-result engineering fixes

1. Corrected the nested experiment-root derivation (`parents[3]` rather than `parents[2]`) before any outer-development metrics were inspected.
2. Corrected screen fold lifecycle so the dataset bundle remains available across all five folds while only the per-fold GPU cache is released. The first partial run stopped before aggregation; checkpoints were retained and resumed under the same protocol lock.
3. Kept `BUGFIX_LOG.md` as Markdown rather than serializing JSON into a `.md` artifact.

## Post-screen holdout execution fix

4. The first post-screen holdout invocation exposed an `UnboundLocalError` in the holdout data-object scope. Renamed the holdout-local object to `hold_bundle` and reran the complete post-screen internal holdout audit. This changes no model, split, normalization, checkpoint, fusion, or selection rule.

The PowerShell wrapper also treated ordinary Python warnings as terminating stderr; holdout was rerun directly with the same script after that wrapper failure. No scientific result was produced by the failed invocation.
