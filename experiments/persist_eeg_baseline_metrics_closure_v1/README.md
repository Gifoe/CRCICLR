# Baseline metrics closure v1

This branch contains a read-only audit and a protocol-gated stop for the six retained PERSIST-EEG baselines. Run `code/audit.py` before any evaluation. The closure is **not** complete: WBCIC internal and final true-outer cohorts were conflated in the requested regression targets, two LiteBN 15-checkpoint matrices are incomplete, and the published PSWA branch still has ModernTCN/Medformer at 0/5 coverage. See `outputs/BASELINE_METRICS_CLOSURE_REPORT.md`.

No new training, checkpoint selection, heldout inference, MAC profiling, or PSWA calculation was performed here. Original experiments remain untouched.
