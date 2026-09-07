# Relative sidecar fold0 decision

Frozen EEGNet-Base BA: 0.7964. Raw-Sidecar BA: 0.7907. Relative-Sidecar BA: 0.7929.

Capacity / relative-specific / total gain: -0.57 / +0.21 / -0.36 pp.

Relative mean / median subject delta: +0.21 / +0.00 pp; improved/tied/harmed: 3 / 10 / 1; 95% CI [-0.07, +0.57] pp. Total 95% CI [-1.14, +0.36] pp.

Base hash unchanged: YES. Zero-init identity: YES. Offset MSE / cosine / zero-MSE: 0.057252 / 0.8168 / 0.173092.

Prediction-change fraction Raw / Relative: 0.0157 / 0.0150. Base-wrong to corrected Raw / Relative: 7 / 8. Base-correct to newly wrong Raw / Relative: 15 / 13.

Oracle Relative diagnostic BA: 0.7964 (TRANSDUCTIVE DIAGNOSTIC ONLY). Relative beat Raw: YES; at least +0.3 pp: NO; Relative beat Base by +0.5 pp: NO.

Reserved holdout accessed: NO. Claim support: WEAK. Terminal: `RELATIVE_SIDECAR_TRIVIAL_STOP`.

Recommended next action: stop centering / relative residual as a constructive core; do not tune the sidecar or offset estimator.
