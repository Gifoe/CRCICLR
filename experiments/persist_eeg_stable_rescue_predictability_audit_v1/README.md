# Stable rescue and predictability audit

Development-only analysis of whether LiteBN-XS rescues recur across seeds and whether Rescue versus Harm can be predicted from label-free inference quantities. The only fitted model is the predeclared L2 logistic-regression diagnostic; no EEG network is trained or changed.

Stage 0 first audits and repairs the five previously failed OpenBMI ERP replay cells using only historically plausible execution semantics. Once exact historical metrics, subjects, labels, trials, checkpoints, and normalizers pass, the final scope is OpenBMI MI, ERP, SSVEP, and WBCIC MI.

The analysis constructs one seed-0 `B0_FIXED` reference per task/fold and aligns all three XS seeds against it. It then extracts confidence and forward-hook mechanistic diagnostics without changing model semantics, performs strict five-fold subject-disjoint cross-fitting, selects switching thresholds on training folds only, and reports fully out-of-fold policy results.

Run the six scripts in filename order: fixed-row construction, XS stability, mechanistic feature extraction, predictability cross-fitting, selective policy, and final aggregation. Runtime-only replay rows remain outside Git; required audit tables and reports are written under `outputs/`.
