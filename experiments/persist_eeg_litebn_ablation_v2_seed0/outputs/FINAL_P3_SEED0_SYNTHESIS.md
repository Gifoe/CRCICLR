# Final P3 seed-0 synthesis

## 1. Architecture attribution

B0_FULL_MATCHED was not trained by explicit user instruction. Consequently, all B1-B4 contrasts use the official frozen Full as a non-matched reference and cannot isolate training-protocol effects.

- B1_SAME_SCALE_63: mean future-BA delta across tasks -0.76 pp; mean WS-BA delta -1.24 pp.
- B2_SCALE_COLLAPSE: mean future-BA delta across tasks -1.85 pp; mean WS-BA delta -1.96 pp.
- B3_SINGLE_SPATIAL_BASIS: mean future-BA delta across tasks -1.04 pp; mean WS-BA delta -0.92 pp.
- B4_ONE_STAGE_BACKEND: mean future-BA delta across tasks -1.74 pp; mean WS-BA delta -2.24 pp.

## 2. Representation bridge

- OpenBMI_MI: Full-SameScale PEEH +2.93 pp; PSWA +0.47 pp; full-model WS-BA +3.46 pp.
- OpenBMI_ERP: Full-SameScale PEEH -1.68 pp; PSWA -1.99 pp; full-model WS-BA -0.38 pp.
- OpenBMI_SSVEP: Full-SameScale PEEH -0.18 pp; PSWA -0.46 pp; full-model WS-BA +0.29 pp.
- WBCIC_MI: Full-SameScale PEEH -1.15 pp; PSWA -3.12 pp; full-model WS-BA +1.61 pp.

PEEH/PSWA are mechanistic probe quantities, not model-quality metrics. Alignment with WS-BA is interpreted task by task and no direction is selected post hoc.

## 3. Normalization mechanism transfer

- OpenBMI_MI adaptation_loss: future BA -0.03 pp [-0.61, +0.59]; WS-BA -0.03 pp [-0.83, +0.77].
- OpenBMI_MI state_restoration: future BA -0.11 pp [-0.53, +0.27]; WS-BA -0.26 pp [-0.64, +0.14].
- OpenBMI_MI state_transplant: future BA +0.09 pp [-0.31, +0.50]; WS-BA -0.09 pp [-0.59, +0.41].
- OpenBMI_MI parameter_only_retained: future BA -0.14 pp [-0.71, +0.40]; WS-BA -0.29 pp [-1.03, +0.41].
- WBCIC_MI adaptation_loss: future BA -0.15 pp [-0.54, +0.23]; WS-BA +0.07 pp [-1.08, +1.10].
- WBCIC_MI state_restoration: future BA +0.03 pp [-0.39, +0.44]; WS-BA -0.16 pp [-0.84, +0.80].
- WBCIC_MI state_transplant: future BA +0.11 pp [-0.32, +0.51]; WS-BA +0.37 pp [-0.55, +1.02].
- WBCIC_MI parameter_only_retained: future BA -0.12 pp [-0.64, +0.42]; WS-BA -0.09 pp [-0.79, +0.62].

## 4. Objective intervention

- OpenBMI_MI: alignment +0.1006; future BA -1.63 pp [-2.53, -0.77]; WS-BA -0.99 pp [-1.83, -0.14].
- WBCIC_MI: alignment +0.0289; future BA -2.43 pp [-3.59, -1.31]; WS-BA -2.28 pp [-3.60, -1.09].

## 5. Claim boundary

- The exact LiteBN architecture is not mathematically derived by PERSIST-EEG.
- A diagnostic improvement is not called a predictive improvement unless BA/WS-BA also improves.
- WBCIC true-outer has already been accessed and is an exposed diagnostic benchmark.
- These are seed-0 results only; no seed1/2 or multiseed selection was run.
- The missing matched-B0 is a material protocol deviation, not hidden by the report.
