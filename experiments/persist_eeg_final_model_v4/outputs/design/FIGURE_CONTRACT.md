# V4 figure contract

## Core conclusion

Constrained dynamic KEEP aggregation produces a robust exploratory OpenBMI gain,
whereas ACTION and PERSIST do not add prospective value and no architecture is
robustly positive on both OpenBMI and WBCIC development.

## Evidence chain

1. Figure 1a: A0-A3 OpenBMI performance relative to the strongest static ensemble.
2. Figure 1b: WBCIC development transfer and adapted controls.
3. Figure 1c: all 52 OpenBMI subject effects, with no subject omitted.
4. Figure 1d: the remaining oracle headroom and the selection bottleneck.
5. Figure 2: paired subject-bootstrap increments for every mandatory component ablation.
6. Figure 3: cross-benchmark gains, separating direct transfer from benchmark adaptation.

## Contract

- Archetype: quantitative grid with a cross-benchmark hero comparison.
- Backend: Python/matplotlib exclusively.
- Exclusions: none; all 52 OpenBMI and 41 WBCIC development subjects are retained.
- Statistics: paired subject bootstrap, 10,000 repetitions; CIs are 95% intervals.
- Export: editable SVG/PDF plus 600-dpi PNG on white background.
- Source data: exact CSVs are stored under `outputs/figures/source_data/`.

## Reviewer risks

OpenBMI is exploratory after repeated development use; WBCIC intervals cross
zero; the initially used WBCIC logit-mean baseline was corrected to the stronger
probability-mean baseline; oracle panels are explicitly non-prospective.
