# LiteBN-X single-model seed-0 decision

Selected before outer-dev reveal: LiteBN_X.
Stage-A robust winner: True.

| Task | LiteBN BA | New Model BA | Delta pp | 95% CI | LiteBN F1 | New F1 |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI MI | 0.7848 | 0.7930 | +0.825 | [-0.875, +2.550] | 0.7800 | 0.7867 |
| OpenBMI ERP | 0.8326 | 0.8350 | +0.243 | [-0.595, +1.104] | 0.7849 | 0.7965 |
| OpenBMI SSVEP | 0.8920 | 0.9133 | +2.125 | [+0.175, +4.225] | 0.8880 | 0.9112 |
| WBCIC MI | 0.7903 | 0.7837 | -0.660 | [-2.226, +0.935] | 0.7858 | 0.7803 |

Equal-task mean delta: +0.633 pp.
OpenBMI three-task mean delta: +1.064 pp.
Dataset-balanced mean delta: +0.202 pp.
Task improvements: 3/4.
Decision terminal: MIXED_SINGLE_MODEL.

## Required answers

1. Selected architecture: LiteBN_X.
2. It was selected only by the frozen Stage-A inner-validation rule: MI-first amendment: MI survivor gate was predeclared; among retained architectures, full four-task inner-val eligibility requires mean delta >0 and worst task >=-0.50 pp; highest equal-task mean, 0.10 pp tie resolved lexicographically.
3. Parameters versus LiteBN are task-specific only through head dimension: OpenBMI_MI 47978 to 219733; OpenBMI_ERP 47978 to 219733; OpenBMI_SSVEP 48108 to 219991; WBCIC_MI 47786 to 219445.
4. OpenBMI MI gain: +0.825 pp.
5. OpenBMI ERP gain: +0.243 pp.
6. OpenBMI SSVEP gain: +2.125 pp.
7. WBCIC MI gain: -0.660 pp.
8. Improved tasks: 3/4.
9. Equal-task mean gain: +0.633 pp.
10. Dataset-balanced mean gain: +0.202 pp.
11. Replacement support is limited to the terminal above; this remains seed-0 SEARCH/development evidence.
12. Seed1/2 are not justified automatically.
13. The inner-val ablation summary identifies the useful module descriptively; no outer result changed the architecture.
14. Held-out/test data accessed: NO.

Boundary: Stage B opened only frozen outer-development SEARCH subjects once after the Stage-A freeze. It did not read final held-out/test labels or predictions.
