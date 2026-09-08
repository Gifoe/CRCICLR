# Final backbone screen decision

## Screen terminal

**BACKBONE_GENERALITY_NOT_SUPPORTED**. The frozen seed-0 screen used five outer-development folds on SEARCH subjects only; primary fusion was fixed raw-logit 50/50 with LiteBN. No holdout subject was loaded during selection.

# Backbone generality matrix

| Dataset | Pair | Base BA | Partner BA | Fusion BA | Delta best (pp) | 95% CI (pp) | Positive folds |
|---|---|---:|---:|---:|---:|---|---:|
| OpenBMI | CBraMod+EEGNet | 0.5098 | 0.7577 | 0.7587 | +0.100 | [-0.125, +0.300] | 3/5 |
| OpenBMI | CBraMod+LiteBN | 0.5098 | 0.7915 | 0.7913 | -0.025 | [-0.200, +0.125] | 0/5 |
| OpenBMI | CodeBrain+EEGNet | 0.6675 | 0.7577 | 0.7518 | -0.600 | [-1.925, +0.675] | 2/5 |
| OpenBMI | CodeBrain+LiteBN | 0.6675 | 0.7915 | 0.7732 | -1.825 | [-3.050, -0.700] | 1/5 |
| OpenBMI | EEGConformer+EEGNet | 0.7578 | 0.7577 | 0.7888 | +3.100 | [+1.275, +3.650] | 5/5 |
| OpenBMI | EEGConformer+LiteBN | 0.7578 | 0.7915 | 0.7897 | -0.175 | [-1.250, +0.950] | 3/5 |
| OpenBMI | EEGNet+LiteBN | 0.7577 | 0.7915 | 0.8000 | +0.850 | [-0.100, +1.850] | 6/10 |
| WBCIC | CBraMod+EEGNet | 0.7348 | 0.7827 | 0.7761 | -0.663 | [-1.516, +0.210] | 2/5 |
| WBCIC | CBraMod+LiteBN | 0.7348 | 0.7870 | 0.7867 | -0.032 | [-0.871, +0.823] | 3/5 |
| WBCIC | CodeBrain+EEGNet | 0.7502 | 0.7827 | 0.7848 | +0.209 | [-0.952, +1.516] | 2/5 |
| WBCIC | CodeBrain+LiteBN | 0.7502 | 0.7870 | 0.7886 | +0.160 | [-0.727, +1.048] | 3/5 |
| WBCIC | EEGConformer+EEGNet | 0.7254 | 0.7827 | 0.7780 | -0.468 | [-1.371, +0.403] | 1/5 |
| WBCIC | EEGConformer+LiteBN | 0.7254 | 0.7870 | 0.7837 | -0.338 | [-1.484, +0.759] | 1/5 |
| WBCIC | EEGNet+LiteBN | 0.7827 | 0.7870 | 0.7940 | +0.694 | [-0.305, +1.258] | 6/10 |


## Answers to the predeclared questions

1. EEGConformer + LiteBN on OpenBMI: **no** (−0.175 pp; CI [−1.250, +0.950]).
2. EEGConformer + LiteBN on WBCIC: **no** (−0.338 pp; CI [−1.484, +0.759]).
3. CBraMod + LiteBN on OpenBMI: **no** (−0.025 pp; CI [−0.200, +0.125]).
4. CBraMod + LiteBN on WBCIC: **no** (−0.032 pp; CI [−0.871, +0.823]).
5. CodeBrain + LiteBN on OpenBMI: **no** (−1.825 pp; CI [−3.050, −0.700]).
6. CodeBrain + LiteBN on WBCIC: **yes descriptively but weakly** (+0.160 pp; CI [−0.727, +1.048]).
7. New backbone families generalizing across both datasets: **0/3** under the predeclared positive rule.
8. Pretrained FM generalizing across both datasets: **no**.
9. LiteBN pairing stronger than generic EEGNet fusion: **not uniformly**; controls are better for EEGConformer on OpenBMI, while LiteBN is slightly better for CodeBrain on WBCIC.
10. Most defensible interpretation: **specialized/partial generality**; existing EEGNet+LiteBN remains the supported reference.
11. Seeds 1/2: **do not proceed automatically**.
12. Exact pairs for further study: **none mandated by this seed-0 screen**.

## Post-screen internal holdout audit

This separate audit used fixed screen checkpoints; holdout was not used for model, checkpoint, fusion, or preprocessing selection. WBCIC true outer subjects were not accessed.

```text
dataset,model,BA,macro_F1,subjects,folds,selection_used_holdout
OpenBMI,CBraMod,0.5047142857142857,0.40640043766406025,14,5,False
OpenBMI,CBraMod+EEGNet,0.7114285714285714,0.7031282538593652,14,5,False
OpenBMI,CBraMod+LiteBN,0.7454285714285713,0.737647266921416,14,5,False
OpenBMI,CodeBrain,0.650857142857143,0.630490609927129,14,5,False
OpenBMI,CodeBrain+EEGNet,0.7044285714285715,0.6939806152058551,14,5,False
OpenBMI,CodeBrain+LiteBN,0.7301428571428571,0.7209712947593383,14,5,False
OpenBMI,EEGConformer,0.7382857142857142,0.7309753235971506,14,5,False
OpenBMI,EEGConformer+EEGNet,0.7487142857142856,0.7425767804558102,14,5,False
OpenBMI,EEGConformer+LiteBN,0.7588571428571428,0.7524481163091175,14,5,False
OpenBMI,EEGNet,0.7117142857142857,0.7037605737126124,14,5,False
OpenBMI,EEGNet+LiteBN,0.7489999999999999,0.741576673629817,14,5,False
WBCIC,CBraMod,0.7448999999999999,0.7407102800641435,10,5,False
WBCIC,CBraMod+EEGNet,0.7891000000000001,0.7849006833883166,10,5,False
WBCIC,CBraMod+LiteBN,0.7979999999999999,0.7938198506592151,10,5,False
WBCIC,CodeBrain,0.7515999999999999,0.7495060973032309,10,5,False
WBCIC,CodeBrain+EEGNet,0.7877,0.7847845085790104,10,5,False
WBCIC,CodeBrain+LiteBN,0.7924000000000001,0.7889149259815902,10,5,False
WBCIC,EEGConformer,0.7557,0.7473078806452902,10,5,False
WBCIC,EEGConformer+EEGNet,0.7944999999999999,0.7887818539096952,10,5,False
WBCIC,EEGConformer+LiteBN,0.7942,0.7889926246768618,10,5,False
WBCIC,EEGNet,0.795,0.7910385237420221,10,5,False
WBCIC,EEGNet+LiteBN,0.7994,0.7942327432606229,10,5,False
```

The holdout audit is descriptive confirmation only, not an all-search refit, and does not reverse the screen terminal.
