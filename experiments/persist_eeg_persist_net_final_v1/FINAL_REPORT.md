# Final report

## Terminal state

**PERSIST_NET_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED**

Strongest legal baseline: **Strong Generic**, BA=0.7924.
FULL BA=0.7639, Macro-F1=0.7614, delta=-2.850 pp,
paired subject-bootstrap 95% CI=[-4.092, -1.625] pp.

## Main table

| Method | BA | Macro-F1 | Delta BA | 95% CI L | 95% CI U | NTR | Worst-quartile Delta | Params | Target-trainable Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla EEGNet | 0.78617 | 0.78234 | -0.00625 | -0.01933 | 0.00575 | nan | nan | 34162.00000 | 0.00000 |
| Strong EEGNet | 0.79150 | 0.78861 | -0.00092 | -0.00808 | 0.00608 | nan | nan | 54773.20000 | 0.00000 |
| Strong Generic | 0.79242 | 0.78973 | 0.00000 | 0.00000 | 0.00000 | 0.42500 | -0.02767 | 54773.20000 | 47026.00000 |
| Dual capacity control | 0.77633 | 0.77186 | -0.01608 | -0.02792 | -0.00467 | 0.37500 | -0.02500 | 47072.80000 | 19212.40000 |
| P-only | 0.76400 | 0.75999 | -0.02100 | -0.03725 | -0.00450 | 0.27500 | -0.02200 | 47072.80000 | 19212.40000 |
| P+U | 0.76000 | 0.75769 | -0.02500 | -0.04350 | -0.00750 | 0.35000 | -0.01800 | 47072.80000 | 19212.40000 |
| P+D | 0.75425 | 0.75245 | -0.03075 | -0.04901 | -0.01325 | 0.35000 | -0.03300 | 47072.80000 | 19212.40000 |
| Identity-protected | 0.76983 | 0.76654 | -0.02258 | -0.03492 | -0.01058 | 0.42500 | -0.02233 | 47072.80000 | 19212.40000 |
| Random-protected | 0.76700 | 0.76406 | -0.02542 | -0.03717 | -0.01408 | 0.47500 | -0.01933 | 47072.80000 | 19212.40000 |
| PCA-protected | 0.76675 | 0.76255 | -0.01825 | -0.03525 | -0.00175 | 0.15000 | -0.01200 | 47072.80000 | 19212.40000 |
| PUD all-adapt | 0.76200 | 0.75924 | -0.03042 | -0.04283 | -0.01883 | 0.37500 | -0.02200 | 47072.80000 | 38424.80000 |
| PUD protected-freeze FULL | 0.76392 | 0.76141 | -0.02850 | -0.04092 | -0.01625 | 0.25000 | -0.00900 | 47072.80000 | 19212.40000 |

## Required scientific answers

1. **Different from P4-SI / Protection-First / Guard?** Yes architecturally: this is source-time functional distillation into an independent pathway with a universal structural freeze, not GRL, coordinate update projection, or prospective routing.
2. **Did P/U/D train an independent protected task pathway?** Protected functional agreement and nonzero intervention metrics are reported in `results/mechanism_metrics.csv`; gate G8 is **PASS**. Architectural independence alone is not evidence of benefit.
3. **More task-consequential than identity/random?** Final performance: FULL-minus-identity=-0.592 pp and FULL-minus-random=-0.308 pp. Protected-branch erasure-harm differences are +13.042 pp versus identity and +12.058 pp versus random; their subject-bootstrap CIs are in `results/statistics.json`. Theory-specificity G6 is **FAIL**.
4. **Is freezing better than all-adapt?** FULL-minus-all-adapt=+0.192 pp; G7 is **PASS**.
5. **Beyond dual-path capacity?** FULL-minus-dual-control=-1.242 pp. G6 requires this to be positive.
6. **Stable across fold/seed/subject?** Positive folds=0/5; positive seeds=0/3; G2/G3/G4=`FAIL`/`FAIL`/`FAIL`.
7. **Negative transfer?** FULL NTR=0.250; Strong Generic NTR=0.425; safety G5 is **PASS**.
8. **WBCIC external development replication?** NOT_AUTHORIZED. It is not run unless frozen OpenBMI G1/G6/G9 authorize it.
9. **Sealed outer untouched?** Yes. OpenBMI internal holdout accessed: **No**. WBCIC sealed outer accessed: **No**.
10. **Strongest legal claim now?** The protection-by-construction hypothesis is not supported under the frozen OpenBMI development gate. Exp1-3 remain descriptive/mechanistic evidence; they do not establish that PUD distillation improves future-session deployment.

## Scope and limitations

Secondary P-only/P+U/P+D/PCA ablations use one predeclared seed; primary gates use five folds by three seeds. The OpenBMI internal holdout is not opened and is not described as independent confirmation. Subject is the statistical unit; no trial-level pseudoreplication is used.
