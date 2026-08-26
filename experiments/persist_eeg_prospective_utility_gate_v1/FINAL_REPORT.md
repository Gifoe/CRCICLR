# Final report — Phase 2.5 Prospective Utility Gate

## Decision

`NO_ACTIONABLE_SUPPRESSION_HEADROOM`

Recommendation: `CLOSE_CONSTRUCTIVE_ROUTE`

## Primary evidence

- Valid runs/cells: 30 / 240.
- Within-run Spearman mean/median: 0.0890 / 0.2137; hierarchical 95% CI [-0.14999049160728484, 0.3027121220700082].
- EEGNet mean rho: 0.1457; EEGConformer: 0.0324.
- Pooled hierarchical Pearson/Spearman: 0.3217 / 0.1049; CIs [-0.017280213634722187, 0.5492458423262284] / [-0.15252995034056438, 0.3327630133550886].
- MIDC minus MALLU held-run RMSE: -0.000043; CI [-0.00013421634121587583, 1.3095868033291272e-05].
- Best diagnostic (MC) minus MU RMSE: -0.000587; CI [-0.00171453726722187, 0.0001757636699252213].
- PseudoUtility-Top1 minus random: 0.095 pp; CI [-0.0011041666666666413, 0.002765755208333321].
- PseudoUtility-Top1 / Random / Oracle utility: -0.029 / -0.124 / 0.558 pp.
- Mean policy regret / recovery fraction: 0.587 pp / 0.121; unstable denominators: 0.
- Helpful/harmful/neutral: 8.75% / 19.17% / 72.08%.
- Permutation policy p=0.23576; prediction p=0.97003.
- Purity: PASS; no internal holdout or WBCIC access.

## Held-run RMSE

| scope   | model   |   mean_run_RMSE |   median_run_RMSE |     ci_low |    ci_high |   relative_improvement_vs_M0 |
|:--------|:--------|----------------:|------------------:|-----------:|-----------:|-----------------------------:|
| all     | M0      |      0.00621153 |        0.00538537 | 0.00444976 | 0.00838441 |                   0          |
| all     | MI      |      0.00618866 |        0.00535578 | 0.00454447 | 0.00817805 |                   0.00368133 |
| all     | MD      |      0.00580772 |        0.00511316 | 0.00458079 | 0.00724989 |                   0.0650089  |
| all     | MC      |      0.00548112 |        0.00485426 | 0.00417191 | 0.00697476 |                   0.117589   |
| all     | MIDC    |      0.00553914 |        0.0048852  | 0.00426085 | 0.00701642 |                   0.108248   |
| all     | MU      |      0.0060682  |        0.00532073 | 0.00461504 | 0.0078892  |                   0.0230743  |
| all     | MALLU   |      0.00558262 |        0.00504256 | 0.00430836 | 0.00708909 |                   0.101248   |

## Interpretation

Only 8.75% of cells were meaningfully helpful at the frozen +0.5 pp threshold, below Gate H's 15% minimum, while 19.17% were harmful and 72.08% neutral. The ranking and Top-1 point estimates were weakly positive but their hierarchical intervals crossed zero; adding U_pseudo did not improve held-run ridge RMSE. The evidence therefore does not justify a selective utility-gated invariance model.

The terminal state follows the pre-frozen gates. No architecture, outcome, scientific definition, or gate was modified after outcome evaluation.
