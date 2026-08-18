# PERSIST-EEG prospective action policy V1

## Terminal interpretation

`STOP_ACTIONABILITY_NOT_PREDICTABLE`

This is exploratory development evidence, not a confirmatory result. The
sealed WBCIC outer subjects were not read or used.

## Direct answers

1. **Original oracle headroom.** Exact subject-balanced OpenBMI reconstruction
   gives `0.118701` BA.
2. **Distribution/concentration.** `1.000`
   of subjects have at least one rescue; the largest subject contributes
   `0.060` and the largest run
   `0.187` of oracle gain. At the trial
   level rescue remains sparse (`0.119`).
3. **Action-selection value.** Best fixed action including NO_OP gains
   `0.000000`; selection value is
   `0.118701` BA.
4. **Rescue actions.** All three actions rescue some errors under oracle
   selection:

- `AMPLIFY`: always-action ΔBA=-0.01490; pair-oracle rescue gain=0.02895; harm fraction=0.657.
- `ERASE`: always-action ΔBA=-0.17341; pair-oracle rescue gain=0.07103; harm fraction=0.995.
- `GEOMETRY`: always-action ΔBA=-0.05613; pair-oracle rescue gain=0.04740; harm fraction=0.902.

5. **Harmful actions.** Every non-trivial fixed OpenBMI intervention is net
   harmful despite its rescue cases.
6. **Predictive features.** Strongest legal associations for the best risk
   family are:

- `f_decision_flip`: Spearman=-0.953, grouped permutation p=0.0033.
- `f_decision_margin`: Spearman=-0.952, grouped permutation p=0.0033.
- `f_decision_tv`: Spearman=-0.950, grouped permutation p=0.0033.
- `f_decision_logit_ratio`: Spearman=-0.928, grouped permutation p=0.0033.
- `f_geometry_strength`: Spearman=-0.881, grouped permutation p=0.0033.

7. **Does D add beyond P/U?**
   `{"dda_leave_run": {"status": "ESTIMATED", "baseline": "P+U", "augmented": "P+U+D", "R2_baseline": 0.2460425923, "R2_augmented": 0.7562932619, "delta_R2": 0.5102506696, "RMSE_baseline": 0.03931601221, "RMSE_augmented": 0.02235269146}, "wbcic_leave_fold": {"status": "ESTIMATED", "baseline": "P+U", "augmented": "P+U+D", "R2_baseline": 0.225965354, "R2_augmented": 0.4661799708, "delta_R2": 0.2402146168, "RMSE_baseline": 0.04026357592, "RMSE_augmented": 0.03343717479}}`
8. **Does geometry add?**
   `{"dda_leave_run": {"status": "ESTIMATED", "baseline": "P+U+D", "augmented": "P+U+D+geometry", "R2_baseline": 0.7562932619, "R2_augmented": 0.7549340129, "delta_R2": -0.0013592490000000623, "RMSE_baseline": 0.02235269146, "RMSE_augmented": 0.02241493969}, "wbcic_leave_fold": {"status": "ESTIMATED", "baseline": "P+U+D", "augmented": "P+U+D+geometry", "R2_baseline": 0.4661799708, "R2_augmented": 0.2647818427, "delta_R2": -0.20139812810000002, "RMSE_baseline": 0.03343717479, "RMSE_augmented": 0.03924101707}}`
9. **Does uncertainty reduce harm?** The best risk policy has unsafe rate
   `0.000`. Its exact comparison to the
   non-conservative ridge is stored in `FINAL_POLICY_CANDIDATE.json`.
10. **Baseline comparisons.**

### openbmi_sample_router
- `AlwaysSuppress`: ΔBA=-0.173106, LCB95=-0.187187, unsafe=0.244.
- `BestFixedTrain`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `ElasticNetEffect`: ΔBA=-0.000074, LCB95=-0.000124, unsafe=0.000.
- `M0_NO_OP`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `Oracle`: ΔBA=0.118607, LCB95=0.108371, unsafe=0.000.
- `P_only_gate`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `P_plus_U_gate`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `P_plus_U_plus_D_gate`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `RidgeEffect`: ΔBA=-0.000099, LCB95=-0.000175, unsafe=0.000.
- `RiskAwarePERSIST`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `SmallHGBEffect`: ΔBA=-0.000470, LCB95=-0.001096, unsafe=0.029.
### openbmi_dda_block
- `AlwaysSuppress`: ΔBA=-0.031501, LCB95=-0.033999, unsafe=0.815.
- `BestFixedTrain`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `ElasticNetEffect`: ΔBA=-0.000045, LCB95=-0.000192, unsafe=0.621.
- `M0_NO_OP`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `Oracle`: ΔBA=0.000632, LCB95=0.000397, unsafe=0.000.
- `P_only_gate`: ΔBA=-0.031501, LCB95=-0.034030, unsafe=0.815.
- `P_plus_U_gate`: ΔBA=-0.006799, LCB95=-0.014434, unsafe=0.687.
- `P_plus_U_plus_D_gate`: ΔBA=-0.003774, LCB95=-0.010289, unsafe=0.333.
- `RidgeEffect`: ΔBA=-0.000215, LCB95=-0.000389, unsafe=0.610.
- `RiskAwarePERSIST`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `SmallHGBEffect`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
### wbcic_development_block
- `AlwaysSuppress`: ΔBA=-0.018023, LCB95=-0.021900, unsafe=0.675.
- `BestFixedTrain`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `ElasticNetEffect`: ΔBA=-0.001769, LCB95=-0.004066, unsafe=0.515.
- `M0_NO_OP`: ΔBA=0.000000, LCB95=0.000000, unsafe=0.000.
- `Oracle`: ΔBA=0.001614, LCB95=0.001063, unsafe=0.000.
- `P_only_gate`: ΔBA=-0.014943, LCB95=-0.018142, unsafe=0.696.
- `P_plus_U_gate`: ΔBA=-0.002495, LCB95=-0.003925, unsafe=0.687.
- `P_plus_U_plus_D_gate`: ΔBA=-0.002828, LCB95=-0.004115, unsafe=0.933.
- `RidgeEffect`: ΔBA=-0.001331, LCB95=-0.003857, unsafe=0.611.
- `RiskAwarePERSIST`: ΔBA=-0.000021, LCB95=-0.000052, unsafe=0.400.
- `SmallHGBEffect`: ΔBA=0.000150, LCB95=-0.000519, unsafe=0.586.

11. **Recovered headroom.** Best risk policy recovers
    `0.000` of its grouped oracle headroom.
12. **Replication across groups.** Positive-group fraction is
    `0.000`; largest positive-group share is
    `N/A`.
13. **Unsafe intervention rate.** `0.000` for the
    best risk-aware candidate.
14. **Freeze decision.** `NO_POLICY_CANDIDATE`. This never
    authorizes opening outer test.

## Limits

- Leave-one-dataset-out is not estimable without pooling incompatible trial
  and block decision units.
- FBCNet is excluded from the WBCIC policy meta-data because its representation
  competence failed near chance.
- OpenBMI router U is unavailable independently of the target trial outcome;
  reporting a P+U router by inserting rescue/harm would be leakage.
- DDA and WBCIC use cross-fitted U only; same-cell realised U is excluded.
- A positive oracle is not a deployable policy and does not justify outer use.
