# PERSIST-EEG prospective action policy V2

## Terminal state

`DEVELOPMENT_HOLDOUT_SUCCESS_NEW_PROTOCOL_REQUIRED`

The autonomous search used 40 exploration subjects. Twelve development
subjects were opened once only after candidate specifications and code hashes
were locked. WBCIC outer data were never accessed.

## Scientific result

The single-run confidence and regularized error models failed. A deterministic
leave-target-run consensus rule passed the exploration stopping gate, so search
stopped. Its development-holdout result is reported below without subsequent
method changes.

| policy_id | mean_subject_delta_BA | bootstrap_CI95_L | bootstrap_CI95_U | action_rate | rescue_precision | unsafe_intervention_rate | positive_run_fraction | recovered_oracle_headroom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_KEEP | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| I003_CROSS_RUN_FULL | 0.008472 | 0.003542 | 0.013750 | 0.080652 | 0.559299 | 0.440701 | 0.666667 | 0.075635 |
| I003_CROSS_RUN_PROTECTED_SAFE | 0.007326 | 0.003681 | 0.010834 | 0.032500 | 0.612040 | 0.387960 | 1.000000 | 0.065406 |
| ORACLE_FULL_MENU | 0.112014 | 0.077847 | 0.143681 | 0.104891 | 1.000000 | 0.000000 | 1.000000 | nan |

## Interpretation

- Best frozen candidate: `I003_CROSS_RUN_FULL`
- Development-holdout Delta BA: `0.008472`
- Grouped bootstrap LCB95: `0.003542`
- Rescue precision / harm: `0.559` / `0.441`
- Positive-run fraction: `0.667`
- Recovered holdout oracle headroom: `0.076`

The method is not a single-model PERSIST router. It requires multiple frozen
run experts at inference, and the full-menu variant may select ERASE. The
protected-safe result must therefore be read separately. A positive V2 result
would still be exploratory and would require a new independent protocol.

`OUTER_TEST_USED = false`
