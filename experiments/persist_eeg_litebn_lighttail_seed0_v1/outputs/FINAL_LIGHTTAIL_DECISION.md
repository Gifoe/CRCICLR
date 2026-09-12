# LiteBN-LightTail seed0 final decision

LIGHTTAIL_NO_USEFUL_SIGNAL

Exact manifest: all 6,300 episodes matched; exposure 21 x 128 trials per epoch, 60 epochs per fold. Tail0 tests passed for all folds in full precision and historical AMP.

Outer BA: 0.791500 -> 0.776250, delta -1.525 pp; positive/negative/tied folds 0/4/1.
Current internal heldout BA: 0.744857 -> 0.738143, delta -0.671 pp; paired bootstrap 95% CI [-1.529, +0.171] pp.
Fold SD: 3.758 -> 3.971 pp (ddof=0).

Lower-tail subjects (delta BA pp):
- outer: median -1.000; q25 -4.000; q10 -6.100; worst -14.000; positive ratio 0.350; baseline-hardest-quartile mean -1.700.
- heldout: median -0.400; q25 -1.750; q10 -2.940; worst -3.400; positive ratio 0.429; baseline-hardest-quartile mean -0.500.

Does not meet the frozen continuation criteria. Do not expand seeds/tasks or change tail weight based on this run.

AMP successful updates 6297/6300 attempts; skipped updates 3. Both attempts and complete sample exposure match the historical manifest. AMP scaler behavior is unchanged; any overflow is logged.

This is the already-open current internal heldout diagnostic, not untouched final data. No new population or split was created.

Post-run optimizer audit: historical and candidate totals are both 6,297 successful
updates out of 6,300 attempts, but per-fold counts differ (historical:
1260/1259/1259/1259/1260; candidate: 1259/1260/1259/1260/1259).
Exposure matching does not mean identical AMP skipped-step timing. The current
Windows runtime differs from the original Linux binary environment; exact
initialization and one-step equivalence do not prove a bitwise-identical full
60-epoch CE trajectory. Interpret this as a seed0 diagnostic, not a fully
runtime-controlled causal estimate. See README.md and ENGINEERING_LEDGER.md.

```text
TASK = OpenBMI_MI
SEED = 0
FOLDS = 5
EXACT_HISTORICAL_MANIFEST = YES
TRAINING_EXPOSURE_MATCHED = YES
TAIL_WEIGHT = 0.1
SAMPLER_CHANGED = NO
BATCH_SIZE_CHANGED = NO
CHECKPOINT_SELECTION_CHANGED = NO
NORMALIZATION_CHANGED = NO
OUTER_DEVELOPMENT_EVALUATED = YES
CURRENT_INTERNAL_HELDOUT_EVALUATED = YES
NEW_SEALED_FINAL_DATA_ACCESSED = NO
```
