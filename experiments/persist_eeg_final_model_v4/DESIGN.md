# Design contract

- OpenBMI primary baseline: frozen `B6_ALL_RUN_LOGIT_MEAN` (`B_STRONG`).
- WBCIC development primary baseline: the audited five-expert probability
  mean. The initially considered logit mean is weaker by `0.158 pp` and is not
  used to inflate the final comparison.
- Primary endpoint: mean subject-balanced accuracy difference from B_STRONG.
- Required information ladder: static B_STRONG, dynamic KEEP, KEEP+ACTION,
  KEEP+ACTION+PERSIST.
- Candidate selection: model-fit subjects only; hyperparameters and thresholds
  on disjoint calibration subjects; one evaluation on held-out subjects.
- OpenBMI: exploratory only.
- WBCIC: development subjects from `DEVELOPMENT_SCOPE_LOCK.json` only.
- Explicitly forbidden: `OUTER_SPLIT_LOCK.json`, outer cache, outer labels,
  outer logits, outer metadata, or raw paths for non-development subjects.
- No positive result may be claimed unless the paired subject bootstrap lower
  confidence bound is above zero and fold/subject stability is acceptable.

Executed families include threshold calibration, generic linear stacking,
shallow boosted trees, bounded anchored residual correction, positive logit
pooling, positive probability pooling, contextual pooling, and DeepSets.

The final selected discovery architecture is an availability-normalized
positive pool over frozen KEEP margins with a learned scale/bias and a narrow
inner-calibrated threshold. ACTION and PERSIST are excluded because the matched
A0-A9 ladder shows no incremental performance or safety value. Direct transfer
to WBCIC development fails, so the freeze is a research freeze and does not
authorize opening the sealed outer set.
