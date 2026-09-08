# Method

The audit reuses the source `FIVEFOLD_SPLIT.json`, cache preprocessing and
normalizer semantics. OpenBMI evaluates S2 from S1 normalization; WBCIC
evaluates S3 from S1+S2 normalization. Only listed V8_SEARCH subjects are
loaded. Subject and trial oracles are label-informed, non-deployable upper
bounds and are never treated as methods.
