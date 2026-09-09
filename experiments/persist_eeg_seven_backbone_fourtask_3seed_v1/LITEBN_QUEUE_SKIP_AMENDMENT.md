# Execution amendment: skip already-complete LiteBN continuation cells

The frozen seven-backbone protocol remains unchanged.  Existing LiteBN
results are available from the prior benchmark work; the current runtime's
partial LiteBN records are retained without modification.  At the user's
request, the continuation queue omits all remaining LiteBN cells and starts
with the next unfinished backbone, TCFormer.  No split, seed, architecture,
training hyperparameter, metric, or held-out access rule is changed.

The queue still reports its own continuation denominator (six unfinished
backbone families × four tasks × five folds × three seeds = 360 cells); the
retained LiteBN records remain discoverable under the same runtime root and
are included by downstream aggregation when available; this continuation does
not claim that the current runtime alone is a complete LiteBN grid.
