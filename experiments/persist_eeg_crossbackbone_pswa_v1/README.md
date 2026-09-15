# Cross-backbone PSWA v1

This is a frozen-artifact, probe-only audit. It never trains a model, reruns
inference, recomputes Protected assignments, recomputes the persistence
spectrum, or resamples random coordinate controls.

The runner first verifies whether the corrected PEEH run persisted every
artifact needed to estimate Protected-only and equal-rank random-only WS-BA.
Cells are marked incomplete rather than reconstructed when any frozen artifact
is absent.
