# Relative Sidecar, OpenBMI fold0

This seed-0 SEARCH-only screen keeps the selected epoch-53 EEGNet-ERM model
bitwise frozen.  It compares an additive Raw-Sidecar with an identically
initialized additive Relative-Sidecar.  The relative input is the frozen-base
embedding minus a separately pre-trained, frozen single-trial offset estimate.
No legal model consumes target subject identity, session identity, history, or
labels at inference.
