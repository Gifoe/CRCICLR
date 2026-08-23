# PERSIST-Net final constructive experiment

This directory contains the final theory-derived protection-by-construction
test for PERSIST-EEG.  It starts from commit
`47e773142cc8cd098eeeb89103bebe68c525d760` and never modifies historical
experiments.

Execution is server-only.  Preflight/audit artifacts are created before model
training; runtime checkpoints and the authorized raw cache stay untracked.
Lightweight CSV/JSON results, reports, and figures are committed after the
frozen development gate is evaluated.
