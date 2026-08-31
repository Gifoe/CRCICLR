# CGR-Fuse: Consensus-Guided Residual Fusion

This directory contains the bounded source-only CGR-Fuse experiment.  The
primary prediction unit is one EEG trial, with six frozen run predictions for
KEEP, AMPLIFY, and GEOMETRY.  Stable, high-confidence KEEP decisions are
reproduced exactly; only the consensus-unstable region is eligible for a
convex action mixture.  Subject IDs are used only as statistical grouping
variables, never as model inputs.

The executable entry point is `code/cgrfuse.py`.  It builds the OpenBMI bank
from the committed historical router cache and trains a fresh WBCIC S0->S1
bank using only the authorized S0/S1 development cache.  Runtime data and
checkpoints are intentionally ignored by git.  The compact parquet action
bank, CSV/JSON summaries, reports, figures, tests, and validation evidence
are the only deliverables.

The protocol is deliberately conservative.  The 12 declared recipes are the
only scientific search, and a failed minimum-support gate terminates the
constructive search with `CGRFUSE_SOURCE_NOT_SUPPORTED`; no WBCIC S2 or outer
resource is opened in that case.
