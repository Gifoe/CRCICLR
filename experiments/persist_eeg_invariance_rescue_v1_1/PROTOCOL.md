# Protocol

V1.1 is an outcome-informed exploratory redesign, not a preregistered
analysis. The protocol is frozen before any V1.1 outcome audit. The dataset is
OpenBMI/NEMAR `nm000273` motor imagery, development folds 0--2, two final
seeds. GRL candidates, train-side thresholds, protected assignment rules,
functional retention, SPL, rescue rank matching, and terminal states are
defined in `configs/protocol_v1_1.json` and `PROTOCOL_FROZEN.json`.

No outer subject membership, EEG, labels, or metadata are enumerated or used.
All output records set `outer_test_used=false` and
`outer_membership_enumerated=false`.
