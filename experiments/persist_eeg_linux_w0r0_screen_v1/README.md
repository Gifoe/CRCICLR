# Linux W0R0 seed-0 screen

This experiment is the user-requested Linux adaptation of the W0R0-only part
of the earlier Windows W/R screen. It trains W0R0 from scratch at seed 0 on the
canonical four-task, five-fold SEARCH/development protocol and strictly replays
the existing matched Linux seed-0 B0 checkpoints.

The exposed cohort is named `DEVELOPMENT_MODEL_SELECTION_DATA`. No new sealed
or final-test cohort is accessed. This single-seed result is not a final-model
claim.

W0R0 is exactly W0 width (8 temporal, 16 spatial, 64 backend), with C/S/M on,
three residual mixers at dilations 1/2/4, and the R0 adaptive eight-bin ordered
readout (512 -> 64 -> task classes).

