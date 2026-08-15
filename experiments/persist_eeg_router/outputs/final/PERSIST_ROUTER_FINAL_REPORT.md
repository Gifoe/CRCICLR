# PERSIST-Router final report

Terminal state: `PERSIST_ROUTER_NO_ROUTEABLE_HEADROOM`

## Outcome

The matched base achieved TRAIN-only subject-disjoint OOF BA `0.826838`. The selected R0 achieved `0.826373` (Delta BA `-0.000466`, `1/6` positive runs). It rescued `83` base errors and harmed `102` base-correct samples.

## Mechanism

ERASE had diagnostic KEEP-union oracle headroom `0.071029` BA, but standalone ERASE changed BA by `-0.173407`. The best diagnostic Router AUROC was `0.963634`, yet its deployable Delta BA was `-0.000172`. High rescue ranking did not separate rescue from harm well enough to improve decisions.

## Protocol decision

R1, R2, and R3 are not authorized because even raw-best R0 was below +0.002 BA. No Router lock was written; OpenBMI development-validation, EEGMMIDB, and OpenBMI outer-test were not accessed. The result is explanatory rather than accuracy-improving.
