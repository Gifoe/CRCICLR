# Confirmation resource ledger

| resource | status before autonomous repair | status during R1 |
|---|---|---|
| OpenBMI session-1 -> session-2 | DEVELOPMENT-KNOWN | source-only development |
| WBCIC S1 -> S2 | DEVELOPMENT-KNOWN | source-only development (R1/R2/R3) |
| ATCNet-Official architecture-level confirmation | UNTOUCHED | UNTOUCHED |
| EEGNeX architecture-level confirmation | UNTOUCHED | UNTOUCHED |
| WBCIC outer / sealed resources | SEALED | SEALED; not opened |

R2 and R3 remained source-only.  Because all three repair rounds failed the
source gate, the conditional architecture-level confirmation (EEGNeX first)
was not authorized and is recorded as `NOT_RUN_SOURCE_GATE`.

If a repair passes all source gates, EEGNeX is the first untouched
architecture-level evaluation as required by the autonomous addendum.  No
future resource is opened during source-only R1.
