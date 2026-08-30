# Autonomous SCST research ledger

| version | prior failure addressed | design-time data | status |
|---|---|---|---|
| SCST-V1 | full SCST did not beat ERM/RandomTransport | historical source and future result | immutable negative |
| ME-HardSCST-V2 | structured transport did not beat matched HardRandom; BA indistinguishable | historical source-only evidence | immutable negative |
| Bures-SCST-V3 | mean-only transport was replaced by second-order Bures geometry | OpenBMI/WBCIC source development | `BURES_SCST_TRANSPORT_NOT_REALIZED`; no confirmation |
| Repair-R1 | V3 class-fidelity/coverage failure and V2 structured~=random | source-only OpenBMI/WBCIC only | `R1_SOURCE_GATE_FAILED`; target affinity positive but class fidelity/coverage and utility failed |
| Repair-R2 | R1 global task-protected map remained locally heterogeneous and class-unsafe | source-only OpenBMI/WBCIC only | `R2_SOURCE_GATE_FAILED`; local OT did not beat controls |
| Repair-R3 | R2 local OT retained class-semantic contamination and no utility advantage | source-only OpenBMI/WBCIC only | `R3_SOURCE_GATE_FAILED`; final constructive round |

Constructive terminal: `SCST_CONSTRUCTIVE_SEARCH_EXHAUSTED`.  Future architecture
confirmation was not run because no repair passed the source gate; outer and
sealed resources remain closed.
