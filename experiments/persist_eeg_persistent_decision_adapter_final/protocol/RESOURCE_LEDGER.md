# PDA resource ledger

| Resource | Status | Use |
|---|---|---|
| CleanRoom ATCNet `model_fit` | DEVELOPMENT_KNOWN | frozen population logits and historical adapter fitting |
| CleanRoom ATCNet `validation` | HISTORICAL_TRAIN | source-only prospective transition metrics and historical fitting |
| CleanRoom ATCNet `outcome` | CONFIRMATORY_USED | source-only held-out transition metrics; not WBCIC S2 |
| WBCIC S2 future-session resource | UNTOUCHED_FUTURE | sealed; source gate failed |
| OpenBMI sealed/outer resource | SEALED | not inspected |
| WBCIC outer resource | SEALED | not inspected |
| raw EEG/checkpoints/cache | SEALED | not included in package |

The source archive metadata contain biological subject IDs, session IDs, trial
indices, labels and frozen features/logits. The future-session utility paths
remain inaccessible until a source-passed lock exists; this run never created
such a lock.
