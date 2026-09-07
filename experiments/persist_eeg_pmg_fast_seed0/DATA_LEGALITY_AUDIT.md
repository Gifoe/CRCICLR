# Data legality audit

The pilot uses only the frozen OpenBMI MI manifest and the frozen WBCIC development cache. For each fold, model-fit subjects provide S1+S2 (OpenBMI) or S1+S2 (WBCIC) source rows; discovery subjects provide the canonical future-session source-only evaluation (S2 OpenBMI or S3 WBCIC). The implementation constructs no outcome indices, never opens WBCIC sealed outer ten, never opens an OpenBMI sealed/internal cohort, and uses no target adaptation, router, task prior, or dataset-specific signal prior.
