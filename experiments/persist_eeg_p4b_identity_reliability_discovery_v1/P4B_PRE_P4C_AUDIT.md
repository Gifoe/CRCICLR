# P4B Pre-P4C Audit

Audit PASS on validated/pushed P4B tip `b4d6fcabf4e8e33501d32c9bb1b70625ac93106b`. Exact P4B terminal: `P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED`.

- RMSE MI=0.007967329; MINT=0.022863854; MINT better=False.
- beta_IxE=0.000094884, CI=[-0.0015074821732454515, 0.0006757057062616963]; required sign negative=False.
- DeltaSlope=-0.000189769, CI=[-0.0013514114125233927, 0.0030149643464909004]; required direction positive=False.
- DeltaRegime=0.006690019, CI=[0.000983032858042108, 0.012903471187988244]; required direction positive=True.
- Per-setting interaction-direction consistency=75.0%.
- Maximum setting/run absolute influence shares=99.2%/35.6%; these diagnostics did not trigger any setting removal or rescue.
- P4A/source hashes and pre-outcome timing: PASS.
- S4/S6 future utility: UNTOUCHED. OpenBMI sealed holdout: UNTOUCHED. WBCIC outer 10: UNTOUCHED.
- P4C_LOCK is intentionally absent because the validated P4B terminal does not authorize one.

The positive DeltaRegime alone is insufficient: frozen MINT transports much worse than MI, and both beta_IxE and DeltaSlope have the wrong point direction.
