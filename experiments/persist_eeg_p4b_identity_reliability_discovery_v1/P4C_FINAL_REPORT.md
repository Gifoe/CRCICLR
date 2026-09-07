# P4C Final Report — Not Authorized

## Decision

- P4B exact terminal: `P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED`.
- P4B validator: PASS on `b4d6fcabf4e8e33501d32c9bb1b70625ac93106b`.
- P4C authorization: **DENIED**.
- CONDITION_STATUS: `P4C_NOT_AUTHORIZED_BY_P4B`.
- ACTIONABILITY_STATUS: `P4C_ACTIONABILITY_NOT_EVALUATED`.
- FINAL_MODEL_AUTHORIZATION: `NOT_AUTHORIZED`.

## Why the gate failed

- Frozen LOSO-setting RMSE MI: 0.007967329.
- Frozen LOSO-setting RMSE MINT: 0.022863854.
- RMSE_MI - RMSE_MINT: -0.014896525, CI=[-0.032068833931208574, 0.0005675010482946285]; the point estimate is negative, so MINT is worse.
- beta_IxE: 0.000094884, CI=[-0.0015074821732454515, 0.0006757057062616963]; the point sign is positive, opposite to the hypothesis.
- DeltaSlope: -0.000189769, CI=[-0.0013514114125233927, 0.0030149643464909004]; the point direction is negative.
- DeltaRegime: 0.006690019, CI=[0.000983032858042108, 0.012903471187988244]; this component is positive, but cannot override the failed predictive and interaction gates.
- Per-setting interaction-direction consistency: 75.0%.

## Prospective evaluation status

Reserved settings remain exactly S4 and S6. Their future BA/F1/CE, per-direction utility, ranking, oracle, and policy outcomes were never opened. No P4C_LOCK, prediction freeze, P4C branch, prospective RMSE, policy comparison, or oracle analysis was created because doing so would violate the authorization gate. Accordingly all P4C outcome quantities are `NOT_EVALUATED`, not missing results.

OpenBMI sealed internal holdout is untouched. WBCIC outer 10 is untouched and unenumerated. There was no post-outcome model, E_task, normalization, threshold, alpha, top-k, or setting modification. No final model was trained.
