# Stage-0 v1 invalidation

Status: **SUPERSEDED — NOT A VALID SCIENTIFIC GATE-B CONCLUSION**

The preserved v1 artifacts are retained for provenance only. They must not be cited as evidence for a scientific Gate-B decision. The v1 run is invalidated because:

1. The task was not first demonstrated to be adequately learnable by a tiny-set overfit and an independent EEG sanity baseline.
2. The source observation was inconsistent: the operator matrices used `B0 = C Ψ` (CAR64), while the model observed the raw cached signal before applying `A` (a `Y/B` mismatch).
3. Gate A could be satisfied by a single worst-operator result rather than a preregistered aggregate over held-out families.
4. Test-only categorical family dimensions could affect B4, introducing an information path unavailable during training.
5. Channel-row positional embeddings assigned unstable identities across heterogeneous operators.

The v1 report, checkpoints, plots, logs, and tables in this directory are frozen unchanged. The v2 workflow starts from `STAGE0_V1_INVALIDATED` and requires the repair and re-falsification gates described in the v2 protocol.
