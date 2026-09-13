# XS counterfactual factorial protocol

This development-only audit replays the frozen XS seeds 0/1/2 for all four tasks and five existing outer folds. It uses the exact fixed B0, trial keys, normalizers, checkpoints, and repaired ERP numerical semantics from the stable-rescue audit.

The eight predeclared states form a complete `C x S x M` cube. `C=0` sets only `lambda_channel=0`; `S=0` sets only `lambda_scale=0`; `M=0` sets only the three trained mixer `gamma` scalars to zero. All interventions occur on an in-memory checkpoint load under `model.eval()` and `torch.no_grad()`. No weight file is changed or saved. `C0S0M0` remains the trained wide XS network and is not LiteBN.

`C1S1M1` must exactly reproduce the authoritative full-XS trial keys, labels, predictions, subject set, and subject-equal BA within `1e-8`; an invalid cell excludes its whole cube. Primary Rescue, Harm, and NET quantities are computed within each real subject and then averaged equally. Subject bootstraps use 10,000 paired real-subject resamples.

Factorial effects use `OFF=-1`, `ON=+1`; every reported effect is `mean(product-sign=+1) - mean(product-sign=-1)`. These are post-training counterfactual associations, not causal estimates or accuracy estimates for an architecture trained without a mechanism.

The global structural screen is fixed in advance: one reduced state must satisfy every Rescue-retention, Harm-reduction, equal-task NET, task consistency, cell consistency, worst-task, and protocol criterion to become `NEXT_RETRAIN_CANDIDATE`. No task-specific architecture is selected.
