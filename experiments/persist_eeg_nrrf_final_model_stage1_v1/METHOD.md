# NRRF-v1

For cached EEG epoch `x`, the exact carrier pair provides
`logits_A = EEGNet(x)` and `logits_E = LiteBN(x)`. The frozen anchor is
always in evaluation mode and executes under `no_grad`. The deployable
prediction is fixed before execution:

`logits_F = 0.5 logits_A + 0.5 logits_E`.

Only LiteBN is trainable. Each optimization batch deterministically draws up
to eight source-session subjects and sixteen trials per chosen subject,
class-balanced 8/8 whenever feasible. Subject losses are used to form
`L_mean`, a worst-`ceil(0.25K)` subject tail loss, anchor-relative positive
regret, and an expressive-branch cross-entropy loss.

`NRRF-v1 = L_mean + 0.5 L_tail + 1.0 L_regret + 0.25 L_expr`.

The sole matched control is `JOINT-CE = L_mean + 0.25 L_expr`; it shares all
information, initialization, batches, normalization, optimizer, 20-epoch
schedule, and EMA handling. Its only omission is the tail and regret terms.

OpenBMI optimization uses inner-train session 1 only. WBCIC optimization uses
inner-train sessions 1+2 (cache session indices 0+1) only. Session 2 for
OpenBMI and session 3 for WBCIC (index 2 in each cache) are outer-development
evaluation only and never participate in optimization.
