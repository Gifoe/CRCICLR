# Frozen LiteBN-TSW seed0 protocol

Exact historical CompactLite plus one scale-only intervention after each branch's
historical first AvgPool/dropout and immediately before original concat. TSW uses
the prior RG gate exactly: branch time/spatial means concatenate to 48 dimensions,
MLP 48-24-3 with GELU, alpha=softmax, and scale
`1+tanh(lambda_scale)*(3*alpha_i-1)`. Lambda starts at zero. StaticScale uses only
three zero logits and weights `3*softmax(beta)`. No RG mixer/readout/widening or
other rejected component is present.

Seed0 only. Run tasks serially in this frozen order: OpenBMI_MI, WBCIC_MI,
OpenBMI_ERP, OpenBMI_SSVEP. For each task, first pass both architecture's five-fold
initialization/function/RNG/module-tree audits, then train five StaticScale cells
and five TSW cells under the exact historical task protocol. Evaluate outer and
the previously exposed internal heldout diagnostic only after all ten cells.

User-requested fail-fast override: a task fails if protocol/execution verification
fails or its five-fold mean outer TSW delta versus exact LiteBN is <=0. In that
case emit `TASK_STOP`, do not launch the next task, and await review. This criterion
is frozen before candidate results. It is stricter than the prompt's final
four-task aggregate gate and prevents unnecessary work once 4/4 positivity is
impossible.

Historical MI uses exact stored 60x episode manifests and batch128; ERP/SSVEP use
historical deterministic session1 full-permutation batch64 (ERP class-weighted CE).
Optimizer, LR3e-4, WD5e-4, clip5, AMP, 60 epochs, canonical subject-equal inner-val
BA, epoch10 eligibility and strict-improvement/earliest tie are unchanged. All
new parameters share the historical optimizer; no auxiliary loss or separate LR.

Common weights/buffers must be bitwise identical. New module construction uses
independent subseed314159 and restores historical RNG. For both controls, initial
logits max difference <=1e-6 and prediction mismatch zero in eval/train FP32/AMP;
equal-RNG forward must retain identical common BN buffers and dropout RNG. TSW's
initial lambda gradient and StaticScale beta gradient must be finite/nonzero.

Re-evaluate hash-verified historical checkpoints with exact normalizers/metrics,
requiring per-subject BA/F1/accuracy match within1e-10. Diagnostics never affect
selection. Record lambda/amplitude, alpha and scale distribution/dominance by
task/fold, subject and class. A nearly constant gate, lambda collapse, or branch
dominance is reported without rescue.

No new sealed cohort, seed1/2, post-result gate change or ablation. Available
Windows CUDA differs from original Linux binaries; exact initialization/sampling
does not prove bitwise full-training cross-runtime identity. AMP attempts and
successful steps are recorded separately.
