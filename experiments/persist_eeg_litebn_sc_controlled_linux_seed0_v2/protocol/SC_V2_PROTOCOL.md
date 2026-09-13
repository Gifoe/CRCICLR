# Controlled LiteBN-SC v2 protocol

## Scope

The locked full design contains B0 (historical one-forward LiteBN), C0
(two-forward DualCE), and C1 (two-forward DualCE plus symmetric KL). Per the
current execution instruction, this run completes only C1 for WBCIC_MI seed 0,
folds 0--4. B0 is replayed from the matched frozen Linux checkpoints. C0 and
Phase 2 are not run in this scope.

## Identity

All methods use the authoritative historical `CompactLite(channels, "bn")`
implementation. C1 has no architecture, parameter, buffer, classifier, or
inference-graph addition. Formal evaluation uses one checkpoint, one model, and
one deterministic eval forward.

## C1 training

Each original manifest batch performs two forwards and exactly one optimizer
step. Forward A consumes the historical main Torch RNG stream. Forward B consumes
an independent persistent secondary Torch RNG stream. The B stream advances over
the entire run; after B, the post-A main stream is restored.

Before A, all buffers are cloned for B. A executes on the live model and advances
the persistent BatchNorm buffers once. B executes with:

```python
torch.func.functional_call(model, (parameters, buffers_B), (x,), strict=True)
```

`parameters` are the original Parameter objects, while `buffers_B` are detached
clones. Both stochastic graphs therefore train the same parameters, B cannot
mutate live buffers, and no live buffer is copied in-place before backward.

The locked objective is:

```text
L_CE = 0.5 * (CE(logits_A, y) + CE(logits_B, y))
J = 0.5 * mean(sum(p_A * (logp_A-logp_B)) + sum(p_B * (logp_B-logp_A)))
L = L_CE + 0.5 * J
```

Network forwards retain historical AMP. `J` is computed explicitly in FP32 with
autocast disabled. Optimizer, GradScaler, clipping, 60 epochs, episode manifests,
normalizers, validation, eligibility from epoch 10, and earliest-tie selection
are unchanged from the matched Linux LiteBN reference.

## Preflight and evidence boundary

Real inner-train batches verify identity, exact one-update BN persistence, shared
parameter gradients from forward B, total graph safety, FP32 and AMP behavior,
lambda-zero C0 equivalence, null/symmetric KL, no-op extra-B replay, and exact
10-step interruption/resume replay including both RNG streams.

All five C1 checkpoints are frozen before outer evaluation. Only after outer is
frozen is the already-open internal-heldout cohort evaluated. It is labeled
`DEVELOPMENT_MODEL_SELECTION_DATA`; no new sealed test is accessed.

Because C0 is outside the current run scope, the mandatory C1-versus-C0 gate
cannot be evaluated. If C1 already fails any C1-versus-B0 condition, the result
is `SC_NO_CONTINUATION_SIGNAL`. Otherwise it is
`C1_WBCIC_COMPLETE_C0_REQUIRED`; neither label authorizes Phase 2.
