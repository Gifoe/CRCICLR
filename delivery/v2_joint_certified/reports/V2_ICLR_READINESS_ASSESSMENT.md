# V2 ICLR readiness assessment

## Decision: NO-GO

The current result does not support the main empirical claim. This judgment is driven by the outcome, not implementation effort:

- Safe-Oracle headroom exists, so the action library is not the sole blocker.
- Nested joint validity is conservative, but TTA selection rate and Safe-Oracle gain captured are both zero.
- Benefit prediction does not reliably outperform the constant-zero baseline.
- Aggregate classification utility is exactly No-TTA rather than better than it.
- Safety materially depends on sentinel full-set fallback.

## What would change the decision

A new development cycle—not the tainted final sets—must demonstrate all of: nonzero positive-benefit certification with subject-level uncertainty, stable selected-TTA PPV, captured Safe-Oracle gain with confidence intervals above zero, nontrivial CSR without dominant full-set fallback, and an advantage over fixed/heuristic policy CRC baselines. Only then should the frozen confirmatory protocol be used.
