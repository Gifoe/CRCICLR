# Iteration ledger

1. **V0 (authorized)**: data/purity preflight, parameter matching, exact-D unit
   test, one source-only smoke fold, gradient/freeze/scale diagnostics.
2. **V1 (authorized)**: frozen architecture and losses from
   `PROTOCOL_FROZEN.json`.
3. **V1.1 (conditional, at most once)**: only a mechanism-preserving repair
   for a documented non-finite value, missing gradient, branch collapse,
   distillation-scale error, or protected-freeze violation.

Outcome-driven modules, routers, attention additions, and expanded searches
are forbidden.  Runtime entries will record whether V1.1 was invoked.
