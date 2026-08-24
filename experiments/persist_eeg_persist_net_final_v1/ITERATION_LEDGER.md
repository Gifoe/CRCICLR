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

V0 repair 1 was a read-only Arrow mask copy fix before any training or outer
outcome access; see `UPSTREAM_REPAIR_LOG.md`.

V0 repair 2 locks constructor-time initialization and pairs matched-control
initialization/minibatch order. Repair 3 fixes an output-only subject-ID delta
lookup. Both were identified before outer outcome evaluation; no scientific
module, hyperparameter, certificate rule, or gate changed.

Repair 4 makes each single-path baseline use its own source-only selected epoch
instead of borrowing another candidate's epoch. It was locked before final
development runs.

Repair 5 pairs target-history minibatch order across all dual-path methods for
each fold/seed/subject. It was identified by static audit before any outcome
marker was emitted; five partial source-training jobs were stopped and all 15
final development jobs are rerun. This is a controlled-comparison repair, not
an outcome-driven scientific iteration.

Repair 6 moves the unchanged frozen run under Windows Task Scheduler after an
OpenSSH connection reset killed its descendant processes. No outcome marker or
completed run existed. This is execution-lifetime hardening only; the entire
15-run development matrix is restarted without inspecting outcomes.
