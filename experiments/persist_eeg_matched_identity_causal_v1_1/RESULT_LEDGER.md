# Result ledger

This ledger records the V1.1 milestones.  COMMIT A must contain the complete
block-wise train-only feasibility audit and frozen protocol before any
validation task outcome.  COMMIT B must contain the held-out manipulation,
causal task results, final report and reproducibility audit.  Any post-freeze
bug fix must state whether it changed scientific semantics and, if so, force a
new freeze and final rerun.

Post-freeze repair log:

* Finalize attempt after frozen validation failed in `bootstrap_v11` because a
  Pandas aggregation left `subject_id` simultaneously as an index level and a
  column.  Commit `POST_FREEZE_BUGFIX` resets the input index before the
  subject-cluster bootstrap.  Old behavior: finalize raised before writing
  statistics.  New behavior: the same already-written raw validation outputs
  are aggregated deterministically.  Scientific semantics: unchanged; no
  control, dose, metric, estimator or gate was modified.

Semantic repair / freeze revision 2:

* The first final audit found that the G2 implementation accepted a zero-drop
  Protected arm whenever the P/N difference was within 0.01 BA. The contract
  requires measurable identity reduction in both arms. The revised rule is
  `mean(Delta_ID_P)>0 AND mean(Delta_ID_N)>0` plus the unchanged 0.01 BA
  tolerance. This changes scientific semantics, so the original freeze is
  superseded and `PROTOCOL_FROZEN.json`/`PROTOCOL_FREEZE_AUDIT.json` receive a
  new revision before the final phase is rerun. The controls, doses, metric,
  estimator and G3 threshold are unchanged. The pre-repair decision is kept
  in server outputs as `*_PRE_SEMANTIC_REPAIR`.

Post-freeze reporting repair:

* The compact first report did not explicitly answer all 15 protocol questions
  or summarize per-block ranks/control counts. `make_report_v11` now emits a
  numbered audit report, including dose direction, random-control diagnostics,
  leakage state, and the final claim. Scientific semantics are unchanged; this
  is a reporting/completeness repair. The code hash was re-frozen and final
  outputs were regenerated before the second finalize pass.
