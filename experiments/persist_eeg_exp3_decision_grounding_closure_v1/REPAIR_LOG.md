# Repair log

1. The report formatter was corrected to consume `tests["A"]`, `tests["B"]`,
   and `tests["C"]` after the code had assembled those aliases from the nested
   `STATISTICAL_TESTS.json.deltas` object. The repair is reporting-only; it does
   not alter identity measurements, decision metrics, model fits, or tests.
2. The server was rerun in the frozen order `audit -> compute -> finalize`
   using code commit `668d0fca06bd1756c935f1997945fc419c391dc0`, so the
   protocol lock and reproducibility record agree.
3. No source DDA-B, DDA-C, or V1.2 artifact was modified. Runtime path and
   cache handling are explicit through `PERSIST_DDA_ROOT`, `PERSIST_V12_ROOT`,
   and `PERSIST_SIGNED_ROOT`; all derived outputs are new files under this
   experiment.
