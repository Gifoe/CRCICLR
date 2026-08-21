# Post-freeze reporting repair

The frozen scientific implementation is commit
`873486f7f215ce3b5287d8220d660e06913ea10f`. After final statistics, the report
generator was changed only to make the negative result harder to overclaim.

The repair adds no method, rerun, outcome selection, rescue, gate, threshold,
or metric calculation. It:

- marks rescue/selectivity as not estimable because no family was eligible;
- explains that C's formal I2-failure label is not affirmative preservation;
- reports matched non-Protected retention and the cross-model-identifiability
  limitation of PRS;
- keeps lower GRL lambdas explicitly exploratory rather than promoting them.

All numeric outputs and `NO_ELIGIBLE_PROTECTED_LOSS_OBSERVED` are unchanged.

A second reporting/provenance-only repair clarifies that the complete
`SPLIT_FREEZE.json` was hashed as opaque bytes even though outer membership was
never indexed or enumerated, and consolidates the full, binding-smoke, and
excluded-debug run ledgers. It does not alter any scientific computation.
