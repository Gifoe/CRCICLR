# Bug description

The first server-side `finalize` attempt exposed a report-generation bug: the
statistics file stores the three deltas under `STATISTICAL_TESTS.json.deltas`,
but the report formatter looked for the same keys at the top level. This raised
`KeyError: 'A_MD_vs_M0'` after all primary computations had already completed.

No historical DDA-B, DDA-C, or V1.2 scientific artifact was changed. The V1.2
matched-identity intervention remains terminally infeasible under its frozen
train-side measurability gate; this experiment measures identity evidence
directly instead of repeating that intervention.
