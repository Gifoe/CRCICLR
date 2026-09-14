# Engineering repair ledger

The first Stage-0 invocation completed all identity, frozen-base, gradient,
future-harm, and 10,000-draw bootstrap computations and wrote their numerical
artifacts. Final Markdown formatting then failed because Pandas renamed the
reserved output column `pass` in a named tuple. The repair changed only report
row access to use the original DataFrame column. Aggregation was rerun from the
already saved gradient observations; all numerical summaries reproduced
exactly. No architecture, data, pairing, optimizer, statistic, decision gate,
or result was changed.
