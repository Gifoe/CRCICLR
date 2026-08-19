# Iteration 006: Fixed variance-reduced local stack

- New hypothesis: Removing inner configuration variance will stabilize the local signal.
- Grouped result: +0.867 pp, positive CI, 5/5 folds.
- Conclusion: `KEEP`
- Evaluation: subject-disjoint WBCIC development folds; target S3 outcomes were evaluation-only for held-out subjects.
- `OUTER_TEST_USED=false`.
