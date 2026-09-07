# V1.2 protocol

The primary continuous identity variable is symmetric cross-session subject-ID
identity skill, `log(K) - cross-entropy`, averaged over S1→S2 and S2→S1. Top-1
BA is retained only as a human-readable secondary metric. Correct-subject
logit margin and same-vs-impostor retrieval margin are also reported.

V1.2 uses a dense alpha grid (0.00–1.00 by 0.02, selected for train-only
computational feasibility before freeze), train-only nondecreasing
isotonic regression, and linear interpolation on the fitted response. The
train noise floor is twice the deterministic subject-bootstrap standard error
at alpha=0.05. Dmax is the largest fitted drop whose 95% train-side lower
bound is positive and exceeds that noise floor. LOW/MEDIUM/HIGH are 25/50/75%
of Dmax, with MEDIUM primary. Each fixed V1.1 Non-Protected control independently
solves the same target; no new controls can enter.

G1 requires at least 5/6 runs, 80% blocks, and 20 fixed controls per eligible
block at MEDIUM. G2 requires both P and N to have measurable held-out
continuous identity reduction and a 90% subject-bootstrap P−N CI wholly within
the train-frozen epsilon. G3 retains the V1.1 threshold: mean Delta_H ≥ 0.01
BA, 95% LCB > 0, at least 5/6 positive runs, and at least 60% nonnegative
subjects. G3 is not run when G2 fails.
