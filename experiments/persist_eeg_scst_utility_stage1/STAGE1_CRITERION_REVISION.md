# Stage-1 criterion revision

Stage-0 correctly ended `NO_ADMISSIBLE_COMPETENT_REPRESENTATION_FOUND` under
its frozen binary `3NN <= 1.25` screen. This result is not overwritten.

For the new hypothesis, `1.25` is treated as a conservative heuristic rather
than a biological or mathematical phase transition. Before any new
future-session utility is inspected, Stage-1 freezes an absolute plausibility
guardrail of `3NN <= 1.30`. Eligibility additionally requires task competence,
positive cross-session residual stability with positive subject-bootstrap CI
lower bound, positive subject fidelity with positive CI lower bound, positive
matched-random advantage with positive CI lower bound, bounded class loss,
independent-probe BA at least 0.55, and off-manifold excess no greater than 0.02.

The thresholds 1.20, 1.25, 1.30, and 1.35 are all reported descriptively. The
historical `<=1.25` result remains labeled `HISTORICAL_STRICT_GATE`. Neither the
1.30 guardrail nor any other gate may be changed after utility is observed.

