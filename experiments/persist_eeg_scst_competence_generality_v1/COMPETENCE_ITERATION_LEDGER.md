# Competence iteration ledger

All entries are recorded before any future-session SCST utility is inspected.

## Iteration 1 — frozen-representation decoder repair

- Hypothesis: the frozen CBraMod geometry contains nonlinear task information not recovered by the current linear head.
- Available information: previous source-validation and held-development task competence; frozen SCST geometry audit.
- Change: globally selected H0/H1/H2 decoder with frozen feature z-scoring.
- Predicted effect: improve task BA without changing representation hashes or geometry.
- Keep/reject: pending source-validation selection and held-development evaluation.

## Iteration 2 — limited R1 representation repair

- Trigger: Phase 1A failed both frozen competence thresholds.
- Change: fine-tune only CBraMod encoder layer 11, official task projector, and task head.
- Search: encoder LR {1e-5, 3e-5}; downstream LR 3e-4; weight decay 1e-2.
- Selection: source validation BA, NLL tie-break.
- SCST utility available: no.
- Keep/reject: selected once per dataset; full admissibility re-audit is mandatory.
