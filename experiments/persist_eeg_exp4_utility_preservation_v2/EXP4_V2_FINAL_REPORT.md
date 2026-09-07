# PERSIST-EEG Experiment 4 V2 — Decision-Grounded Utility-Preserving Adaptation

## Terminal decision

`EXP4_V2_NO_DEPLOYMENT_MATCHED_PROTECTED_DIRECTION`

The prospective direction gate failed before Guard training. This is a valid
negative terminal state, not a missing-value success claim. The 41 development
subjects were used; no sealed outer subject ID, raw file, label, or embedding
was opened. The final lock was therefore not created and no outer command was
authorized.

## Baseline audit

The exact S1-only EEGNet anchor and the S2 residual-adapter route were
reproduced on five deterministic development folds. Generic was selected by
S2 subject-held-out validation alone (`GEN_LINEAR_LR1E3_E25`). With three
optimization seeds averaged at the logit level, the held-development S3 audit
gave:

| method | subject-balanced S3 BA | macro-F1 | negative transfer vs Frozen |
|---|---:|---:|---:|
| Frozen | 0.70116 | 0.66944 | 0/41 |
| Generic | 0.77299 | 0.77146 | 7/41 (17.1%) |

Generic therefore retains substantial future-session headroom (`+7.18 pp`
relative to Frozen). Its worst-subject change was `−2.50 pp` and its
worst-quartile mean change was `−0.589 pp`. The three-seed S3 means are stored
in `results/SEED_ROBUSTNESS.csv`; the seed audit is not an independent
biological sample analysis.

## Required protocol questions

1. **Previous Generic reproduced?** Yes, with the exact current deployment
   route; the full table is in `results/DEV_METHOD_SUMMARY.csv`.
2. **Why not historical P01_04?** It was defined in the earlier S1+S2→S3
   representation protocol, whereas V2 is S1-only anchor→S2 adapter→S3. A
   same-dimensional block is not automatically the same functional direction.
3. **How constructed?** For each fold, all legal non-outcome subjects' S1/S2
   anchor centroids were pooled, centered, ridge-whitened, and used to form a
   symmetric cross-session centroid covariance; its ordered directions were
   QR-orthonormalized.
4. **Candidate count?** Eight direction-level candidates per fold.
5. **Multiplicity?** One-sided sign-flip evidence with Holm correction across
   the eight candidates within each fold.
6. **Persistence pass?** The leading persistence eigenvalues were positive in
   every fold; persistence alone was not the limiting gate.
7. **Signed-utility pass?** None. Fold-level maximum signed-utility means
   were approximately `−0.0018` to `+0.0024` CE units; the positive folds did
   not pass the Holm utility gate.
8. **Decision-dependence pass?** Finite centered-logit responses were positive,
   but no direction passed utility and decision gates simultaneously.
9. **Final protected rank?** Zero; no proposed protected set was frozen.
10. **Any S3 used for selection?** No. S3 was used only for the predeclared
    baseline audit after the direction gate had already failed.
11. **Did Generic reduce protected utility?** Not estimable: no direction was
    certified, so the protocol correctly did not define a protected-utility
    headroom target.
12. **How many subjects/folds?** Not applicable for protected utility.
13. **Collapse enriched in negative-transfer subjects?** Not tested; there was
    no certified utility target.
14. **Did ΔG predict ΔBA?** Not estimable under the failed G1 gate.
15. **Compared with coordinate drift?** Not estimable for a proposed set.
16. **Compared with exact response drift?** Not estimable for a proposed set.
17. **Final Generic adapter?** Frozen S1 EEGNet anchor/classifier plus a
    zero-initialized 32×32 linear residual adapter, AdamW, learning rate
    `1e−3`, 25 epochs, weight decay `5e−4`, trained on legal S2.
18. **Guard equation?** Predeclared but not run:
    `hψ=h0+Aψ(h0)` with `CE(raw)+λΣw_j ReLU(0−G_j)`, `λ=0.5`.
19. **Conditional or always active?** Conditional (`τ=0`); it would be
    inactive when certified utility remained nonnegative.
20. **Constraint activity?** Not applicable; no Guard was trained.
21. **Frozen S3 BA?** `0.70116`.
22. **Generic S3 BA?** `0.77299`.
23. **Guard S3 BA?** Not run.
24. **Guard−Generic delta?** Not estimable.
25. **95% CI?** Not estimable for an unrun Guard comparison.
26. **Permutation p-value?** Not estimable for an unrun Guard comparison.
27. **Subjects favoring Guard?** Not estimable.
28. **Generic negative transfer?** `7/41 = 17.1%`.
29. **Guard negative transfer?** Not run.
30. **Worst quartile?** Generic `−0.589 pp` relative to Frozen; worst subject
    `−2.50 pp`.
31. **Generic-harmed subjects rescued?** Not applicable.
32. **Previously helped subjects harmed by Guard?** Not applicable.
33. **Did Guard preserve positive utility?** Not tested because G1 failed.
34. **Complement adaptation?** Not tested for a Guard.
35. **Historical hard Guard?** Fixed V1 evidence was `+0.134 pp` Guard−Generic,
    95% CI crossing zero, 18/41 subjects favoring; it failed specificity and
    safety. Those artifacts remain unchanged in the historical directory.
36. **Deployment-matched hard Guard?** Not run after the prospective utility
    direction gate failed.
37. **Persistence-only control?** Not run; no matched proposed utility target.
38. **Identity control?** Not run; identity remains a control, not a selection
    criterion.
39. **PCA control?** Not run.
40. **Random control?** Not run.
41. **Specificity?** Not estimable because no Guard/control family was opened.
42. **Seed robustness?** Generic was audited with seeds 0/1/2; the per-fold
    values and checkpoint hashes are in `results/SEED_ROBUSTNESS.csv`.
43. **Fold robustness?** The same direction gate was applied to all five
    folds; no fold selected a direction. Generic baseline tables include all
    41 outcome subjects.
44. **Second backbone?** Not run; EEGNet failed the required G1 gate.
45. **Outer accessed?** No.
46. **Final lock before outer?** No final lock was generated because the gate
    failed; `protocol/OUTER_LOCK.json` remains `OUTER_SEALED`.
47. **One-time outer result?** None; outer evaluation count is zero.
48. **Did outer confirm?** Not applicable.
49. **Justified claim?** In this exact S1-only deployment setting, Generic
    adaptation has real future-session headroom, but the proposed
    deployment-matched decision-grounded utility direction was not certified.
50. **Stronger claim not justified?** The data do not support claiming that a
    utility-preserving Guard improves generalization, reduces negative transfer,
    or works on the sealed cohort.
51. **Terminal state?** `EXP4_V2_NO_DEPLOYMENT_MATCHED_PROTECTED_DIRECTION`.
52. **Remaining reviewer weakness?** The negative direction gate has limited
    power and depends on the EEGNet anchor and a finite utility intervention;
    a different competent representation could expose a different functional
    persistence structure. That limitation is reported, not repaired by
    post-hoc threshold relaxation.

## Reproducibility and delivery

The server execution directory is
`D:\nips-temp\TotalP\P1\CRCICLR_EXP4_UTILITY_PRESERVATION_V2\experiments\persist_eeg_exp4_utility_preservation_v2`.
Compact code, protocol locks, result tables, reports, and figures are on the
GitHub branch `codex/persist-eeg-exp4-utility-preservation-v2`. Raw EEG,
cache, and checkpoints were not added to Git.
