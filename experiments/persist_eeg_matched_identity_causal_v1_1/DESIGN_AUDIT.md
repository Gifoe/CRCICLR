# Design audit — Experiment 3 V1.1

V1 did not obtain a negative causal result.  Its G1 failure was a train-only
feasibility problem: fold 2 / seed 1 had Protected blocks 5 and 6, each rank
4, but V1 unioned them into rank 8 while the non-Protected supported pool also
had rank 8, so C(8,8)=1 rather than the required 20 controls.  No validation
task outcome was used in identifying this problem.

The upstream V3.1 spectrum is 128-dimensional and is produced by the frozen
EEGNet checkpoint.  The V1.1 representation cache is 64-dimensional and is a
different model/projection; using it with the V3.1 whitening matrices would be
invalid.  Experiment 3 therefore extracts the same V3.1 checkpoint
representation on the server and uses the persisted canonical spectrum,
rather than silently substituting the V1.1 cache.

Protected is the exact union of the MI block IDs in each V3.1
`SIGNED_ASSIGNMENTS_V3_1.json`.  No block is reselected here.  Controls are
formed only from non-overlapping, persistence-supported canonical coordinates.
The matching score was fixed before any validation outcome is read.  It is a
standardized train-only distance over structural and train-probe summaries;
validation BA is never available to this stage.  The initial implementation
also required every candidate feature to have absolute standardized deviation
at most 2.5.  A train-only diagnostic showed that this conjunction accepted no
legal candidate (the closest candidate had joint RMS distance 1.38 but one
correlated feature above that arbitrary cutoff).  That numerical rule was
repaired before freeze: all legal exact-rank candidates are deterministically
sorted by the joint distance and the first 50 are retained.  This preserves
exact rank, no overlap, the non-Protected persistence-supported source, and
the full structural diagnostics, while avoiding an empty set caused by a
redundant per-feature hard cutoff.  The repair used no validation outcomes,
does not select a Protected block, and is recorded in the matching payload.

V1.1 changes only the causal unit to one frozen Protected block.  This is
scientifically aligned with the Signed-V3.1 block-level persistence and utility
definitions and preserves the primary estimand.  Controls are certified by
train-only same-vs-mismatched cross-session persistence (R_persist > 0), then
matched on the predeclared structural features using deterministic top-K
standardized distance.  Identity calibration is P-anchored per block, so a
weak individual control cannot compress the common dose range for all other
controls.  The human-readable subject-ID BA remains in every manipulation
record.

The loader reads only the frozen train and development-validation subject
fields needed for this experiment.  It does not extract, enumerate, or
materialize outer subject membership, EEG, labels, or features.

All block/control eligibility, matching, persistence certification, alpha and
dose choices occur before validation task outcomes.  If a block cannot meet
the frozen coverage rule, it is reported as unavailable rather than replaced
with unsupported coordinates or duplicate rotations of a full-rank span.

The preceding rank-8 statement describes the preserved V1 union-level audit;
it is not the V1.1 result.  After the block-wise redesign, the V1.1 train-only
audit has 10/10 eligible Protected blocks and 6/6 eligible runs, with no
unsupported coordinate or duplicate rotation admitted.  The final V1.1
terminal state is determined by the held-out manipulation check and is
reported in `outputs/FINAL_DECISION.json`.

## Post-freeze semantic repair (freeze revision 2)

The first V1.1 finalization exposed an omission in the G2 implementation. The
implementation checked only whether the difference between the P and matched-N
held-out identity drops was within 0.01 BA. The scientific contract also
requires both interventions to produce measurable identity reduction. The
pre-repair final output had mean MEDIUM identity drop `P=0.0` and
`N=0.0028747358`; therefore the old implementation could label a non-removing
P arm as an identity-equivalent match.

The repair adds the condition
`mean(identity_drop_P)>0 AND mean(identity_drop_N)>0` while retaining the
unchanged 0.01 BA equivalence tolerance. It does not change Protected blocks,
controls, doses, alpha values, the identity metric, task estimator, bootstrap,
or the G3 threshold. This is a scientific-semantics repair, so the original
freeze is superseded, a new protocol/configuration hash is recorded, and the
held-out final phase is rerun from the same raw validation measurements. The
pre-repair decision and freeze audit are retained in server outputs as
`*_PRE_SEMANTIC_REPAIR` diagnostics. No new matching choice was made from the
task outcome and no outer data were accessed.
