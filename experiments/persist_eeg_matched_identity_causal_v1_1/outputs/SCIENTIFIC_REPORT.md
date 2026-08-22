# PERSIST-EEG Experiment 3 V1.1 scientific report

This is a development-resource closure, not an untouched or independent replication. The report preserves the negative result; it does not retune the experiment after observing validation outcomes.

## Required questions

1. **Why block-wise rather than union-level?** V1's union of two frozen Protected blocks at fold 2/seed 1 had rank 8, while the supported Non-Protected pool also had dimension 8, yielding only C(8,8)=1 control. V1.1 tests each already-frozen Protected block as the causal unit, matching the upstream block-level definition; Protected assignments and the estimand are unchanged.

2. **Frozen before validation outcome?** Yes. Freeze revision 2 was written before the rerun of final validation; protocol SHA256 is `775d189d4252f2b7f843ac28fbce9db8117e8b9d6dafa59a07c605b0bb7b9fcf`. The revision only repairs the omitted G2 measurability clause. `validation_outcome_used_for_design=false`.

3. **Number of frozen MI Protected blocks:** 10.

4. **Rank per run/block:** fold=0,seed=0,B5,rank=4,controls=50,MEDIUM_eligible=50; fold=0,seed=1,B5,rank=4,controls=50,MEDIUM_eligible=50; fold=0,seed=1,B6,rank=3,controls=50,MEDIUM_eligible=50; fold=1,seed=0,B4,rank=4,controls=50,MEDIUM_eligible=50; fold=1,seed=0,B6,rank=3,controls=50,MEDIUM_eligible=48; fold=1,seed=1,B4,rank=4,controls=50,MEDIUM_eligible=50; fold=2,seed=0,B5,rank=4,controls=50,MEDIUM_eligible=50; fold=2,seed=0,B6,rank=4,controls=50,MEDIUM_eligible=50; fold=2,seed=1,B5,rank=4,controls=50,MEDIUM_eligible=50; fold=2,seed=1,B6,rank=4,controls=50,MEDIUM_eligible=50.

5. **Legal matched controls per block:** the same summary reports the retained controls and MEDIUM-eligible controls for every block; all eligible blocks have at least 20 MEDIUM controls.

6. **Run coverage:** 6/6 runs have a legal block-wise causal estimate; 10/10 blocks are eligible; failures=[].

7. **Held-out persistence replication:** G0=True; R_persist mean=0.5662742334319298, 95% bootstrap CI=[0.436542084990335, 0.6985015572320464], unique subjects=23.

8. **MEDIUM identity reductions:** ΔID_P=0.0; ΔID_N=0.0028747358091787438; P−N=-0.0028747358091787438.

9. **Identity equivalence:** False. Both arms must have strictly positive mean held-out identity reduction plus |P−N|≤0.01 BA; measurable(P)=False, measurable(N)=True. The Protected arm has zero mean reduction, so the matched-manipulation check fails.

10. **MEDIUM task outcome:** H_P=0.0; H_N=0.0001092693236714974; ΔH=H_P−H_N=-0.0001092693236714974.

11. **ΔH statistics:** mean=-0.0001092693236714974; median=0.0; 95% CI=[-0.0006240111714975841, 0.0003498565821256038]; sign probability=0.3506; positive-subject fraction=0.43478260869565216; nonnegative-subject fraction=0.6956521739130435; worst subject=-0.0033499999999999997; positive run means=1/6.

12. **Dose-response direction:** {"HIGH": true, "LOW": false, "MEDIUM": false}. The direction is not consistent across LOW/MEDIUM/HIGH; only HIGH has P task harm greater than N, while LOW and MEDIUM do not.

13. **Secondary controls:** same-rank random-control diagnostics are retained in `RANDOM_CONTROL_DIAGNOSTICS.csv` (mean task harm=0.0); Neutral-only was not used as a primary comparator or to alter coverage/gates and no separate Neutral-only causal estimate is claimed.

14. **Outer data:** untouched. All final artifacts set `outer_test_used=false` and `outer_membership_enumerated=false`.

15. **Utility—not identity determines the consequence of invariance:** PARTIAL. G2 fails because P removes no measurable held-out identity, so the causal comparison is not identified; G3 is not evidence for the claim.

## Final decision

- Terminal state: `IDENTITY_MATCH_FAILED`.
- G0=True; G1=True; G2=False; G3=False.
- READY_FOR_EXPERIMENT_4: `NO`.
- The pre-repair decision, freeze audit, identity statistics, primary effect, and report remain on the server as `*_PRE_SEMANTIC_REPAIR` diagnostics.
