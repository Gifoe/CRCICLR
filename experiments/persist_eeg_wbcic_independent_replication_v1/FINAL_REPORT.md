# Final report

Primary terminal state: `WBCIC_INVARIANCE_MANIPULATION_INCONCLUSIVE`.

1. **Competence:** Yes; EEGNet ERM S3 BA=0.7891, Macro-F1=0.7877.
2. **Persistent structure consequence:** R1=`R1_PARTIAL_SUPPORT`.
3. **Reliable predeclared block harm:** P01_04.
4. **Greater than matched random:** none.
5. **D_finite vs identity:** R2=`R2_STRONG_SUPPORT`.
6. **RMSE:** MI=0.016506; MD=0.011869.
7. **MI−MD:** 0.004636, 95% CI [0.001237, 0.007194].
8. **DANN meaningful identity reduction:** no.
9. **CORAL meaningful identity reduction:** no.
10. **MMD meaningful identity reduction:** no.
11. **Did lower identity reliably improve S3 BA?** No reliable positive guarantee.
12. **Global slope:** -0.2230, 95% CI [-0.4469, -0.0469].
13. **Counterexamples:** [].
14. **Replication components:** R1=R1_PARTIAL_SUPPORT; R2=R2_STRONG_SUPPORT; R3=R3_MANIPULATION_INCONCLUSIVE.
15. **Sealed WBCIC outer accessed:** NO.
16. **OpenBMI sealed holdout accessed:** NO.
17. **Exact terminal state:** `WBCIC_INVARIANCE_MANIPULATION_INCONCLUSIVE`.
18. **Strongest defensible conclusion:** At least two predeclared representation-level distinctions reproduce on WBCIC under a frozen EEGNet S1+S2→S3 protocol; effect sizes and uncertainty remain dataset- and protocol-specific.

## R1 blocks

- P01_04: BA harm 0.0144 (95% CI 0.0055, 0.0266); persistent−random 0.0060 (95% CI -0.0027, 0.0167).
- P05_08: BA harm 0.0002 (95% CI -0.0039, 0.0042); persistent−random -0.0010 (95% CI -0.0044, 0.0021).

## R3 fixed grid

- CORAL λ=0.01: S_I=0.0011 [-0.0015, 0.0049], ΔBA=-0.0024 [-0.0082, 0.0021], meaningful=False, counterexample=NONE.
- CORAL λ=0.1: S_I=-0.0048 [-0.0092, 0.0004], ΔBA=-0.0013 [-0.0068, 0.0030], meaningful=False, counterexample=NONE.
- CORAL λ=1: S_I=-0.0062 [-0.0093, -0.0026], ΔBA=0.0005 [-0.0027, 0.0033], meaningful=False, counterexample=NONE.
- DANN λ=0.01: S_I=0.0105 [0.0069, 0.0143], ΔBA=-0.0026 [-0.0081, 0.0007], meaningful=False, counterexample=NONE.
- DANN λ=0.1: S_I=0.0129 [0.0092, 0.0166], ΔBA=-0.0025 [-0.0080, 0.0020], meaningful=False, counterexample=NONE.
- DANN λ=1: S_I=0.0094 [0.0049, 0.0142], ΔBA=-0.0152 [-0.0265, -0.0049], meaningful=False, counterexample=NONE.
- MMD λ=0.01: S_I=0.0026 [-0.0007, 0.0067], ΔBA=-0.0016 [-0.0067, 0.0032], meaningful=False, counterexample=NONE.
- MMD λ=0.1: S_I=0.0030 [-0.0002, 0.0070], ΔBA=-0.0021 [-0.0046, 0.0005], meaningful=False, counterexample=NONE.
- MMD λ=1: S_I=0.0017 [-0.0039, 0.0069], ΔBA=-0.0012 [-0.0060, 0.0033], meaningful=False, counterexample=NONE.

## Claim boundary

The evidence concerns frozen WBCIC EEGNet representations under this S1+S2→S3 protocol. It does not establish physiological causality, a universal EEG law, or that subject invariance is universally harmful.
