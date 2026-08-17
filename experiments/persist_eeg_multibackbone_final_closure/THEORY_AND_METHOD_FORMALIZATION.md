# PERSIST-EEG theory and method formalization

For a frozen encoder-head pair, let `h=Eθ(x)` and `z=Wh+b`. For a rank block
`U_j` in the prospectively ordered cross-session subject-persistence basis:

- `P_j` (persistence) is the same-subject S1/S2 centroid-matching advantage of
  `U_j` over norm-matched Haar-random subspaces.
- `U_j` (signed utility) is
  `u_spec=(CE_erase-CE_raw)-E_R(CE_random-CE_raw)`. Negative values identify
  harmful utility; positive values identify protected utility.
- `D_j` (decision dependence) combines local centered-head energy and the
  finite centered-logit/margin response, each relative to same-rank controls.
- `A_j` (actionability) is
  `ΔBA_specific=(BA_erase-BA_raw)-E_R(BA_random-BA_raw)` together with the
  practical, confidence, multiplicity and stability gates.

The discrete action is:

```text
PERSISTENT + PROTECTED                         -> PRESERVE
not jointly persistent/harmful/active/actionable -> NO-OP
PERSISTENT + HARMFUL + ACTIVE + ACTIONABLE     -> SUPPRESS
```

Persistence does not imply harmfulness. Negative signed utility does not imply
decision dependence. Decision dependence does not imply beneficial
removability. These are tested implications, not definitions collapsed into a
single nuisance score.

If a qualified harmful union `U_H` exists and protected union `U_P` has
priority, define `U_H,res=orth[(I-U_P U_Pᵀ)U_H]` and

`W_α = W_0(I-α U_H,res U_H,resᵀ)`, with `α∈{0,.25,.5,.75,1}`.

This conditional AGDI projection has four structural properties:

1. Intervention locality: `ΔW` lies only in the certified harmful span.
2. Protected preservation: `W_α U_P = W_0 U_P`.
3. Full harmful invariance at `α=1`: `W_1 U_H,res=0`.
4. Identity fixed point: without a replicated actionable harmful span,
   `α=0` and `W_α=W_0`.

None of these identities proves improved generalization. The improvement and
subject-safety claims require H4/H5 and, if AGDI is authorized, a separate
development and sealed-outer empirical evaluation.
