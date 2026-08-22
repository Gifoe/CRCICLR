# Iteration ledger

## V0 — generic headroom audit

* **Question:** Is there enough nontrivial S2-to-S3 adaptation for a protection test?
* **Frozen protocol:** EEGNet S1 anchor, S2 linear residual update, five subject-disjoint development folds.
* **Observed:** Frozen BA `0.7012`; Generic BA `0.7719`; Generic protected drift `0.3512`; six of 41 subjects had negative Generic transfer.
* **Decision:** retain Generic as a competent baseline and proceed. This was not a positive Guard result.

## V1 — hard projected Guard (selected for final negative report)

* **Change:** apply `P_perp = I-U_P U_P^T` to the matched residual; no extra parameters or optimizer budget.
* **Expected:** preserve task-protected coordinates while allowing complement adaptation.
* **Observed:** Guard BA `0.7732`, Guard−Generic `+0.134 pp`, CI `[-0.085,+0.354] pp`, `p=0.13663`; negative transfer `17.1%` versus `14.6%`; protected drift `3.7e-8`; decision-response drift `1.349` versus Generic `1.477`.
* **Decision:** retained as the simplest and highest-mean variant, but rejected as a successful method because G2, G3, G4, and G5 fail.

## V2 — exact decision-response correction

* **Failure diagnosis:** V1 structurally preserves coordinates but does not preserve the frozen-head response to protected perturbations, the mechanism identified in Experiment 3.
* **Change:** subtract the minimum-norm complement correction so `W delta(U_P)=0` up to numerical precision; parameter count, data, seed family, and epochs unchanged.
* **Expected:** restore decision response without reopening the protected coordinates.
* **Observed:** decision-response drift `4.5e-9`, protected drift `3.4e-8`, but Guard−Generic `-0.660 pp` (95% CI `[-1.171,-0.173] pp`), and negative transfer remained `17.1%`.
* **Decision:** reject; the constraint is too restrictive for this deployment setting.

## V3 — fixed soft response strength `alpha=0.25`

* **Failure diagnosis:** V2 over-constrained useful adaptation.
* **Change:** retain hard coordinate protection but apply 25% of the response correction, fixed prospectively before running V3.
* **Expected:** intermediate mechanism preservation with less utility loss.
* **Observed:** Guard−Generic `+0.085 pp` (95% CI `[-0.147,+0.305] pp`, `p=0.26705`), decision-response drift `1.068`, protected drift `4.5e-8`; negative transfer again `17.1%`.
* **Decision:** reject; it does not establish performance, safety, or specificity.

No further blind search was run. The three variants span the simple hard guard, the mechanistically exact correction, and one fixed intermediate correction. Continuing to tune the interpolation solely on these development outcomes would be p-hacking rather than a clean test of the frozen causal story.
