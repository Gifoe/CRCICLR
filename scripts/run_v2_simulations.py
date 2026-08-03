#!/usr/bin/env python
from pathlib import Path
import matplotlib.pyplot as plt

from hsc_tta.v2.simulation import simulation_grid

ROOT=Path("/root/autodl-tmp/hsc_tta_eeg")
OUT=ROOT/"outputs/v2_joint_certified/theory"
OUT.mkdir(parents=True,exist_ok=True)
frame=simulation_grid(5000); frame.to_csv(OUT/"SIMULATION_V2_RESULTS.csv",index=False)
figures=OUT/"SIMULATION_V2_FIGURES"; figures.mkdir(exist_ok=True)
data=frame[(frame.scenario=="one_beneficial")&(frame.site_shift==0)&(~frame.heteroskedastic)]
data.to_csv(figures/"validity_by_calibration_size.csv",index=False)
fig,ax=plt.subplots(figsize=(6,4))
for actions,g in data.groupby("action_count"):
    ax.plot(g.calibration_size,g.joint_simultaneous_validity,marker="o",label=f"A={actions}")
ax.axhline(.9,color="black",linestyle="--"); ax.set(xlabel="calibration subjects",ylabel="joint validity")
ax.legend(); fig.tight_layout(); fig.savefig(figures/"validity_by_calibration_size.png",dpi=180); plt.close(fig)
report="# V2 joint-certificate simulation\n\nEach configuration uses 5,000 independent episode repetitions. Calibration is subject-level; within-episode windows are abstracted into one residual.\n\n"
report+=frame.to_markdown(index=False)+"\n\nThe adversarial selector uses only U. Site shift is intentionally outside exchangeability and is expected to break nominal coverage; it is a limitation diagnostic, not a theorem counterexample.\n"
(OUT/"SIMULATION_V2_REPORT.md").write_text(report,encoding="utf-8")
theory="""# Joint risk-and-benefit certificate

## Assumptions

1. Episodes are exchangeable at the subject level. Training, calibration, and evaluation episodes are independent.
2. Windows within one episode may be arbitrarily dependent; each calibration subject contributes exactly one score.
3. Every action and diagnostic is measurable with respect to U. All predictors and meta-OOF scales are frozen before calibration.
4. The selector uses only U, joint bounds, fixed action availability, and fixed costs. V is opened only after decision hashing.

## Results

**Theorem 1 (simultaneous critical-index upper bound).** Applying the finite-sample split-conformal quantile to the subject maximum of normalized critical-index underestimation scores gives a marginal, actionwise simultaneous upper bound on every frozen action's critical index with probability at least 1-delta.

**Theorem 2 (simultaneous relative-benefit lower bound).** Including the normalized benefit-overestimation score for every frozen TTA action in the same subject maximum gives simultaneous lower bounds on action gain relative to No-TTA with the same marginal probability.

**Corollary (U-only post-selection).** On the joint event, any U-only selector restricted to non-sentinel risk bounds and strictly positive benefit lower bounds selects an action whose future prediction-set miscoverage is at most alpha; if it selects TTA, its future argmax-error gain relative to No-TTA is nonnegative.

This is a marginal episode-level guarantee, not conditional validity inside the certified subgroup and not a deterministic per-subject guarantee. It does not certify Macro-F1, nontrivial set existence, or the existence of a beneficial TTA. Exchangeability-breaking site shift is not covered.
"""
(OUT/"THEORY_V2.md").write_text(theory,encoding="utf-8")
print(frame.groupby("scenario").joint_simultaneous_validity.mean())
