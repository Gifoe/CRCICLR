"""Render the precomputed frozen-audit decision; does not load models or data."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs"
def main():
 s=pd.read_csv(ROOT.parent/"persist_eeg_carrier_dualdataset_screen_v1"/"outputs"/"CARRIER_SCREEN_SUMMARY.csv")
 # When run from the audit experiment, source is its sibling under experiments.
 if not len(s): raise RuntimeError("missing source summary")
 a=pd.read_csv(OUT/"SUBJECT_ASSOCIATIONS.csv");p=pd.read_csv(OUT/"PERTURBATION_RESULTS.csv")
 lite=a[a.model.eq("LiteBN")].copy();w=lite[lite.dataset.eq("WBCIC")].sort_values("rho").iloc[0];o=lite[lite.dataset.eq("OpenBMI")].sort_values("rho").iloc[0]
 g=p.groupby(["dataset","model","perturbation"]).delta_BA_pp.mean().unstack("model");motor=g.loc[("WBCIC","motor_mask")];non=g.loc[("WBCIC","nonmotor_mask")]
 boot=json.loads((OUT/"DATASET_SHIFT_BOOTSTRAP.json").read_text())
 e=s # source summary only contains carriers; EEGNet baseline separate
 b=pd.read_csv(ROOT.parent/"persist_eeg_carrier_dualdataset_screen_v1"/"outputs"/"EEGNET_BASELINE.csv").iloc[0];l=e[e.model.eq("LiteBN")].iloc[0]
 tri={"spatial":{"leg1_wbcic_covariance_shift_larger":boot["cov_fro_shift"],"leg2_wbcic_litebn_rho":float(w.rho),"leg3_wbcic_motor_diff_sensitivity_pp":float(motor.LiteBN-motor.EEGNet),"matched_nonmotor_diff_sensitivity_pp":float(non.LiteBN-non.EEGNet),"secondary_compatible":False},"amplitude":{"supported":False},"spectral":{"supported":False},"terminal":"REVERSAL_MECHANISM_PARTIALLY_SUPPORTED"}
 (OUT/"MECHANISM_TRIANGULATION.json").write_text(json.dumps(tri,indent=2)+"\n")
 text=f'''# Cross-dataset carrier reversal audit

POST-HOC DIAGNOSTIC ONLY. All models were frozen and hash-verified; no training, optimizer, or backward pass occurred.

1. Original reversal: OpenBMI EEGNet {b.openbmi_BA:.4f}, LiteBN {l.openbmi_BA:.4f}, delta {l.openbmi_delta_pp:+.3f} pp. WBCIC EEGNet {b.wbcic_BA:.4f}, LiteBN {l.wbcic_BA:.4f}, delta {l.wbcic_delta_pp:+.3f} pp.
2. WBCIC signal-scale shift larger: **NO**. Mean log-RMS difference WBCIC-minus-OpenBMI is {boot['abs_log_RMS_shift']['mean_difference_wbcic_minus_openbmi']:+.3f}, CI {boot['abs_log_RMS_shift']['ci95']}.
3. WBCIC mu evidence less stable: **UNCLEAR**; the mean contrast-shift difference is {boot['mu_contrast_shift']['mean_difference_wbcic_minus_openbmi']:+.3f} with CI crossing zero.
4. WBCIC beta evidence less stable: **UNCLEAR**; the mean contrast-shift difference is {boot['beta_contrast_shift']['mean_difference_wbcic_minus_openbmi']:+.3f} with CI crossing zero.
5. WBCIC spatial covariance less stable: **YES**. Normalized covariance shift mean difference is {boot['cov_fro_shift']['mean_difference_wbcic_minus_openbmi']:+.3f}, CI {boot['cov_fro_shift']['ci95']}.
6. Latent class-direction stability alone explains reversal: **NO**. LiteBN direction cosine is high in both environments (see `REPRESENTATION_SHIFT.csv`), while the sign of BA reversal changes.
7. Separation magnitude/margin is more informative than cosine alone: **UNCLEAR**. It changes in WBCIC but this audit did not establish a cross-model performance association for it.
8. Strongest primary separating perturbation: **WBCIC motor-channel mask**. LiteBN differential sensitivity is {motor.LiteBN-motor.EEGNet:+.3f} pp; the matched non-motor differential is {non.LiteBN-non.EEGNet:+.3f} pp. OpenBMI named motor masking is unavailable because its cache lacks a verified channel-name map.
9. Strongest LiteBN subject association: OpenBMI `{o.metric}` rho={o.rho:+.3f}; WBCIC `{w.metric}` rho={w.rho:+.3f}. Both bootstrap intervals are reported in `SUBJECT_ASSOCIATIONS.csv`; they are small-n post-hoc evidence, not proof.
10. Compact and LiteGN compatible evidence: **NO** for the primary WBCIC motor-mask differential; neither reproduces LiteBN's increased vulnerability relative to EEGNet.
11. Strongest supported explanation: a **spatial-covariance / motor-channel reliability clue** for LiteBN on WBCIC: greater population covariance shift, negative per-subject covariance-shift association (rho={w.rho:+.3f}), and stronger motor-mask vulnerability. This is not a causal claim.
12. Strongest remaining alternative: fold-0 checkpoint-selection variance and the small WBCIC inner-validation/outer-development subject counts; the spatial sensitivity does not triangulate across Compact and LiteGN.
13. Enough evidence to constrain a future constructive model: **PARTIAL**. The future principle should test reliability-aware handling of spatial evidence, rather than force global invariance.
14. Final terminal: **REVERSAL_MECHANISM_PARTIALLY_SUPPORTED**.
'''
 (OUT/"DECISION.md").write_text(text,encoding="utf8")
 nxt='''# Next constructive target

No model is implemented or selected here.

## Preserve

Test whether task-relevant spatial/electrode evidence can be retained while its session reliability is explicitly measured.

## Do not trust unconditionally

Do not assume spatial covariance or named motor-channel evidence remains equally reliable across sessions, especially in WBCIC.

## One evidence-based design principle

If a future preregistered development study is run, it should modulate the contribution of spatial evidence by a source-derived reliability estimate rather than force global representation invariance.

## Do not do

- No naive absolute prototype alignment.
- No PRD-only direction alignment as the constructive core.
- No channel-permutation-invariant carrier.
- No blind per-channel normalization that removes useful EEG structure.
- No EEGNet plus tiny-module search.
- No Compact/Lite hyperparameter rescue.
'''
 (OUT/"NEXT_CONSTRUCTIVE_TARGET.md").write_text(nxt,encoding="utf8")
 print("REVERSAL_AUDIT_DECISION_FINALIZED")
if __name__=="__main__":main()
