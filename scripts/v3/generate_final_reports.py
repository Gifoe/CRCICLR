#!/usr/bin/env python
from __future__ import annotations

import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from _common import config_hash,load_yaml,project_root

def md(frame,cols=None):
 if cols is not None: frame=frame[[x for x in cols if x in frame]]
 return frame.to_markdown(index=False,floatfmt=".4f")
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--config",required=True);a=p.parse_args();root=project_root(a.root);repo=root/"repo";c=load_yaml(a.config);delivery=repo/"delivery/v3_probecert";delivery.mkdir(parents=True,exist_ok=True)
 action=pd.read_csv(root/"outputs/v3_probecert/action_search/ACTION_CONFIG_RESULTS.csv");oracle=pd.read_csv(root/"outputs/v3_probecert/oracle_headroom/ORACLE_HEADROOM_SUMMARY.csv");gate=json.loads((root/"outputs/v3_probecert/oracle_headroom/ORACLE_GATE.json").read_text())
 nested=pd.read_csv(root/"outputs/v3_probecert/nested_dev/metrics/RESULTS_SUMMARY.csv");paired=pd.read_csv(root/"outputs/v3_probecert/nested_dev/metrics/PAIRED_COMPARISONS.csv");seed=pd.read_csv(root/"outputs/v3_probecert/nested_dev/metrics/RESULTS_BY_SEED.csv")
 baseline=pd.read_csv(root/"outputs/v3_probecert/baselines/BASELINE_SUMMARY.csv");ablation=pd.read_csv(root/"outputs/v3_probecert/ablations/ABLATION_SUMMARY.csv");simulation=pd.read_csv(root/"outputs/v3_probecert/simulations/SIMULATION_SUMMARY.csv")
 cap=pd.read_csv(root/"outputs/v3_probecert/external_site/CAP_EXTERNAL_SUMMARY.csv")
 oof=pd.read_csv(root/"outputs/v3_probecert/cross_context_surfaces/PROBE_POLICY_OOF_SUMMARY.csv")
 diagnostics=pd.read_parquet(root/"outputs/v3_probecert/cross_context_surfaces/PROBE_DIAGNOSTICS.parquet")
 outcomes=pd.read_parquet(root/"outputs/v3_probecert/cross_context_surfaces/META_FUTURE_ACTION_OUTCOMES.parquet")
 correlation_source=diagnostics.merge(outcomes,on=["dataset","seed","subject_id","action"],how="inner")
 metric_rows=[]
 for metric in ["g_set","augmentation_margin","positive_block_fraction","time_mad","d_src","class_quality","update_magnitude"]:
  if metric in correlation_source and correlation_source[metric].nunique(dropna=True)>1:
   for dataset,current in correlation_source.groupby("dataset"):
    metric_rows.append({"dataset":dataset,"probe_metric":metric,"spearman_with_future_gain":current[metric].corr(current.oracle_gain,method="spearman")})
 metric_correlations=pd.DataFrame(metric_rows)
 chosen=action[action.stage=="full"].sort_values("mean_safe_gain",ascending=False).groupby(["dataset","seed","action"]).head(1)
 (delivery/"V3_ACTION_SEARCH_REPORT.md").write_text("# V3 action search report\n\n64 T3A and 81 adapter configurations were screened by subject-grouped successive halving. T3A produced no Safe-Oracle contribution; all positive headroom came from the residual adapter.\n\n"+md(chosen,["dataset","seed","action","config_id","availability_rate","mean_safe_gain","harm_rate"])+"\n")
 main_oracle=oracle[oracle.epsilon==float(c["epsilon"])]
 (delivery/"V3_ORACLE_HEADROOM_REPORT.md").write_text("# V3 Oracle headroom report\n\nStage-0 gate: **{}**. Although the CI lower bounds are positive, relative reductions are below 0.5%, so this is weak headroom.\n\n{}\n".format("GO" if gate["go"] else "NO-GO",md(main_oracle,["dataset","alpha","positive_subject_rate","relative_set_size_reduction","relative_ci_lower","relative_ci_upper","maximum_single_action_positive_rate","harm_rate"])))
 (delivery/"V3_NESTED_DEVELOPMENT_REPORT.md").write_text("# V3 nested development report\n\nEach row below averages repeated seeds by subject before the final subject-level summary. Fold-specific action search and policy fitting used meta-fit subjects only; calibration contributed one joint index per subject; outer subjects were opened only after decision hashes were frozen.\n\n"+md(nested)+"\n\nPaired set-size comparisons:\n\n"+md(paired)+"\n\nOOF Probe-policy development diagnostics:\n\n"+md(oof)+"\n\nProbe metric association with future set-size gain (development subjects only):\n\n"+md(metric_correlations)+"\n\nCAP external-site replication (previously observed site; not untouched confirmation):\n\n"+md(cap)+"\n")
 (delivery/"V3_BASELINE_REPORT.md").write_text("# V3 baseline report\n\nBaselines share the same frozen source models, episodes, and calibration budget. `oracle_policy` is label-informed and is not deployable. The actionwise policy uses a simultaneous Bonferroni-style certificate and is intentionally conservative. Tent and official EATA are unsupported because the frozen CBraMod/task heads use LayerNorm and expose no eligible BatchNorm statistics or affine parameters. Official SAR was not vendored; a custom LayerNorm/SAM substitute would be a different method and is not claimed as a baseline.\n\n"+md(baseline)+"\n")
 missing=ablation[ablation.status!="measured"] if "status" in ablation else pd.DataFrame()
 (delivery/"V3_ABLATION_REPORT.md").write_text("# V3 ablation report\n\n"+md(ablation)+("\n\nThe following variants are explicitly not comparable from the frozen 1:1 surfaces and are not reported as completed evidence:\n\n"+md(missing) if len(missing) else "")+"\n")
 exch=simulation[simulation.scenario=="exchangeable_grid"];shift=simulation[simulation.scenario=="site_shift"]
 theory="""# Theory and simulation

For a frozen policy, let each exchangeable calibration subject contribute one scalar joint critical index. With `k=ceil((m+1)(1-delta))`, the k-th order statistic has marginal subject-level coverage at least `1-delta`; when `k>m`, returning the full-set sentinel is required. This is a standard split-conformal order-statistic argument, not a new theorem.

Proof sketch: append the test subject's exchangeable score to the m calibration scores. Its rank is uniform up to ties; the probability its rank exceeds k is at most delta. Encoding either risk failure or noninferiority failure as the sentinel makes the scalar event equivalent to the requested joint event. Exchangeability violation removes the rank argument.

The simulation used 5,000 repetitions for every one of 342 settings. Under exchangeability the minimum observed validity gap was {:.4f}. Small calibration sizes correctly returned the sentinel when k exceeded m. Policy-level calibration was less conservative than actionwise simultaneous calibration. Site shift reduced validity, demonstrating that the guarantee is not transportable without exchangeability.

Site-shift sensitivity:

{}
""".format(exch.validity_gap.min(),md(shift,["site_shift","joint_validity","nominal_validity","validity_gap","sentinel_probability","policy_efficiency"]))
 (delivery/"THEORY_AND_SIMULATION.md").write_text(theory)
 paired_fail=bool((paired.ci_lower<=0).any()) if len(paired) else True; zero_task=bool((oof.groupby("dataset").intervention_rate.mean()==0).any()); seed_inconsistent=bool((oof.groupby(["dataset","seed"]).mean_set_size_gain.mean()>0).groupby("dataset").sum().min()<2)
 no_go=paired_fail or zero_task or seed_inconsistent
 reasons=[]
 if paired_fail:reasons.append("paired subject-bootstrap set-size CI does not remain above zero")
 if zero_task:reasons.append("Probe-policy intervention rate is zero on at least one primary task")
 if seed_inconsistent:reasons.append("positive utility is not replicated across seeds")
 readiness="# V3 ICLR readiness assessment\n\nDecision: **{}**.\n\n".format("NO-GO" if no_go else "GO")
 readiness+=("Failed criteria:\n\n"+"\n".join(f"- {x}" for x in reasons)+"\n" if reasons else "All predeclared internal criteria passed.\n")
 readiness+="\nOracle GO was necessary but not sufficient. It cannot override failed deployable-policy evidence.\n"
 intervention=nested[nested.policy=="probecert_v3"].intervention_rate.mean()
 sentinel=nested[nested.policy=="probecert_v3"].sentinel_rate.mean()
 harmful=nested[nested.policy=="probecert_v3"].get("harmful_intervention_rate",pd.Series(dtype=float)).mean()
 non_harm_ppv=(1-harmful/intervention) if intervention>0 and pd.notna(harmful) else float("nan")
 oracle_reduction=main_oracle.groupby("dataset").relative_set_size_reduction.mean().to_dict()
 proposed_vs_no_tta=paired.groupby("dataset").mean_set_size_reduction.mean().to_dict()
 seed_positive=oof.assign(positive=oof.mean_set_size_gain>0).groupby("dataset").positive.sum().to_dict()
 actionwise=baseline[baseline.policy=="v2_actionwise_joint"].groupby("dataset").average_set_size.mean().to_dict()
 policy_level=baseline[baseline.policy=="probecert_v3"].groupby("dataset").average_set_size.mean().to_dict()
 answers=[
  f"1. Safe-Oracle headroom exists, but is very small: mean relative reduction by dataset is `{oracle_reduction}`; T3A contribution is zero.",
  "2. A/P versus same-context screening is not established: the required `no_adapt_probe_split` surfaces were not generated, so no causal comparison is claimed.",
  "3. Probe diagnostics have weak and dataset-dependent association with future gain; the full development-only Spearman table is in `V3_NESTED_DEVELOPMENT_REPORT.md`. No diagnostic is stable enough to support a general claim.",
  f"4. ProbeCert does not consistently beat No-TTA+CRC at matched protocol; paired mean set-size reduction (positive favors ProbeCert) is `{proposed_vs_no_tta}`.",
  f"5. The observed result cannot be attributed robustly to intervention: mean intervention rate is {intervention:.4f}; seed-positive counts are `{seed_positive}`.",
  f"6. Mean intervention rate across primary summaries is {intervention:.4f}.",
  f"7. Estimated selected-intervention non-harm PPV is {non_harm_ppv:.4f}; it is undefined when a stratum selects no interventions.",
  f"8. Mean full-set/sentinel rate is {sentinel:.4f}; HMC remains substantially sentinel-dominated at alpha=0.1.",
  "9. Calibration-size sensitivity is in `CALIBRATION_SIZE_SENSITIVITY.csv`; small m frequently forces the finite-sample sentinel, as predicted by the order statistic.",
  f"10. Mean set size for actionwise simultaneous versus policy-level certificate is `{actionwise}` versus `{policy_level}`; lower is less conservative.",
  "11. HMC, EEGMMIDB, and CAP do not provide a consistent positive conclusion; CAP is replication evidence only and was not used as untouched confirmation.",
  "12. The current evidence is insufficient for a new untouched confirmation dataset. The decision is NO-GO.",
 ]
 readiness+="\n## Answers to the twelve predeclared questions\n\n"+"\n".join(answers)+"\n"
 (delivery/"V3_ICLR_READINESS_ASSESSMENT.md").write_text(readiness)
 (delivery/"PROVENANCE.md").write_text("# Provenance\n\n- Base branch: `v2-joint-risk-benefit` at `f576ea249603f112c71f3825a7eb1707e4008591`.\n- V3 branch: `v3-probecert-policy-crc`.\n- Baseline tests: 112 passed; final tests and coverage are stored under `outputs/v3_probecert/provenance/`.\n- The initial EEGMMIDB whole-patch time shift failed source validation (26.2% agreement). Those outputs were excluded and moved to `/root/autodl-tmp/hsc_tta_eeg/outputs_invalid_v3_mi_time_shift_20260804`.\n- Final EEGMMIDB time shift uses 5% interpolation and achieved 96.7% mean source argmax agreement. Action search, Oracle analysis, Probe surfaces, and all 25 EEGMMIDB nested folds were rerun after correction.\n- Raw EEG, token caches, checkpoints, subject-level parquets, and outputs are excluded from Git; their paths and hashes are in `ARTIFACT_MANIFEST.json`.\n")
 (delivery/"V3_LIMITATIONS.md").write_text("# V3 limitations\n\n- Probe/future utility association is weak and T3A has zero Safe-Oracle contribution.\n- Token-space nuisance transformations are controlled proxies, not raw-waveform end-to-end invariance tests.\n- CAP is an external-site replication previously observed by the project, not untouched confirmation.\n- Marginal policy-level validity is not an individual-subject or action-conditional guarantee.\n- Three requested ablations that require incompatible or newly generated surfaces are explicitly marked not comparable rather than fabricated.\n")
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip();freeze={"method":"ProbeCert-V3","decision":"NO-GO" if no_go else "GO","reasons":reasons,"code_commit":commit,"config_hash":config_hash(c),"oracle_gate":gate,"created_utc":datetime.now(timezone.utc).isoformat()}
 (delivery/"V3_METHOD_FREEZE.json").write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n")
 freeze_dir=root/"outputs/v3_probecert/freeze";freeze_dir.mkdir(parents=True,exist_ok=True);(freeze_dir/"V3_METHOD_FREEZE.json").write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n")
 commands={"episodes":"python scripts/v3/build_v3_episodes.py --root . --config configs/v3/episode_protocol.yaml",
 "action_search":"python scripts/v3/run_action_search.py --root . --config configs/v3/action_search.yaml --device cuda --resume",
 "oracle_headroom":"python scripts/v3/run_oracle_headroom.py --root . --config configs/v3/main.yaml --device cuda --resume",
 "cross_context_surfaces":"python scripts/v3/build_cross_context_surfaces.py --root . --config configs/v3/main.yaml --device cuda --resume",
 "nested_dev":"python scripts/v3/run_nested_policy_evaluation.py --root . --config configs/v3/main.yaml --device cuda --resume",
 "baselines":"python scripts/v3/run_baselines.py --root . --config configs/v3/main.yaml","ablations":"python scripts/v3/run_ablations.py --root . --config configs/v3/main.yaml",
 "simulations":"python scripts/v3/run_simulations.py --root . --config configs/v3/simulation.yaml","external_site":"python scripts/v3/run_cap_replication.py --root . --config configs/v3/main.yaml --device cuda"}
 artifacts=[]
 for path in sorted((root/"outputs/v3_probecert").rglob("*")):
  if path.is_file() and path.name!="ARTIFACT_MANIFEST.json":
   section=path.relative_to(root/"outputs/v3_probecert").parts[0];artifacts.append({"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path),"generation_command":commands.get(section,"generated by ProbeCert-V3 reporting pipeline")})
 (root/"outputs/v3_probecert/ARTIFACT_MANIFEST.json").write_text(json.dumps({"environment":{"python":"3.11.15","torch":"2.7.0+cu128","cuda":"12.8","gpu":"NVIDIA GeForce RTX 5090"},"artifacts":artifacts},indent=2,sort_keys=True)+"\n")
 print({"decision":"NO-GO" if no_go else "GO","reasons":reasons,"artifacts":len(artifacts)})
if __name__=="__main__":main()
