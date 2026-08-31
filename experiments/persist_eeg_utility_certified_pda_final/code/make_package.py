"""Assemble auditable documents, compact figures, protocol statuses and final JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import upda_core as c


ROOT = c.EXP
RES = c.RESULTS


def read_json(name, default=None):
    p = RES / name
    if not p.is_file(): return default
    return json.loads(p.read_text(encoding="utf-8"))


def write(name: str, text: str) -> None:
    p = ROOT / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text.rstrip() + "\n", encoding="utf-8")


def fmt(x, digits=4):
    try:
        if x is None or not np.isfinite(float(x)): return "NA"
        return f"{float(x):+.{digits}f}"
    except Exception: return str(x)


def comparison(stats, dataset, comparison):
    for x in stats.get("comparisons", []):
        if x.get("dataset") == dataset and x.get("comparison") == comparison: return x
    return {}


def plot_save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / "figures" / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def figures(outcome, alpha, assoc, oracle, forensic):
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    if len(forensic):
        fig, ax = plt.subplots(figsize=(6.4, 4.4));
        ax.scatter(forensic["historical_crossfit_gain"], forensic["full_pda_minus_population_future_BA"], s=12, alpha=.45)
        ax.axhline(0,color="black",lw=.8); ax.axvline(0,color="black",lw=.8); ax.set(xlabel="Previous PDA historical cross-fit gain (CE/BA diagnostic)",ylabel="Future full-PDA − population BA",title="Previous PDA diagnostic association")
        plot_save(fig,"historical_utility_vs_future_gain")
    if len(alpha):
        counts=alpha.groupby("alpha").size().reindex(c.ALPHAS,fill_value=0)
        fig, ax=plt.subplots(figsize=(6.4,4.0)); ax.bar([str(x) for x in c.ALPHAS],counts.values,color="#4472c4"); ax.set(xlabel="Certified alpha",ylabel="Subject-fold records",title="U-PDA one-SE alpha distribution"); plot_save(fig,"alpha_distribution")
    if len(assoc):
        fig, ax=plt.subplots(figsize=(6.4,4.4)); ax.scatter(assoc["historical_utility"],assoc["future_gain"],s=14,alpha=.45); ax.axhline(0,color="black",lw=.8); ax.axvline(0,color="black",lw=.8); ax.set(xlabel="Historical held-block CE utility",ylabel="Future BA gain",title="Certificate utility versus source future gain"); plot_save(fig,"alpha_vs_future_gain")
    st=read_json("STATISTICS.json",{}) or {}; ds=[]; pop=[]; always=[]; cert=[]
    primary=(read_json("SOURCE_GATE.json",{}) or {}).get("primary_method","u_pda")
    for d in c.DATASETS:
        ds.append(d); pop.append(outcome[(outcome.dataset==d)&(outcome.method=="population")].BA.mean()); always.append(outcome[(outcome.dataset==d)&(outcome.method=="persistent_ce")].BA.mean()); cert.append(outcome[(outcome.dataset==d)&(outcome.method==primary)].BA.mean())
    fig,ax=plt.subplots(figsize=(7,4.3)); x=np.arange(len(ds)); w=.24; ax.bar(x-w,pop,w,label="population"); ax.bar(x,always,w,label="always-on CE"); ax.bar(x+w,cert,w,label="certified"); ax.set_xticks(x,ds); ax.set_ylabel("Future BA"); ax.set_title("Source future population / always-on / certified"); ax.legend(); plot_save(fig,"population_vs_always_on_vs_certified")
    comp=pd.DataFrame([comparison(st,d,"correct_adapter-wrong_adapter")["delta_BA"] if comparison(st,d,"correct_adapter-wrong_adapter") else np.nan for d in c.DATASETS],index=ds,columns=["correct-wrong"]); comp["correct-shuffled"]=[comparison(st,d,"correct_adapter-shuffled_adapter").get("delta_BA",np.nan) for d in c.DATASETS]; fig,ax=plt.subplots(figsize=(7,4)); comp.plot.bar(ax=ax,color=["#70ad47","#ed7d31"]); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Delta BA"); ax.set_title("Correct versus wrong / shuffled adapters"); ax.legend(); plot_save(fig,"correct_wrong_shuffled")
    if len(oracle):
        fig,ax=plt.subplots(figsize=(6.4,4.2)); oracle.groupby("dataset")["oracle_minus_population_BA"].mean().reindex(c.DATASETS).plot.bar(ax=ax,color="#a5a5a5"); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Oracle − population BA"); ax.set_title("Diagnostic oracle headroom (not a method)"); plot_save(fig,"oracle_headroom")
    # No cross-backbone future is authorized after a failed source gate.  The
    # figure therefore records source datasets only and labels the status.
    fig,ax=plt.subplots(figsize=(6.6,4.2)); vals=[comparison(st,d,f"{primary}-population").get("delta_BA",np.nan) for d in c.DATASETS]; ax.bar(c.DATASETS,vals,color="#4472c4"); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Certified − population BA"); ax.set_title("Cross-backbone replication: NOT AUTHORIZED (source gate failed)"); plot_save(fig,"cross_backbone_gain")


def main():
    gate=read_json("SOURCE_GATE.json",{}) or {}; stats=read_json("STATISTICS.json",{}) or {}; prev=read_json("PREVIOUS_PDA_STATS.json",{}) or {}
    outcome=pd.read_csv(RES/"SOURCE_PER_SUBJECT.csv"); alpha=pd.read_csv(RES/"SUBJECT_ALPHA.csv"); assoc=pd.read_csv(RES/"CERTIFICATE_FUTURE_ASSOCIATION.csv"); oracle=pd.read_csv(RES/"ORACLE_HEADROOM.csv"); forensic=pd.read_csv(RES/"PREVIOUS_PDA_UTILITY_PREDICTIVENESS.csv")
    primary=gate.get("primary_method","u_pda"); selected=gate.get("selected_recipe",{})
    c.write_json(RES/"ALPHA_DISTRIBUTION.json",{str(float(a)):int((alpha.alpha==a).sum()) for a in c.ALPHAS})
    figures(outcome,alpha,assoc,oracle,forensic)
    status=gate.get("terminal","U_PDA_SOURCE_NOT_SUPPORTED")
    source_comps={d:comparison(stats,d,f"{primary}-population") for d in c.DATASETS}
    def line(d):
        x=source_comps[d]; return f"{d}: ΔBA {fmt(x.get('delta_BA'))}, 95% CI [{fmt(x.get('CI95_L'))}, {fmt(x.get('CI95_U'))}], n={x.get('subjects','NA')}"
    write("PREVIOUS_PDA_FORENSIC_AUDIT.md",f"""# Previous PDA forensic audit (development-known source only)

This is a forensic audit of the already committed PERSIST-PDA source outcome. It does not reinterpret the prior negative result and does not open WBCIC S2, EEGNeX future outcomes, outer, or sealed resources.

- Full-PDA minus population future BA and ordinary-adapter minus population are in `results/PREVIOUS_PDA_UTILITY_PREDICTIVENESS.csv`.
- Historical cross-fit diagnostic versus future full-PDA gain: Pearson `{fmt(prev.get('pearson_r'))}`, Spearman `{fmt(prev.get('spearman_r'))}`, AUROC for future gain > 0 `{fmt(prev.get('auroc_future_gain_gt_0'))}` (n={prev.get('n','NA')}).
- Historical-positive mean future gain: `{fmt(prev.get('positive_historical_mean_future_gain'))}`; historical-negative mean: `{fmt(prev.get('negative_historical_mean_future_gain'))}`.
- Persistent/transient norms, ratios, correct-vs-wrong and correct-vs-shuffled differences are included per subject in the CSV.

The old PDA always reused a subject adapter despite this mixed/negative transfer evidence. U-PDA treats that evidence as a reason to certify or fall back to population, not as evidence of universal personalization.
""")
    write("SCIENTIFIC_RATIONALE.md","""# Scientific rationale

Diagnosability is not actionability. U-PDA asks whether a same-subject decision intervention improves held historical likelihood before allowing it to persist. Population logits and representation remain frozen; identity alone never licenses an intervention.

The primary certificate is a fixed-alpha one-standard-error rule over four deterministic historical blocks. EB-U-PDA is one predeclared weak-evidence shrinkage variant. The source grid is bounded at 12 recipes and uses one recipe across OpenBMI and WBCIC.
""")
    write("METHOD.md","""# U-PDA method

For frozen feature `z` and population logits `p`, the adapter is `delta(z) = V diag(a) U^T LayerNorm(z) + c`. `a,c` minimize class-balanced label cross-entropy of `p + delta` plus `lambda_A ||a||² + lambda_C ||c||²`; no artificial desired-logit target is used by the primary method. `U,V` are learned from model-fit historical labels only and frozen before validation/outcome fitting.

Each subject's earliest natural session is sorted by archive index and split into four contiguous blocks. Leave-one-block-out adapters produce historical CE utility curves for alpha `{0,.25,.50,.75,1}`. The smallest alpha within one standard error of the best held-block CE is selected. Alpha zero exactly reproduces population. The final adapter uses all allowed history, and future labels/data are metrics-only.
""")
    write("RELATED_METHOD_BOUNDARY.md","""# Boundary of the claim

This package does not claim novelty for personalization, low-rank adaptation, cross-validation, or subject identity. The bounded claim is a historical intervention-utility certificate controlling prospective reuse. DANN, MMD, CORAL, SCST, Bures/local OT, cross-subject transport, identity suppression, black-box routers, neural gates, new backbones, and future-label selectors are outside the primary method.
""")
    al_counts={str(float(a)):int((alpha.alpha==a).sum()) for a in c.ALPHAS}
    write("ACTIONABILITY_CERTIFICATE_REPORT.md",f"""# Actionability certificate report

Selected recipe: `{selected}`; primary certificate method: `{primary}`.

Alpha counts (subject-fold records): `{json.dumps(al_counts,sort_keys=True)}`. Alpha is not forced on: alpha=0 is the exact population fallback. Historical held-block CE utility and subsequent development-known future gain are in `HISTORICAL_CROSSFIT_UTILITY.csv` and `CERTIFICATE_FUTURE_ASSOCIATION.csv`.

The source gate is **{status}**. U-PDA beats always-on CE on the source outcome comparison, but it does not meet the required positive-vs-population and mechanism/random-gate criteria. Therefore no WBCIC S2 or EEGNeX future run is authorized.
""")
    write("SOURCE_DEVELOPMENT_REPORT.md",f"""# Source development report

Only ATCNet-CleanRoom source representations were used. Selection used the bounded 12-recipe grid and the minimum OpenBMI/WBCIC validation delta. Selected recipe: `{selected.get('id','NA')}`. No future S2/outer/sealed resource was opened.

{line('OpenBMI')}

{line('WBCIC')}

Terminal: `{status}`.
""")
    write("SOURCE_MECHANISM_REPORT.md","""# Source mechanism report

The certified-alpha and control rows are biological-subject rows. Correct, wrong, shuffled, random-gate, ordinary CE, previous PDA, and oracle controls are retained. The correct-versus-wrong and correct-versus-shuffled directions are not positive with a pooled subject-bootstrap lower bound above zero. This is evidence against a successful actionability certificate on these source transitions, not evidence that an adapter can never be useful.
""")
    write("WBCIC_S2_REPORT.md","""# WBCIC S2 report

`NOT_RUN`: WBCIC S2 remained sealed because the ATCNet-CleanRoom source gate failed. `results/WBCIC_S2_PER_SUBJECT.csv` and `results/WBCIC_S2_PER_FOLD.csv` contain only a compact status schema and no S2 observations.
""")
    write("EEGNEX_REPORT.md","""# EEGNeX report

`NOT_RUN`: the protocol permits EEGNeX future replication only after ATCNet WBCIC S2 success. That prerequisite was not met; no EEGNeX future archive was read.
""")
    write("CONTROL_REPORT.md","""# Controls

M0 population, M1 intercept-only, M2 ordinary CE adapter, M3 previous full PDA comparator, M4 persistent CE alpha=1, M5 U-PDA one-SE, M6 EB-U-PDA, M7 random gate, M8 wrong-subject, M9 shuffled-subject, and the diagnostic oracle upper bound are represented in the source table. Oracle alpha is explicitly diagnostic and never enters recipe or alpha selection.
""")
    oracle_mean=float(oracle.oracle_minus_population_BA.mean()) if len(oracle) else float('nan')
    write("ORACLE_HEADROOM_REPORT.md",f"""# Oracle headroom

Oracle alpha is a `DIAGNOSTIC_UPPER_BOUND_ONLY` that uses future labels only to quantify family headroom. Mean source oracle-minus-population BA is `{fmt(oracle_mean)}`. It is not a fair prospective method and was not used for recipe selection, alpha selection, or gate authorization.
""")
    write("LEAKAGE_AUDIT.md","""# Leakage audit

All source rows assert `future_session_used_for_fit=false`, `future_labels_used_for_fit=false`, and `future_labels_used_for_selection=false`. Shared basis fitting reads model-fit historical labels only. Historical held blocks are excluded from their own leave-one-block-out fit. S2, EEGNeX future, outer, sealed, utility_metrics, and utility_units were not read. Population archive identities remain unchanged.
""")
    write("CLAIM_AUDIT.md",f"""# Claim audit

Supported: a true label-likelihood adapter and a deterministic historical certificate were implemented and audited on authorized source transitions.

Not supported: prospective U-PDA utility on both source datasets under the predeclared gate. The exact terminal is `{status}`. No S2, EEGNeX, cross-backbone, or outer claim is made.
""")
    write("ITERATION_LEDGER.md","""# Iteration ledger

1. Preserved the previous PDA negative result and completed a development-known forensic audit.
2. Implemented the predeclared true class-balanced CE adapter, four-block leave-one-block-out certificate, fixed alpha set, EB shrinkage, and controls.
3. Ran the bounded 12-recipe source grid once. The source gate failed; no future resource was opened and no outcome-driven scientific repair was performed.
""")
    write("REPRODUCIBILITY.md","""# Reproducibility

Run from the repository root with the authorized environment:

```text
set PYTHONPATH=experiments/persist_eeg_utility_certified_pda_final/code
python experiments/persist_eeg_utility_certified_pda_final/code/run_source.py --datasets OpenBMI WBCIC --backbone ATCNet-CleanRoom
python experiments/persist_eeg_utility_certified_pda_final/code/make_package.py
python experiments/persist_eeg_utility_certified_pda_final/code/validate.py
```

The source archives are referenced by path but not copied into this package. Runtime/checkpoint/cache/raw EEG files are excluded by `.gitignore`.
""")
    write("README.md",f"""# U-PDA — Utility-Certified Persistent Decision Adapter

This is a source-only, auditable implementation of historical actionability certification for repeated-session EEG. The package preserves prior negative terminals and does not claim success when the objective gate fails.

Selected recipe: `{selected}`. Final terminal: **{status}**. See `FINAL_REPORT.md`, `results/SOURCE_GATE.json`, and `LEAKAGE_AUDIT.md`.
""")
    write("protocol/RESOURCE_LEDGER.md","""# Resource ledger

- Authorized: ATCNet-CleanRoom `model_fit`, `validation`, and development-known `outcome` archives under `specialist_representations` for OpenBMI and WBCIC.
- Sealed/not accessed: WBCIC S2, WBCIC outer, OpenBMI sealed holdout, EEGNeX future outcomes, `utility_metrics`, and `utility_units`.
- Package excludes runtime, checkpoints, caches, raw EEG, and large embeddings.
""")
    lock_base={"status":"NOT_AUTHORIZED_SOURCE_GATE_FAILED","terminal":status,"source_gate_pass":False,"future_resource_opened":False,"reason":"No S2/future/outer method lock is valid after a failed source gate."}
    c.write_json(ROOT/"protocol"/"DATA_ACCESS_LOCK.json",{"status":"SEALED_SOURCE_ONLY","authorized_source":"ATCNet-CleanRoom model_fit/validation/development-known outcome","forbidden":["WBCIC S2","EEGNeX future","outer","sealed holdout","utility_metrics","utility_units"],"future_resource_opened":False})
    for fn in ("U_PDA_SOURCE_LOCK.json","U_PDA_FINAL_METHOD_LOCK.json","U_PDA_OUTER_CONFIRMATION_LOCK.json"): c.write_json(ROOT/"protocol"/fn,lock_base)
    final={"branch":"codex/persist-eeg-utility-certified-pda-final","final_commit":"see git tip","oracle_personalization_headroom_mean_BA":oracle_mean,"previous_pda_association":prev,"selected_recipe":selected,"alpha_distribution":al_counts,"OpenBMI_source":source_comps.get("OpenBMI",{}),"WBCIC_source":source_comps.get("WBCIC",{}),"u_pda_vs_always_on":{"OpenBMI":comparison(stats,"OpenBMI",f"{primary}-persistent_ce"),"WBCIC":comparison(stats,"WBCIC",f"{primary}-persistent_ce")},"u_pda_vs_random_gate":comparison(stats,"POOLED",f"{primary}-random_gate"),"correct_vs_wrong":comparison(stats,"POOLED","correct_adapter-wrong_adapter"),"correct_vs_shuffled":comparison(stats,"POOLED","correct_adapter-shuffled_adapter"),"WBCIC_S2_ATCNet":None,"EEGNeX":None,"cross_backbone_status":"NOT_AUTHORIZED_SOURCE_GATE_FAILED","outer_status":"SEALED_NOT_INSPECTED","strongest_supported_claim":"True CE U-PDA certificate was implemented, but the predeclared source gate failed on OpenBMI and WBCIC; prospective utility is unsupported.","exact_terminal":status}
    c.write_json(ROOT/"FINAL_REPORT.json",final)
    write("FINAL_REPORT.md",f"""# Final report

Branch: `codex/persist-eeg-utility-certified-pda-final`

Selected recipe: `{selected}`. Exact terminal: **{status}**.

{line('OpenBMI')}

{line('WBCIC')}

The certified method improves over the always-on CE comparator on the source outcome table, but the required population, random-gate, and correct-identity criteria are not met. Oracle headroom is diagnostic only. WBCIC S2, EEGNeX future replication, and outer/sealed confirmation were not run.
""")


if __name__ == "__main__": main()
