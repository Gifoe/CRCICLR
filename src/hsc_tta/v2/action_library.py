from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_action_library(root:str|Path,bootstrap_repetitions:int=2000)->dict[str,object]:
    root=Path(root); base=root/"outputs/v2_joint_certified"; features=pd.read_parquet(base/"actions/DEVELOPMENT_CONTEXT_FEATURES.parquet")
    outcomes=pd.read_parquet(base/"actions/DEVELOPMENT_ACTION_SURFACE.parquet")
    rows=[]; safe_rows=[]; win_rows=[]; fold_subject=[]
    for dataset in ("hmc","eegmmidb"):
        for seed in range(5):
            split=json.loads((root/"data/splits"/dataset/f"seed_{seed}.json").read_text()); meta=set(split["roles"]["meta_risk_train"])
            local=outcomes[(outcomes.dataset==dataset)&(outcomes.seed==seed)&outcomes.subject_id.isin(meta)].copy()
            no=local[local.action=="no_tta"][["subject_id","alpha","argmax_error"]].rename(columns={"argmax_error":"no_tta_error"})
            local=local.merge(no,on=["subject_id","alpha"]); local["gain"]=local.no_tta_error-local.argmax_error
            for (action,alpha),g in local.groupby(["action","alpha"]):
                unavailable=features[(features.dataset==dataset)&(features.seed==seed)&features.subject_id.isin(meta)&(features.action==action)].action_available
                rows.append({"dataset":dataset,"seed":seed,"action":action,"alpha":alpha,"mean_gain":g.gain.mean(),
                    "win_rate":np.mean(g.gain>0),"harm_rate":np.mean(g.gain<0),"safe_rate":np.mean(g.true_critical_index<20),
                    "safe_beneficial_rate":np.mean((g.true_critical_index<20)&(g.gain>0)),"unavailable_rate":1-unavailable.mean()})
            for alpha,g in local.groupby("alpha"):
                tta=g[(g.action!="no_tta")&(g.true_critical_index<20)&(g.gain>0)]
                best=tta.groupby("subject_id").gain.max(); subjects=sorted(meta)
                gains=np.asarray([float(best.get(s,0)) for s in subjects])
                rng=np.random.default_rng(74000+seed*100+int(alpha*100))
                boot=np.asarray([rng.choice(gains,len(gains),replace=True).mean() for _ in range(bootstrap_repetitions)])
                safe_rows.append({"dataset":dataset,"seed":seed,"alpha":alpha,"safe_oracle_gain":gains.mean(),
                    "ci_lower":np.quantile(boot,.025),"ci_upper":np.quantile(boot,.975),
                    "safe_beneficial_subject_rate":np.mean(gains>0)})
                fold_subject.extend({"dataset":dataset,"seed":seed,"alpha":alpha,"subject_id":s,"safe_gain":gain}
                                    for s,gain in zip(subjects,gains))
            best=local.sort_values(["subject_id","alpha","argmax_error","action"]).groupby(["subject_id","alpha"]).first().reset_index()
            for action,rate in best.action.value_counts(normalize=True).items(): win_rows.append({"dataset":dataset,"seed":seed,"action":action,"oracle_win_rate":rate})
    out=base/"actions"; candidates=pd.DataFrame(rows); candidates.to_csv(out/"ACTION_CANDIDATE_RESULTS.csv",index=False)
    pd.DataFrame(win_rows).to_csv(out/"ACTION_WIN_RATE.csv",index=False); safe=pd.DataFrame(safe_rows); safe.to_csv(out/"ACTION_SAFE_ORACLE_HEADROOM.csv",index=False)
    subject=pd.DataFrame(fold_subject); wide=subject.pivot_table(index=["dataset","seed","subject_id"],columns="alpha",values="safe_gain")
    diversity=[]
    for dataset in ("hmc","eegmmidb"):
        local=outcomes[(outcomes.dataset==dataset)&np.isclose(outcomes.alpha,.1)].copy()
        no=local[local.action=="no_tta"][["seed","subject_id","argmax_error"]].rename(columns={"argmax_error":"no"})
        local=local.merge(no,on=["seed","subject_id"]); local["gain"]=local.no-local.argmax_error
        matrix=local.pivot_table(index=["seed","subject_id"],columns="action",values="gain").corr()
        for a in matrix.index:
            for b in matrix.columns: diversity.append({"dataset":dataset,"action_a":a,"action_b":b,"gain_correlation":matrix.loc[a,b]})
    pd.DataFrame(diversity).to_csv(out/"ACTION_DIVERSITY_MATRIX.csv",index=False)
    hmc=safe[safe.dataset=="hmc"]; gate=bool((hmc.groupby("alpha").safe_oracle_gain.mean()>0).all() and (hmc.groupby("alpha").ci_lower.mean()>0).all())
    report="# V2 action library selection\n\n"+candidates.groupby(["dataset","action","alpha"])[["mean_gain","win_rate","harm_rate","safe_beneficial_rate","unavailable_rate"]].mean().reset_index().to_markdown(index=False)+"\n\n"
    report+=safe.to_markdown(index=False)+"\n\n"
    report+=("Gate: **PASS**. Formal library is `no_tta`, `official_t3a`, and `robust_residual_adapter`.\n" if gate else
             "Gate: **NO ACTION HEADROOM HARD BLOCKER**. Selector training must stop.\n")
    report+=("Tent and official EATA configure BatchNorm statistics/affine parameters; the frozen CBraMod and selected heads contain LayerNorm and no eligible BatchNorm, so they are marked incompatible rather than relabeled. "
             "Official SAR was not vendored or claimed: a custom LayerNorm/SAM variant would be a different method and is outside the frozen two-TTA library.\n")
    (out/"ACTION_LIBRARY_SELECTION.md").write_text(report,encoding="utf-8")
    if not gate: (base/"NO_ACTION_HEADROOM_HARD_BLOCKER.md").write_text(report,encoding="utf-8")
    return {"gate":gate,"hmc_safe_gain":hmc.groupby("alpha").safe_oracle_gain.mean().to_dict()}
