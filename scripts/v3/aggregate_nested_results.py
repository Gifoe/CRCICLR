#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.v3.statistics import subject_cluster_bootstrap

from _common import load_yaml, project_root


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default="."); parser.add_argument("--config",required=True)
    args=parser.parse_args(); root=project_root(args.root); config=load_yaml(args.config)
    base=root/"outputs/v3_probecert/nested_dev"; frame=pd.read_parquet(base/"counterfactuals/OUTER_COUNTERFACTUALS.parquet")
    frame["risk_violation"] = frame.future_risk > frame.alpha
    frame["noninferiority_violation"] = frame.degradation > float(config["epsilon"])
    frame["harmful_intervention"] = frame.intervention & frame.noninferiority_violation
    frame["normalized_set_size"] = frame.average_set_size / frame.groupby("dataset").average_set_size.transform("max").clip(lower=1)
    by_seed=frame.groupby(["dataset","seed","alpha","policy"],as_index=False).agg(
        n_subjects=("subject_id","nunique"),joint_violation_rate=("joint_violation","mean"),
        risk_violation_rate=("risk_violation","mean"),noninferiority_violation_rate=("noninferiority_violation","mean"),
        average_set_size=("average_set_size","mean"),normalized_set_size=("normalized_set_size","mean"),
        singleton_rate=("singleton_rate","mean"),argmax_error=("argmax_error","mean"),macro_f1=("macro_f1","mean"),
        intervention_rate=("intervention","mean"),sentinel_rate=("sentinel","mean"))
    harmful=frame.groupby(["dataset","seed","alpha","policy"],as_index=False).harmful_intervention.mean().rename(columns={"harmful_intervention":"harmful_intervention_rate"})
    by_seed=by_seed.merge(harmful,on=["dataset","seed","alpha","policy"]);by_seed["non_adaptation_rate"]=1-by_seed.intervention_rate
    by_seed["joint_validity"]=1-by_seed.joint_violation_rate
    by_seed.to_csv(base/"metrics/RESULTS_BY_SEED.csv",index=False)
    averaged=frame.groupby(["dataset","subject_id","alpha","policy"],as_index=False).agg(
        average_set_size=("average_set_size","mean"),joint_violation=("joint_violation","mean"),
        risk_violation=("risk_violation","mean"),noninferiority_violation=("noninferiority_violation","mean"),
        singleton_rate=("singleton_rate","mean"),argmax_error=("argmax_error","mean"),macro_f1=("macro_f1","mean"),
        intervention=("intervention","mean"),sentinel=("sentinel","mean"))
    averaged_harm=frame.groupby(["dataset","subject_id","alpha","policy"],as_index=False).harmful_intervention.mean()
    averaged=averaged.merge(averaged_harm,on=["dataset","subject_id","alpha","policy"])
    summaries=[]; cis=[]
    for keys,group in averaged.groupby(["dataset","alpha","policy"]):
        dataset,alpha,policy=keys
        summaries.append({"dataset":dataset,"alpha":alpha,"policy":policy,"n_unique_subjects":group.subject_id.nunique(),
            "joint_violation_rate":group.joint_violation.mean(),"joint_validity":1-group.joint_violation.mean(),
            "average_set_size":group.average_set_size.mean(),"singleton_rate":group.singleton_rate.mean(),
            "argmax_error":group.argmax_error.mean(),"macro_f1":group.macro_f1.mean(),"intervention_rate":group.intervention.mean(),
            "non_adaptation_rate":1-group.intervention.mean(),"harmful_intervention_rate":group.harmful_intervention.mean(),
            "sentinel_rate":group.sentinel.mean()})
        for metric in ["average_set_size","joint_violation","intervention","sentinel"]:
            boot=subject_cluster_bootstrap(group.rename(columns={metric:"value"}),"value",repetitions=int(config["bootstrap_repetitions"]))
            cis.append({"dataset":dataset,"alpha":alpha,"policy":policy,"metric":metric,**boot})
    summary=pd.DataFrame(summaries); summary.to_csv(base/"metrics/RESULTS_SUMMARY.csv",index=False)
    pd.DataFrame(cis).to_csv(base/"metrics/SUBJECT_BOOTSTRAP_CI.csv",index=False)
    comparisons=[]
    for (dataset,alpha),group in averaged.groupby(["dataset","alpha"]):
        pivot=group.pivot(index="subject_id",columns="policy",values="average_set_size").dropna()
        if {"probecert_v3","no_tta_global_crc"} <= set(pivot):
            difference=(pivot.no_tta_global_crc-pivot.probecert_v3).to_numpy(); rng=np.random.default_rng(7201)
            draw=rng.integers(0,len(difference),size=(int(config["bootstrap_repetitions"]),len(difference))); means=difference[draw].mean(1)
            comparisons.append({"dataset":dataset,"alpha":alpha,"proposed":"probecert_v3","baseline":"no_tta_global_crc",
                "mean_set_size_reduction":difference.mean(),"ci_lower":np.quantile(means,.025),"ci_upper":np.quantile(means,.975),"n_subjects":len(difference)})
    pd.DataFrame(comparisons).to_csv(base/"metrics/PAIRED_COMPARISONS.csv",index=False)
    main_table=summary.rename(columns={"joint_violation_rate":"Violation","joint_validity":"CSR","average_set_size":"Set Size",
        "non_adaptation_rate":"NAR","harmful_intervention_rate":"HER","macro_f1":"Macro-F1"})
    main_table[["dataset","policy","alpha","Violation","CSR","Set Size","NAR","HER","Macro-F1"]].to_csv(base/"metrics/MAIN_RESULTS_TABLE.csv",index=False)
    proposed=frame[frame.policy=="probecert_v3"].copy();source=frame[frame.policy=="no_tta_global_crc"][["dataset","seed","alpha","subject_id","future_risk"]].rename(columns={"future_risk":"no_tta_future_risk"})
    proposed=proposed.merge(source,on=["dataset","seed","alpha","subject_id"]);proposed["selected_vs_no_tta_risk_change"]=proposed.future_risk-proposed.no_tta_future_risk
    selection=proposed.groupby(["dataset","seed","alpha","selected_action"],as_index=False).agg(selection_rate=("subject_id","size"),
        uncertified_rate=("sentinel","mean"),full_set_fallback_rate=("sentinel","mean"),
        selected_vs_no_tta_risk_change=("selected_vs_no_tta_risk_change","mean"),harmful_selected_rate=("harmful_intervention","mean"))
    totals=proposed.groupby(["dataset","seed","alpha"]).subject_id.size();selection["selection_rate"]=[row.selection_rate/totals.loc[(row.dataset,row.seed,row.alpha)] for row in selection.itertuples()]
    selection.to_csv(base/"metrics/ACTION_SELECTION_SUMMARY.csv",index=False)
    print(summary.to_string(index=False))


if __name__=="__main__": main()
