"""Run the bounded, source-only U-PDA development experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import upda_core as c


def recipes() -> list[dict[str, object]]:
    out = []
    for rank in c.RANKS:
        for l2 in c.L2S:
            for cert in ("one-SE U-PDA", "EB-U-PDA"):
                out.append({"id": f"upda_r{rank}_l2{l2:g}_{'one_se' if cert.startswith('one') else 'eb'}", "rank": rank, "l2": l2, "certificate": cert})
    return out


def fit_all(rep: dict[str, np.ndarray], role: str, basis: c.Basis, recipe: dict[str, object], cache: dict) -> tuple[list[c.SubjectTransition], dict[str, dict[str, object]]]:
    transitions = c.make_transitions(rep, n_blocks=4)
    fitted = {}
    for tr in transitions:
        key = (role, tuple(tr.history_blocks[0]["indices"].tolist()), tr.subject, int(recipe["rank"]), float(recipe["l2"]))
        if key not in cache:
            cache[key] = c.fit_subject(tr, basis, float(recipe["l2"]), cache=cache)
        fitted[tr.subject] = cache[key]
    return transitions, fitted


def eval_rows(dataset: str, role: str, fold: int, seed: int, transitions: list[c.SubjectTransition], fitted: dict[str, dict[str, object]], basis: c.Basis, recipe: dict[str, object], checkpoint: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = [tr.subject for tr in transitions]
    alpha_values = {s: float(fitted[s]["alpha"]) for s in ids}
    rng = np.random.default_rng(c.stable_seed("random-gate", dataset, role, fold, seed, recipe["id"]))
    shuffled_values = list(alpha_values.values()); rng.shuffle(shuffled_values)
    random_map = dict(zip(ids, shuffled_values))
    rows=[]; curves=[]; hist=[]
    adapter_map = fitted
    for tr in transitions:
        r, cr = c.evaluate_subject(tr, fitted[tr.subject], basis, ids, adapter_map, role, dataset, fold, seed, str(recipe["id"]), random_map)
        for x in r: x["population_checkpoint_id"] = checkpoint; x["certificate"] = recipe["certificate"]; x["rank"] = recipe["rank"]; x["l2"] = recipe["l2"]
        for x in cr: x["population_checkpoint_id"] = checkpoint; x["certificate"] = recipe["certificate"]; x["rank"] = recipe["rank"]; x["l2"] = recipe["l2"]
        rows.extend(r); curves.extend(cr)
        params=fitted[tr.subject]
        for k, block in enumerate(tr.history_blocks):
            loo=params["loo"][k]
            pm=c.metric(block,np.zeros(basis.U.shape[1]),np.zeros(c.CLASSES),basis)
            am=c.metric(block,np.asarray(loo["a"]),np.asarray(loo["c"]),basis)
            hist.append({"dataset":dataset,"role":role,"fold":fold,"seed":seed,"subject":tr.subject,"block":k,"recipe":recipe["id"],
                         "population_CE":pm["CE"],"adapter_CE":am["CE"],"utility_CE":pm["CE"]-am["CE"],
                         "population_BA":pm["BA"],"adapter_BA":am["BA"],"delta_BA":am["BA"]-pm["BA"],
                         "held_block_used_for_fit":False,"future_labels_used_for_fit":False})
    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(hist)


def val_search() -> tuple[pd.DataFrame, dict[str, object], dict]:
    cache={}; rows=[]
    for ds in c.DATASETS:
        for fold in c.FOLDS:
            for seed in c.SEEDS:
                fd=c.load_fold(ds,fold,seed)
                bases={r:c.fit_shared_basis(fd.model_fit,r) for r in c.RANKS}
                checkpoint=c.population_checkpoint_id(fd.model_fit)
                for rec in recipes():
                    transitions,fitted=fit_all(fd.validation,"validation",bases[int(rec["rank"])],rec,cache)
                    full,_,_=eval_rows(ds,"validation",fold,seed,transitions,fitted,bases[int(rec["rank"])],rec,checkpoint)
                    primary="u_pda" if rec["certificate"]=="one-SE U-PDA" else "eb_u_pda"
                    g=full[full.method.isin(["population","persistent_ce",primary])]
                    means=g.groupby("method").BA.mean()
                    rows.append({"dataset":ds,"fold":fold,"seed":seed,"recipe":rec["id"],"rank":rec["rank"],"l2":rec["l2"],"certificate":rec["certificate"],
                                 "validation_population_BA":float(means.get("population",np.nan)),"validation_persistent_ce_BA":float(means.get("persistent_ce",np.nan)),
                                 "validation_certified_BA":float(means.get(primary,np.nan)),"validation_delta_BA":float(means.get(primary,np.nan)-means.get("population",np.nan)),
                                 "validation_delta_vs_always_on":float(means.get(primary,np.nan)-means.get("persistent_ce",np.nan)),"subjects":int(full.subject.nunique())})
    frame=pd.DataFrame(rows)
    summary=frame.groupby(["dataset","recipe","rank","l2","certificate"],as_index=False).agg(validation_population_BA=("validation_population_BA","mean"),validation_persistent_ce_BA=("validation_persistent_ce_BA","mean"),validation_certified_BA=("validation_certified_BA","mean"),validation_delta_BA=("validation_delta_BA","mean"),validation_delta_vs_always_on=("validation_delta_vs_always_on","mean"),validation_subjects=("subjects","sum"))
    piv=summary.pivot_table(index=["recipe","rank","l2","certificate"],columns="dataset",values="validation_delta_BA")
    for ds in c.DATASETS:
        if ds not in piv: piv[ds]=np.nan
    piv["minimum_dataset_delta"]=piv[list(c.DATASETS)].min(axis=1); piv["mean_dataset_delta"]=piv[list(c.DATASETS)].mean(axis=1)
    # Deterministic tie-break follows recipe order after the scientific score.
    order={r["id"]:i for i,r in enumerate(recipes())}
    p=piv.reset_index(); p["_order"]=p.recipe.map(order)
    selected_row=p.sort_values(["minimum_dataset_delta","mean_dataset_delta","_order"],ascending=[False,False,True]).iloc[0]
    selected=next(r for r in recipes() if r["id"]==selected_row.recipe)
    out=summary.merge(p.drop(columns=["_order"]),on=["recipe","rank","l2","certificate"],how="left")
    return out,selected,cache


def run_selected(selected: dict[str, object], cache: dict) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    all_rows=[]; all_curves=[]; all_hist=[]
    for ds in c.DATASETS:
        for fold in c.FOLDS:
            for seed in c.SEEDS:
                fd=c.load_fold(ds,fold,seed); basis=c.fit_shared_basis(fd.model_fit,int(selected["rank"])); checkpoint=c.population_checkpoint_id(fd.model_fit)
                transitions,fitted=fit_all(fd.outcome,"outcome",basis,selected,cache)
                rows,curves,hist=eval_rows(ds,"outcome",fold,seed,transitions,fitted,basis,selected,checkpoint)
                all_rows.append(rows); all_curves.append(curves); all_hist.append(hist)
    return pd.concat(all_rows,ignore_index=True),pd.concat(all_curves,ignore_index=True),pd.concat(all_hist,ignore_index=True)


def forensic_audit() -> tuple[pd.DataFrame,dict]:
    oldr=c.OLD_EXP/"results"/"SOURCE_PER_SUBJECT.csv"; oldc=c.OLD_EXP/"results"/"ADAPTER_COMPONENTS.csv"
    if not oldr.is_file(): return pd.DataFrame(),{"status":"OLD_PDA_RESULTS_MISSING"}
    r=pd.read_csv(oldr); comp=pd.read_csv(oldc) if oldc.is_file() else pd.DataFrame()
    keep=r[r.method.isin(["population","ordinary_adapter","full_pda","correct_adapter","wrong_adapter","shuffled_adapter"])].copy()
    piv=keep.pivot_table(index=["dataset","fold","seed","subject"],columns="method",values="BA",aggfunc="mean").reset_index()
    for col in ["population","ordinary_adapter","full_pda","correct_adapter","wrong_adapter","shuffled_adapter"]:
        if col not in piv: piv[col]=np.nan
    piv["full_pda_minus_population_future_BA"]=piv.full_pda-piv.population
    piv["ordinary_adapter_minus_population_future_BA"]=piv.ordinary_adapter-piv.population
    piv["correct_minus_wrong_future_BA"]=piv.correct_adapter-piv.wrong_adapter
    piv["correct_minus_shuffled_future_BA"]=piv.correct_adapter-piv.shuffled_adapter
    if len(comp):
        use=comp[["dataset","fold","seed","subject","persistent_norm","transient_norm","persistent_transient_ratio","historical_crossfit_gain"]]
        piv=piv.merge(use,on=["dataset","fold","seed","subject"],how="left")
    x=piv.dropna(subset=["historical_crossfit_gain","full_pda_minus_population_future_BA"])
    stats={"n":int(len(x))}
    if len(x)>=3 and x.historical_crossfit_gain.nunique()>1 and x.full_pda_minus_population_future_BA.nunique()>1:
        stats["pearson_r"]=float(__import__("scipy").stats.pearsonr(x.historical_crossfit_gain,x.full_pda_minus_population_future_BA).statistic)
        stats["spearman_r"]=float(__import__("scipy").stats.spearmanr(x.historical_crossfit_gain,x.full_pda_minus_population_future_BA).statistic)
        y=(x.full_pda_minus_population_future_BA>0).astype(int)
        stats["auroc_future_gain_gt_0"]=float(__import__("sklearn").metrics.roc_auc_score(y,x.historical_crossfit_gain)) if y.nunique()==2 else None
    else: stats.update(pearson_r=None,spearman_r=None,auroc_future_gain_gt_0=None)
    stats["positive_historical_mean_future_gain"]=float(x.loc[x.historical_crossfit_gain>0,"full_pda_minus_population_future_BA"].mean()) if np.any(x.historical_crossfit_gain>0) else None
    stats["negative_historical_mean_future_gain"]=float(x.loc[x.historical_crossfit_gain<=0,"full_pda_minus_population_future_BA"].mean()) if np.any(x.historical_crossfit_gain<=0) else None
    return piv,stats


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--datasets",nargs="+",choices=c.DATASETS,default=list(c.DATASETS)); ap.add_argument("--backbone",default="ATCNet-CleanRoom")
    args=ap.parse_args()
    if tuple(args.datasets)!=c.DATASETS: raise SystemExit("U-PDA source recipe selection requires both authorized source datasets")
    if args.backbone not in {"ATCNet","ATCNet-CleanRoom"}: raise SystemExit("primary backbone is fixed to ATCNet-CleanRoom")
    c.RESULTS.mkdir(parents=True,exist_ok=True); c.RUNTIME.mkdir(parents=True,exist_ok=True)
    forensic, forensic_stats=forensic_audit(); c.write_csv(c.RESULTS/"PREVIOUS_PDA_UTILITY_PREDICTIVENESS.csv",forensic)
    c.write_json(c.RESULTS/"PREVIOUS_PDA_STATS.json",forensic_stats)
    search,selected,cache=val_search(); c.write_csv(c.RESULTS/"SOURCE_RECIPE_SEARCH.csv",search)
    c.write_json(c.RESULTS/"SOURCE_RECIPE_SELECTION.json",{"selected":selected,"selection_rule":"maximize minimum OpenBMI/WBCIC validation delta; tie-break mean then recipe order","validation_future_labels_used_for_fit":False,"future_S2_opened":False})
    outcome,curves,hist=run_selected(selected,cache)
    c.write_csv(c.RESULTS/"SOURCE_PER_SUBJECT.csv",outcome); c.write_csv(c.RESULTS/"SOURCE_PER_FOLD.csv",outcome.groupby(["dataset","role","fold","seed","method"],as_index=False).agg(BA=("BA","mean"),macro_F1=("macro_F1","mean"),CE=("CE","mean"),subjects=("subject","nunique")))
    alpha_keys=["dataset","role","fold","seed","subject","recipe"]
    subject_alpha=curves.groupby(alpha_keys,as_index=False).first()
    subject_alpha["alpha"]=subject_alpha["selected_one_se"] if selected["certificate"]=="one-SE U-PDA" else subject_alpha["selected_eb"]
    c.write_csv(c.RESULTS/"SUBJECT_ALPHA.csv",subject_alpha); c.write_csv(c.RESULTS/"ALPHA_CALIBRATION.csv",curves); c.write_csv(c.RESULTS/"HISTORICAL_CROSSFIT_UTILITY.csv",hist)
    primary="u_pda" if selected["certificate"]=="one-SE U-PDA" else "eb_u_pda"
    cert=outcome[outcome.method==primary].copy(); pop=outcome[outcome.method=="population"].copy(); join=cert.merge(pop,on=["dataset","role","fold","seed","subject"],suffixes=("_cert","_pop")); join["future_gain"]=join.BA_cert-join.BA_pop
    hist_cols=["dataset","role","fold","seed","subject","alpha","selected_one_se","selected_eb","historical_utility"]
    hist_for_assoc=curves[hist_cols].drop_duplicates(["dataset","role","fold","seed","subject"])
    join=join.merge(hist_for_assoc,on=["dataset","role","fold","seed","subject"],how="left")
    assoc=join[["dataset","role","fold","seed","subject","alpha","historical_utility","future_gain","BA_cert","BA_pop"]]; c.write_csv(c.RESULTS/"CERTIFICATE_FUTURE_ASSOCIATION.csv",assoc)
    c.write_csv(c.RESULTS/"CORRECT_WRONG_SHUFFLED.csv",outcome[outcome.method.isin(["u_pda","correct_adapter","wrong_adapter","shuffled_adapter"])])
    c.write_csv(c.RESULTS/"RANDOM_GATE_CONTROL.csv",outcome[outcome.method=="random_gate"])
    ora=outcome[outcome.method=="oracle_alpha"].merge(pop,on=["dataset","role","fold","seed","subject"],suffixes=("_oracle","_pop")); ora["oracle_minus_population_BA"]=ora.BA_oracle-ora.BA_pop; ora["oracle_label"]=ora.get("oracle_label_oracle","DIAGNOSTIC_UPPER_BOUND_ONLY"); c.write_csv(c.RESULTS/"ORACLE_HEADROOM.csv",ora[["dataset","role","fold","seed","subject","alpha_oracle","oracle_minus_population_BA","oracle_label"]])
    comparisons=[]
    for ds in c.DATASETS:
        for left,right in [(primary,"population"),(primary,"persistent_ce"),(primary,"ordinary_adapter"),(primary,"random_gate"),("correct_adapter","wrong_adapter"),("correct_adapter","shuffled_adapter")]: comparisons.append(c.paired_delta(outcome,left,right,ds,"source"))
    comparisons.append(c.pooled_paired_delta(outcome,primary,"population","source")); comparisons.append(c.pooled_paired_delta(outcome,primary,"random_gate","source")); comparisons.append(c.pooled_paired_delta(outcome,"correct_adapter","wrong_adapter","source")); comparisons.append(c.pooled_paired_delta(outcome,"correct_adapter","shuffled_adapter","source"))
    c.write_json(c.RESULTS/"STATISTICS.json",{"selected_recipe":selected,"primary_method":primary,"comparisons":comparisons,"bootstrap_unit":"biological_subject","n_bootstrap":c.N_BOOTSTRAP,"previous_pda":forensic_stats})
    def cmp(ds,l,r): return next((x for x in comparisons if x["dataset"]==ds and x["comparison"]==f"{l}-{r}"),None)
    gate={"selected_recipe":selected,"primary_method":primary,"OpenBMI_delta_ge_0_003":bool(cmp("OpenBMI",primary,"population") and cmp("OpenBMI",primary,"population")["delta_BA"]>=.003),"WBCIC_delta_ge_0_003":bool(cmp("WBCIC",primary,"population") and cmp("WBCIC",primary,"population")["delta_BA"]>=.003),"OpenBMI_ci_lower_gt_zero":bool(cmp("OpenBMI",primary,"population") and cmp("OpenBMI",primary,"population")["CI95_L"]>0),"WBCIC_ci_lower_gt_zero":bool(cmp("WBCIC",primary,"population") and cmp("WBCIC",primary,"population")["CI95_L"]>0),"U_PDA_gt_always_on":bool(all((cmp(ds,primary,"persistent_ce") or {"delta_BA":-1})["delta_BA"]>0 for ds in c.DATASETS)),"U_PDA_gt_random_pooled_ci":bool(comparisons[-3]["CI95_L"]>0),"correct_gt_wrong_pooled_ci":bool(comparisons[-2]["CI95_L"]>0),"correct_gt_shuffled_pooled_ci":bool(comparisons[-1]["CI95_L"]>0)}
    alpha=outcome[outcome.method==primary].groupby("subject").alpha.mean(); gain=assoc.groupby("subject").future_gain.mean(); positive=alpha[alpha>0].index.intersection(gain.index)
    gate["alpha_positive_subjects_future_gain_positive"]=bool(len(positive) and float(gain.loc[positive].mean())>0); gate["alpha_not_all_zero"]=bool(np.any(alpha.to_numpy()>0)); gate["alpha_not_all_one"]=bool(np.any(alpha.to_numpy()<1)); gate["no_future_labels_in_fit"]=bool((outcome.future_labels_used_for_fit==False).all()); gate["no_future_session_in_fit"]=bool((outcome.future_session_used_for_fit==False).all()); gate["population_checkpoint_unchanged"]=bool(outcome.population_checkpoint_unchanged.all()); gate["source_gate_pass"]=bool(all(v for k,v in gate.items() if k not in {"selected_recipe","primary_method"})); gate["terminal"]="U_PDA_SOURCE_ONLY_SUPPORTED" if gate["source_gate_pass"] else "U_PDA_SOURCE_NOT_SUPPORTED"; c.write_json(c.RESULTS/"SOURCE_GATE.json",gate); c.write_json(c.RESULTS/"ITERATION_STATE.json",{"finished":True,"terminal":gate["terminal"],"future_resource_opened":False,"source_gate_pass":gate["source_gate_pass"]})
    # Explicit sealed-resource placeholders make the no-access decision auditable.
    for name in ("WBCIC_S2_PER_SUBJECT.csv","WBCIC_S2_PER_FOLD.csv","EEGNEX_PER_SUBJECT.csv","EEGNEX_PER_FOLD.csv"):
        c.write_csv(c.RESULTS/name,pd.DataFrame(columns=["status","reason"]))
    print(json.dumps({"terminal":gate["terminal"],"selected_recipe":selected,"primary_method":primary,"gate":gate},indent=2,sort_keys=True))


if __name__=="__main__": main()
