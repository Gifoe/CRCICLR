from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import modern_common as c


HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    env = os.environ.copy(); env["PYTHONPATH"] = str(HERE)
    print("[exec] " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def pred_path(model: str, dataset: str, fold: int) -> Path:
    slug = model.lower() if model in ("EEGNet", "LiteBN") else c.modern_slug(model)
    return c.RUNTIME / "predictions" / f"stageB_{dataset.lower()}_fold{fold}_{slug}.npz"


def ensure_predictions(model: str, dataset: str, fold: int) -> Path:
    path = pred_path(model, dataset, fold)
    if not path.is_file():
        run([sys.executable, str(HERE / "predict_model.py"), "--model", model, "--dataset", dataset, "--fold", str(fold), "--stage", "B"])
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def metric(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    pred = logits.argmax(1); subjects = subjects.astype(str)
    ba=[]; f1=[]; acc=[]
    for s in c.subject_sort(np.unique(subjects)):
        m=subjects==s; ba.append(balanced_accuracy_score(labels[m],pred[m])); f1.append(f1_score(labels[m],pred[m],average="macro",zero_division=0)); acc.append(accuracy_score(labels[m],pred[m]))
    return {"BA":float(np.mean(ba)),"macro_F1":float(np.mean(f1)),"accuracy":float(np.mean(acc))}


def bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    rng=np.random.default_rng(0); draw=rng.choice(values,size=(10000,len(values)),replace=True).mean(1)
    return float(values.mean()),float(np.quantile(draw,.025)),float(np.quantile(draw,.975))


def write_text(path: Path, text: str) -> None:
    tmp=path.with_suffix(path.suffix+".part"); tmp.write_text(text.rstrip()+"\n",encoding="utf-8"); os.replace(tmp,path)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--allow-outer-reveal",action="store_true"); args=parser.parse_args()
    if not args.allow_outer_reveal: raise SystemExit("Stage B is an explicit post-lock operation; pass --allow-outer-reveal")
    lock_path=c.PROTOCOL/"STAGE_A_LOCK.json"
    if not lock_path.is_file(): raise SystemExit("missing Stage A lock")
    lock=c.read_json(lock_path)
    if lock.get("outer_predictions_generated") is not False or lock.get("fresh_pair_outer_utility_generated") is not False: raise SystemExit("invalid Stage A lock")
    folds,search,split_sha=c.load_split()
    blind=pd.read_csv(c.PROTOCOL/"STAGE_A_BLIND_PREDICTIONS.csv")
    status=pd.read_csv(c.PROTOCOL/"PAIR_OUTCOME_STATUS.csv")
    prediction={}
    for dataset in c.DATASETS:
        for fold in range(5):
            for model in c.MODELS:
                prediction[(dataset,fold,model)]=load_npz(ensure_predictions(model,dataset,fold))
    # Per-backbone outer metrics.
    model_rows=[]
    for dataset in c.DATASETS:
        for fold in range(5):
            for model in c.MODELS:
                z=prediction[(dataset,fold,model)]; m=metric(z["labels"],z["logits"],z["subjects"])
                model_rows.append({"dataset":dataset,"fold":fold,"model":model,**m})
    model_df=pd.DataFrame(model_rows); model_df.to_csv(c.OUT/"BACKBONE_FOLD_METRICS.csv",index=False)
    pair_rows=[]
    for dataset in c.DATASETS:
        for fold in range(5):
            for a,b in itertools.combinations(c.MODELS,2):
                za,zb=prediction[(dataset,fold,a)],prediction[(dataset,fold,b)]
                if not np.array_equal(za["labels"],zb["labels"]) or not np.array_equal(za["subjects"],zb["subjects"]): raise RuntimeError("outer prediction alignment mismatch")
                labels=za["labels"]; subjects=za["subjects"].astype(str); logit=.5*(za["logits"]+zb["logits"]); prob=.5*(np.exp(za["logits"]-za["logits"].max(1,keepdims=True))/np.exp(za["logits"]-za["logits"].max(1,keepdims=True)).sum(1,keepdims=True)+np.exp(zb["logits"]-zb["logits"].max(1,keepdims=True))/np.exp(zb["logits"]-zb["logits"].max(1,keepdims=True)).sum(1,keepdims=True));
                ma=metric(labels,za["logits"],subjects); mb=metric(labels,zb["logits"],subjects); ml=metric(labels,logit,subjects); mp=metric(labels,prob,subjects)
                st=status[(status.model_a==a)&(status.model_b==b)].iloc[0]
                pair_rows.append({"dataset":dataset,"fold":fold,"pair":f"{a}+{b}","model_a":a,"model_b":b,"historical":st.status=="HISTORICAL","competence_eligible":bool(blind[(blind.dataset==dataset)&(blind.pair==f"{a}+{b}")].competence_eligible.iloc[0]),"EEGNet_or_component_A_BA":ma["BA"],"component_B_BA":mb["BA"],"LOGIT50_BA":ml["BA"],"PROB50_BA":mp["BA"],"G_test":ml["BA"]-max(ma["BA"],mb["BA"]),"G_prob":mp["BA"]-max(ma["BA"],mb["BA"]),"LOGIT50_macro_F1":ml["macro_F1"],"PROB50_macro_F1":mp["macro_F1"]})
    pair_df=pd.DataFrame(pair_rows); pair_df.to_csv(c.OUT/"PAIR_OUTER_METRICS.csv",index=False)
    # Pair-level actionability correlations; pair family, not folds, is the unit.
    rows=[]
    for dataset in c.DATASETS:
        p=pair_df[pair_df.dataset==dataset].groupby(["pair","model_a","model_b","historical","competence_eligible"],as_index=False).agg(G_test=("G_test","mean"),G_prob=("G_prob","mean"))
        d=blind[blind.dataset==dataset][["pair","EC","disagreement","margin_diversity","G_dev","fresh_eligible","EC_selected_pair","Gdev_selected_pair"]]
        p=p.merge(d,on="pair",how="left"); eligible=p[(~p.historical)&p.competence_eligible].copy()
        for x in ("EC","disagreement","margin_diversity","G_dev"):
            rho=spearmanr(eligible[x],eligible.G_test).statistic if len(eligible)>=3 and eligible[x].nunique()>1 and eligible.G_test.nunique()>1 else np.nan
            rows.append({"dataset":dataset,"diagnostic":x,"n_fresh_eligible_pairs":len(eligible),"spearman_rho":None if not np.isfinite(rho) else float(rho),"mean_G_test":float(eligible.G_test.mean()) if len(eligible) else None,"min_G_test":float(eligible.G_test.min()) if len(eligible) else None,"max_G_test":float(eligible.G_test.max()) if len(eligible) else None})
    corr_df=pd.DataFrame(rows); corr_df.to_csv(c.OUT/"ACTIONABILITY_CORRELATIONS.csv",index=False)
    # Selected-vs-random-vs-oracle, using Stage-A-frozen selections.
    selected=[]
    for dataset in c.DATASETS:
        p=pair_df[pair_df.dataset==dataset].groupby(["pair","historical","competence_eligible"],as_index=False).agg(G_test=("G_test","mean"),G_prob=("G_prob","mean"))
        e=p[(~p.historical)&p.competence_eligible]
        d=blind[blind.dataset==dataset]; ec_pair=str(d[d.fresh_eligible].sort_values(["EC","pair"],ascending=[False,True]).iloc[0].pair) if len(d[d.fresh_eligible]) else None; gd_pair=str(d[d.fresh_eligible].sort_values(["G_dev","pair"],ascending=[False,True]).iloc[0].pair) if len(d[d.fresh_eligible]) else None
        def gain(pair):
            q=e[e.pair==pair]; return float(q.G_test.iloc[0]) if len(q) else None
        selected.append({"dataset":dataset,"n_fresh_eligible_pairs":int(len(e)),"EC_selected_pair":ec_pair,"Gdev_selected_pair":gd_pair,"EC_selected_gain":gain(ec_pair),"Gdev_selected_gain":gain(gd_pair),"random_pair_mean_gain":float(e.G_test.mean()) if len(e) else None,"oracle_best_pair":str(e.sort_values(["G_test","pair"],ascending=[False,True]).iloc[0].pair) if len(e) else None,"oracle_gain":float(e.G_test.max()) if len(e) else None,"oracle_minus_random":float(e.G_test.max()-e.G_test.mean()) if len(e) else None})
    sel_df=pd.DataFrame(selected); sel_df.to_csv(c.OUT/"SELECTED_VS_ORACLE.csv",index=False)
    # Full matrix (all historical/fresh pairs) with dataset-specific deltas.
    matrix=[]
    p=pair_df.groupby(["pair","model_a","model_b","historical","competence_eligible","dataset"],as_index=False).agg(G_test=("G_test","mean"),G_prob=("G_prob","mean"),LOGIT50_BA=("LOGIT50_BA","mean"),PROB50_BA=("PROB50_BA","mean"))
    for pair in sorted(p.pair.unique()):
        base=p[p.pair==pair].iloc[0]; row={"pair":pair,"model_a":base.model_a,"model_b":base.model_b,"historical":bool(base.historical),"competence_eligible_all_datasets":bool(len(p[(p.pair==pair)&(~p.competence_eligible)])==0)}
        for dataset in c.DATASETS:
            q=p[(p.pair==pair)&(p.dataset==dataset)]
            if len(q): row.update({f"{dataset}_delta_best_pp":float(q.G_test.iloc[0]*100),f"{dataset}_delta_prob_best_pp":float(q.G_prob.iloc[0]*100),f"{dataset}_LOGIT50_BA":float(q.LOGIT50_BA.iloc[0]),f"{dataset}_PROB50_BA":float(q.PROB50_BA.iloc[0]),f"{dataset}_positive":bool(q.G_test.iloc[0]>0),f"{dataset}_competence_eligible":bool(q.competence_eligible.iloc[0])})
        matrix.append(row)
    pd.DataFrame(matrix).to_csv(c.OUT/"MULTIBACKBONE_MATRIX.csv",index=False)
    # Compact dataset summary and final decision.
    summary=[]
    for dataset in c.DATASETS:
        q=model_df[model_df.dataset==dataset].groupby("model",as_index=False).agg(BA=("BA","mean"),macro_F1=("macro_F1","mean"),accuracy=("accuracy","mean"));
        for r in q.itertuples(): summary.append({"dataset":dataset,"model":r.model,"mean_outer_dev_BA":r.BA,"mean_outer_dev_macro_F1":r.macro_F1,"mean_outer_dev_accuracy":r.accuracy})
    pd.DataFrame(summary).to_csv(c.OUT/"BACKBONE_DATASET_SUMMARY.csv",index=False)
    comp=pd.read_csv(c.OUT/"BACKBONE_DATASET_SUMMARY.csv")
    sel=sel_df.set_index("dataset"); fresh_counts={d:int(sel.loc[d,"n_fresh_eligible_pairs"]) for d in c.DATASETS}
    terminals=[]
    for d in c.DATASETS:
        # Actionability support requires a positive, nontrivial rank signal and
        # selected pair above random.  With fewer than three fresh pairs the
        # preregistered outcome is necessarily inconclusive.
        cr=corr_df[(corr_df.dataset==d)&(corr_df.diagnostic=="EC")].iloc[0]
        if fresh_counts[d] < 3: terminals.append("INCONCLUSIVE")
        elif pd.notna(cr.spearman_rho) and cr.spearman_rho >= .5 and float(sel.loc[d,"EC_selected_gain"]) > float(sel.loc[d,"random_pair_mean_gain"]): terminals.append("ACTIONABILITY_SUPPORTED")
        elif float(sel.loc[d,"oracle_minus_random"]) > .005: terminals.append("ACTIONABILITY_GAP_SUPPORTED")
        else: terminals.append("INCONCLUSIVE")
    terminal="ACTIONABILITY_SUPPORTED" if all(x=="ACTIONABILITY_SUPPORTED" for x in terminals) else "ACTIONABILITY_GAP_SUPPORTED" if any(x=="ACTIONABILITY_GAP_SUPPORTED" for x in terminals) and all(x in ("ACTIONABILITY_GAP_SUPPORTED","INCONCLUSIVE") for x in terminals) else "INCONCLUSIVE"
    table=["# FINAL_SEED0_DECISION", "", "## Backbone competence", "", "See `BACKBONE_COMPETENCE_SUMMARY.csv`; competence was frozen using inner-validation BA before outer reveal.", "", "## Outcome-blind actionability", "", "| Dataset | fresh eligible pairs | rho EC | rho Gdev | EC-selected gain | random gain | oracle gain |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in c.DATASETS:
        ec=corr_df[(corr_df.dataset==d)&(corr_df.diagnostic=="EC")].iloc[0]; gd=corr_df[(corr_df.dataset==d)&(corr_df.diagnostic=="G_dev")].iloc[0]; r=sel.loc[d]
        fmt=lambda x:"NA" if x is None or (isinstance(x,float) and not np.isfinite(x)) else f"{float(x):+.4f}"
        table.append(f"| {d} | {int(r.n_fresh_eligible_pairs)} | {fmt(ec.spearman_rho)} | {fmt(gd.spearman_rho)} | {fmt(r.EC_selected_gain)} | {fmt(r.random_pair_mean_gain)} | {fmt(r.oracle_gain)} |")
    table += ["", "## Multi-backbone", "", "See `MULTIBACKBONE_MATRIX.csv` for all 15 fixed pairs, including historical pairs.", "", "## Answers", "", "1. EC 是否能提前 rank actual fusion utility？见 rho EC；本轮只把 pair family 当作 ranking unit。", "2. Development fusion gain 是否更能预测未来 utility？见 rho Gdev。", "3. 是否存在真实 positive oracle opportunity？见 oracle gain 与 oracle_minus_random。", "4. EC-selected pair 是否优于 random？见表。", "5. EEGNet+LiteBN 是否仍是特殊 positive case？该 pair 保留为 HISTORICAL，未进入 fresh statistic。", "6. 现代 FM pair 是否表现出稳定 generality？由完整矩阵和 competence 结果判断，不能由单一正例外推。", "7. seed0 是否值得扩展 seed1/2？只根据本轮冻结规则给出，不自动扩展。", "", f"Final terminal: **{terminal}**", "", "Claim boundary: any oracle value is recoverable conditional headroom, not a deployable gate; fixed fusion does not solve subject reliability by itself.", "", "Next action (exactly one): " + ("design a no-regret conditional fusion mechanism using only deployment-available information" if terminal=="ACTIONABILITY_GAP_SUPPORTED" else "formalize a unified conservative-residual architecture" if terminal=="ACTIONABILITY_SUPPORTED" else "stop or treat this seed0 screen as inconclusive; do not claim actionability")]
    write_text(c.OUT/"FINAL_SEED0_DECISION.md","\n".join(table))
    c.write_json(c.PROTOCOL/"STAGE_B_REVEAL.json", {"outer_predictions_generated":True,"fresh_pair_outer_utility_generated":True,"stage_a_lock_commit":c.read_json(c.PROTOCOL/"STAGE_A_LOCK.json").get("stage_a_commit"),"holdout_isolation":{"V8_INTERNAL_HOLDOUT_loaded":False,"V8_INTERNAL_HOLDOUT_labels_loaded":False,"WBCIC_true_outer_loaded":False,"WBCIC_true_outer_labels_loaded":False},"terminal":terminal})
    print(terminal, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

