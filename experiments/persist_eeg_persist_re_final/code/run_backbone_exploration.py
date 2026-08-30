"""Exploratory same-method source screen for additional verified backbones.

This script is deliberately separate from the pre-registered CleanRoom run.
It applies the unchanged PERSIST-RE equations, fixed optimizer values, and the
same source gate to historical FBCNet/EEGInceptionMI feature archives.  It does
not open any future or sealed resource and cannot alter the CleanRoom result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import persist_re_core as c
from run_source import recipes, subject_metric, bootstrap_delta


def fit_cached(backbone: str, method: str, rep: dict[str, np.ndarray], dataset: str, fold: int, seed: int, recipe: dict[str, object], device: torch.device):
    key = f"explore_{c.FIT_VERSION}_{backbone}_{dataset}_{method}_r{recipe['rank']}_lr{recipe['lambda_R']}_lp{recipe['lambda_P']}_f{fold}_s{seed}_n{len(rep['indices'])}"
    path = c.RUNTIME / "exploration_fits" / f"{key}.pt"
    subjects, _ = c.subject_index(rep["subjects"])
    if path.is_file():
        payload = torch.load(path, map_location=device, weights_only=False)
        model = c.PERSISTRE(rep["features"].shape[1], len(subjects), int(recipe["rank"])).to(device)
        model.load_state_dict(payload["state_dict"]); model.eval()
        return model, payload.get("diagnostics", {})
    model, diagnostics = c.fit_model(method, rep, int(recipe["rank"]), float(recipe["lambda_R"]), float(recipe["lambda_P"]), c.stable_seed("explore", backbone, dataset, method, fold, seed, recipe["id"]), device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "diagnostics": diagnostics}, path)
    return model, diagnostics


def search(backbone: str, dataset: str, device: torch.device) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for recipe in recipes():
        vals = []
        for fold in c.FOLDS:
            fd = c.load_fold(dataset, fold, 0, backbone)
            model, _ = fit_cached(backbone, "PERSIST-RE", fd.model_fit, dataset, fold, 0, recipe, device)
            _, mapping = c.subject_index(fd.model_fit["subjects"])
            metric = pd.DataFrame(c.metric_rows(dataset, fold, 0, "PERSIST-RE", c.predict(model, fd.validation, mapping, device)))
            value = subject_metric(metric); vals.append(value)
            rows.append({"backbone": backbone, "dataset": dataset, "recipe": recipe["id"], "rank": recipe["rank"], "lambda_R": recipe["lambda_R"], "lambda_P": recipe["lambda_P"], "fold": fold, "seed": 0, "validation_BA": value, "validation_subjects": int(metric.subject_id.nunique()), "future_session_used": False})
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["backbone", "dataset", "recipe", "rank", "lambda_R", "lambda_P"], as_index=False).agg(mean_validation_BA=("validation_BA", "mean"), minimum_fold_BA=("validation_BA", "min"))
    selected_row = grouped.sort_values(["mean_validation_BA", "minimum_fold_BA", "recipe"], ascending=[False, False, True]).iloc[0]
    selected = next(r for r in recipes() if r["id"] == selected_row.recipe)
    return selected, rows


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    per_fold = frame.groupby(["backbone", "dataset", "method", "fold", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"), subjects=("subject_id", "nunique"))
    subject = frame.groupby(["backbone", "dataset", "method", "subject_id"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"))
    summaries=[]; deltas=[]
    for (backbone, dataset), ds in subject.groupby(["backbone", "dataset"]):
        for method in c.METHODS:
            vals=ds[ds.method==method].set_index("subject_id").BA
            if len(vals):
                mean,lo,hi=bootstrap_delta(vals.to_numpy(),c.stable_seed("explore-summary",backbone,dataset,method)); summaries.append({"backbone":backbone,"dataset":dataset,"method":method,"BA":mean,"CI95_L":lo,"CI95_U":hi,"subjects":len(vals)})
        erm=ds[ds.method=="ERM"].set_index("subject_id").BA
        for method in c.METHODS:
            if method=="ERM": continue
            other=ds[ds.method==method].set_index("subject_id").BA; common=erm.index.intersection(other.index)
            if len(common):
                delta,lo,hi=bootstrap_delta((other.loc[common]-erm.loc[common]).to_numpy(),c.stable_seed("explore-paired",backbone,dataset,method)); fd=per_fold[(per_fold.backbone==backbone)&(per_fold.dataset==dataset)&(per_fold.method==method)].set_index(["fold","seed"]).BA-per_fold[(per_fold.backbone==backbone)&(per_fold.dataset==dataset)&(per_fold.method=="ERM")].set_index(["fold","seed"]).BA; deltas.append({"backbone":backbone,"dataset":dataset,"comparison":f"{method}-ERM","delta_BA":delta,"CI95_L":lo,"CI95_U":hi,"positive_fold_seed_units":int((fd>0).sum()),"fold_seed_units":len(fd)})
    return per_fold, subject, {"method_summary":summaries,"deltas":deltas}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--backbones",nargs="+",choices=("FBCNet","EEGInceptionMI","ATCNet-Official","EEGNeX","EEGNet","EEGConformer"),default=["ATCNet-Official","EEGNeX"]); parser.add_argument("--datasets",nargs="+",choices=c.DATASETS,default=list(c.DATASETS)); parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto"); args=parser.parse_args()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)); selections={}; search_rows=[]; outcome_rows=[]; diag_rows=[]
    for backbone in args.backbones:
        for dataset in args.datasets:
            selected,rows=search(backbone,dataset,device); selections[f"{backbone}/{dataset}"]=selected; search_rows.extend(rows)
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    fd=c.load_fold(dataset,fold,seed,backbone); train=c.concat_rep(fd.model_fit,fd.validation); _,mapping=c.subject_index(train["subjects"])
                    for method in c.METHODS:
                        model,diag=fit_cached(backbone,method,train,dataset,fold,seed,selected,device); pred=c.predict(model,fd.outcome,mapping,device); outcome_rows.extend([{**r,"backbone":backbone} for r in c.metric_rows(dataset,fold,seed,method,pred)]); diag_rows.append({"backbone":backbone,"dataset":dataset,"fold":fold,"seed":seed,"method":method,"rank":selected["rank"],"lambda_R":selected["lambda_R"],"lambda_P":selected["lambda_P"],"center_e_norm":diag.get("center_e_norm",0.0),"center_a_norm":diag.get("center_a_norm",0.0),"center_e_mean_norm":diag.get("center_e_mean_norm",0.0),"center_a_mean_norm":diag.get("center_a_mean_norm",0.0),"random_effect_variance":diag.get("random_effect_variance",0.0),"random_effect_parameter_norm":diag.get("random_effect_parameter_norm",0.0),"future_session_used":False}); del model
                    if device.type=="cuda": torch.cuda.empty_cache()
                print(f"[explore-outcome] {backbone} {dataset} fold={fold} seed={seed} recipe={selected['id']}",flush=True)
    frame=pd.DataFrame(outcome_rows); per_fold,subject,statistics=summarize(frame); c.write_csv(c.RESULTS/"EXPLORATORY_PER_SUBJECT.csv",frame); c.write_csv(c.RESULTS/"EXPLORATORY_PER_FOLD.csv",per_fold); c.write_csv(c.RESULTS/"EXPLORATORY_METHOD_SUMMARY.csv",pd.DataFrame(statistics["method_summary"])); c.write_csv(c.RESULTS/"EXPLORATORY_ABLATION_SUMMARY.csv",pd.DataFrame(statistics["deltas"])); c.write_csv(c.RESULTS/"EXPLORATORY_RECIPE_SEARCH.csv",pd.DataFrame(search_rows)); c.write_csv(c.RESULTS/"EXPLORATORY_RANDOM_EFFECT_STATISTICS.csv",pd.DataFrame(diag_rows)); c.write_json(c.RESULTS/"EXPLORATORY_STATISTICS.json",statistics); c.write_json(c.RESULTS/"EXPLORATORY_SELECTION.json",selections)
    print(json.dumps(selections,indent=2,sort_keys=True),flush=True)


if __name__=="__main__": main()
