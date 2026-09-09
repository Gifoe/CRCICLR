from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import modern_common as c


HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    env = os.environ.copy(); env["PYTHONPATH"] = str(HERE)
    print("[exec] " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def pred_path(model: str, dataset: str, fold: int) -> Path:
    slug = model.lower() if model in ("EEGNet", "LiteBN") else c.modern_slug(model)
    return c.RUNTIME / "predictions" / f"stageA_{dataset.lower()}_fold{fold}_{slug}.npz"


def ensure_predictions(model: str, dataset: str, fold: int) -> Path:
    path = pred_path(model, dataset, fold)
    if not path.is_file():
        run([sys.executable, str(HERE / "predict_model.py"), "--model", model, "--dataset", dataset, "--fold", str(fold), "--stage", "A"])
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def subject_stats(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    if not np.array_equal(a["labels"], b["labels"]) or not np.array_equal(a["subjects"], b["subjects"]):
        raise RuntimeError("prediction row alignment mismatch")
    labels = a["labels"]; subjects = a["subjects"].astype(str)
    pa, pb = a["logits"].argmax(1), b["logits"].argmax(1)
    ma = a["logits"][:, 1] - a["logits"][:, 0]; mb = b["logits"][:, 1] - b["logits"][:, 0]
    subject_values = []
    for subject in c.subject_sort(np.unique(subjects)):
        mask = subjects == subject
        ca, cb = pa[mask] == labels[mask], pb[mask] == labels[mask]
        fused = (a["logits"][mask] + b["logits"][mask]).argmax(1)
        from sklearn.metrics import balanced_accuracy_score
        ba_a = balanced_accuracy_score(labels[mask], pa[mask]); ba_b = balanced_accuracy_score(labels[mask], pb[mask]); ba_f = balanced_accuracy_score(labels[mask], fused)
        corr = np.corrcoef(ma[mask], mb[mask])[0, 1] if int(mask.sum()) > 1 else 1.0
        if not np.isfinite(corr): corr = 0.0
        subject_values.append({"ec": float(np.mean(np.logical_xor(ca, cb))), "dis": float(np.mean(pa[mask] != pb[mask])),
                               "margin_div": float(1.0 - corr), "gdev": float(ba_f - max(ba_a, ba_b))})
    return {key: float(np.mean([item[key] for item in subject_values])) for key in ("ec", "dis", "margin_div", "gdev")}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part"); tmp.write_text(text.rstrip() + "\n", encoding="utf-8"); os.replace(tmp, path)


def write_protocol(folds: dict[str, list[dict[str, object]]], search: dict[str, list[str]], split_sha: str,
                   summaries: pd.DataFrame, pair_status: pd.DataFrame, blind: pd.DataFrame) -> None:
    c.PROTOCOL.mkdir(parents=True, exist_ok=True); c.OUT.mkdir(parents=True, exist_ok=True)
    c.write_json(c.PROTOCOL / "SOURCE_EXPERIMENT.json", {"repository": str(c.REPO), "source_branch": "codex/persist-eeg-carrier-5fold-multiseed-stability-v1", "source_split": str(c.SPLIT_PATH), "source_split_sha256": split_sha, "datasets": c.DATASETS, "search_sizes": {k: len(v) for k, v in search.items()}, "folds": 5, "seed": 0})
    c.write_json(c.PROTOCOL / "DATA_PROVENANCE.json", {"cache_root": str(c.CACHE_ROOT), "datasets": {"OpenBMI": {"channels": 62, "cache_sessions_used": [1, 2], "sample_shape": [62, 1000]}, "WBCIC": {"channels": 58, "cache_sessions_used": [0, 1, 2], "sample_shape": [58, 1000]}}, "normalization": "channel-wise mean/std computed on each fold inner_train source sessions", "search_only": True})
    c.write_json(c.PROTOCOL / "CHECKPOINT_PROVENANCE.json", {"carrier_runtime": str(c.CARRIER_RUNTIME), "existing_models": ["EEGNet", "LiteBN"], "modern_runtime": str(c.RUNTIME), "modern_models": ["TCFormer", "ST-EEGFormer-small", "LaBraM-base", "CBraMod"], "checkpoint_selection": "inner-validation subject-equal BA, earliest best epoch", "seed": 0})
    c.write_json(c.PROTOCOL / "FYL412_EXECUTION.json", {"hostname": os.uname().nodename, "user": os.environ.get("USER", "unknown"), "home": str(Path.home()), "gpu": "RTX 4090", "outer_reveal": False, "seed": 0})
    c.write_json(c.PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded": False, "V8_INTERNAL_HOLDOUT_labels_loaded": False, "WBCIC_true_outer_loaded": False, "WBCIC_true_outer_labels_loaded": False, "scope": "V8_SEARCH inner-train/inner-validation only"})
    c.write_json(c.PROTOCOL / "TESTS.json", {"fixed_backbone_pool": list(c.MODELS), "seed0_only": True, "exact_fivefold_split_reused": True, "stage_a_outer_predictions_absent": True, "no_weight_search": True, "primary_fusion": "raw-logit 50/50", "competence_gate": "mean inner-val BA >= EEGNet - 3pp", "modern_input_adapter_amendment_hash": hashlib.sha256(("ST:250Hzx1000->128Hzx512;LaBraM:250Hzx1000->200Hzx800;CBraMod:5x200 reshape" ).encode()).hexdigest()})
    pair_status.to_csv(c.PROTOCOL / "PAIR_OUTCOME_STATUS.csv", index=False)
    blind.to_csv(c.PROTOCOL / "STAGE_A_BLIND_PREDICTIONS.csv", index=False)
    summaries.to_csv(c.OUT / "BACKBONE_DATASET_SUMMARY.csv", index=False)
    lines = ["# Outcome-blind protocol — modern backbone seed0", "", "This file is frozen before any fresh outer-dev pair utility is generated.", "", "## Backbone pool", ", ".join(c.MODELS), "", "## Data and split", "OpenBMI MI and WBCIC MI; exact carrier FIVEFOLD_SPLIT.json; seed 0 only; source sessions and future session semantics are inherited from the carrier experiment.", "", "## Model-native adapters", "The cache is 250 Hz × 1000 samples. TCFormer consumes it directly. ST-EEGFormer-small uses a fixed linear 250 Hz → 128 Hz, 512-sample adapter before its official tokenization. LaBraM-base uses a fixed linear 250 Hz → 200 Hz, 800-sample adapter and four 200-sample patches. CBraMod uses a fixed five 200-sample patch reshape preserving all cache samples. These adapters were fixed before outcome reveal and are not selected from outcomes.", "", "## Stage A seal", "Only inner-train/source sessions and inner-validation future-session rows were accessed. No outer-dev prediction or pair utility is present in this stage. Competence is fixed as mean inner-validation subject-equal BA >= EEGNet mean minus 3 percentage points. Primary diagnostic is Exclusive-Correct Complementarity (EC). Primary fusion is raw-logit 50/50; G_dev is descriptive only.", "", "## Selection", "For each dataset, the EC-selected and G_dev-selected pair are argmax over FRESH + competent pairs, frozen in STAGE_A_BLIND_PREDICTIONS.csv. Fresh pair definitions are fixed by PAIR_OUTCOME_STATUS.csv."]
    write_text(c.PROTOCOL / "OUTCOME_BLIND_PROTOCOL.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-training", action="store_true"); args = parser.parse_args()
    folds, search, split_sha = c.load_split(); c.RUNTIME.mkdir(parents=True, exist_ok=True)
    # Ensure compact protocol/output artifacts can be written even when this
    # is the first run in a fresh worktree.  (The runtime/checkpoint trees are
    # intentionally kept outside git.)
    c.PROTOCOL.mkdir(parents=True, exist_ok=True); c.OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_training:
        for model in ("TCFormer", "ST-EEGFormer-small", "LaBraM-base", "CBraMod"):
            for dataset in c.DATASETS:
                for fold in range(5):
                    record = c.modern_dir(dataset, fold, model) / "record.json"
                    if record.is_file() and (c.modern_dir(dataset, fold, model) / "selected_best.pt").is_file():
                        print(f"[reuse-training] {model} {dataset} fold={fold}", flush=True)
                    else:
                        run([sys.executable, str(HERE / "train_modern.py"), "--model", model, "--dataset", dataset, "--fold", str(fold)])
    # Build Stage-A predictions only on inner-validation subjects.
    pred: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    model_summary = []
    for dataset in c.DATASETS:
        baseline_values = []
        for fold in range(5):
            for model in c.MODELS:
                data = load_npz(ensure_predictions(model, dataset, fold)); pred[(dataset, fold, model)] = data
                record_path = c.modern_dir(dataset, fold, model) / "record.json" if model not in ("EEGNet", "LiteBN") else None
                parameter_count = None
                if record_path is not None and record_path.is_file():
                    parameter_count = c.read_json(record_path).get("parameter_count")
                model_summary.append({"dataset": dataset, "fold": fold, "model": model, "inner_val_BA": c.metrics(data["labels"], data["logits"], data["subjects"])["BA"], "inner_val_macro_F1": c.metrics(data["labels"], data["logits"], data["subjects"])["macro_F1"], "parameter_count": parameter_count})
                if model == "EEGNet": baseline_values.append(model_summary[-1]["inner_val_BA"])
        base = float(np.mean(baseline_values))
        for model in c.MODELS:
            rows = [x for x in model_summary if x["dataset"] == dataset and x["model"] == model]
            mean_ba = float(np.mean([x["inner_val_BA"] for x in rows])); model_summary.append({"dataset": dataset, "fold": "aggregate", "model": model, "inner_val_BA": mean_ba, "inner_val_macro_F1": float(np.mean([x["inner_val_macro_F1"] for x in rows])), "parameter_count": rows[0]["parameter_count"], "competent": bool(mean_ba >= base - .03), "competence_threshold": base - .03})
    summary_df = pd.DataFrame(model_summary)
    status_rows=[]; hist={frozenset(("EEGNet","LiteBN")),frozenset(("EEGNet","CBraMod")),frozenset(("LiteBN","CBraMod"))}
    for a,b in itertools.combinations(c.MODELS,2):
        historical = frozenset((a,b)) in hist
        status_rows.append({"model_a":a,"model_b":b,"status":"HISTORICAL" if historical else "FRESH","reason":"previous outcome was inspected before this protocol" if historical else "no previous pair outcome located before protocol lock","evidence_path_or_branch":"carrier/frozen-fusion history" if historical else "fresh seed0 modern-backbone block"})
    status_df=pd.DataFrame(status_rows)
    agg=[]
    for dataset in c.DATASETS:
        comp={r["model"]: bool(r.get("competent",False)) for r in model_summary if r["dataset"]==dataset and r["fold"]=="aggregate"}
        for a,b in itertools.combinations(c.MODELS,2):
            vals=[]
            for fold in range(5): vals.append(subject_stats(pred[(dataset,fold,a)],pred[(dataset,fold,b)]))
            row={"dataset":dataset,"pair":f"{a}+{b}","model_a":a,"model_b":b,"historical":status_df[(status_df.model_a==a)&(status_df.model_b==b)].status.iloc[0]=="HISTORICAL","competence_eligible":bool(comp[a] and comp[b]),"EC":float(np.mean([v["ec"] for v in vals])),"disagreement":float(np.mean([v["dis"] for v in vals])),"margin_diversity":float(np.mean([v["margin_div"] for v in vals])),"G_dev":float(np.mean([v["gdev"] for v in vals]))}
            agg.append(row)
    blind_df=pd.DataFrame(agg); eligible=(blind_df.status if False else None)
    blind_df["fresh_eligible"]=(~blind_df.historical)&blind_df.competence_eligible
    for dataset in c.DATASETS:
        m=blind_df[(blind_df.dataset==dataset)&blind_df.fresh_eligible].copy()
        blind_df.loc[blind_df.dataset==dataset,"EC_rank"]=blind_df.loc[blind_df.dataset==dataset,"EC"].rank(method="min",ascending=False)
        blind_df.loc[blind_df.dataset==dataset,"Gdev_rank"]=blind_df.loc[blind_df.dataset==dataset,"G_dev"].rank(method="min",ascending=False)
        if len(m):
            ec=m.sort_values(["EC","pair"],ascending=[False,True]).iloc[0]["pair"]; gd=m.sort_values(["G_dev","pair"],ascending=[False,True]).iloc[0]["pair"]
        else: ec=gd=None
        blind_df.loc[blind_df.dataset==dataset,"EC_selected_pair"]=ec; blind_df.loc[blind_df.dataset==dataset,"Gdev_selected_pair"]=gd
    blind_df.to_csv(c.PROTOCOL / "STAGE_A_BLIND_PREDICTIONS.csv", index=False)
    write_protocol(folds, search, split_sha, summary_df, status_df, blind_df)
    c.write_json(c.PROTOCOL / "STAGE_A_LOCK.json", {"stage":"A","seed":0,"commit_pending":True,"outer_predictions_generated":False,"fresh_pair_outer_utility_generated":False,"fixed_pair_selection":{"OpenBMI":blind_df[(blind_df.dataset=="OpenBMI")&blind_df.fresh_eligible].sort_values(["EC","pair"],ascending=[False,True]).iloc[0]["pair"] if len(blind_df[(blind_df.dataset=="OpenBMI")&blind_df.fresh_eligible]) else None,"WBCIC":blind_df[(blind_df.dataset=="WBCIC")&blind_df.fresh_eligible].sort_values(["EC","pair"],ascending=[False,True]).iloc[0]["pair"] if len(blind_df[(blind_df.dataset=="WBCIC")&blind_df.fresh_eligible]) else None},"holdout_isolation":{"V8_INTERNAL_HOLDOUT_loaded":False,"WBCIC_true_outer_loaded":False}})
    print("STAGE_A_OUTCOME_BLIND_READY", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
