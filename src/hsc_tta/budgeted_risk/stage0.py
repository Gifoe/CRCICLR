from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mord
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from hsc_tta.contextual_risk.features import context_features
from hsc_tta.contextual_risk.families import TPSFamily
from hsc_tta.contextual_risk.io import atomic_parquet
from hsc_tta.contextual_risk.quantiles import split_conformal_upper
from hsc_tta.contextual_risk.statistics import paired_bootstrap_ci

from .access import BudgetedAccessController
from .inclusion_index import (
    critical_index_from_kappa, inclusion_index_table, inclusion_indices,
)
from .query_oracle import QueryOracle


FEATURE_SCHEMA = "budgeted-risk-full-context-features-v1"
MODEL_SPECS = (
    "constant", "direct", "isotonic",
    "ridge_0.01", "ridge_0.1", "ridge_1", "ridge_10", "ridge_100",
    "ordinal_0.01", "ordinal_0.1", "ordinal_1", "ordinal_10",
)
COMPLEXITY = {"constant": 0, "direct": 1, "isotonic": 2}


def _safe_rho(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(prediction)) < 2:
        return -1.0
    value = spearmanr(y, prediction).statistic
    return float(value) if np.isfinite(value) else -1.0


def _model_complexity(name: str) -> int:
    return COMPLEXITY.get(name, 3 if name.startswith("ridge") else 4)


def _fit(name: str, x: np.ndarray, y: np.ndarray, j_context: np.ndarray) -> dict[str, Any]:
    if name == "constant":
        return {"name": name, "value": float(np.median(y))}
    if name == "direct":
        return {"name": name}
    if name == "isotonic":
        model = IsotonicRegression(y_min=0, y_max=20, out_of_bounds="clip").fit(j_context, y)
        return {"name": name, "model": model}
    scaler = RobustScaler().fit(x)
    transformed = scaler.transform(x)
    strength = float(name.rsplit("_", 1)[1])
    if name.startswith("ridge"):
        model = Ridge(alpha=strength).fit(transformed, y)
    else:
        classes=np.unique(y.astype(int)); encoded=np.searchsorted(classes,y.astype(int))
        model = mord.LogisticAT(alpha=strength, max_iter=2000).fit(transformed,encoded)
        return {"name": name, "model": model, "scaler": scaler, "ordinal_classes": classes}
    return {"name": name, "model": model, "scaler": scaler}


def _predict(fitted: dict[str, Any], x: np.ndarray, j_context: np.ndarray) -> np.ndarray:
    name = fitted["name"]
    if name == "constant":
        result = np.full(len(x), fitted["value"], dtype=float)
    elif name == "direct":
        result = np.asarray(j_context, dtype=float)
    elif name == "isotonic":
        result = fitted["model"].predict(j_context)
    elif name.startswith("ordinal"):
        model = fitted["model"]
        probability = model.predict_proba(fitted["scaler"].transform(x))
        result = probability @ fitted["ordinal_classes"].astype(float)
    else:
        result = fitted["model"].predict(fitted["scaler"].transform(x))
    return np.clip(np.asarray(result, dtype=float), 0, 20)


def _select_model(frame: pd.DataFrame, feature_columns: list[str], specs: tuple[str,...]=MODEL_SPECS) -> tuple[str, pd.DataFrame]:
    rows = []
    for name in specs:
        true, prediction = [], []
        for held_out in sorted(frame.screening_fold.unique()):
            train = frame[frame.screening_fold != held_out]
            valid = frame[frame.screening_fold == held_out]
            if train.empty or valid.empty:
                continue
            fitted = _fit(name, train[feature_columns].to_numpy(), train.j_future.to_numpy(), train.j_context.to_numpy())
            prediction.extend(_predict(fitted, valid[feature_columns].to_numpy(), valid.j_context.to_numpy()))
            true.extend(valid.j_future.to_numpy())
        y = np.asarray(true, float); p = np.asarray(prediction, float)
        rows.append({
            "candidate": name, "mae": float(np.mean(np.abs(y-p))),
            "spearman": _safe_rho(y, p), "underestimation_rate": float(np.mean(p < y)),
            "complexity": _model_complexity(name),
        })
    scores = pd.DataFrame(rows)
    eligible = scores[scores.mae <= scores.mae.min() + .05].copy()
    selected = eligible.sort_values(
        ["spearman", "underestimation_rate", "complexity", "candidate"],
        ascending=[False, True, True, True],
    ).iloc[0].candidate
    return str(selected), scores


@dataclass
class SubjectRecord:
    row: dict[str, Any]
    context_probabilities: np.ndarray
    future_probabilities: np.ndarray
    future_labels: np.ndarray
    controller: BudgetedAccessController
    query_hash: str
    source_model_hash: str
    episode_hash: str


def _prepare_subject(
    cache_path: Path, dataset: str, subject_id: str, seed: int, fold: int,
    role: str, screening_fold: int, alpha: float,
) -> tuple[SubjectRecord, list[dict[str, Any]], dict[str, float]]:
    with np.load(cache_path, allow_pickle=False) as z:
        ci = z["context_sample_indices"].astype(int)
        cp = z["context_probabilities"].astype(float)
        ce = z["context_embeddings"].astype(float)
        guarded_context_labels = z["context_labels_guarded"].astype(int)
        fp = z["future_probabilities"].astype(float)
        guarded_future_labels = z["future_labels_guarded"].astype(int)
        source_hash = str(z["source_model_hash"]); episode_hash = str(z["episode_hash"])
    oracle = QueryOracle(dataset, subject_id, seed, ci, guarded_context_labels, budget=len(ci), strategy="full_context")
    controller = BudgetedAccessController(dataset, subject_id, seed, role)
    controller.begin_queries(); table = inclusion_index_table(cp); observed = []
    for position, sample_index in enumerate(ci):
        label = oracle.query(int(sample_index), predicted_class=int(cp[position].argmax()), kappa_by_label=table[position])
        observed.append(int(table[position, label]))
    query_hash = oracle.freeze(); controller.freeze_queries(query_hash)
    kappa = np.asarray(observed, int); j_context = critical_index_from_kappa(kappa, alpha)
    unlabeled = context_features(cp)
    prefixes = [critical_index_from_kappa(kappa[:max(1, int(np.ceil(len(kappa)*fraction)))], alpha) for fraction in (.25, .5, .75, 1.)]
    labeled = {
        "j_context": float(j_context), "kappa_q50": float(np.quantile(kappa, .5)),
        "kappa_q80": float(np.quantile(kappa, .8)), "kappa_q90": float(np.quantile(kappa, .9)),
        "kappa_mean": float(kappa.mean()), "kappa_std": float(kappa.std()),
        "kappa_tail_mass": float(np.mean(kappa >= 15)),
        "prefix_instability": float(np.mean(np.abs(np.asarray(prefixes)-prefixes[-1]))),
    }
    row = {
        "dataset": dataset, "subject_id": subject_id, "seed": seed, "outer_fold": fold,
        "screening_fold": screening_fold, "rotation_role": role, **unlabeled, **labeled,
        "context_embedding_norm_mean": float(np.linalg.norm(ce, axis=1).mean()),
    }
    feature_row = {
        "schema_version": FEATURE_SCHEMA, "dataset": dataset, "subject_id": subject_id,
        "seed": seed, "outer_fold": fold, "rotation_role": role, **unlabeled,
        "context_embedding_norm_mean": row["context_embedding_norm_mean"],
    }
    record = SubjectRecord(row, cp, fp, guarded_future_labels, controller, query_hash, source_hash, episode_hash)
    return record, list(oracle.transcript), feature_row


def _open_future(record: SubjectRecord, predicted_index: float, repo: Path, alpha: float, delta: float) -> int:
    row = record.row
    path = repo/"outputs/budgeted_risk/risk_decisions/stage0_full_context"/row["dataset"]/f"fold_{row['outer_fold']}"/f"seed_{row['seed']}"/f"{row['subject_id'].replace(':','_')}.json"
    decision = {
        "dataset": row["dataset"], "subject_id": row["subject_id"], "seed": row["seed"],
        "role": row["rotation_role"], "budget": int(round(row["context_sample_count"])),
        "strategy": "full_context", "alpha": alpha, "delta": delta,
        "query_hash": record.query_hash, "source_model_hash": record.source_model_hash,
        "episode_hash": record.episode_hash, "certified_index": int(np.clip(np.ceil(predicted_index), 0, 20)),
    }
    record.controller.freeze_decision(decision, path)
    labels = record.controller.open_future(record.future_labels, path)
    return critical_index_from_kappa(inclusion_indices(record.future_probabilities, labels), alpha)


def run_full_context(project_root: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(project_root); repo = root/"repo"
    cohorts = pd.read_parquet(repo/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet")
    dev = cohorts[cohorts.master_cohort == "method_development"].copy()
    cache_manifest = pd.read_parquet(repo/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet")
    alpha = float(config["alpha"]); delta = float(config["delta"])
    results: list[dict[str, Any]] = []; transcripts = []; unlabeled_features = []; selection_rows = []
    for dataset in ("hmc", "eegmmidb"):
        feature_columns: list[str] | None = None
        fold_map = dev[dev.dataset == dataset].set_index("subject_id").screening_fold.astype(int).to_dict()
        subjects = sorted(fold_map)
        for fold in range(5):
            for seed in range(5):
                records: dict[str, SubjectRecord] = {}
                for subject in subjects:
                    role = "evaluation" if fold_map[subject] == fold else "calibration" if fold_map[subject] == (fold+1)%5 else "meta"
                    path = Path(cache_manifest[(cache_manifest.dataset==dataset)&(cache_manifest.fold==fold)&(cache_manifest.seed==seed)&(cache_manifest.subject_id==subject)].iloc[0].cache_path)
                    record, current_transcript, feature_row = _prepare_subject(path, dataset, subject, seed, fold, role, fold_map[subject], alpha)
                    records[subject] = record; transcripts.extend(current_transcript); unlabeled_features.append(feature_row)
                # Meta/calibration Future is opened only after a full-context query and frozen direct decision.
                for subject, record in records.items():
                    if record.row["rotation_role"] != "evaluation":
                        record.row["j_future"] = _open_future(record, record.row["j_context"], repo, alpha, delta)
                meta = pd.DataFrame([r.row for r in records.values() if r.row["rotation_role"] == "meta"])
                cal = pd.DataFrame([r.row for r in records.values() if r.row["rotation_role"] == "calibration"])
                identifiers = {"dataset","subject_id","seed","outer_fold","screening_fold","rotation_role","j_future"}
                if feature_columns is None:
                    feature_columns = sorted(column for column in meta.columns if column not in identifiers)
                selected, scores = _select_model(meta, feature_columns)
                scores.insert(0,"dataset",dataset);scores.insert(1,"outer_fold",fold);scores.insert(2,"seed",seed);scores["selected"]=scores.candidate==selected
                selection_rows.extend(scores.to_dict("records"))
                fitted = _fit(selected, meta[feature_columns].to_numpy(), meta.j_future.to_numpy(), meta.j_context.to_numpy())
                cal_prediction = _predict(fitted, cal[feature_columns].to_numpy(), cal.j_context.to_numpy())
                correction = split_conformal_upper(np.maximum(cal.j_future.to_numpy()-cal_prediction,0), delta, insufficient=20)
                constant = float(np.median(meta.j_future))
                for subject, record in records.items():
                    if record.row["rotation_role"] != "evaluation": continue
                    x = pd.DataFrame([record.row])[feature_columns].to_numpy(); raw = float(_predict(fitted,x,np.asarray([record.row["j_context"]]))[0])
                    cert = int(np.clip(np.ceil(raw+correction),0,20)); true = _open_future(record, cert, repo, alpha, delta)
                    _, future_sizes, repairs = TPSFamily().future_curve(record.future_probabilities, record.future_labels)
                    results.append({
                        **record.row, "j_future": true, "selected_model": selected,
                        "raw_prediction": raw, "certified_index": cert, "conformal_correction": correction,
                        "constant_prediction": constant, "absolute_error": abs(raw-true),
                        "constant_absolute_error": abs(constant-true), "underestimated": raw < true,
                        "violation": bool(cert < true), "set_size": float(future_sizes[cert]),
                        "constant_set_size": float(future_sizes[int(np.clip(np.ceil(constant),0,20))]),
                        "oracle_set_size": float(future_sizes[true]), "sentinel": cert == 20,
                        "monotonicity_repairs": repairs, "source_model_hash": record.source_model_hash,
                        "episode_hash": record.episode_hash,
                    })
    result_frame = pd.DataFrame(results); transcript_frame = pd.DataFrame(transcripts); selection_frame = pd.DataFrame(selection_rows)
    output = repo/"outputs/budgeted_risk/stage0"; output.mkdir(parents=True,exist_ok=True)
    atomic_parquet(result_frame, output/"FULL_CONTEXT_RESULTS.parquet")
    atomic_parquet(transcript_frame, output/"QUERY_TRANSCRIPTS.parquet")
    atomic_parquet(selection_frame, output/"FULL_CONTEXT_MODEL_SELECTION.parquet")
    atomic_parquet(pd.DataFrame(unlabeled_features), repo/"outputs/budgeted_risk/features/UNLABELED_CONTEXT_FEATURES.parquet")
    return result_frame, transcript_frame, selection_frame


def full_context_gate(results: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    rows = []
    gate = config["full_context_gate"]
    for dataset, current in results.groupby("dataset"):
        aggregated = current.groupby("subject_id",as_index=False).agg(
            j_future=("j_future","mean"), raw_prediction=("raw_prediction","mean"),
            constant_prediction=("constant_prediction","mean"),
        )
        y=aggregated.j_future.to_numpy();p=aggregated.raw_prediction.to_numpy();c=aggregated.constant_prediction.to_numpy()
        mae=float(np.mean(abs(y-p)));constant_mae=float(np.mean(abs(y-c)))
        improvement=float((constant_mae-mae)/constant_mae) if constant_mae else 0.0
        differences=abs(y-c)-abs(y-p);ci_low,ci_high=paired_bootstrap_ci(differences,reps=int(config["bootstrap_repetitions"]),seed=int(config["bootstrap_seed"]))
        slope=float(np.polyfit(y,p,1)[0]) if len(np.unique(y))>1 else 0.0
        rho=_safe_rho(y,p)
        passed=bool(rho>=float(gate["minimum_spearman"]) and improvement>=float(gate["minimum_mae_improvement"]) and slope>0 and ci_low>0)
        rows.append({
            "dataset":dataset,"n_subjects":len(aggregated),"spearman":rho,"mae":mae,
            "constant_mae":constant_mae,"relative_mae_improvement":improvement,
            "predicted_vs_true_slope":slope,"underestimation_rate":float(np.mean(p<y)),
            "paired_improvement_mean":float(differences.mean()),"paired_ci_low":ci_low,
            "paired_ci_high":ci_high,"full_context_pass":passed,
        })
    summary=pd.DataFrame(rows)
    return summary, bool(len(summary)==2 and summary.full_context_pass.all())
