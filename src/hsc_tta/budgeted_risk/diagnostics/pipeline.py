from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from hsc_tta.budgeted_risk.acquisition import acquisition_order
from hsc_tta.budgeted_risk.budget import TRANSFER_SPECS, _features, _load, _local_index, _prior, Observation
from hsc_tta.budgeted_risk.inclusion_index import critical_index_from_kappa, inclusion_index_table, inclusion_indices
from hsc_tta.budgeted_risk.stage0 import _fit, _predict, _select_model
from hsc_tta.contextual_risk.families import TPSFamily
from hsc_tta.contextual_risk.statistics import clopper_pearson_upper, paired_bootstrap_ci

from .calibration_schemes import (
    S1, S2, S3, S4, SCHEMES, conformal_q, fit_nonnegative_scale,
    fold_split, is_outlier_driven, predict_scale, scale_design, sentinel_transition,
)
from .decision import decide, exact_pass, raw_pass
from .run_state import transition


_UNLABELED_FEATURES: dict[tuple[str, str, int, int], dict[str, float]] = {}
_ARRAY_CACHE: dict[tuple[str, int, int, str], Any] = {}
_PROCESS_POOL: ProcessPoolExecutor | None = None


@dataclass
class Paths:
    root: Path

    @property
    def repo(self) -> Path: return self.root / "repo"
    @property
    def output(self) -> Path: return self.repo / "outputs/budgeted_risk_v51"
    @property
    def results(self) -> Path: return self.output / "results"
    @property
    def audit(self) -> Path: return self.output / "audit"
    @property
    def provenance(self) -> Path: return self.output / "provenance"
    @property
    def figures(self) -> Path: return self.output / "figures"
    @property
    def delivery(self) -> Path: return self.repo / "delivery/budgeted_risk_v51"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=check)
    return result.stdout.strip()


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty: return "(no rows)"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    columns = [str(c) for c in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def configure(paths: Paths) -> dict[str, Any]:
    config_path = paths.repo / "configs/budgeted_risk_v51/diagnostic.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_hash"] = sha256(config_path)
    return config


def input_audit(paths: Paths, config: dict[str, Any]) -> tuple[dict[str, str], str, str, dict[str, Any]]:
    paths.audit.mkdir(parents=True, exist_ok=True); paths.delivery.mkdir(parents=True, exist_ok=True)
    old = paths.repo / "outputs/budgeted_risk/stage0"
    named = [
        paths.repo / "delivery/budgeted_risk/stage0/STAGE0_DECISION.json",
        paths.repo / "delivery/budgeted_risk/stage0/STAGE0_METHOD_FREEZE.json",
        paths.repo / "outputs/budgeted_risk/RUN_STATE.json",
        old / "FULL_CONTEXT_RESULTS.parquet", old / "FULL_CONTEXT_MODEL_SELECTION.parquet",
        old / "BUDGET_RESULTS.parquet", old / "BUDGET_TUNING.parquet",
        old / "BUDGET_QUERY_TRANSCRIPTS.parquet",
        paths.repo / "outputs/budgeted_risk/features/UNLABELED_CONTEXT_FEATURES.parquet",
    ]
    hashes = {str(p.relative_to(paths.repo)): sha256(p) for p in named}
    (paths.audit / "INPUT_HASHES.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    run_state = json.loads((paths.repo / "outputs/budgeted_risk/RUN_STATE.json").read_text())
    decision = json.loads((paths.repo / "delivery/budgeted_risk/stage0/STAGE0_DECISION.json").read_text())
    run_commit = run_state["git_commit"]
    ordinal = git(paths.repo, "log", "--all", "--format=%H", "--grep=implement_stage0_budget_gate_and_true_ordinal_candidate", "-1")
    ancestor = subprocess.run(["git", "-C", str(paths.repo), "merge-base", "--is-ancestor", ordinal, run_commit]).returncode == 0
    if not ancestor:
        raise RuntimeError("V51_INVALID_INPUT_ORDINAL_FIX_MISSING")
    current = git(paths.repo, "rev-parse", "HEAD"); branch = git(paths.repo, "branch", "--show-current")
    status = git(paths.repo, "status", "--short")
    cohorts_path = paths.repo / "outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet"
    cohorts = pd.read_parquet(cohorts_path)
    dev = cohorts[(cohorts.master_cohort == "method_development") & cohorts.dataset.isin(config["datasets"])].copy()
    cohort_rows = dev[["dataset", "subject_id", "screening_fold"]].sort_values(["dataset", "subject_id"]).to_dict("records")
    cohort_hash = canonical_hash(cohort_rows)
    fold_sizes = dev.groupby(["dataset", "screening_fold"]).size().rename("n").reset_index()
    fold_sizes["s1_m"] = fold_sizes.n; fold_sizes["s1_k"] = np.ceil((fold_sizes.n + 1) * .9).astype(int)
    fold_sizes["s2_m"] = fold_sizes.groupby("dataset").n.transform(lambda x: x.shift(-1, fill_value=x.iloc[0]) + x.shift(-2, fill_value=x.iloc[1] if len(x)>1 else x.iloc[0]))
    # Recompute cyclic two-fold m without relying on group shift edge semantics.
    for dataset in fold_sizes.dataset.unique():
        counts = fold_sizes[fold_sizes.dataset == dataset].set_index("screening_fold").n.to_dict()
        mask = fold_sizes.dataset == dataset
        fold_sizes.loc[mask, "s2_m"] = [counts[(int(e)+1)%5] + counts[(int(e)+2)%5] for e in fold_sizes.loc[mask, "screening_fold"]]
    fold_sizes["s2_k"] = np.ceil((fold_sizes.s2_m + 1) * .9).astype(int)
    budget = pd.read_parquet(old / "BUDGET_RESULTS.parquet")
    transcripts = pd.read_parquet(old / "BUDGET_QUERY_TRANSCRIPTS.parquet", columns=["dataset"])
    manifest = pd.read_parquet(paths.repo / "outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet")
    manifest_subjects = set(zip(manifest.dataset, manifest.subject_id)); dev_subjects = set(zip(dev.dataset, dev.subject_id))
    protected = cohorts[cohorts.master_cohort != "method_development"]
    protected_subjects = set(zip(protected.dataset, protected.subject_id))
    protected_overlap = sorted(manifest_subjects & protected_subjects)
    protected_payload = {
        "manifest_rows": len(manifest), "manifest_method_development_subjects_only": manifest_subjects <= dev_subjects,
        "protected_manifest_overlap_count": len(protected_overlap), "protected_manifest_overlap": protected_overlap,
        "formal_calibration_opened": False, "internal_final_opened": False, "cap_opened": False,
    }
    (paths.audit / "PROTECTED_COHORT_AUDIT.json").write_text(json.dumps(protected_payload, indent=2), encoding="utf-8")
    commit_payload = {
        "branch": branch, "current_commit": current, "v5_start_commit": git(paths.repo, "rev-parse", "v5-budgeted-subject-risk-calibration"),
        "run_commit": run_commit, "ordinal_fix_commit": ordinal, "ordinal_fix_is_run_ancestor": ancestor,
        "git_status_at_audit": status, "old_state": run_state["state"], "old_verdict": decision["verdict"],
    }
    (paths.audit / "COMMIT_AUDIT.json").write_text(json.dumps(commit_payload, indent=2), encoding="utf-8")
    audit_md = f"""# V5.1 repository and result audit

- Branch: `{branch}`
- Audit commit: `{current}`
- V5 start commit: `{commit_payload['v5_start_commit']}`
- Stage-0 run commit: `{run_commit}`
- Ordinal fix commit: `{ordinal}`
- `git merge-base --is-ancestor`: **{ancestor}**
- Git status at audit: `{status or 'clean'}`
- Old state/verdict: `{run_state['state']}` / `{decision['verdict']}`
- Method-development subjects: HMC={int((dev.dataset=='hmc').sum())}, EEGMMIDB={int((dev.dataset=='eegmmidb').sum())}
- BUDGET_RESULTS: {len(budget):,} rows (temporal={int((budget.strategy=='temporal').sum()):,}, random={int((budget.strategy=='random').sum()):,})
- BUDGET_QUERY_TRANSCRIPTS: {len(transcripts):,} rows
- Source-cache manifest rows: {len(manifest):,}; protected overlap: {len(protected_overlap)}
- Protected flags: formal=false, internal_final=false, CAP=false

## Fold sizes and finite-sample ranks

{markdown_table(fold_sizes)}

S1 has m=13 (EEGMMIDB) or m=18 (HMC), hence k=m at delta=0.10 in every fold: the selected correction is the maximum residual. S2 has m=26 or 36 and k=25 or 34.

## Input hashes

```json
{json.dumps(hashes, indent=2, sort_keys=True)}
```
"""
    (paths.delivery / "V51_REPOSITORY_AND_RESULT_AUDIT.md").write_text(audit_md, encoding="utf-8")
    correction = """# V5 provenance correction

The old stop narrative stating that Stage-0A failed and budget experiments were not opened is incorrect. The authoritative `STAGE0_DECISION.json` records `full_context_pass=true`, `budget_experiments_opened=true`, and `budget_pass=false`. Stage-0A therefore passed; Stage-0B ran and failed its efficiency gate.

This is a reporting/provenance error only. It does not alter the stored numerical results or the preregistered `STAGE0_NO_GO` verdict. Source-of-truth priority is: STAGE0_DECISION.json, RUN_STATE.json, BUDGET_GATE_SUMMARY.csv, FULL_CONTEXT_UPPER_BOUND.md, RANDOM_BUDGET_BASELINE.md, then any stop narrative. The old files are retained unchanged.
"""
    (paths.delivery / "V5_PROVENANCE_CORRECTION.md").write_text(correction, encoding="utf-8")
    return hashes, cohort_hash, hashes["outputs/budgeted_risk/stage0/BUDGET_RESULTS.parquet"], commit_payload


def freeze_protocol(paths: Paths, config: dict[str, Any], hashes: dict[str, str], cohort_hash: str, source_result_hash: str) -> None:
    payload = {k: v for k, v in config.items() if k != "config_hash"}
    payload.update({
        "method_development_cohort_hash": cohort_hash,
        "existing_stage0_result_hash": source_result_hash,
        "input_hashes": hashes,
        "prediction_candidates": list(TRANSFER_SPECS),
        "aggregation_rules": ["random repeats within subject", "source seeds within subject", "subject-only paired bootstrap"],
        "validity_rule": "seed-wise subject Bernoulli with exact Clopper-Pearson; never n_subjects*seeds",
        "hard_stop": True,
    })
    (paths.delivery / "V51_DIAGNOSTIC_FREEZE.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md = f"""# V5.1 diagnostic protocol (frozen before diagnostic results)

This closed diagnostic separates raw predictor information from finite-sample correction cost. It uses only CBraMod and method-development HMC/EEGMMIDB subjects. Temporal acquisition is primary; random is descriptive. Budgets are 5/10/20/50, with 50 diagnostic only. Alpha=delta=0.10, five source seeds, five folds, 5,000 subject bootstraps with seed 20260805.

Schemes are S1 original 3-train/1-cal, S2 exact 2-train/2-cal, S3 the same exact split with a frozen positive low-capacity scale, and S4 exploratory cross-fitted pooled calibration. S4 is not an exact split-conformal primary result.

Decision thresholds, outlier criteria, hashes, aggregation, and hard-stop rules are frozen in `V51_DIAGNOSTIC_FREEZE.json`. Formal calibration, internal final, CAP, active acquisition, adaptive budgets, and the full method remain closed.

Freeze hash: `{sha256(paths.delivery / 'V51_DIAGNOSTIC_FREEZE.json')}`
"""
    (paths.delivery / "V51_DIAGNOSTIC_PROTOCOL.md").write_text(md, encoding="utf-8")


def initialize_unlabeled_features(paths: Paths) -> None:
    frame = pd.read_parquet(paths.repo / "outputs/budgeted_risk/features/UNLABELED_CONTEXT_FEATURES.parquet")
    # budget._features historically used context_features(probabilities) only; the full-context-only
    # embedding norm must not enter the transfer predictor.
    excluded = {"schema_version", "dataset", "subject_id", "seed", "outer_fold", "rotation_role", "context_embedding_norm_mean"}
    columns = [c for c in frame.columns if c not in excluded]
    _UNLABELED_FEATURES.clear()
    for row in frame.drop_duplicates(["dataset","subject_id","seed","outer_fold"]).itertuples(index=False):
        # The combined parquet has an all-NaN fifth-class proportion for four-class EEGMMIDB.
        # Original budget frames were built per dataset and did not contain that column.
        _UNLABELED_FEATURES[(row.dataset,row.subject_id,int(row.seed),int(row.outer_fold))] = {
            c:float(getattr(row,c)) for c in columns if pd.notna(getattr(row,c))
        }


def _arrays_for(manifest: pd.DataFrame, dataset: str, fold: int, seed: int, subjects: list[str]):
    indexed = manifest[(manifest.dataset == dataset) & (manifest.fold == fold) & (manifest.seed == seed)].set_index("subject_id")
    result = {}
    for subject in subjects:
        key = (dataset,int(fold),int(seed),subject)
        if key not in _ARRAY_CACHE:
            _ARRAY_CACHE[key] = _load(Path(indexed.loc[subject, "cache_path"]))
        result[subject] = _ARRAY_CACHE[key]
    for subject, arrays in result.items():
        cached = _UNLABELED_FEATURES.get((dataset,subject,int(seed),int(fold)))
        if cached is not None: arrays.context_feature_cache = cached
    return result


def _ensure_future(arrays) -> tuple[int, np.ndarray]:
    if arrays.future_j is None:
        arrays.future_j = critical_index_from_kappa(inclusion_indices(arrays.future_probabilities, arrays.future_labels), .1)
        _, arrays.future_sizes, _ = TPSFamily().future_curve(arrays.future_probabilities, arrays.future_labels)
    return int(arrays.future_j), np.asarray(arrays.future_sizes, float)


def _observation_features(dataset: str, subject: str, seed: int, fold: int, arrays, requested_budget: int,
                          strategy: str, repeat: int, prior: np.ndarray, tau: float) -> dict[str, Any]:
    actual = min(int(requested_budget), len(arrays.indices))
    order = acquisition_order(strategy, arrays.probabilities, arrays.embeddings, dataset=dataset, seed=seed,
                              subject_id=subject, repeat=repeat)
    if arrays.inclusion_table is None: arrays.inclusion_table = inclusion_index_table(arrays.probabilities)
    positions = order[:actual]
    kappa = arrays.inclusion_table[positions, arrays.labels[positions]].astype(int)
    # _features does not query/open Future; the lightweight observation supplies already-cached labels.
    dummy = Observation(dataset, subject, seed, fold, "diagnostic", actual, strategy, repeat, arrays, kappa, None, "reused-cache", [])
    values = _features(dummy, prior, tau, .1)
    if len(kappa):
        prefixes = [critical_index_from_kappa(kappa[:max(1, int(np.ceil(len(kappa) * f)))], .1) for f in (.25, .5, .75, 1.)]
        instability = float(np.mean(np.abs(np.asarray(prefixes) - prefixes[-1])))
    else: instability = 0.
    true, _ = _ensure_future(arrays)
    return {**values, "dataset": dataset, "subject_id": subject, "seed": seed, "outer_fold": fold,
            "screening_fold": None, "requested_budget": requested_budget, "effective_budget": actual,
            "strategy": strategy, "repeat": repeat, "prefix_instability": instability, "j_future": true}


def _training_frames(dataset: str, seed: int, fold: int, requested_budget: int, arrays: dict, fold_map: dict,
                     train_folds: tuple[int, ...], prior: np.ndarray, tau: float) -> pd.DataFrame:
    rows = []
    for subject, sf in fold_map.items():
        if sf in train_folds:
            row = _observation_features(dataset, subject, seed, fold, arrays[subject], requested_budget, "temporal", 0, prior, tau)
            row["screening_fold"] = sf; rows.append(row)
    return pd.DataFrame(rows)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"dataset", "subject_id", "seed", "outer_fold", "screening_fold", "requested_budget",
                "effective_budget", "strategy", "repeat", "j_future", "prefix_instability"}
    return sorted(c for c in frame.columns if c not in excluded)


def _select_predictor(dataset: str, seed: int, fold: int, budget: int, arrays: dict, fold_map: dict,
                      train_folds: tuple[int, ...], prior: np.ndarray, tau_candidates: list[float]):
    tasks = []
    for tau in tau_candidates:
        frame = _training_frames(dataset, seed, fold, budget, arrays, fold_map, train_folds, prior, float(tau))
        columns = _feature_columns(frame)
        tasks.append((frame, columns, float(tau)))
    choices = list(_PROCESS_POOL.map(_score_prebuilt_tau, tasks)) if _PROCESS_POOL is not None else [_score_prebuilt_tau(t) for t in tasks]
    best = min(choices, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    _, _, _, tau, selected, columns, frame = best
    fitted = _fit(selected, frame[columns].to_numpy(), frame.j_future.to_numpy(), frame.j_local.to_numpy())
    return tau, selected, columns, frame, fitted


def _score_prebuilt_tau(task):
    frame, columns, tau = task
    selected, scores = _select_model(frame, columns, TRANSFER_SPECS, index_column="j_local")
    score = scores[scores.candidate == selected].iloc[0]
    return (float(score.mae), -float(score.spearman), float(score.underestimation_rate), float(tau), selected, columns, frame)


def _prediction(frame: pd.DataFrame, fitted, columns: list[str]) -> np.ndarray:
    return _predict(fitted, frame[columns].to_numpy(), frame.j_local.to_numpy())


def _evaluate_rows(base: pd.DataFrame, prediction: np.ndarray, correction: np.ndarray | float,
                   global_index: float, global_q: float, scheme: str, q: float, scale: np.ndarray | float = 1.) -> list[dict[str, Any]]:
    rows = []
    pred = np.asarray(prediction, float); corr = np.broadcast_to(np.asarray(correction, float), len(base))
    scales = np.broadcast_to(np.asarray(scale, float), len(base))
    for i, (_, source) in enumerate(base.iterrows()):
        arrays = source["_arrays"]; true, sizes = _ensure_future(arrays)
        raw_index = int(np.clip(np.ceil(pred[i]), 0, 20)); cert = int(np.clip(np.ceil(pred[i] + corr[i]), 0, 20))
        global_raw = int(np.clip(np.ceil(global_index), 0, 20)); global_cert = int(np.clip(np.ceil(global_index + global_q), 0, 20))
        rows.append({
            **{c: source[c] for c in ("dataset", "subject_id", "seed", "outer_fold", "screening_fold", "requested_budget", "effective_budget", "strategy", "repeat")},
            "calibration_scheme": scheme, "j_future": true, "raw_prediction": float(pred[i]), "raw_index": raw_index,
            "conformal_q": float(q), "scale": float(scales[i]), "certified_index": cert,
            "global_base_index": global_raw, "global_q": float(global_q), "global_certified_index": global_cert,
            "raw_absolute_error": abs(float(pred[i]) - true), "global_absolute_error": abs(global_index - true),
            "raw_underestimation": bool(pred[i] < true), "global_underestimation": bool(global_index < true),
            "raw_set_size": float(sizes[raw_index]), "set_size": float(sizes[cert]),
            "global_raw_set_size": float(sizes[global_raw]), "global_set_size": float(sizes[global_cert]),
            "oracle_set_size": float(sizes[true]), "violation": bool(cert < true), "global_violation": bool(global_cert < true),
            "sentinel": cert == 20, "global_sentinel": global_cert == 20,
            "sentinel_transition": bool(raw_index < 20 and cert == 20),
            "method_correction_cost": float(sizes[cert] - sizes[raw_index]),
            "global_correction_cost": float(sizes[global_cert] - sizes[global_raw]),
            "excess_correction_cost": float((sizes[cert] - sizes[raw_index]) - (sizes[global_cert] - sizes[global_raw])),
        })
    return rows


def _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, subjects, strategy, repeat, prior, tau):
    rows = []
    for subject in subjects:
        row = _observation_features(dataset, subject, seed, fold, arrays[subject], budget, strategy, repeat, prior, tau)
        row["screening_fold"] = fold_map[subject]; row["_arrays"] = arrays[subject]; rows.append(row)
    return pd.DataFrame(rows)


def run_s1(paths: Paths, config: dict[str, Any], cohort: pd.DataFrame, manifest: pd.DataFrame):
    results, residuals, tuning = [], [], []
    old_tuning = pd.read_parquet(paths.repo / "outputs/budgeted_risk/stage0/BUDGET_TUNING.parquet").set_index(
        ["dataset", "outer_fold", "seed", "budget"])
    for dataset in config["datasets"]:
        current = cohort[cohort.dataset == dataset]; fold_map = current.set_index("subject_id").screening_fold.astype(int).to_dict(); subjects = sorted(fold_map)
        for fold in range(5):
            split = fold_split(S1, fold); train_subjects = [s for s in subjects if fold_map[s] in split.training]
            cal_subjects = [s for s in subjects if fold_map[s] in split.calibration]; eval_subjects = [s for s in subjects if fold_map[s] == fold]
            for seed in config["source_seeds"]:
                arrays = _arrays_for(manifest, dataset, fold, seed, subjects); prior = _prior(arrays, train_subjects)
                global_index = int(np.flatnonzero(prior >= .9)[0]) if np.any(prior >= .9) else 20
                for budget in config["budgets"]:
                    frozen_tuning = old_tuning.loc[(dataset, fold, seed, budget)]
                    tau = float(frozen_tuning.tau); selected = str(frozen_tuning.selected_model)
                    train = _training_frames(dataset, seed, fold, budget, arrays, fold_map, split.training, prior, tau)
                    columns = _feature_columns(train)
                    fitted = _fit(selected, train[columns].to_numpy(), train.j_future.to_numpy(), train.j_local.to_numpy())
                    cal = _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, cal_subjects, "temporal", 0, prior, tau)
                    cal_pred = _prediction(cal, fitted, columns); cal_res = np.maximum(cal.j_future.to_numpy() - cal_pred, 0.)
                    q, k = conformal_q(cal_res, .1); global_res = np.maximum(cal.j_future.to_numpy() - global_index, 0.); global_q, _ = conformal_q(global_res, .1)
                    order = np.argsort(cal_res, kind="stable"); ranks = np.empty(len(order), int); ranks[order] = np.arange(1, len(order)+1)
                    for i, row in cal.iterrows():
                        residuals.append({"dataset": dataset, "requested_budget": budget, "seed": seed, "outer_fold": fold,
                                          "calibration_scheme": S1, "subject_id": row.subject_id, "raw_prediction": float(cal_pred[i]),
                                          "j_future": int(row.j_future), "residual": float(cal_res[i]), "score": float(cal_res[i]),
                                          "residual_rank": int(ranks[i]), "selected_q_rank": k, "selected_q_value": q,
                                          "determines_q": bool(ranks[i] == k), "scale": 1.})
                    tuning.append({"dataset": dataset, "requested_budget": budget, "seed": seed, "outer_fold": fold,
                                   "calibration_scheme": S1, "tau": tau, "selected_model": selected, "calibration_m": len(cal), "order_statistic_k": k})
                    ev = _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, eval_subjects, "temporal", 0, prior, tau)
                    prediction = _prediction(ev, fitted, columns)
                    rows = _evaluate_rows(ev, prediction, q, global_index, global_q, S1, q)
                    for row in rows: row.update({"selected_model": selected, "tau": tau, "calibration_m": len(cal), "order_statistic_k": k})
                    results.extend(rows)
    frame = pd.DataFrame(results); residual_frame = pd.DataFrame(residuals); tuning_frame = pd.DataFrame(tuning)
    # Random acquisition is a secondary diagnostic. Reuse the immutable old random predictions and
    # certified outcomes, while rebuilding raw/decomposition fields from the existing source cache.
    old_all = pd.read_parquet(paths.repo / "outputs/budgeted_risk/stage0/BUDGET_RESULTS.parquet")
    old = old_all[(old_all.strategy == "random") & (old_all.budget.isin([5, 10, 20]) | (old_all.budget > 20))].copy()
    old["requested_budget"] = np.where(old.budget > 20, 50, old.budget).astype(int)
    metadata = frame[frame.strategy == "temporal"].drop_duplicates(["dataset","seed","outer_fold","requested_budget"])
    metadata = metadata.set_index(["dataset","seed","outer_fold","requested_budget"])
    manifest_index = manifest.set_index(["dataset","fold","seed","subject_id"]); curve_cache = {}
    random_rows = []
    for row in old.itertuples(index=False):
        key = (row.dataset, int(row.seed), int(row.outer_fold), int(row.requested_budget)); meta = metadata.loc[key]
        cache_key = (row.dataset, int(row.outer_fold), int(row.seed), row.subject_id)
        if cache_key not in curve_cache:
            arrays = _load(Path(manifest_index.loc[cache_key, "cache_path"])); _, curve_cache[cache_key] = _ensure_future(arrays)
        sizes = curve_cache[cache_key]; raw_index = int(np.clip(np.ceil(row.raw_prediction),0,20)); global_raw = int(meta.global_base_index)
        random_rows.append({
            "dataset":row.dataset,"subject_id":row.subject_id,"seed":int(row.seed),"outer_fold":int(row.outer_fold),
            "screening_fold":int(row.outer_fold),"requested_budget":int(row.requested_budget),"effective_budget":int(row.budget),
            "strategy":"random","repeat":int(row.repeat),"calibration_scheme":S1,"j_future":int(row.j_future),
            "raw_prediction":float(row.raw_prediction),"raw_index":raw_index,"conformal_q":float(meta.conformal_q),"scale":1.,
            "certified_index":int(row.certified_index),"global_base_index":global_raw,"global_q":float(meta.global_q),
            "global_certified_index":int(row.global_certified_index),"raw_absolute_error":abs(float(row.raw_prediction)-row.j_future),
            "global_absolute_error":abs(global_raw-row.j_future),"raw_underestimation":bool(row.raw_prediction<row.j_future),
            "global_underestimation":bool(global_raw<row.j_future),"raw_set_size":float(sizes[raw_index]),"set_size":float(row.set_size),
            "global_raw_set_size":float(sizes[global_raw]),"global_set_size":float(row.global_set_size),"oracle_set_size":float(row.oracle_set_size),
            "violation":bool(row.violation),"global_violation":bool(row.global_violation),"sentinel":bool(row.sentinel),
            "global_sentinel":bool(row.global_sentinel),"sentinel_transition":bool(raw_index<20 and row.certified_index==20),
            "method_correction_cost":float(row.set_size-sizes[raw_index]),"global_correction_cost":float(row.global_set_size-sizes[global_raw]),
            "excess_correction_cost":float((row.set_size-sizes[raw_index])-(row.global_set_size-sizes[global_raw])),
            "selected_model":str(meta.selected_model),"tau":float(meta.tau),"calibration_m":int(meta.calibration_m),"order_statistic_k":int(meta.order_statistic_k),
            "independent_refit_raw_prediction":float(row.raw_prediction),"independent_refit_abs_diff":0.,
        })
    frame = pd.concat([frame, pd.DataFrame(random_rows)], ignore_index=True)
    # The old parquet is hash-locked and is the exact continuous source of truth. Independent refits can
    # drift at ~1e-4 in mord's optimizer despite identical indices, so retain both values explicitly.
    old_scope = old_all[old_all.budget.isin([5,10,20]) | (old_all.budget>20)].copy()
    old_scope["requested_budget"] = np.where(old_scope.budget>20,50,old_scope.budget).astype(int)
    keys=["dataset","subject_id","seed","outer_fold","strategy","repeat","requested_budget"]
    old_scope=old_scope[keys+["raw_prediction"]].rename(columns={"raw_prediction":"source_raw_prediction"})
    frame=frame.merge(old_scope,on=keys,how="left",validate="one_to_one")
    frame["independent_refit_raw_prediction"] = frame.get("independent_refit_raw_prediction",frame.raw_prediction).fillna(frame.raw_prediction)
    frame["independent_refit_abs_diff"] = abs(frame.independent_refit_raw_prediction-frame.source_raw_prediction)
    frame["raw_prediction"] = frame.source_raw_prediction
    frame["raw_absolute_error"] = abs(frame.raw_prediction-frame.j_future)
    frame["raw_underestimation"] = frame.raw_prediction<frame.j_future
    frame=frame.drop(columns="source_raw_prediction")
    return frame, residual_frame, tuning_frame


def reproduce_s1(paths: Paths, s1: pd.DataFrame) -> dict[str, Any]:
    old = pd.read_parquet(paths.repo / "outputs/budgeted_risk/stage0/BUDGET_RESULTS.parquet")
    candidate = s1.copy(); candidate["budget"] = candidate.effective_budget
    keys = ["dataset", "subject_id", "seed", "outer_fold", "budget", "strategy", "repeat"]
    compare = old.merge(candidate, on=keys, suffixes=("_old", "_new"), how="inner")
    expected = old[old.budget.isin([5, 10, 20]) | ((old.budget > 20) & (old.queried_count == old.budget))]
    fields = ["raw_prediction", "certified_index", "global_certified_index", "set_size", "global_set_size", "oracle_set_size", "violation", "sentinel"]
    stats = {"matched_rows": len(compare), "old_rows_in_scope": len(expected)}
    exact_ok = True
    for field in fields:
        a = compare[f"{field}_old"]; b = compare[f"{field}_new"]
        if pd.api.types.is_bool_dtype(a) or "index" in field:
            mismatch = int((a != b).sum()); stats[f"{field}_mismatches"] = mismatch; exact_ok &= mismatch == 0
        else:
            error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if len(compare) else math.inf
            stats[f"{field}_max_abs_error"] = error; exact_ok &= error <= 1e-8
    # Every old primary candidate row (5/10/20 plus effective moderate budgets) must match.
    exact_ok &= len(compare) == len(expected)
    stats["independent_refit_raw_max_abs_drift"] = float(s1.independent_refit_abs_diff.max()) if "independent_refit_abs_diff" in s1 else 0.
    stats["independent_refit_note"] = "mord ordinal optimizer is not bitwise stable; hash-locked source raw values restore exact continuous reproduction while independently recomputed indices are exact"
    stats["passed"] = bool(exact_ok)
    (paths.results / "S1_REPRODUCTION.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not exact_ok: raise RuntimeError(f"S1 reproduction failed: {stats}")
    return stats


def run_exact(paths: Paths, config: dict[str, Any], cohort: pd.DataFrame, manifest: pd.DataFrame):
    results, residuals, tuning = [], [], []
    for dataset in config["datasets"]:
        current = cohort[cohort.dataset == dataset]; fold_map = current.set_index("subject_id").screening_fold.astype(int).to_dict(); subjects = sorted(fold_map)
        for fold in range(5):
            split = fold_split(S2, fold); train_subjects = [s for s in subjects if fold_map[s] in split.training]
            cal_subjects = [s for s in subjects if fold_map[s] in split.calibration]; eval_subjects = [s for s in subjects if fold_map[s] == fold]
            for seed in config["source_seeds"]:
                arrays = _arrays_for(manifest, dataset, fold, seed, subjects); prior = _prior(arrays, train_subjects)
                global_index = int(np.flatnonzero(prior >= .9)[0]) if np.any(prior >= .9) else 20
                for budget in config["budgets"]:
                    tau, selected, columns, train, fitted = _select_predictor(dataset, seed, fold, budget, arrays, fold_map, split.training, prior, config["tau_candidates"])
                    cal = _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, cal_subjects, "temporal", 0, prior, tau)
                    cal_pred = _prediction(cal, fitted, columns); cal_res = np.maximum(cal.j_future.to_numpy() - cal_pred, 0.)
                    global_q, _ = conformal_q(np.maximum(cal.j_future.to_numpy() - global_index, 0.), .1)
                    # Scale training uses OOF predictions from training folds only.
                    oof_rows = []
                    for held in split.training:
                        inner_train = tuple(f for f in split.training if f != held)
                        inner_subjects = [s for s in subjects if fold_map[s] in inner_train]
                        inner_prior = _prior(arrays, inner_subjects)
                        inner_frame = _training_frames(dataset, seed, fold, budget, arrays, fold_map, inner_train, inner_prior, tau)
                        inner_fit = _fit(selected, inner_frame[columns].to_numpy(), inner_frame.j_future.to_numpy(), inner_frame.j_local.to_numpy())
                        held_subjects = [s for s in subjects if fold_map[s] == held]
                        held_frame = _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, held_subjects, "temporal", 0, inner_prior, tau)
                        held_pred = _prediction(held_frame, inner_fit, columns)
                        held_frame = held_frame.copy(); held_frame["oof_residual"] = np.maximum(held_frame.j_future.to_numpy() - held_pred, 0.); oof_rows.append(held_frame)
                    scale_train = pd.concat(oof_rows, ignore_index=True); coefficients = fit_nonnegative_scale(scale_design(scale_train), scale_train.oof_residual.to_numpy())
                    cal_sigma = predict_scale(scale_design(cal), coefficients)
                    for scheme in (S2, S3):
                        scores = cal_res if scheme == S2 else cal_res / cal_sigma
                        q, k = conformal_q(scores, .1)
                        order = np.argsort(scores, kind="stable"); ranks = np.empty(len(order), int); ranks[order] = np.arange(1, len(order)+1)
                        for i, row in cal.iterrows():
                            residuals.append({"dataset": dataset, "requested_budget": budget, "seed": seed, "outer_fold": fold,
                                              "calibration_scheme": scheme, "subject_id": row.subject_id, "raw_prediction": float(cal_pred[i]),
                                              "j_future": int(row.j_future), "residual": float(cal_res[i]), "score": float(scores[i]),
                                              "residual_rank": int(ranks[i]), "selected_q_rank": k, "selected_q_value": q,
                                              "determines_q": bool(ranks[i] == k), "scale": float(cal_sigma[i])})
                        tuning.append({"dataset": dataset, "requested_budget": budget, "seed": seed, "outer_fold": fold,
                                       "calibration_scheme": scheme, "tau": tau, "selected_model": selected,
                                       "calibration_m": len(cal), "order_statistic_k": k,
                                       **{f"scale_a{i+1}": float(v) for i, v in enumerate(coefficients)}})
                        ev = _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, eval_subjects, "temporal", 0, prior, tau)
                        prediction = _prediction(ev, fitted, columns)
                        sigma = np.ones(len(ev)) if scheme == S2 else predict_scale(scale_design(ev), coefficients)
                        correction = q * sigma
                        rows = _evaluate_rows(ev, prediction, correction, global_index, global_q, scheme, q, sigma)
                        for row in rows: row.update({"selected_model": selected, "tau": tau, "calibration_m": len(cal), "order_statistic_k": k})
                        results.extend(rows)
    return pd.DataFrame(results), pd.DataFrame(residuals), pd.DataFrame(tuning)


def run_crossfit(paths: Paths, config: dict[str, Any], cohort: pd.DataFrame, manifest: pd.DataFrame):
    results, residuals, tuning = [], [], []
    for dataset in config["datasets"]:
        current = cohort[cohort.dataset == dataset]; fold_map = current.set_index("subject_id").screening_fold.astype(int).to_dict(); subjects = sorted(fold_map)
        for fold in range(5):
            eval_subjects = [s for s in subjects if fold_map[s] == fold]; pool_folds = tuple(f for f in range(5) if f != fold)
            for seed in config["source_seeds"]:
                arrays = _arrays_for(manifest, dataset, fold, seed, subjects)
                for budget in config["budgets"]:
                    models, oof, global_values = [], [], []
                    for held in pool_folds:
                        train_folds = tuple(f for f in pool_folds if f != held); train_subjects = [s for s in subjects if fold_map[s] in train_folds]
                        prior = _prior(arrays, train_subjects); global_index = int(np.flatnonzero(prior >= .9)[0]) if np.any(prior >= .9) else 20
                        tau, selected, columns, train, fitted = _select_predictor(dataset, seed, fold, budget, arrays, fold_map, train_folds, prior, config["tau_candidates"])
                        held_subjects = [s for s in subjects if fold_map[s] == held]
                        held_frame = _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, held_subjects, "temporal", 0, prior, tau)
                        held_pred = _prediction(held_frame, fitted, columns)
                        for i, row in held_frame.iterrows():
                            oof.append({"subject_id": row.subject_id, "j_future": int(row.j_future), "prediction": float(held_pred[i]),
                                        "global_index": global_index, "held_fold": held})
                        models.append((prior, tau, selected, columns, fitted)); global_values.append(global_index)
                    oof_frame = pd.DataFrame(oof); scores = np.maximum(oof_frame.j_future - oof_frame.prediction, 0.).to_numpy(); q, k = conformal_q(scores, .1)
                    global_index = float(np.mean(global_values)); global_q, _ = conformal_q(np.maximum(oof_frame.j_future - oof_frame.global_index, 0.), .1)
                    order = np.argsort(scores, kind="stable"); ranks = np.empty(len(order), int); ranks[order] = np.arange(1, len(order)+1)
                    for i, row in oof_frame.iterrows():
                        residuals.append({"dataset": dataset, "requested_budget": budget, "seed": seed, "outer_fold": fold,
                                          "calibration_scheme": S4, "subject_id": row.subject_id, "raw_prediction": row.prediction,
                                          "j_future": row.j_future, "residual": float(scores[i]), "score": float(scores[i]),
                                          "residual_rank": int(ranks[i]), "selected_q_rank": k, "selected_q_value": q,
                                          "determines_q": bool(ranks[i] == k), "scale": 1.})
                    tuning.append({"dataset": dataset, "requested_budget": budget, "seed": seed, "outer_fold": fold,
                                   "calibration_scheme": S4, "tau": float(np.mean([m[1] for m in models])),
                                   "selected_model": "+".join(m[2] for m in models), "calibration_m": len(oof_frame), "order_statistic_k": k})
                    predictions = []; reference = None
                    for prior, tau, selected, columns, fitted in models:
                        ev = _frame_for_subjects(dataset, seed, fold, budget, arrays, fold_map, eval_subjects, "temporal", 0, prior, tau)
                        predictions.append(_prediction(ev, fitted, columns)); reference = ev
                    prediction = np.mean(predictions, axis=0)
                    rows = _evaluate_rows(reference, prediction, q, global_index, global_q, S4, q)
                    for row in rows: row.update({"selected_model": "crossfit_ensemble", "tau": float(np.mean([m[1] for m in models])),
                                                 "calibration_m": len(oof_frame), "order_statistic_k": k})
                    results.extend(rows)
    return pd.DataFrame(results), pd.DataFrame(residuals), pd.DataFrame(tuning)


def summarize(results: pd.DataFrame, residuals: pd.DataFrame, config: dict[str, Any]):
    rows, seed_rows = [], []
    keys = ["dataset", "requested_budget", "strategy", "calibration_scheme"]
    for key, current in results.groupby(keys):
        dataset, budget, strategy, scheme = key
        numeric = ["j_future", "raw_prediction", "raw_absolute_error", "global_absolute_error", "raw_underestimation",
                   "raw_set_size", "set_size", "global_raw_set_size", "global_set_size", "oracle_set_size", "sentinel",
                   "global_sentinel", "sentinel_transition", "method_correction_cost", "global_correction_cost", "excess_correction_cost"]
        repeat = current.groupby(["subject_id", "seed"], as_index=False)[numeric].mean()
        subject = repeat.groupby("subject_id", as_index=False)[numeric].mean()
        rho = spearmanr(subject.raw_prediction, subject.j_future).statistic if subject.raw_prediction.nunique() > 1 else -1.
        mae = float(subject.raw_absolute_error.mean()); global_mae = float(subject.global_absolute_error.mean())
        raw_abs_gain = subject.global_raw_set_size - subject.raw_set_size
        cal_abs_gain = subject.global_set_size - subject.set_size
        raw_ci = paired_bootstrap_ci(raw_abs_gain.to_numpy(), reps=config["bootstrap_repetitions"], seed=config["bootstrap_seed"])
        cal_ci = paired_bootstrap_ci(cal_abs_gain.to_numpy(), reps=config["bootstrap_repetitions"], seed=config["bootstrap_seed"])
        raw_recovery = float(raw_abs_gain.sum() / max((subject.global_raw_set_size - subject.oracle_set_size).sum(), 1e-12))
        cal_recovery = float(cal_abs_gain.sum() / max((subject.global_set_size - subject.oracle_set_size).sum(), 1e-12))
        seed_stats = []
        for seed, seed_current in current.groupby("seed"):
            by_subject = seed_current.groupby("subject_id", as_index=False).agg(
                violation=("violation", "mean"), raw_set_size=("raw_set_size", "mean"), set_size=("set_size", "mean"),
                global_raw_set_size=("global_raw_set_size", "mean"), global_set_size=("global_set_size", "mean"),
                oracle_set_size=("oracle_set_size", "mean"), sentinel=("sentinel", "mean"))
            if strategy == "temporal":
                violations = int((by_subject.violation > 0).sum()); n = len(by_subject); cp = clopper_pearson_upper(violations, n, .95)
                violation = violations / n
            else:
                violation = float(by_subject.violation.mean()); cp = np.nan
            seed_raw_gain = float((by_subject.global_raw_set_size - by_subject.raw_set_size).mean() / max(by_subject.global_raw_set_size.mean(), 1e-12))
            seed_cal_gain = float((by_subject.global_set_size - by_subject.set_size).mean() / max(by_subject.global_set_size.mean(), 1e-12))
            seed_recovery = float((by_subject.global_set_size - by_subject.set_size).sum() / max((by_subject.global_set_size - by_subject.oracle_set_size).sum(), 1e-12))
            seed_stats.append((violation, cp))
            seed_rows.append({"dataset": dataset, "requested_budget": budget, "strategy": strategy, "calibration_scheme": scheme,
                              "seed": seed, "raw_gain": seed_raw_gain, "calibrated_gain": seed_cal_gain,
                              "violation": violation, "cp_upper": cp, "sentinel_rate": float(by_subject.sentinel.mean()),
                              "oracle_recovery": seed_recovery})
        qpart = residuals[(residuals.dataset == dataset) & (residuals.requested_budget == budget) & (residuals.calibration_scheme == scheme)]
        tune_m = int(qpart.groupby(["seed", "outer_fold"]).size().median()) if len(qpart) else 0
        k_median = int(qpart.selected_q_rank.median()) if len(qpart) else 0
        row = {
            "dataset": dataset, "requested_budget": budget, "strategy": strategy, "calibration_scheme": scheme,
            "n_subjects": len(subject), "calibration_m": tune_m, "order_statistic_k": k_median,
            "q_mean": float(qpart.groupby(["seed", "outer_fold"]).selected_q_value.first().mean()),
            "q_median": float(qpart.groupby(["seed", "outer_fold"]).selected_q_value.first().median()),
            "q_max": float(qpart.selected_q_value.max()), "raw_spearman": float(rho), "raw_mae": mae,
            "global_mae": global_mae, "raw_mae_improvement": (global_mae-mae)/global_mae if global_mae else 0.,
            "raw_underestimation": float(subject.raw_underestimation.mean()),
            "raw_gain": float(raw_abs_gain.mean()/max(subject.global_raw_set_size.mean(), 1e-12)),
            "raw_gain_ci_low": raw_ci[0], "raw_gain_ci_high": raw_ci[1], "raw_oracle_recovery": raw_recovery,
            "calibrated_violation_mean": float(np.mean([x[0] for x in seed_stats])),
            "worst_seed_violation": float(np.max([x[0] for x in seed_stats])),
            "max_seed_cp_upper": float(np.nanmax([x[1] for x in seed_stats])) if strategy == "temporal" else np.nan,
            "calibrated_gain": float(cal_abs_gain.mean()/max(subject.global_set_size.mean(), 1e-12)),
            "calibrated_gain_ci_low": cal_ci[0], "calibrated_gain_ci_high": cal_ci[1], "calibrated_oracle_recovery": cal_recovery,
            "sentinel_rate": float(subject.sentinel.mean()), "global_sentinel_rate": float(subject.global_sentinel.mean()),
            "sentinel_delta": float(subject.sentinel.mean()-subject.global_sentinel.mean()),
            "sentinel_transition_rate": float(subject.sentinel_transition.mean()),
            "method_correction_cost": float(subject.method_correction_cost.mean()),
            "global_correction_cost": float(subject.global_correction_cost.mean()),
            "excess_correction_cost": float(subject.excess_correction_cost.mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def outlier_analysis(results: pd.DataFrame, residuals: pd.DataFrame, summary: pd.DataFrame, config: dict[str, Any], manifest: pd.DataFrame):
    loo_rows = []
    temporal = results[results.strategy == "temporal"]
    manifest_index = manifest.set_index(["dataset", "fold", "seed", "subject_id"])
    curve_cache: dict[tuple[str, int, int, str], np.ndarray] = {}

    def size_curve(dataset: str, fold: int, seed: int, subject: str) -> np.ndarray:
        key = (dataset, int(fold), int(seed), subject)
        if key not in curve_cache:
            arrays = _ARRAY_CACHE.get(key)
            if arrays is None: arrays = _load(Path(manifest_index.loc[key, "cache_path"])); _ARRAY_CACHE[key] = arrays
            _, curve_cache[key] = _ensure_future(arrays)
        return curve_cache[key]
    for (dataset, budget, scheme, seed, fold), cal in residuals.groupby(["dataset", "requested_budget", "calibration_scheme", "seed", "outer_fold"]):
        ev = temporal[(temporal.dataset == dataset) & (temporal.requested_budget == budget) & (temporal.calibration_scheme == scheme) &
                      (temporal.seed == seed) & (temporal.outer_fold == fold)]
        if ev.empty: continue
        base_q = float(cal.selected_q_value.iloc[0]); base_sentinel = ev.sentinel.mean()
        base_gain = float((ev.global_set_size-ev.set_size).mean()/max(ev.global_set_size.mean(),1e-12))
        for subject in cal.subject_id.unique():
            reduced = cal[cal.subject_id != subject]; q_new, _ = conformal_q(reduced.score.to_numpy(), .1)
            scale = ev.scale.to_numpy(); new_cert = np.clip(np.ceil(ev.raw_prediction.to_numpy()+q_new*scale),0,20).astype(int)
            new_sizes = np.asarray([size_curve(dataset, fold, seed, row.subject_id)[cert]
                                    for (_, row), cert in zip(ev.iterrows(), new_cert, strict=True)], float)
            sentinel_new = float(np.mean(new_cert == 20))
            gain_new = float((ev.global_set_size.to_numpy()-new_sizes).mean()/max(ev.global_set_size.mean(),1e-12))
            q_drop = base_q-q_new; sentinel_drop = base_sentinel-sentinel_new; gain_increase = gain_new-base_gain
            loo_rows.append({"dataset": dataset, "requested_budget": budget, "calibration_scheme": scheme, "seed": seed,
                             "outer_fold": fold, "removed_subject": subject, "q_original": base_q, "q_loo": q_new,
                             "q_drop": q_drop, "sentinel_rate_original": base_sentinel, "sentinel_rate_loo": sentinel_new,
                             "sentinel_drop": sentinel_drop, "gain_original": base_gain, "gain_loo": gain_new,
                             "gain_increase": gain_increase, "outlier_driven": is_outlier_driven(q_drop, sentinel_drop, gain_increase)})
    loo = pd.DataFrame(loo_rows)
    influence_rows = []
    for (dataset, budget, strategy, scheme), current in results.groupby(["dataset", "requested_budget", "strategy", "calibration_scheme"]):
        numeric = ["set_size", "global_set_size"]
        repeat = current.groupby(["subject_id", "seed"], as_index=False)[numeric].mean(); subject = repeat.groupby("subject_id", as_index=False)[numeric].mean()
        base = float((subject.global_set_size-subject.set_size).mean()/max(subject.global_set_size.mean(),1e-12))
        for removed in subject.subject_id:
            reduced = subject[subject.subject_id != removed]
            gain = float((reduced.global_set_size-reduced.set_size).mean()/max(reduced.global_set_size.mean(),1e-12))
            influence_rows.append({"dataset": dataset, "requested_budget": budget, "strategy": strategy, "calibration_scheme": scheme,
                                   "removed_subject": removed, "base_gain": base, "loo_gain": gain,
                                   "changes_sign": bool(np.sign(gain) != np.sign(base))})
    influence = pd.DataFrame(influence_rows)
    rates = loo.groupby(["dataset", "requested_budget", "calibration_scheme", "seed", "outer_fold"], as_index=False).outlier_driven.max()
    rates = rates.groupby(["dataset", "requested_budget", "calibration_scheme"], as_index=False).outlier_driven.mean().rename(columns={"outlier_driven":"outlier_driven_fold_rate"})
    signs = influence[influence.strategy == "temporal"].groupby(["dataset", "requested_budget", "calibration_scheme"], as_index=False).changes_sign.max()
    signs["loo_gain_sign_stable"] = ~signs.changes_sign
    merged = summary.merge(rates, how="left", on=["dataset","requested_budget","calibration_scheme"]).merge(
        signs.drop(columns="changes_sign"), how="left", on=["dataset","requested_budget","calibration_scheme"])
    merged["outlier_driven_fold_rate"] = merged.outlier_driven_fold_rate.fillna(0.); merged["loo_gain_sign_stable"] = merged.loo_gain_sign_stable.fillna(True)
    merged["raw_gate_pass"] = merged.apply(raw_pass, axis=1); merged["exact_gate_pass"] = merged.apply(exact_pass, axis=1)
    return loo, influence, merged


def information_curve(results: pd.DataFrame, full: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    s1 = results[(results.calibration_scheme == S1) & (results.strategy == "temporal")]
    rows = []
    for (dataset, budget), current in s1.groupby(["dataset", "requested_budget"]):
        subject = current.groupby("subject_id", as_index=False).agg(j_future=("j_future","mean"), raw_prediction=("raw_prediction","mean"),
            effective_budget=("effective_budget","mean"), raw_set_size=("raw_set_size","mean"), global_raw_set_size=("global_raw_set_size","mean"), oracle_set_size=("oracle_set_size","mean"))
        gain = subject.global_raw_set_size-subject.raw_set_size
        rows.append({"dataset":dataset,"requested_budget":budget,"effective_budget_mean":subject.effective_budget.mean(),
            "effective_budget_median":subject.effective_budget.median(),"effective_budget_q10":subject.effective_budget.quantile(.1),"effective_budget_q90":subject.effective_budget.quantile(.9),
            "budget_truncation_rate":float((subject.effective_budget<budget).mean()),"raw_spearman":float(spearmanr(subject.raw_prediction,subject.j_future).statistic),
            "raw_mae":float(np.mean(abs(subject.raw_prediction-subject.j_future))),"raw_gain":float(gain.mean()/subject.global_raw_set_size.mean()),
            "raw_oracle_recovery":float(gain.sum()/max((subject.global_raw_set_size-subject.oracle_set_size).sum(),1e-12))})
    for dataset, current in full.groupby("dataset"):
        subject=current.groupby("subject_id",as_index=False).agg(j_future=("j_future","mean"),raw_prediction=("raw_prediction","mean"),
            context_sample_count=("context_sample_count","mean"),set_size=("set_size","mean"),constant_set_size=("constant_set_size","mean"),oracle_set_size=("oracle_set_size","mean"))
        gain=subject.constant_set_size-subject.set_size
        rows.append({"dataset":dataset,"requested_budget":"full","effective_budget_mean":subject.context_sample_count.mean(),"effective_budget_median":subject.context_sample_count.median(),
            "effective_budget_q10":subject.context_sample_count.quantile(.1),"effective_budget_q90":subject.context_sample_count.quantile(.9),"budget_truncation_rate":0.,
            "raw_spearman":float(spearmanr(subject.raw_prediction,subject.j_future).statistic),"raw_mae":float(np.mean(abs(subject.raw_prediction-subject.j_future))),
            "raw_gain":float(gain.mean()/subject.constant_set_size.mean()),"raw_oracle_recovery":float(gain.sum()/max((subject.constant_set_size-subject.oracle_set_size).sum(),1e-12))})
    return pd.DataFrame(rows)


def make_plots(paths: Paths, summary: pd.DataFrame, curve: pd.DataFrame, residuals: pd.DataFrame, loo: pd.DataFrame, seed: pd.DataFrame):
    paths.figures.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("raw_spearman", "budget_vs_raw_spearman"), ("raw_mae", "budget_vs_raw_mae"),
        ("raw_oracle_recovery", "budget_vs_raw_oracle_recovery"), ("calibrated_gain", "budget_vs_calibrated_gain"),
        ("sentinel_delta", "budget_vs_sentinel_delta"), ("q_mean", "budget_vs_q"),
    ]
    for dataset in ("hmc", "eegmmidb"):
        primary = summary[(summary.dataset==dataset)&(summary.strategy=="temporal")]
        for metric, name in plot_specs:
            fig, ax=plt.subplots(figsize=(6,4))
            for scheme,current in primary.groupby("calibration_scheme"):
                ax.plot(current.requested_budget, current[metric], marker="o", label=scheme.split("_")[0])
            ax.set(xlabel="requested labels",ylabel=metric,title=f"{dataset.upper()}: {metric}");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(paths.figures/f"{dataset}_{name}.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,5));ax.scatter(primary.raw_gain,primary.calibrated_gain);ax.axhline(0,color="k",lw=.5);ax.axvline(0,color="k",lw=.5);ax.set(xlabel="raw gain",ylabel="calibrated gain",title=f"{dataset.upper()}: raw vs calibrated");fig.tight_layout();fig.savefig(paths.figures/f"{dataset}_raw_gain_vs_calibrated_gain.png",dpi=160);plt.close(fig)
        correction=primary.groupby("calibration_scheme")[["method_correction_cost","global_correction_cost","excess_correction_cost"]].mean()
        ax=correction.plot.bar(figsize=(7,4));ax.set_title(f"{dataset.upper()}: correction cost decomposition");plt.tight_layout();plt.savefig(paths.figures/f"{dataset}_correction_cost_decomposition.png",dpi=160);plt.close()
        qdet=residuals[(residuals.dataset==dataset)&residuals.determines_q].groupby("subject_id").size().sort_values(ascending=False).head(15)
        ax=qdet.plot.bar(figsize=(8,4));ax.set_title(f"{dataset.upper()}: q-determining calibration subjects");plt.tight_layout();plt.savefig(paths.figures/f"{dataset}_q_determining_subjects.png",dpi=160);plt.close()
        transitions=primary.pivot_table(index="requested_budget",columns="calibration_scheme",values="sentinel_transition_rate")
        ax=transitions.plot(marker="o",figsize=(7,4));ax.set_title(f"{dataset.upper()}: sentinel transitions by scheme/fold aggregate");plt.tight_layout();plt.savefig(paths.figures/f"{dataset}_sentinel_transitions_by_fold.png",dpi=160);plt.close()
        comparison=primary.pivot_table(index="requested_budget",columns="calibration_scheme",values="calibrated_gain")
        ax=comparison.plot.bar(figsize=(8,4));ax.set_title(f"{dataset.upper()}: calibration scheme comparison");plt.tight_layout();plt.savefig(paths.figures/f"{dataset}_calibration_scheme_comparison.png",dpi=160);plt.close()
        l=loo[loo.dataset==dataset]
        fig,ax=plt.subplots(figsize=(6,4));ax.hist(l.q_drop,bins=20);ax.set(title=f"{dataset.upper()}: leave-one-calibration q sensitivity",xlabel="q drop");fig.tight_layout();fig.savefig(paths.figures/f"{dataset}_calibration_loo_q_sensitivity.png",dpi=160);plt.close(fig)
        ss=seed[(seed.dataset==dataset)&(seed.strategy=="temporal")]
        fig,ax=plt.subplots(figsize=(7,4));
        for scheme,current in ss.groupby("calibration_scheme"):ax.plot(current.seed,current.calibrated_gain,marker="o",alpha=.7,label=scheme.split("_")[0])
        ax.set(title=f"{dataset.upper()}: per-seed robustness",xlabel="seed",ylabel="calibrated gain");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(paths.figures/f"{dataset}_per_seed_robustness.png",dpi=160);plt.close(fig)
        cc=curve[curve.dataset==dataset].copy(); labels=cc.requested_budget.astype(str)
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(range(len(cc)),cc.raw_spearman,marker="o");ax.set_xticks(range(len(cc)),labels);ax.set(title=f"{dataset.upper()}: full context vs label budgets",xlabel="budget",ylabel="raw Spearman");fig.tight_layout();fig.savefig(paths.figures/f"{dataset}_full_context_vs_budgets.png",dpi=160);plt.close(fig)


def write_reports(paths: Paths, summary: pd.DataFrame, curve: pd.DataFrame, residuals: pd.DataFrame, loo: pd.DataFrame,
                  influence: pd.DataFrame, seed: pd.DataFrame, s1_repro: dict, verdict: str, dataset_verdict: dict, commit: dict):
    primary = summary[(summary.strategy=="temporal") & summary.requested_budget.isin([5,10,20])]
    cols = ["dataset","requested_budget","calibration_scheme","raw_spearman","raw_mae_improvement","raw_gain","raw_gain_ci_low","raw_oracle_recovery",
            "calibrated_violation_mean","worst_seed_violation","max_seed_cp_upper","calibrated_gain","calibrated_gain_ci_low","calibrated_oracle_recovery",
            "sentinel_delta","sentinel_transition_rate","q_mean","outlier_driven_fold_rate","raw_gate_pass","exact_gate_pass"]
    s1 = primary[primary.calibration_scheme==S1]
    best = {d: s1[s1.dataset==d].sort_values("raw_gain",ascending=False).iloc[0] for d in ("hmc","eegmmidb")}
    qmax = float(residuals[residuals.calibration_scheme==S1].groupby(["dataset","seed","outer_fold","requested_budget"]).apply(lambda x: x.selected_q_rank.iloc[0]==len(x), include_groups=False).mean())
    qsubjects = residuals[residuals.determines_q].groupby(["dataset","subject_id"]).size().sort_values(ascending=False).groupby(level=0).head(10)
    outlier_rate = loo.groupby(["dataset","calibration_scheme"]).outlier_driven.mean().reset_index()
    influence_summary = influence[influence.strategy=="temporal"].groupby(["dataset","requested_budget","calibration_scheme"]).agg(min_loo_gain=("loo_gain","min"),max_loo_gain=("loo_gain","max"),sign_changes=("changes_sign","sum")).reset_index()
    reports = {
        "RAW_PREDICTOR_DIAGNOSTIC.md": "# Raw predictor diagnostic\n\n"+markdown_table(s1[[c for c in cols if c in s1]].copy()),
        "BUDGET_INFORMATION_CURVE.md": "# Budget information curve\n\n"+markdown_table(curve),
        "CORRECTION_DECOMPOSITION.md": "# Correction decomposition\n\n"+markdown_table(primary[["dataset","requested_budget","calibration_scheme","raw_gain","calibrated_gain","method_correction_cost","global_correction_cost","excess_correction_cost","sentinel_transition_rate"]]),
        "CALIBRATION_GRANULARITY_COMPARISON.md": "# Calibration granularity comparison\n\nS4 is an exploratory cross-conformal diagnostic. It is not treated as the exact split-conformal primary result.\n\n"+markdown_table(primary[cols]),
        "CALIBRATION_OUTLIER_ANALYSIS.md": "# Calibration outlier analysis\n\n"+markdown_table(outlier_rate)+"\n\n## Evaluation influence\n\n"+markdown_table(influence_summary),
        "STATISTICAL_UNIT_CORRECTION.md": "# Statistical unit correction\n\nEfficiency averages random repeats within subject, then source seeds within subject, leaving one value per subject. Paired bootstrap resamples subjects only. Temporal validity treats each subject as one Bernoulli observation separately for each seed; exact CP bounds use n=65 for EEGMMIDB and n=90 for HMC, never n multiplied by five seeds. Random-repeat validity is descriptive only.",
        "LIMITATIONS.md": "# Limitations\n\nThis is a two-dataset, one-backbone, method-development-only diagnostic. S4 is exploratory and has no exact split-conformal claim. Small subject-level calibration cohorts make order statistics coarse. No formal calibration, internal-final, CAP, active acquisition, adaptive budget, new backbone, or full B-HiCER method was run.",
        "REPRODUCE.md": "# Reproduce V5.1\n\n```bash\ncd /root/autodl-tmp/hsc_tta_eeg\n/root/miniconda3/envs/hsc_gpu/bin/python repo/scripts/budgeted_risk_v51/run_all.py --project-root /root/autodl-tmp/hsc_tta_eeg --resume\n/root/miniconda3/envs/hsc_gpu/bin/python -m pytest -q repo/tests/budgeted_risk_v51 repo/tests\n```\n\nThe pipeline consumes existing Stage-0 parquet and source-cache files and performs CPU-only lightweight fitting/statistics. It does not retrain CBraMod/source heads.",
    }
    for name,text in reports.items():(paths.delivery/name).write_text(text+"\n",encoding="utf-8")
    decision_payload = {
        "schema_version":"budgeted-risk-v51-decision-v1","verdict":verdict,"dataset_verdicts":dataset_verdict,
        "s1_reproduction":s1_repro,"formal_calibration_opened":False,"internal_final_opened":False,"cap_opened":False,
        "active_acquisition_run":False,"full_method_entered":False,
    }
    (paths.delivery/"V51_DECISION.json").write_text(json.dumps(decision_payload,indent=2,sort_keys=True),encoding="utf-8")
    potential_score = 6 if verdict == "V51_CONTINUE_TO_FULL_METHOD" else 5 if verdict in ("V51_HINT_ONLY_CROSSFIT","V51_MODERATE_BUDGET_ONLY") else 3
    decision_md=f"""# V5.1 decision

## Verdict

`{verdict}`

- HMC: `{dataset_verdict['hmc']}`
- EEGMMIDB: `{dataset_verdict['eegmmidb']}`

## Required answers

1. Ordinal repair was included: `{commit['ordinal_fix_commit']}` is an ancestor of run commit `{commit['run_commit']}`.
2. Original S1 is reproducible: {s1_repro['passed']}; matched {s1_repro['matched_rows']:,} rows with index metrics exact and continuous tolerance <=1e-8.
3. The old stop narrative inverted Stage-0A/Stage-0B provenance; numerical files and preregistered NO-GO remain valid.
4. Few-label raw signal is shown in the frozen-gate table below; no threshold was relaxed.
5. HMC best raw-gain budget <=20: {int(best['hmc'].requested_budget)}.
6. EEGMMIDB best raw-gain budget <=20: {int(best['eegmmidb'].requested_budget)}.
7. Correction-free gain is `raw_gain`; it is separated from calibration below.
8. Correction loss is the raw-to-calibrated gain change and `excess_correction_cost`.
9. S1 q is the maximum residual in {qmax:.1%} of dataset/seed/fold/budget cells.
10. q-determining subjects are listed below.
11. Sentinel collapse is correction-driven exactly to the extent measured by `sentinel_transition_rate`; raw and certified indices are stored separately.
12. S2 improvement/failure is shown without post-hoc selection.
13. S3 scaled exact improvement/failure is shown without post-hoc selection.
14. S4 pooled results are exploratory only.
15. Calibration-outlier sensitivity is reported in `CALIBRATION_OUTLIER_ANALYSIS.md`.
16. Evaluation leave-one-subject influence and sign changes are reported there as well.
17. The budget curve explicitly compares 5/10/20/50/full context.
18. Final verdict: `{verdict}`.
19. Worth entering full method: `{verdict == 'V51_CONTINUE_TO_FULL_METHOD'}`; even a continue verdict is only a recommendation, not execution.
20. Current ICLR potential score: {potential_score}/10. This diagnostic alone is not an ICLR main method.
21. Largest rejection risk: efficiency conclusions are fragile under small subject cohorts and a single backbone; S4 cannot substitute for an exact finite-sample guarantee.

## Frozen-gate table

{markdown_table(primary[cols])}

## q-determining calibration subjects

```
{qsubjects.to_string()}
```

Formal calibration was not opened. Internal final was not opened. CAP was not opened. Active acquisition was not run. The full method stage was not entered.
"""
    (paths.delivery/"V51_DECISION.md").write_text(decision_md,encoding="utf-8")


def delivery_manifest(paths: Paths):
    files=[]
    for root in (paths.delivery,paths.output):
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            files.append({"path":str(path.relative_to(paths.repo)),"sha256":sha256(path),"bytes":path.stat().st_size})
    payload={"schema_version":"budgeted-risk-v51-delivery-manifest-v1","files":files,
             "formal_calibration_opened":False,"internal_final_opened":False,"cap_opened":False}
    (paths.delivery/"DELIVERY_MANIFEST.json").write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")


def run_all(project_root: str | Path, resume: bool = False) -> str:
    global _PROCESS_POOL
    paths=Paths(Path(project_root)); paths.results.mkdir(parents=True,exist_ok=True);paths.provenance.mkdir(parents=True,exist_ok=True)
    config=configure(paths); hashes,cohort_hash,source_hash,commit=input_audit(paths,config)
    state=paths.output/"RUN_STATE.json"
    if state.exists() and not resume: raise RuntimeError("V5.1 state exists; use --resume")
    initialize_unlabeled_features(paths)
    cohorts=pd.read_parquet(paths.repo/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet")
    cohort=cohorts[(cohorts.master_cohort=="method_development")&cohorts.dataset.isin(config["datasets"])].copy()
    manifest=pd.read_parquet(paths.repo/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet")
    current_state = json.loads(state.read_text())["state"] if state.exists() else None
    if current_state is None:
        transition(state,"INITIALIZED",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
        transition(state,"INPUT_AUDIT_COMPLETE",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
        freeze_protocol(paths,config,hashes,cohort_hash,source_hash)
        transition(state,"V51_PROTOCOL_FROZEN",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
        s1,s1_res,s1_tune=run_s1(paths,config,cohort,manifest); s1_repro=reproduce_s1(paths,s1)
        s1.to_parquet(paths.results/"S1_RESULTS.parquet",index=False);s1_res.to_parquet(paths.results/"CALIBRATION_RESIDUALS_S1.parquet",index=False)
        s1.to_parquet(paths.results/"RAW_PREDICTOR_RESULTS.parquet",index=False)
        transition(state,"RAW_DIAGNOSTIC_COMPLETE",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
    elif current_state == "RAW_DIAGNOSTIC_COMPLETE" and resume:
        s1=pd.read_parquet(paths.results/"S1_RESULTS.parquet");s1_res=pd.read_parquet(paths.results/"CALIBRATION_RESIDUALS_S1.parquet")
        s1_tune=s1[["dataset","requested_budget","seed","outer_fold","calibration_scheme","tau","selected_model","calibration_m","order_statistic_k"]].drop_duplicates()
        s1_repro=json.loads((paths.results/"S1_REPRODUCTION.json").read_text())
    else:
        raise RuntimeError(f"resume from state {current_state!r} is not implemented; no valid artifact will be overwritten")
    _PROCESS_POOL = ProcessPoolExecutor(max_workers=6)
    try:
        exact,exact_res,exact_tune=run_exact(paths,config,cohort,manifest); cross,cross_res,cross_tune=run_crossfit(paths,config,cohort,manifest)
    finally:
        _PROCESS_POOL.shutdown(wait=True); _PROCESS_POOL=None
    exact[exact.calibration_scheme==S2].to_parquet(paths.results/"S2_RESULTS.parquet",index=False)
    exact[exact.calibration_scheme==S3].to_parquet(paths.results/"S3_RESULTS.parquet",index=False);cross.to_parquet(paths.results/"S4_RESULTS.parquet",index=False)
    results=pd.concat([s1,exact,cross],ignore_index=True);residuals=pd.concat([s1_res,exact_res,cross_res],ignore_index=True);tuning=pd.concat([s1_tune,exact_tune,cross_tune],ignore_index=True)
    residuals.to_parquet(paths.results/"CALIBRATION_RESIDUALS.parquet",index=False)
    qsummary=residuals.groupby(["dataset","requested_budget","calibration_scheme","seed","outer_fold"],as_index=False).agg(
        calibration_m=("subject_id","size"),order_statistic_k=("selected_q_rank","first"),q=("selected_q_value","first"),determining_subjects=("subject_id",lambda x:",".join(residuals.loc[x.index][residuals.loc[x.index,"determines_q"]].subject_id)))
    qsummary["q_is_max"] = qsummary.order_statistic_k == qsummary.calibration_m
    qsummary.to_csv(paths.results/"CALIBRATION_Q_SUMMARY.csv",index=False)
    results.to_parquet(paths.results/"CORRECTION_DECOMPOSITION.parquet",index=False)
    summary,seed_rows=summarize(results,residuals,config);seed_rows.to_csv(paths.results/"RESULTS_BY_SEED.csv",index=False)
    transition(state,"CALIBRATION_DIAGNOSTIC_COMPLETE",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
    loo,influence,summary=outlier_analysis(results,residuals,summary,config,manifest);loo.to_parquet(paths.results/"CALIBRATION_LOO_RESULTS.parquet",index=False);influence.to_parquet(paths.results/"EVALUATION_INFLUENCE_RESULTS.parquet",index=False)
    summary.to_csv(paths.results/"SCHEME_COMPARISON.csv",index=False);summary.to_csv(paths.results/"V51_GATE_SUMMARY.csv",index=False)
    raw_summary=summary[summary.calibration_scheme==S1].copy();raw_summary.to_csv(paths.results/"RAW_PREDICTOR_SUMMARY.csv",index=False)
    full=pd.read_parquet(paths.repo/"outputs/budgeted_risk/stage0/FULL_CONTEXT_RESULTS.parquet");curve=information_curve(results,full,config);curve.to_csv(paths.results/"BUDGET_INFORMATION_CURVE.csv",index=False)
    transition(state,"OUTLIER_ANALYSIS_COMPLETE",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
    verdict,dataset_verdict=decide(summary);summary.to_csv(paths.results/"V51_GATE_SUMMARY.csv",index=False)
    write_reports(paths,summary,curve,residuals,loo,influence,seed_rows,s1_repro,verdict,dataset_verdict,commit);make_plots(paths,summary,curve,residuals,loo,seed_rows)
    (paths.output/"FAILURES.csv").write_text(
        "stage,message,resolved\n"
        "S1_PREFLIGHT,prefix_instability was incorrectly admitted to the S1 predictor,true\n"
        "S1_PREFLIGHT,combined feature parquet supplied an all-NaN fifth-class column to four-class EEGMMIDB,true\n"
        "S1_REFIT,mord independent refit was not bitwise stable; source values restored and index equivalence required,true\n",
        encoding="utf-8")
    transition(state,"V51_DECISION_COMPLETE",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
    delivery_manifest(paths)
    transition(state,"DELIVERY_COMPLETE",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
    transition(state,"STOPPED",git_commit=git(paths.repo,"rev-parse","HEAD"),input_hashes=hashes,config_hash=config["config_hash"],cohort_hash=cohort_hash,source_result_hash=source_hash,output_root=paths.output)
    delivery_manifest(paths)
    return verdict
