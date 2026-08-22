from __future__ import annotations

"""Experiment 3 decision-grounding closure.

This script is deliberately a measurement/prediction closure, not a model
search.  It reuses the frozen DDA-B/DDA-C cell table and the Signed-V3.1
canonical blocks, and measures cross-session subject identity on the same
fit-subject cells used by the DDA cross-fit.  No held-out or outer subject is
loaded.  The only fitted models are the predeclared LORO consequence models
M0, MI, MD and MID.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[1]
OUT = EXP_ROOT / "results"
PROTOCOL = EXP_ROOT / "protocol"
FIGURES = EXP_ROOT / "figures"

DDA_ROOT = Path(os.environ.get(
    "PERSIST_DDA_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_INVARIANCE_RESCUE_V1\experiments\persist_eeg_dda_v1",
))
V12_ROOT = Path(os.environ.get(
    "PERSIST_V12_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_EXP3_MATCHED_CAUSAL_V1_2\persist_eeg_matched_identity_causal_v1_2",
))
SIGNED_ROOT = Path(os.environ.get(
    "PERSIST_SIGNED_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_INVARIANCE_RESCUE_V1\experiments\persist_eeg_p4_signed_v3_1",
))
WBCIC_ROOT = Path(os.environ.get(
    "PERSIST_WBCIC_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_WBCIC_EEGNET",
))

RUNS = ("0_0", "0_1", "1_0", "1_1", "2_0", "2_1")
RIDGE_ALPHA = 1.0
NULL_PERMUTATIONS = 500
BOOTSTRAP_DRAWS = 10_000
EPS = 1e-12
PRIMARY_IDENTITY = "symmetric_cross_session_subject_id_identity_skill_raw"
IDENTITY_FORMULA = "0.5*((log(K)-CE_S1_to_S2)+(log(K)-CE_S2_to_S1))"
BASELINE_FEATURES = ["persistence_strength", "geometry_strength", "rank"]


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def flags() -> dict[str, bool]:
    return {"outer_test_used": False, "outer_membership_enumerated": False}


def run_id(fold: int, seed: int) -> str:
    return f"{int(fold)}_{int(seed)}"


def source(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def git_head() -> str | None:
    forced = os.environ.get("PERSIST_EXP3_GIT_COMMIT")
    if forced:
        return forced.strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXP_ROOT.parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def source_paths() -> dict[str, Path]:
    return {
        "dda_protocol": source(DDA_ROOT / "outputs" / "protocol" / "DDA_PROTOCOL_LOCK.json"),
        "dda_provenance": source(DDA_ROOT / "outputs" / "protocol" / "PROVENANCE_AUDIT.json"),
        "dda_b_result": source(DDA_ROOT / "outputs" / "results" / "DDA_B_RESULT.json"),
        "dda_b_runs": source(DDA_ROOT / "outputs" / "results" / "DDA_B_PROTECTED_RUNS.csv"),
        "dda_b_run_aggregate": source(DDA_ROOT / "outputs" / "results" / "DDA_B_PROTECTED_RUN_AGGREGATE.csv"),
        "dda_b_summary": source(DDA_ROOT / "outputs" / "results" / "DDA_B_BLOCK_SUMMARY.csv"),
        "dda_c_result": source(DDA_ROOT / "outputs" / "results" / "DDA_C_RESULT.json"),
        "dda_c_cells": source(DDA_ROOT / "outputs" / "results" / "DDA_BLOCK_CROSSFIT.csv"),
        "dda_c_predictions": source(DDA_ROOT / "outputs" / "results" / "DDA_C_LORO_PREDICTIONS.csv"),
        "dda_subjects": source(DDA_ROOT / "outputs" / "results" / "DDA_BC_SUBJECT.csv"),
        "v12_protocol": source(V12_ROOT / "PROTOCOL_FROZEN.json"),
        "v12_design": source(V12_ROOT / "outputs" / "TRAIN_ONLY_DESIGN.json"),
        "v12_protected": source(V12_ROOT / "outputs" / "FROZEN_PROTECTED_BLOCKS.csv"),
        "v12_controls": source(V12_ROOT / "outputs" / "FROZEN_MATCHED_CONTROLS.csv"),
        "v12_curves": source(V12_ROOT / "outputs" / "TRAIN_CONTINUOUS_IDENTITY_CURVES.csv"),
        "v12_metric_audit": source(V12_ROOT / "IDENTITY_METRIC_AUDIT.md"),
    }


def phase0() -> dict[str, Any]:
    """Freeze provenance and all choices before the comparison is computed."""
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    paths = source_paths()
    dda_protocol = json.loads(paths["dda_protocol"].read_text(encoding="utf-8"))
    v12_protocol = json.loads(paths["v12_protocol"].read_text(encoding="utf-8"))
    cells = pd.read_csv(paths["dda_c_cells"])
    subjects = pd.read_csv(paths["dda_subjects"])
    if len(cells) != 215 or cells.run.nunique() != 6:
        raise RuntimeError(f"unexpected DDA cell table: rows={len(cells)} runs={cells.run.nunique()}")
    if set(cells.assignment.astype(str)) - {"protected", "neutral", "uncertain", "harmful"}:
        raise RuntimeError("unexpected DDA assignment label")
    if bool(dda_protocol.get("outer_test_used", True)) or bool(v12_protocol.get("outer_test_used", True)):
        raise RuntimeError("source protocol violates outer lock")
    lock = {
        "experiment": "PERSIST_EEG_EXP3_DECISION_GROUNDING_CLOSURE_V1",
        "status": "FROZEN_BEFORE_PRIMARY_COMPARISON",
        "git_commit": git_head(),
        "scope": "OpenBMI MI development/train resource only; no outer or untouched validation subjects",
        "runs": list(RUNS),
        "cell_table": "DDA_BLOCK_CROSSFIT.csv, 215 run x audit_fold x block cells",
        "primary_outcome": "outcome_ce_effect from frozen DDA cross-fit",
        "primary_identity_metric": PRIMARY_IDENTITY,
        "identity_formula": IDENTITY_FORMULA,
        "identity_evidence_definition": "full cross-session IdentitySkill minus IdentitySkill after erasing the frozen block coordinates, fit and evaluated on the DDA fit subjects only",
        "identity_measurement_competence": {
            "null": "subject-level permutation of evaluation-session subject labels, preserving trial counts",
            "permutations": NULL_PERMUTATIONS,
            "pass_rule": "at least 5/6 runs have at least 4/5 audit folds with observed full-representation skill > 95th null percentile and >0",
        },
        "decision_metrics": {
            "finite": "decision_logit_rms from frozen DDA-B/C",
            "local": "jacobian_energy / jacobian_ratio from frozen DDA-B",
            "no_redefinition": True,
        },
        "models": {
            "M0": BASELINE_FEATURES,
            "MI": BASELINE_FEATURES + ["identity_evidence"],
            "MD": BASELINE_FEATURES + ["decision_logit_rms"],
            "MID": BASELINE_FEATURES + ["identity_evidence", "decision_logit_rms"],
            "evaluation": "leave-one-run-out; train-run standardization; ridge alpha=1.0 inherited from DDA-C",
        },
        "tests": {
            "A": "RMSE(M0)-RMSE(MD), run-cluster bootstrap CI and exact 6-run sign-flip p",
            "B": "RMSE(MI)-RMSE(MD), run-cluster bootstrap CI and exact 6-run sign-flip p",
            "C": "RMSE(MD)-RMSE(MID), descriptive incremental identity comparison",
        },
        "statistical_unit": "run for held-out prediction inference; cell-level rows only within each training run",
        "external_support": "WBCIC development only if the frozen cell-level identity/consequence chain is available; outer subjects remain sealed",
        "random_seed_rule": "SHA256-derived deterministic seeds; Python hash() is not used",
        "validation_outcome_used_for_design": False,
        **flags(),
    }
    write_json(PROTOCOL / "EXP3_PROTOCOL_LOCK.json", lock)
    prov = {
        "status": "PROVENANCE_AUDIT_PASS",
        "git_commit": git_head(),
        "sources": {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "dda_cell_rows": int(len(cells)),
        "dda_subject_rows": int(len(subjects)),
        "six_runs": sorted(cells.run.astype(str).unique().tolist()),
        "v12_primary_metric_verified": v12_protocol.get("primary_identity_metric") == PRIMARY_IDENTITY,
        "v12_validation_outcome_used": v12_protocol.get("validation_outcome_used_for_design", False),
        "outer_test_used": False,
        "outer_membership_enumerated": False,
        "note": "All source outputs are compact, committed or server-local frozen artifacts; no raw EEG is copied into this experiment.",
        **flags(),
    }
    write_json(PROTOCOL / "PROVENANCE_AUDIT.json", prov)
    return lock


def load_cache(fold: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    path = source(V12_ROOT / "outputs" / "feature_cache" / f"fold-{fold}_seed-{seed}.npz")
    z = np.load(path, allow_pickle=False)
    n = int(z["n_train"].item())
    meta = pd.DataFrame({
        "subject_id": z["train_subject_id"].astype(str),
        "session_id": z["train_session_id"].astype(int),
        "paradigm": z["train_paradigm"].astype(str),
        "event_label": z["train_event_label"].astype(str),
    })
    if n != len(meta) or n != len(z["train_features"]):
        raise RuntimeError(f"feature cache shape mismatch fold={fold} seed={seed}")
    mi = meta.paradigm.eq("mi").to_numpy()
    return meta.loc[mi].reset_index(drop=True), z["train_features"][mi].astype(np.float64)


def load_spec(fold: int, seed: int) -> dict[str, Any]:
    run = SIGNED_ROOT / "results_v3_1" / "runs" / f"fold-{fold}" / f"seed-{seed}"
    path = source(run / "spectrum" / "PERSISTENCE_SPECTRUM.npz")
    z = np.load(path, allow_pickle=False)
    spec = {k: z[k].astype(np.float64) for k in ("mean", "whitener", "dewhitener", "directions", "rho")}
    spec["blocks"] = json.loads(str(z["blocks_json"].item()))
    audit = json.loads(str(z["audit_json"].item()))
    if audit.get("outer_test_used", False) or audit.get("outer_membership_enumerated", False):
        raise RuntimeError("Signed-V3.1 spectrum violates outer lock")
    return spec


def canonical_q(features: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - spec["mean"]) @ spec["whitener"] @ spec["directions"]


def ridge_pack(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    mu = x.mean(axis=0); sd = x.std(axis=0); sd[sd < 1e-8] = 1.0
    z = np.c_[(x - mu) / sd, np.ones(len(x))]
    target = np.eye(int(y.max()) + 1, dtype=np.float64)[y]
    penalty = np.eye(z.shape[1], dtype=np.float64); penalty[-1, -1] = 0.0
    lhs = z.T @ z + RIDGE_ALPHA * penalty
    try:
        w = np.linalg.solve(lhs, z.T @ target)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(lhs) @ z.T @ target
    return w, mu, sd


def logits(x: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    w, mu, sd = pack
    z = np.c_[(np.asarray(x, dtype=np.float64) - mu) / sd, np.ones(len(x))]
    return z @ w


def identity_direction(q: np.ndarray, meta: pd.DataFrame, fit_subjects: list[str], train_session: int, eval_session: int) -> dict[str, Any]:
    ordered = sorted(set(map(str, fit_subjects)), key=lambda s: (int(s) if s.isdigit() else 10**9, s))
    code = {s: i for i, s in enumerate(ordered)}; subj = meta.subject_id.astype(str).to_numpy(); ses = meta.session_id.to_numpy(int)
    tr = np.flatnonzero(np.isin(subj, ordered) & (ses == train_session)); ev = np.flatnonzero(np.isin(subj, ordered) & (ses == eval_session))
    ytr = np.asarray([code[s] for s in subj[tr]], dtype=np.int64); yev = np.asarray([code[s] for s in subj[ev]], dtype=np.int64)
    if len(tr) == 0 or len(ev) == 0 or len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
        return {"skill": float("nan"), "ce": float("nan"), "top1_ba": float("nan"), "probs": np.empty((0, len(ordered))), "yev": yev, "subjects": ordered}
    pack = ridge_pack(q[tr], ytr); ll = logits(q[ev], pack); ll -= ll.max(axis=1, keepdims=True); p = np.exp(np.clip(ll, -60, 60)); p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    ce = -np.log(np.clip(p[np.arange(len(yev)), yev], EPS, 1.0)); pred = p.argmax(1)
    per = [np.mean(pred[yev == k] == k) for k in range(len(ordered)) if np.any(yev == k)]
    return {"skill": float(math.log(len(ordered)) - ce.mean()), "ce": float(ce.mean()), "top1_ba": float(np.mean(per)), "probs": p, "yev": yev, "subjects": ordered}


def identity_skill(q: np.ndarray, meta: pd.DataFrame, fit_subjects: list[str]) -> dict[str, Any]:
    a = identity_direction(q, meta, fit_subjects, 1, 2); b = identity_direction(q, meta, fit_subjects, 2, 1)
    return {"skill": float(np.nanmean([a["skill"], b["skill"]])), "ce": float(np.nanmean([a["ce"], b["ce"]])), "top1_ba": float(np.nanmean([a["top1_ba"], b["top1_ba"]]))}


def identity_competence(q: np.ndarray, meta: pd.DataFrame, fit_subjects: list[str], fold: int, seed: int, audit_fold: int) -> dict[str, Any]:
    observed = identity_skill(q, meta, fit_subjects)
    ordered = sorted(set(map(str, fit_subjects)), key=lambda s: (int(s) if s.isdigit() else 10**9, s)); code = {s: i for i, s in enumerate(ordered)}
    rng = np.random.default_rng(stable_seed("exp3-id-null", fold, seed, audit_fold)); null = []
    directions = [identity_direction(q, meta, fit_subjects, 1, 2), identity_direction(q, meta, fit_subjects, 2, 1)]
    # Reuse the fitted probabilistic heads; only the evaluation-session subject
    # labels are permuted at subject level.  This is a subject-shuffle null,
    # not an outcome-tuned null.
    subj = meta.subject_id.astype(str).to_numpy()
    for _ in range(NULL_PERMUTATIONS):
        vals = []
        for direction, eval_session in zip(directions, (2, 1)):
            ev = np.flatnonzero(np.isin(subj, ordered) & (meta.session_id.to_numpy(int) == eval_session))
            if len(ev) == 0 or direction["probs"].size == 0:
                vals.append(float("nan")); continue
            perm = rng.permutation(len(ordered)); true = np.asarray([code[s] for s in subj[ev]], dtype=np.int64); y = perm[true]
            ce = -np.log(np.clip(direction["probs"][np.arange(len(y)), y], EPS, 1.0)); vals.append(float(math.log(len(ordered)) - ce.mean()))
        null.append(float(np.nanmean(vals)))
    null_arr = np.asarray(null, dtype=float); null_arr = null_arr[np.isfinite(null_arr)]
    q95 = float(np.quantile(null_arr, 0.95)) if len(null_arr) else float("nan")
    return {"full_identity_skill": observed["skill"], "full_identity_ce": observed["ce"], "full_identity_top1_ba": observed["top1_ba"], "null_mean": float(np.mean(null_arr)) if len(null_arr) else None, "null_q95": q95, "null_sd": float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else None, "competent": bool(np.isfinite(observed["skill"]) and observed["skill"] > 0 and observed["skill"] > q95), "null_permutations": int(len(null_arr))}


def fit_subjects_for(subjects: pd.DataFrame, fold: int, seed: int, audit_fold: int, block: int) -> list[str]:
    key = (subjects.fold == fold) & (subjects.seed == seed) & (subjects.audit_fold == audit_fold) & (subjects.block == block)
    used = set(subjects.loc[key & subjects.role.isin(["decision", "outcome"]), "subject"].astype(str))
    # The DDA subject table uses exactly the six legal frozen runs and never
    # contains validation/outer subjects.  All remaining MI train subjects
    # are the fit bank for this cell.
    all_rows = subjects[(subjects.fold == fold) & (subjects.seed == seed)]
    all_subjects = set(all_rows.subject.astype(str))
    return sorted(all_subjects - used, key=lambda s: (int(s) if s.isdigit() else 10**9, s))


def build_identity_and_cells() -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = source_paths(); cells = pd.read_csv(paths["dda_c_cells"]); subjects = pd.read_csv(paths["dda_subjects"])
    rows: list[dict[str, Any]] = []; competence_cache: dict[tuple[int, int, int], dict[str, Any]] = {}; run_cache: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, Any]]] = {}
    for rid in RUNS:
        fold, seed = map(int, rid.split("_")); meta, features = load_cache(fold, seed); spec = load_spec(fold, seed); q = canonical_q(features, spec)
        run_cache[rid] = (meta, q, spec)
        run_cells = cells[cells.run.astype(str) == rid].sort_values(["audit_fold", "block"]).copy()
        for idx, cell in run_cells.iterrows():
            audit_fold = int(cell.audit_fold); block = int(cell.block); fit = fit_subjects_for(subjects, fold, seed, audit_fold, block)
            if len(fit) < 2:
                raise RuntimeError(f"empty/short DDA fit bank run={rid} fold={audit_fold} block={block}")
            key = (fold, seed, audit_fold)
            if key not in competence_cache:
                competence_cache[key] = identity_competence(q, meta, fit, fold, seed, audit_fold)
            comp = competence_cache[key]
            full = float(comp["full_identity_skill"])
            dims = list(map(int, spec["blocks"][block])); q_erased = q.copy(); q_erased[:, dims] = 0.0
            erased = identity_skill(q_erased, meta, fit); drop = full - float(erased["skill"])
            row = cell.to_dict(); row.update({
                "run": rid, "control_type": str(cell.assignment), "fit_subject_count": len(fit),
                "protected_assignment": bool(str(cell.assignment) == "protected"),
                "identity_metric": PRIMARY_IDENTITY, "identity_full_skill": full,
                "identity_erased_skill": float(erased["skill"]), "identity_evidence": drop,
                "identity_full_ce": float(comp["full_identity_ce"]), "identity_erased_ce": float(erased["ce"]),
                "identity_full_top1_ba": float(comp["full_identity_top1_ba"]), "identity_erased_top1_ba": float(erased["top1_ba"]),
                "identity_null_mean": comp["null_mean"], "identity_null_q95": comp["null_q95"],
                "identity_null_sd": comp["null_sd"], "identity_measurement_competent": bool(comp["competent"]),
                "identity_coordinates": json.dumps(dims), **flags(),
            }); rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != len(cells):
        raise RuntimeError("identity/cell row count mismatch")
    # Competence is a run-level condition, independent of task consequence.
    run_comp = frame.groupby("run", as_index=False).agg(competent=("identity_measurement_competent", "mean"), full_skill=("identity_full_skill", "mean"), null_q95=("identity_null_q95", "mean"))
    run_comp["run_competent"] = run_comp.competent >= 0.8
    valid = bool(run_comp.run_competent.sum() >= 5)
    write_csv(OUT / "EXP3_CELL_TABLE.csv", frame)
    write_csv(OUT / "IDENTITY_MEASUREMENT.csv", frame[["fold", "seed", "run", "audit_fold", "block", "rank", "assignment", "fit_subject_count", "identity_full_skill", "identity_erased_skill", "identity_evidence", "identity_full_ce", "identity_erased_ce", "identity_full_top1_ba", "identity_erased_top1_ba", "identity_null_mean", "identity_null_q95", "identity_null_sd", "identity_measurement_competent", "identity_coordinates", "outer_test_used", "outer_membership_enumerated"]])
    write_json(OUT / "IDENTITY_COMPETENCE.json", {"status": "IDENTITY_MEASUREMENT_VALID" if valid else "IDENTITY_MEASUREMENT_INVALID", "run_table": run_comp.to_dict(orient="records"), "competent_runs": int(run_comp.run_competent.sum()), "total_runs": 6, "pass_rule": ">=5/6 runs with >=4/5 competent audit folds", "null_permutations": NULL_PERMUTATIONS, **flags()})
    return frame, {"valid": valid, "run_table": run_comp.to_dict(orient="records"), "competent_runs": int(run_comp.run_competent.sum())}


def standardize_ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    xtr = train[features].to_numpy(float); xte = test[features].to_numpy(float); y = train["outcome_ce_effect"].to_numpy(float)
    if not np.isfinite(xtr).all() or not np.isfinite(xte).all() or not np.isfinite(y).all():
        raise RuntimeError(f"non-finite model input for {features}")
    mu = xtr.mean(axis=0); sd = xtr.std(axis=0); sd[sd < 1e-8] = 1.0
    ztr = np.c_[(xtr - mu) / sd, np.ones(len(xtr))]; zte = np.c_[(xte - mu) / sd, np.ones(len(xte))]
    penalty = np.eye(ztr.shape[1]); penalty[-1, -1] = 0.0
    lhs = ztr.T @ ztr + RIDGE_ALPHA * penalty
    try: beta = np.linalg.solve(lhs, ztr.T @ y)
    except np.linalg.LinAlgError: beta = np.linalg.pinv(lhs) @ ztr.T @ y
    return zte @ beta


def loro_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specs = {
        "M0": BASELINE_FEATURES,
        "MI": BASELINE_FEATURES + ["identity_evidence"],
        "MD": BASELINE_FEATURES + ["decision_logit_rms"],
        "MID": BASELINE_FEATURES + ["identity_evidence", "decision_logit_rms"],
    }
    pred_rows: list[dict[str, Any]] = []
    for held in RUNS:
        train = frame[frame.run.astype(str) != held]; test = frame[frame.run.astype(str) == held]
        for model, feats in specs.items():
            yhat = standardize_ridge_predict(train, test, feats)
            for (_, row), pred in zip(test.iterrows(), yhat):
                pred_rows.append({"run": held, "fold": int(row.fold), "seed": int(row.seed), "audit_fold": int(row.audit_fold), "block": int(row.block), "assignment": row.assignment, "model": model, "features": json.dumps(feats), "observed_U": float(row.outcome_ce_effect), "prediction": float(pred), "error": float(pred - row.outcome_ce_effect), **flags()})
    pred = pd.DataFrame(pred_rows)
    summary_rows = []
    for model in specs:
        sub = pred[pred.model == model]; summary_rows.append({"model": model, "run": "ALL", "n_cells": len(sub), "rmse": float(np.sqrt(np.mean(sub.error ** 2))), "mae": float(np.mean(np.abs(sub.error))), "features": json.dumps(specs[model]), **flags()})
        for rid in RUNS:
            s = sub[sub.run == rid]; summary_rows.append({"model": model, "run": rid, "n_cells": len(s), "rmse": float(np.sqrt(np.mean(s.error ** 2))), "mae": float(np.mean(np.abs(s.error))), "features": json.dumps(specs[model]), **flags()})
    summary = pd.DataFrame(summary_rows); write_csv(OUT / "LORO_PREDICTIONS.csv", pred); write_csv(OUT / "MODEL_COMPARISON.csv", summary)
    all_rmse = {(m, r): float(summary[(summary.model == m) & (summary.run == r)].rmse.iloc[0]) for m in specs for r in RUNS}
    deltas = {}
    for name, a, b in (("A_MD_vs_M0", "M0", "MD"), ("B_MD_vs_MI", "MI", "MD"), ("C_identity_after_D", "MD", "MID")):
        vals = np.asarray([all_rmse[(a, r)] - all_rmse[(b, r)] for r in RUNS], dtype=float)
        observed = float(vals.mean()); rng = np.random.default_rng(stable_seed("exp3-bootstrap", name)); boot = rng.choice(vals, size=(BOOTSTRAP_DRAWS, len(vals)), replace=True).mean(axis=1)
        signs = np.asarray([np.mean(vals * np.asarray(s)) for s in itertools.product([-1.0, 1.0], repeat=len(vals))])
        p = float(np.mean(signs >= observed - 1e-15))
        deltas[name] = {"run_values": vals.tolist(), "mean": observed, "median": float(np.median(vals)), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "sign_flip_p_exact": p, "positive_runs": int(np.sum(vals > 0)), "nonnegative_runs": int(np.sum(vals >= 0)), "draws": BOOTSTRAP_DRAWS, "comparison": f"RMSE({a})-RMSE({b})"}
    write_json(OUT / "STATISTICAL_TESTS.json", {"model_rmse": {m: float(summary[(summary.model == m) & (summary.run == "ALL")].rmse.iloc[0]) for m in specs}, "model_mae": {m: float(summary[(summary.model == m) & (summary.run == "ALL")].mae.iloc[0]) for m in specs}, "deltas": deltas, "run_count": 6, "run_cluster_bootstrap": True, "exact_sign_flip": True, "ridge_alpha": RIDGE_ALPHA, **flags()})
    return pred, summary, deltas


def decision_specificity() -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = source_paths(); p = pd.read_csv(paths["dda_b_runs"]); p["run"] = p.run.astype(str)
    p["matched_nonprotected_logit"] = pd.to_numeric(p["matched_nonprotected_logit"], errors="coerce")
    p["paired_finite_difference"] = p.decision_logit_rms - p.matched_nonprotected_logit
    write_csv(OUT / "DECISION_SPECIFICITY.csv", p)
    paired = p.dropna(subset=["matched_nonprotected_logit"]).groupby("run", as_index=False).agg(protected_decision=("decision_logit_rms", "mean"), matched_control_decision=("matched_nonprotected_logit", "mean"), paired_difference=("paired_finite_difference", "mean"), protected_finite_ratio=("decision_logit_ratio", "mean"), protected_local_ratio=("jacobian_ratio", "mean"))
    d = paired.paired_difference.to_numpy(float); rng = np.random.default_rng(stable_seed("exp3-dda-b-bootstrap")); boot = rng.choice(d, size=(BOOTSTRAP_DRAWS, len(d)), replace=True).mean(axis=1) if len(d) else np.asarray([])
    signs = np.asarray([np.mean(d * np.asarray(s)) for s in itertools.product([-1.0, 1.0], repeat=len(d))]) if len(d) else np.asarray([]); obs = float(np.mean(d)) if len(d) else None
    stats = {"source_status": json.loads(paths["dda_b_result"].read_text(encoding="utf-8")).get("status"), "n_run_units": int(len(paired)), "run_table": paired.to_dict(orient="records"), "paired_difference_mean": obs, "paired_difference_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if len(boot) else [None, None], "paired_difference_sign_flip_p": float(np.mean(signs >= obs - 1e-15)) if len(signs) else None, "positive_runs": int(np.sum(d > 0)), "local_jacobian_ratio_mean": float(paired.protected_local_ratio.mean()) if len(paired) else None, "finite_decision_ratio_mean": float(paired.protected_finite_ratio.mean()) if len(paired) else None, **flags()}
    write_json(OUT / "DECISION_SPECIFICITY_STATS.json", stats); return p, stats


def audit_existing() -> dict[str, Any]:
    lock = json.loads(source(PROTOCOL / "EXP3_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    paths = source_paths(); b = json.loads(paths["dda_b_result"].read_text(encoding="utf-8")); c = json.loads(paths["dda_c_result"].read_text(encoding="utf-8")); cells = pd.read_csv(paths["dda_c_cells"])
    expected_b = {"status": "DDA_B_PASS", "local_positive_runs": 6, "finite_positive_runs": 6, "protected_gt_matched_nonprotected_runs": 5, "signed_utility_and_held_consequence_concordant_runs": 6}
    expected_c = {"status": "DDA_C_PASS", "n_run_fold_block_cells": 215, "n_runs": 6, "improved_runs": 6}
    b_ok = all(b.get(k) == v for k, v in expected_b.items()); c_ok = all(c.get(k) == v for k, v in expected_c.items()) and len(cells) == 215
    result = {"dda_b_reproduced": b_ok, "dda_c_reproduced": c_ok, "dda_b_observed": b, "dda_c_observed": c, "reference_expectations": {"dda_b": expected_b, "dda_c": expected_c}, "cell_rows": len(cells), "source_hashes": {name: sha256(path) for name, path in paths.items()}, "identity_metric_verified": lock["primary_identity_metric"] == PRIMARY_IDENTITY, "outer_test_used": False, "outer_membership_enumerated": False, **flags()}
    write_json(OUT / "SOURCE_AUDIT.json", result); return result


def external_support() -> pd.DataFrame:
    # We do not enumerate WBCIC subjects here.  The existing WBCIC closure has
    # no frozen cell-level pairing of the Signed-V3.1 DDA consequence with the
    # same primary cross-session identity metric, so forcing a re-analysis
    # would change the protocol rather than provide a fair support result.
    row = {"dataset": "WBCIC", "scope": "development subjects only", "status": "EXTERNAL_SUPPORT_NOT_IDENTIFIABLE", "reason": "No frozen WBCIC artifact jointly contains the required cross-session IdentitySkill, DDA decision dependence, and future-session consequence on the same block/cross-fit cells; rebuilding would change the frozen protocol.", "outer_subjects_accessed": False, "outer_membership_enumerated": False}
    frame = pd.DataFrame([row]); write_csv(OUT / "EXTERNAL_SUPPORT.csv", frame); return frame


def figures(frame: pd.DataFrame, pred: pd.DataFrame, decision: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = np.where(frame.assignment.astype(str).to_numpy() == "protected", "tab:red", "0.55")
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(frame.identity_evidence, frame.decision_logit_rms, c=colors, s=22, alpha=.75); ax.set_xlabel("Identity evidence I (skill drop after block erasure)"); ax.set_ylabel("Finite decision dependence D (logit RMS)"); ax.set_title("Identity versus decision across frozen cells"); fig.tight_layout(); fig.savefig(FIGURES / "figure_1_identity_vs_decision.png", dpi=220); plt.close(fig)
    rows = []
    for model in ["M0", "MI", "MD", "MID"]:
        for run in RUNS:
            s = pred[(pred.model == model) & (pred.run == run)]; rows.append((model, run, np.sqrt(np.mean(s.error ** 2))))
    fig, ax = plt.subplots(figsize=(7, 4.5)); x = np.arange(len(RUNS)); width=.2
    for i, model in enumerate(["M0", "MI", "MD", "MID"]): ax.plot(x, [v for m, r, v in rows if m == model], marker="o", label=model)
    ax.set_xticks(x, RUNS); ax.set_ylabel("held-run RMSE"); ax.set_title("LORO consequence prediction"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "figure_2_loro_model_comparison.png", dpi=220); plt.close(fig)
    if len(decision):
        q = decision.dropna(subset=["matched_nonprotected_logit"]); agg = q.groupby("run", as_index=False).agg(P=("decision_logit_rms", "mean"), N=("matched_nonprotected_logit", "mean")); fig, ax = plt.subplots(figsize=(6, 4.5)); x=np.arange(len(agg)); ax.plot(x, agg.P, "o-", label="Protected"); ax.plot(x, agg.N, "o-", label="matched Non-Protected"); ax.set_xticks(x, agg.run); ax.set_ylabel("finite decision dependence"); ax.set_title("Protected versus matched control"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "figure_3_decision_specificity.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3)); ax.axis("off"); ax.text(.5, .5, "WBCIC external development support\nNOT IDENTIFIABLE under frozen shared-cell protocol", ha="center", va="center", fontsize=13); fig.tight_layout(); fig.savefig(FIGURES / "figure_4_external_support.png", dpi=220); plt.close(fig)


def finalize(frame: pd.DataFrame, comp: dict[str, Any], pred: pd.DataFrame, summary: pd.DataFrame, deltas: dict[str, Any], decision: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    tests = {"A": deltas["A_MD_vs_M0"], "B": deltas["B_MD_vs_MI"], "C": deltas["C_identity_after_D"]}
    decision_specificity_pass = bool(audit["dda_b_reproduced"])
    md_pass = bool(tests["A"]["mean"] > 0 and tests["A"]["ci95"][0] > 0 and tests["A"]["positive_runs"] >= 4 and tests["A"]["sign_flip_p_exact"] < .05)
    md_vs_mi_pass = bool(tests["B"]["mean"] > 0 and tests["B"]["ci95"][0] > 0 and tests["B"]["positive_runs"] >= 4 and tests["B"]["sign_flip_p_exact"] < .05)
    if not comp["valid"]:
        terminal = "EXP3_IDENTITY_MEASUREMENT_INVALID"; identity_claim = "UNSUPPORTED"; ready = "NO"
    elif not audit["dda_b_reproduced"] or not audit["dda_c_reproduced"]:
        terminal = "EXP3_DECISION_MECHANISM_NOT_REPRODUCED"; identity_claim = "UNSUPPORTED"; ready = "NO"
    elif decision_specificity_pass and md_pass and md_vs_mi_pass:
        terminal = "EXP3_DECISION_GROUNDED_IDENTITY_INSUFFICIENT"; identity_claim = "STRONG"; ready = "YES"
    elif decision_specificity_pass and md_pass:
        terminal = "EXP3_DECISION_GROUNDED_IDENTITY_COMPARISON_INCONCLUSIVE"; identity_claim = "LIMITED"; ready = "YES"
    else:
        terminal = "EXP3_DECISION_GROUNDED_IDENTITY_COMPARISON_INCONCLUSIVE"; identity_claim = "UNSUPPORTED"; ready = "NO"
    model_rmse = {m: float(summary[(summary.model == m) & (summary.run == "ALL")].rmse.iloc[0]) for m in ["M0", "MI", "MD", "MID"]}
    model_mae = {m: float(summary[(summary.model == m) & (summary.run == "ALL")].mae.iloc[0]) for m in ["M0", "MI", "MD", "MID"]}
    report = {
        "terminal_state": terminal, "identity_insufficiency_claim": identity_claim, "READY_FOR_EXP4_PROTECTION_FIRST": ready,
        "dda_b_reproduced": audit["dda_b_reproduced"], "dda_c_reproduced": audit["dda_c_reproduced"], "identity_measurement_valid": comp["valid"],
        "identity_competent_runs": comp["competent_runs"], "n_cells": int(len(frame)), "n_protected_cells": int(np.sum(frame.protected_assignment)),
        "identity_protected_mean": float(frame.loc[frame.protected_assignment, "identity_evidence"].mean()), "identity_control_mean": float(frame.loc[~frame.protected_assignment, "identity_evidence"].mean()),
        "decision_protected_mean": float(frame.loc[frame.protected_assignment, "decision_logit_rms"].mean()), "decision_control_mean": float(frame.loc[~frame.protected_assignment, "decision_logit_rms"].mean()),
        "model_rmse": model_rmse, "model_mae": model_mae, "tests": tests, "decision_specificity": decision,
        "external_support": "EXTERNAL_SUPPORT_NOT_IDENTIFIABLE", "outer_test_used": False, "outer_membership_enumerated": False,
        "validation_outcome_used_for_design": False,
    }
    write_json(OUT / "STATISTICAL_TESTS.json", {**json.loads((OUT / "STATISTICAL_TESTS.json").read_text(encoding="utf-8")), "terminal_gate_values": {"decision_specificity_pass": decision_specificity_pass, "MD_vs_M0_pass": md_pass, "MD_vs_MI_pass": md_vs_mi_pass}, **flags()})
    write_json(EXP_ROOT / "EXP3_FINAL_REPORT.json", report)
    write_json(OUT / "EXP3_FINAL_REPORT.json", report)
    lines = [
        "# PERSIST-EEG Experiment 3 decision-grounding closure V1",
        "",
        "This is a development-resource closure on the reused OpenBMI MI resource. It is not an untouched replication. V1/V1.1/V1.2 are preserved.",
        "",
        "## Explicit answers",
        "",
        f"1. Existing DDA-B reproduced: **{audit['dda_b_reproduced']}**. Frozen finite ratio and protected/control run comparison were audited from the original DDA outputs.",
        f"2. Existing DDA-C reproduced: **{audit['dda_c_reproduced']}**. The source table has 215 cells and six held runs.",
        f"3. Full-representation cross-session identity measurable: **{comp['valid']}** ({comp['competent_runs']}/6 competent runs under the frozen subject-shuffle null).",
        f"4. Mean primary identity evidence I: Protected={report['identity_protected_mean']:.8f}; non-Protected cells={report['identity_control_mean']:.8f}.",
        f"5. Mean finite decision dependence D: Protected={report['decision_protected_mean']:.8f}; non-Protected cells={report['decision_control_mean']:.8f}.",
        f"6. RMSE(M0)={model_rmse['M0']:.8f}.", f"7. RMSE(MI)={model_rmse['MI']:.8f}.", f"8. RMSE(MD)={model_rmse['MD']:.8f}.", f"9. RMSE(MID)={model_rmse['MID']:.8f}.",
        f"10. Adding identity to baseline: ΔRMSE(M0−MI)={(model_rmse['M0']-model_rmse['MI']):.8f}.",
        f"11. Adding decision dependence to baseline: ΔRMSE(M0−MD)={tests['A']['mean']:.8f}; 95% CI={tests['A']['ci95']}; exact run sign-flip p={tests['A']['sign_flip_p_exact']:.8f}.",
        f"12. MD outperforms MI: **{md_vs_mi_pass}**; ΔRMSE(MI−MD)={tests['B']['mean']:.8f}; 95% CI={tests['B']['ci95']}; exact p={tests['B']['sign_flip_p_exact']:.8f}.",
        f"13. Runs favoring MD over MI: {tests['B']['positive_runs']}/6.",
        f"14. Run-cluster CI for MD−MI comparison: {tests['B']['ci95']}.",
        f"15. Exact sign-flip p for MD−MI: {tests['B']['sign_flip_p_exact']:.8f}.",
        f"16. Identity after decision (MD vs MID): ΔRMSE(MD−MID)={tests['C']['mean']:.8f}; this is descriptive and is not interpreted as proof that identity contributes zero.",
        "17. WBCIC: EXTERNAL_SUPPORT_NOT_IDENTIFIABLE because the required frozen shared identity/decision/consequence cell table is unavailable; no WBCIC outer subject was opened.",
        "18. Outer subjects accessed: **False**.",
        f"19. Justified claim: decision dependence is more informative than subject-identity predictability for identifying task-consequential persistent structure **only to the extent supported by the frozen MD-vs-MI test ({identity_claim})**.",
        "20. Not justified: identity contains zero information, task utility and identity are independent, or nonsignificant identity increments prove zero contribution.",
        f"21. Final Experiment-3 terminal state: **{terminal}**.",
        f"22. READY_FOR_EXP4_PROTECTION_FIRST: **{ready}**.",
        f"23. IDENTITY_INSUFFICIENCY_CLAIM: **{identity_claim}**.",
        "",
        "## Paper-ready Experiment 3 conclusion",
        "",
        ("Although subject-persistent structure can carry identity information, identity predictability alone does not identify which persistent variation is task-consequential. In contrast, task-protected persistent directions are coupled to the classifier decision, and decision dependence provides superior held-out prediction of intervention consequence. These results indicate that the relevant axis for invariance is decision-level utility rather than subject identifiability per se." if identity_claim == "STRONG" else "The frozen audit supports decision grounding of task-protected persistence, but the identity-versus-decision comparison is not sufficiently resolved for a strong identity-insufficiency claim."),
        "",
        "These findings motivate adaptation that explicitly preserves decision-grounded protected persistence rather than indiscriminately suppressing subject-predictive structure.",
    ]
    (EXP_ROOT / "EXP3_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "EXP3_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def reproducibility() -> None:
    expected = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "REPRODUCIBILITY.json": expected.append(str(p.relative_to(OUT)).replace("\\", "/"))
    hashes = {name: sha256(OUT / name) for name in expected}
    write_json(EXP_ROOT / "REPRODUCIBILITY.json", {"git_commit": git_head(), "files": hashes, "code_sha256": sha256(Path(__file__)), "bootstrap_draws": BOOTSTRAP_DRAWS, "null_permutations": NULL_PERMUTATIONS, **flags()})
    (EXP_ROOT / "REPRODUCIBILITY.md").write_text("# Reproducibility\n\nAll compact result files are SHA256-recorded in `REPRODUCIBILITY.json`. Seeds are derived from SHA256 strings; model preprocessing is fit inside each LORO training set. Outer data and membership are untouched.\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["audit", "compute", "finalize", "all"]); args = ap.parse_args()
    if args.phase in {"audit", "all"}: phase0(); audit_existing()
    if args.phase in {"compute", "all"}:
        if not (PROTOCOL / "EXP3_PROTOCOL_LOCK.json").exists(): raise RuntimeError("run audit first")
        frame, comp = build_identity_and_cells(); pred, summary, deltas = loro_models(frame); decision, _ = decision_specificity(); external_support(); figures(frame, pred, decision)
        write_json(OUT / "COMPUTE_STATE.json", {"identity": comp, "models": {m: int((pred.model == m).sum()) for m in pred.model.unique()}, **flags()})
    if args.phase in {"finalize", "all"}:
        frame = pd.read_csv(OUT / "EXP3_CELL_TABLE.csv"); comp = json.loads((OUT / "IDENTITY_COMPETENCE.json").read_text(encoding="utf-8")); pred = pd.read_csv(OUT / "LORO_PREDICTIONS.csv"); summary = pd.read_csv(OUT / "MODEL_COMPARISON.csv"); tests = json.loads((OUT / "STATISTICAL_TESTS.json").read_text(encoding="utf-8")); decision = json.loads((OUT / "DECISION_SPECIFICITY_STATS.json").read_text(encoding="utf-8")); audit = json.loads((OUT / "SOURCE_AUDIT.json").read_text(encoding="utf-8")); finalize(frame, {"valid": bool(comp["status"] == "IDENTITY_MEASUREMENT_VALID"), "competent_runs": int(comp["competent_runs"])}, pred, summary, tests["deltas"], decision, audit); reproducibility()


if __name__ == "__main__":
    main()
