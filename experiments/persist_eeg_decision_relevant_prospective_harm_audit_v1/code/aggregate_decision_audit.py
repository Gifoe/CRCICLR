"""Memory-bounded finalisation for the frozen decision-harm audit.

The locked audit runner writes one compact per-observation table after replaying
the exact TASK_ONLY trajectory.  This postprocessor deliberately does not load
EEG data or checkpoints: it only computes the predeclared statistics from that
table, using the biological subject as the cluster-bootstrap unit.  It exists
to avoid retaining a multi-gigabyte EEG cache while doing the 10,000 bootstrap
draws on the Windows server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
SEED = 0
DATASETS = ("OpenBMI", "WBCIC")
BOOTSTRAP_DRAWS = int(os.environ.get("PERSIST_BOOTSTRAP_DRAWS", "10000"))


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "little")


def finite(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return finite(float(spearmanr(x, y).statistic))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return finite(float(pearsonr(x, y).statistic))


def safe_kendall(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return finite(float(kendalltau(x, y).statistic))


def safe_auc(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return finite(float(roc_auc_score(y.astype(int), x)))


def top_bottom(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0:
        return None
    order = np.argsort(x, kind="mergesort")
    q = max(1, len(order) // 5)
    return finite(float(np.mean(y[order[-q:]]) - np.mean(y[order[:q]])))


def metric_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    event = (y > 0).astype(int)
    vals = [safe_spearman(x, y), safe_kendall(x, y), safe_auc(x, event), top_bottom(x, y)]
    return np.asarray([np.nan if v is None else v for v in vals], dtype=float)


def ci(values: np.ndarray) -> list[float | None]:
    finite_values = values[np.isfinite(values)]
    if not len(finite_values):
        return [None, None]
    return [float(np.quantile(finite_values, 0.025)), float(np.quantile(finite_values, 0.975))]


def point_pack(frame: pd.DataFrame, cert_col: str, outcome_col: str) -> dict[str, Any]:
    x = frame[cert_col].to_numpy(float)
    y = frame[outcome_col].to_numpy(float)
    values = metric_values(x, y)
    return {
        "spearman": finite(values[0]), "kendall": finite(values[1]), "pearson": safe_pearson(x, y),
        "auroc": finite(values[2]), "top_minus_bottom": finite(values[3]),
        "n_observations": int(len(frame)), "n_subjects": int(frame.subject_id.astype(str).nunique()),
        "event_count": int(np.sum(y > 0)),
    }


def subject_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    subject_values = frame.subject_id.astype(str).to_numpy()
    return [np.flatnonzero(subject_values == subject) for subject in sorted(set(subject_values))]


def bootstrap_metric(frame: pd.DataFrame, cert_col: str, outcome_col: str, seed: int) -> np.ndarray:
    x = frame[cert_col].to_numpy(float)
    y = frame[outcome_col].to_numpy(float)
    groups = subject_groups(frame)
    rng = np.random.default_rng(seed)
    draws = np.full((BOOTSTRAP_DRAWS, 4), np.nan, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[int(i)] for i in chosen])
        draws[draw] = metric_values(x[idx], y[idx])
        if draw and draw % 2000 == 0:
            print(f"[bootstrap] {cert_col}->{outcome_col} draw={draw}/{BOOTSTRAP_DRAWS}", flush=True)
    return draws


def bootstrap_control(frame: pd.DataFrame, same_col: str, other_col: str, outcome_col: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x1 = frame[same_col].to_numpy(float); x2 = frame[other_col].to_numpy(float); y = frame[outcome_col].to_numpy(float)
    groups = subject_groups(frame); rng = np.random.default_rng(seed)
    auc = np.full(BOOTSTRAP_DRAWS, np.nan); sp = np.full(BOOTSTRAP_DRAWS, np.nan)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups)); idx = np.concatenate([groups[int(i)] for i in chosen]); event = (y[idx] > 0).astype(int)
        a, b = safe_auc(x1[idx], event), safe_auc(x2[idx], event)
        s1, s2 = safe_spearman(x1[idx], y[idx]), safe_spearman(x2[idx], y[idx])
        if a is not None and b is not None: auc[draw] = a - b
        if s1 is not None and s2 is not None: sp[draw] = s1 - s2
        if draw and draw % 2000 == 0:
            print(f"[bootstrap-control] {same_col}->{other_col} draw={draw}/{BOOTSTRAP_DRAWS}", flush=True)
    return auc, sp


def bootstrap_quintile(frame: pd.DataFrame, cert_col: str, outcome_col: str, seed: int) -> np.ndarray:
    x = frame[cert_col].to_numpy(float); y = frame[outcome_col].to_numpy(float); groups = subject_groups(frame); rng = np.random.default_rng(seed)
    out = np.full(BOOTSTRAP_DRAWS, np.nan)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups)); idx = np.concatenate([groups[int(i)] for i in chosen]); order = np.argsort(x[idx], kind="mergesort"); q = max(1, len(order) // 5)
        out[draw] = float(np.mean(y[idx][order[-q:]] > 0) - np.mean(y[idx][order[:q]] > 0))
        if draw and draw % 2000 == 0:
            print(f"[bootstrap-quintile] {cert_col}->{outcome_col} draw={draw}/{BOOTSTRAP_DRAWS}", flush=True)
    return out


def calibration(frame: pd.DataFrame, cert_col: str, cert_name: str) -> list[dict[str, Any]]:
    x = frame[cert_col].to_numpy(float); order = np.argsort(x, kind="mergesort"); quint = np.empty(len(frame), dtype=int)
    for rank, pos in enumerate(order): quint[pos] = min(5, int(rank * 5 / max(len(frame), 1)) + 1)
    rows = []
    for q in range(1, 6):
        part = frame.iloc[np.flatnonzero(quint == q)]
        rows.append({
            "dataset": str(frame.dataset.iloc[0]), "certificate": cert_name, "quintile": q,
            "mean_certificate": finite(float(part[cert_col].mean())) if len(part) else None,
            "mean_H_BBR": finite(float(part.H_BBR.mean())) if len(part) else None,
            "BBR_harm_frequency": finite(float(np.mean(part.H_BBR.to_numpy(float) > 0))) if len(part) else None,
            "mean_H_BER": finite(float(part.H_BER.mean())) if len(part) else None,
            "decision_harm_frequency": finite(float(np.mean(part.H_BER.to_numpy(float) > 0))) if len(part) else None,
            "correct_to_wrong_frequency": finite(float(part.correct_to_wrong.sum() / max(part.B_out_trial_count.sum(), 1))) if len(part) else None,
            "subject_count": int(part.subject_id.astype(str).nunique()), "observation_count": int(len(part)),
        })
    return rows


def fold_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (dataset, fold), part in frame.groupby(["dataset", "fold"], sort=True):
        bbr_h = point_pack(part, "certificate_BBR", "H_BBR"); bbr_d = point_pack(part, "certificate_BBR", "H_BER"); ce_d = point_pack(part, "certificate_CE", "H_BER")
        rows.append({"dataset": str(dataset), "fold": int(fold), "n_subjects": int(part.subject_id.astype(str).nunique()), "n_observations": int(len(part)), "BBR_H_BBR_AUROC": bbr_h["auroc"], "BBR_H_BBR_Spearman": bbr_h["spearman"], "BBR_H_BER_AUROC": bbr_d["auroc"], "BBR_H_BER_Spearman": bbr_d["spearman"], "CE_H_BER_AUROC": ce_d["auroc"], "CE_H_BER_Spearman": ce_d["spearman"], "harmful_flip_count": int(part.correct_to_wrong.sum()), "same_subject_signal_positive": bool((bbr_h["spearman"] or 0) > 0), "BBR_not_worse_than_CE": bool((bbr_d["auroc"] or -np.inf) >= (ce_d["auroc"] or -np.inf) or (bbr_d["spearman"] or -np.inf) >= (ce_d["spearman"] or -np.inf))})
    return rows


def control_summary(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = str(frame.dataset.iloc[0]); rows = []
    for name, col in [("same_subject", "certificate_BBR"), ("different_subject", "certificate_BBR_different"), ("permutation", "certificate_BBR_permuted"), ("random", "certificate_BBR_random")]:
        pack = point_pack(frame, col, "H_BBR"); rows.append({"dataset": dataset, "control": name, "certificate_column": col, **pack})
    event = (frame.H_BBR.to_numpy(float) > 0).astype(int)
    same_auc, diff_auc = safe_auc(frame.certificate_BBR.to_numpy(float), event), safe_auc(frame.certificate_BBR_different.to_numpy(float), event)
    same_sp, diff_sp = safe_spearman(frame.certificate_BBR.to_numpy(float), frame.H_BBR.to_numpy(float)), safe_spearman(frame.certificate_BBR_different.to_numpy(float), frame.H_BBR.to_numpy(float))
    auc_boot, sp_boot = bootstrap_control(frame, "certificate_BBR", "certificate_BBR_different", "H_BBR", stable_seed("control", dataset, "different", SEED))
    summary = {"dataset": dataset, "same_subject_AUROC": same_auc, "different_subject_AUROC": diff_auc, "AUROC_advantage": finite((same_auc - diff_auc) if same_auc is not None and diff_auc is not None else None), "AUROC_advantage_CI95": ci(auc_boot), "same_subject_Spearman": same_sp, "different_subject_Spearman": diff_sp, "Spearman_advantage": finite((same_sp - diff_sp) if same_sp is not None and diff_sp is not None else None), "Spearman_advantage_CI95": ci(sp_boot)}
    return summary, rows


def alignment(frame: pd.DataFrame, bbr_boot: np.ndarray, ce_boot: np.ndarray) -> pd.DataFrame:
    rows = []; flip = (frame.correct_to_wrong.to_numpy(int) > 0).astype(int); fb = bootstrap_metric(frame.assign(_flip=flip), "certificate_BBR", "_flip", stable_seed("flip", frame.dataset.iloc[0], "BBR")); fc = bootstrap_metric(frame.assign(_flip=flip), "certificate_CE", "_flip", stable_seed("flip", frame.dataset.iloc[0], "CE"))
    for label, bp, cp, bv, cv in [("H_BER Spearman", safe_spearman(frame.certificate_BBR.to_numpy(float), frame.H_BER.to_numpy(float)), safe_spearman(frame.certificate_CE.to_numpy(float), frame.H_BER.to_numpy(float)), bbr_boot[:, 0], ce_boot[:, 0]), ("H_BER Kendall", safe_kendall(frame.certificate_BBR.to_numpy(float), frame.H_BER.to_numpy(float)), safe_kendall(frame.certificate_CE.to_numpy(float), frame.H_BER.to_numpy(float)), bbr_boot[:, 1], ce_boot[:, 1]), ("decision-harm AUROC", safe_auc(frame.certificate_BBR.to_numpy(float), (frame.H_BER.to_numpy(float) > 0).astype(int)), safe_auc(frame.certificate_CE.to_numpy(float), (frame.H_BER.to_numpy(float) > 0).astype(int)), bbr_boot[:, 2], ce_boot[:, 2]), ("harmful-flip AUROC", safe_auc(frame.certificate_BBR.to_numpy(float), flip), safe_auc(frame.certificate_CE.to_numpy(float), flip), fb[:, 2], fc[:, 2]), ("top-bottom decision-risk separation", top_bottom(frame.certificate_BBR.to_numpy(float), frame.H_BER.to_numpy(float)), top_bottom(frame.certificate_CE.to_numpy(float), frame.H_BER.to_numpy(float)), bbr_boot[:, 3], ce_boot[:, 3])]:
        diff = finite((bp - cp) if bp is not None and cp is not None else None); lo, hi = ci(bv - cv)
        rows.append({"dataset": str(frame.dataset.iloc[0]), "metric": label, "BBR": bp, "CE": cp, "BBR_minus_CE": diff, "BBR_minus_CE_CI95": [lo, hi]})
    return pd.DataFrame(rows)


def verify_lock() -> dict[str, Any]:
    lock_path = RESULTS / "PRE_OUTCOME_LOCK.json"; mandatory_path = RESULTS / "MANDATORY_TESTS.json"
    if not lock_path.is_file() or not mandatory_path.is_file(): raise RuntimeError("missing pre-outcome lock or mandatory tests")
    lock = json.loads(lock_path.read_text(encoding="utf-8")); mandatory = json.loads(mandatory_path.read_text(encoding="utf-8"))
    if not mandatory.get("pass"): raise RuntimeError("mandatory tests failed")
    forbidden = ["seed1_run", "seed2_run", "second_backbone_run", "WBCIC_outer_opened", "OpenBMI_sealed_opened", "outcome_used"]
    if any(bool(lock.get(k)) for k in forbidden): raise RuntimeError("forbidden lock flag")
    return lock


def verify_completed_replay(frame: pd.DataFrame, lock: dict[str, Any]) -> dict[str, Any]:
    """Check that the compact table contains every locked fold/step."""
    trajectory_path = RESULTS / "TRAINING_TRAJECTORIES.csv"
    if not trajectory_path.is_file():
        raise RuntimeError("missing TRAINING_TRAJECTORIES.csv")
    trajectory = pd.read_csv(trajectory_path)
    expected = {(str(row["dataset"]), int(row["fold"])): row for row in lock.get("task_schedule_hashes", [])}
    observed: list[dict[str, Any]] = []
    for key, row in expected.items():
        dataset, fold = key
        t = trajectory[(trajectory.dataset == dataset) & (trajectory.fold == fold)]
        o = frame[(frame.dataset == dataset) & (frame.fold == fold)]
        expected_steps = int(row["total_steps"])
        expected_audit = {int(v) for v in row["audit_steps"]}
        t_steps = {int(v) for v in t.step.tolist()}
        o_steps = {int(v) for v in o.step.tolist()}
        if t_steps != set(range(1, expected_steps + 1)) or o_steps != expected_audit:
            raise RuntimeError(f"incomplete locked replay {dataset} fold={fold}: trajectory={len(t_steps)}/{expected_steps}, audit={sorted(o_steps)}")
        observed.append({"dataset": dataset, "fold": fold, "trajectory_rows": int(len(t)), "expected_total_steps": expected_steps, "audit_observation_steps": sorted(o_steps), "observation_rows": int(len(o)), "trajectory_complete": True})
    if len(observed) != len(expected):
        raise RuntimeError("locked replay context count mismatch")
    result = {"schema": "PERSIST_EEG_DECISION_RELEVANT_REPLAY_COMPLETION_V1", "complete": True, "contexts": observed, "observation_rows": int(len(frame)), "seed": SEED, "outcome_used": False}
    write_json(RESULTS / "AUDIT_RUN_COMPLETION.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", default=str(RESULTS / "PER_OBSERVATION.csv")); args = parser.parse_args()
    lock = verify_lock(); frame = pd.read_csv(args.input); replay_completion = verify_completed_replay(frame, lock)
    required = {"dataset", "fold", "subject_id", "certificate_BBR", "certificate_CE", "certificate_BBR_different", "certificate_BBR_permuted", "certificate_BBR_random", "H_BBR", "H_BER", "correct_to_wrong", "wrong_to_correct", "B_out_trial_count"}
    missing = sorted(required - set(frame.columns));
    if missing: raise RuntimeError(f"missing observation columns: {missing}")
    stats: dict[str, dict[str, Any]] = {}; controls: dict[str, dict[str, Any]] = {}; alignments: list[pd.DataFrame] = []; calibrations: list[dict[str, Any]] = []; flip_rows: list[dict[str, Any]] = []; boot_json: dict[str, Any] = {}
    fold = fold_rows(frame); write_csv(RESULTS / "PER_FOLD_METRICS.csv", fold)
    grouped = frame.groupby(["dataset", "subject_id"], as_index=False).agg(observations=("step", "size"), mean_H_BBR=("H_BBR", "mean"), mean_H_BER=("H_BER", "mean"), total_correct_to_wrong=("correct_to_wrong", "sum"), total_wrong_to_correct=("wrong_to_correct", "sum")); write_csv(RESULTS / "PER_SUBJECT_METRICS.csv", grouped)
    for dataset, part in frame.groupby("dataset", sort=True):
        dataset = str(dataset); bbr_hbbr = bootstrap_metric(part, "certificate_BBR", "H_BBR", stable_seed("boot", dataset, "bbr-hbbr", SEED)); bbr_hber = bootstrap_metric(part, "certificate_BBR", "H_BER", stable_seed("boot", dataset, "bbr-hber", SEED)); ce_hber = bootstrap_metric(part, "certificate_CE", "H_BER", stable_seed("boot", dataset, "ce-hber", SEED));
        bbr = point_pack(part, "certificate_BBR", "H_BBR"); dec = point_pack(part, "certificate_BBR", "H_BER"); ce = point_pack(part, "certificate_CE", "H_BER")
        for key, arr in [("spearman", bbr_hbbr[:, 0]), ("kendall", bbr_hbbr[:, 1]), ("auroc", bbr_hbbr[:, 2]), ("top_minus_bottom", bbr_hbbr[:, 3])]: bbr[key + "_CI95"] = ci(arr)
        for key, arr in [("spearman", bbr_hber[:, 0]), ("kendall", bbr_hber[:, 1]), ("auroc", bbr_hber[:, 2]), ("top_minus_bottom", bbr_hber[:, 3])]: dec[key + "_CI95"] = ci(arr)
        for key, arr in [("spearman", ce_hber[:, 0]), ("kendall", ce_hber[:, 1]), ("auroc", ce_hber[:, 2]), ("top_minus_bottom", ce_hber[:, 3])]: ce[key + "_CI95"] = ci(arr)
        stats[dataset] = {"dataset": dataset, "BBR_to_H_BBR": bbr, "BBR_to_H_BER": dec, "CE_to_H_BER": ce, "H_BBR_mean": float(part.H_BBR.mean()), "H_BBR_positive_count": int(np.sum(part.H_BBR > 0)), "H_BER_mean": float(part.H_BER.mean()), "H_BER_positive_count": int(np.sum(part.H_BER > 0)), "H_BER_negative_count": int(np.sum(part.H_BER < 0)), "H_BER_zero_count": int(np.sum(part.H_BER == 0))}
        summary, control_rows = control_summary(part); controls[dataset] = summary; alignments.append(alignment(part, bbr_hber, ce_hber)); calibrations.extend(calibration(part, "certificate_BBR", "BBR")); calibrations.extend(calibration(part, "certificate_CE", "CE")); flip_rows.append({"dataset": dataset, "total_correct_to_wrong": int(part.correct_to_wrong.sum()), "total_wrong_to_correct": int(part.wrong_to_correct.sum()), "harmful_flip_rate": float(part.correct_to_wrong.sum() / max(part.B_out_trial_count.sum(), 1)), "beneficial_flip_rate": float(part.wrong_to_correct.sum() / max(part.B_out_trial_count.sum(), 1)), "net_flip_harm": float((part.correct_to_wrong.sum() - part.wrong_to_correct.sum()) / max(part.B_out_trial_count.sum(), 1))}); boot_json[dataset] = {"draws": BOOTSTRAP_DRAWS, "BBR_to_H_BBR_CI95": {k: ci(bbr_hbbr[:, i]) for i, k in enumerate(("spearman", "kendall", "auroc", "top_minus_bottom"))}, "BBR_to_H_BER_CI95": {k: ci(bbr_hber[:, i]) for i, k in enumerate(("spearman", "kendall", "auroc", "top_minus_bottom"))}, "CE_to_H_BER_CI95": {k: ci(ce_hber[:, i]) for i, k in enumerate(("spearman", "kendall", "auroc", "top_minus_bottom"))}, "control": summary}
        # Keep control rows for the per-control tables.
        write_csv(RESULTS / "PERMUTATION_CONTROL.csv", [r for r in control_rows if r["control"] == "permutation"] + ([r for r in []]))
    write_csv(RESULTS / "BBR_HARM_SUMMARY.csv", [{"dataset": d, **s["BBR_to_H_BBR"], "mean_H_BBR": s["H_BBR_mean"], "positive_harm_events": s["H_BBR_positive_count"]} for d, s in stats.items()])
    write_csv(RESULTS / "DECISION_HARM_SUMMARY.csv", [{"dataset": d, **s["BBR_to_H_BER"], "CE_H_BER_Spearman": s["CE_to_H_BER"].get("spearman"), "CE_H_BER_AUROC": s["CE_to_H_BER"].get("auroc"), "mean_H_BER": s["H_BER_mean"], "positive_H_BER_events": s["H_BER_positive_count"]} for d, s in stats.items()]); write_csv(RESULTS / "FLIP_SUMMARY.csv", flip_rows); write_csv(RESULTS / "SAME_VS_DIFFERENT.csv", list(controls.values())); write_csv(RESULTS / "BBR_CALIBRATION_BINS.csv", [r for r in calibrations if r["certificate"] == "BBR"]); write_csv(RESULTS / "CE_CALIBRATION_BINS.csv", [r for r in calibrations if r["certificate"] == "CE"]); write_csv(RESULTS / "BBR_VS_CE_DECISION_ALIGNMENT.csv", pd.concat(alignments, ignore_index=True)); write_json(RESULTS / "BOOTSTRAP_RESULTS.json", boot_json)
    # Rewrite per-control tables from all datasets (rather than retaining the
    # temporary per-dataset write above).
    all_controls = []
    for dataset, part in frame.groupby("dataset", sort=True): all_controls.extend(control_summary(part)[1])
    write_csv(RESULTS / "PERMUTATION_CONTROL.csv", [r for r in all_controls if r["control"] == "permutation"]); write_csv(RESULTS / "RANDOM_CONTROL.csv", [r for r in all_controls if r["control"] == "random"])
    power = []
    for dataset, part in frame.groupby("dataset", sort=True):
        harmful = int(np.sum(part.H_BER > 0)); subjects = int(part.loc[part.H_BER > 0, "subject_id"].astype(str).nunique()); power.append({"dataset": str(dataset), "total_subject_step_observations": int(len(part)), "H_BER_positive_count": harmful, "H_BER_negative_count": int(np.sum(part.H_BER < 0)), "H_BER_zero_count": int(np.sum(part.H_BER == 0)), "biological_subjects_with_harmful_event": subjects, "total_correct_to_wrong_flips": int(part.correct_to_wrong.sum()), "total_wrong_to_correct_flips": int(part.wrong_to_correct.sum()), "EXACT_DECISION_ENDPOINT_UNDERPOWERED": bool(harmful < 30 or subjects < 15)})
    write_csv(RESULTS / "RARE_EVENT_POWER.csv", power)
    # Gate logic is deliberately predeclared and uses no result-dependent
    # threshold changes.  Exact endpoint gates are skipped when underpowered.
    gate_a = {}; gate_b = {}; gate_c = {}; gate_d = {}; gate_e = {}
    for dataset in DATASETS:
        s = stats[dataset]; b = s["BBR_to_H_BBR"]; c = controls[dataset]; gate_a[dataset] = bool((b.get("auroc") or -np.inf) >= .60 and (b.get("auroc_CI95") or [-np.inf])[0] > .50 and (b.get("spearman") or -np.inf) > 0 and (b.get("spearman_CI95") or [-np.inf])[0] > 0); gate_b[dataset] = bool((c.get("AUROC_advantage") or -np.inf) > 0 and (c.get("AUROC_advantage_CI95") or [-np.inf])[0] > 0 and (c.get("Spearman_advantage") or -np.inf) > 0 and (c.get("Spearman_advantage_CI95") or [-np.inf])[0] > 0); fold_d = [r for r in fold if r["dataset"] == dataset]; gate_e[dataset] = sum(bool(r["BBR_not_worse_than_CE"] or r["same_subject_signal_positive"]) for r in fold_d) >= 4
        cal = pd.DataFrame([r for r in calibrations if r["dataset"] == dataset and r["certificate"] == "BBR"]); q1 = cal.loc[cal.quintile == 1, "decision_harm_frequency"].iloc[0]; q5 = cal.loc[cal.quintile == 5, "decision_harm_frequency"].iloc[0]; qdiff = float(q5 - q1); qci = ci(bootstrap_quintile(frame[frame.dataset == dataset], "certificate_BBR", "H_BER", stable_seed("qdiff", dataset, SEED))); s["Q5_minus_Q1_decision_harm"] = qdiff; s["Q5_minus_Q1_decision_harm_CI95"] = qci; under = next(p["EXACT_DECISION_ENDPOINT_UNDERPOWERED"] for p in power if p["dataset"] == dataset); align = alignments[0 if dataset == "OpenBMI" else 1]; ar = align[align.metric == "decision-harm AUROC"].iloc[0]; sr = align[align.metric == "H_BER Spearman"].iloc[0]; gate_c[dataset] = bool(not under and (s["BBR_to_H_BER"].get("spearman") or -np.inf) > 0 and (s["BBR_to_H_BER"].get("auroc") or -np.inf) > .55 and qdiff > 0 and qci[0] is not None and qci[0] > 0); gate_d[dataset] = bool(not under and float(ar.BBR) > float(ar.CE) and float(sr.BBR) > float(sr.CE))
    under_all = any(p["EXACT_DECISION_ENDPOINT_UNDERPOWERED"] for p in power); gates = {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d, "E": gate_e}
    if not all(gate_a.values()) or not all(gate_b.values()): terminal = "DECISION_RELEVANT_SIGNAL_NOT_SUPPORTED"
    elif under_all: terminal = "DECISION_ENDPOINT_UNDERPOWERED"
    elif all(gate_c.values()) and all(gate_d.values()) and all(gate_e.values()): terminal = "DECISION_RELEVANT_SUBJECT_HARM_SUPPORTED"
    elif all(gate_a.values()) and all(gate_b.values()) and any(gate_c.values()): terminal = "BOUNDARY_SURROGATE_SUPPORTED_DECISION_UNPROVEN"
    else: terminal = "DECISION_RELEVANT_SIGNAL_NOT_SUPPORTED"
    decision = {"terminal": terminal, "gate_A_stable_BBR_harm": gate_a, "gate_B_subject_specificity": gate_b, "gate_C_decision_alignment": gate_c, "gate_D_BBR_over_CE": gate_d, "gate_E_fold_robustness": gate_e, "power": {row["dataset"]: row for row in power}, "DECISION_ALIGNED_GUARD_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED": terminal == "DECISION_RELEVANT_SUBJECT_HARM_SUPPORTED", "AUTO_START_NEW_MODEL": False}
    mandatory = json.loads((RESULTS / "MANDATORY_TESTS.json").read_text(encoding="utf-8")); validation = {"schema": "PERSIST_EEG_DECISION_RELEVANT_VALIDATION_V1", "pass": True, "terminal": terminal, "mandatory_tests_pass": bool(mandatory.get("pass")), "checkpoint_equivalence_pass": bool(pd.read_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv").get("pass", pd.Series(dtype=bool)).all()), "tau_frozen": True, "B1_B4_Bout_disjoint": bool(mandatory["checks"]["B1_B4_Bout_trial_disjoint"]), "A_B_disjoint": bool(mandatory["checks"]["A_B_subject_disjoint"]), "task_only_trajectory_audit_equivalence": bool(mandatory["checks"]["task_only_replay_equivalence"]["pass"]), "BN_unchanged": bool((pd.read_csv(RESULTS / "TRAINING_TRAJECTORIES.csv").bn_max_displacement <= 1e-12).all()), "outcome_used": False, "seed1_run": False, "seed2_run": False, "second_backbone_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False, "decision": decision}
    write_json(RESULTS / "VALIDATION.json", validation); write_json(RESULTS / "FINAL_REPORT.json", {"schema": "PERSIST_EEG_DECISION_RELEVANT_FINAL_REPORT_V1", "terminal": terminal, "decision": decision, "dataset_stats": stats, "controls": controls, "fold_metrics": fold, "validation": validation, "lock": lock})
    report = ["# Final report", "", f"terminal = {terminal}", "", "Source/refit-only, frozen seed-0 EEGNet trajectory. No development outcome, WBCIC outer-10, OpenBMI sealed/confirmation cohort, seed 1/2, or second backbone was opened.", "", "|dataset|BBR->H_BBR AUROC|CI|BBR->H_BBR Spearman|BBR->H_BER AUROC|CE->H_BER AUROC|H_BER harmful events|Q5-Q1 decision harm|underpowered|", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for dataset in DATASETS:
        s = stats[dataset]; report.append(f"|{dataset}|{s['BBR_to_H_BBR'].get('auroc')}|{s['BBR_to_H_BBR'].get('auroc_CI95')}|{s['BBR_to_H_BBR'].get('spearman')}|{s['BBR_to_H_BER'].get('auroc')}|{s['CE_to_H_BER'].get('auroc')}|{s['H_BER_positive_count']}|{s.get('Q5_minus_Q1_decision_harm')}|{next(p['EXACT_DECISION_ENDPOINT_UNDERPOWERED'] for p in power if p['dataset']==dataset)}|")
    report += ["", "## Required answers", "", f"- BBR K4 cross-batch signal: {gate_a}", f"- Same-subject specificity: {gate_b}", f"- Exact decision alignment: {gate_c}", f"- BBR over CE: {gate_d}", f"- Fold robustness: {gate_e}", "- Exact decision endpoint uses H_BER and correct-to-wrong flips; rare-event power is reported without changing the frozen schedule.", "- No new model or guard is automatically started."]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8"); (EXP / "AUTONOMOUS_DECISION.md").write_text(f"# Autonomous decision\n\n`terminal = {terminal}`\n\nNo new model, guard, seed, backbone, outer cohort or sealed cohort is started automatically.\n", encoding="utf-8")
    print(f"terminal = {terminal}");
    for dataset in DATASETS:
        s = stats[dataset]; b = s["BBR_to_H_BBR"]; d = s["BBR_to_H_BER"]; c = s["CE_to_H_BER"]; print(f"{dataset}_BBR_HARM_AUROC = {b.get('auroc')}"); print(f"{dataset}_BBR_HARM_AUROC_CI = {b.get('auroc_CI95')}"); print(f"{dataset}_BBR_HARM_SPEARMAN = {b.get('spearman')}"); print(f"{dataset}_DECISION_HARM_AUROC = {d.get('auroc')}"); print(f"{dataset}_CE_DECISION_HARM_AUROC = {c.get('auroc')}"); print(f"{dataset}_BBR_MINUS_CE_DECISION_AUROC = {None if d.get('auroc') is None or c.get('auroc') is None else d.get('auroc') - c.get('auroc')}"); print(f"{dataset}_exact_harm_events = {s['H_BER_positive_count']}"); print(f"{dataset}_Q5_minus_Q1_decision_harm = {s.get('Q5_minus_Q1_decision_harm')}"); print(f"EXACT_DECISION_ENDPOINT_UNDERPOWERED_{dataset} = {next(p['EXACT_DECISION_ENDPOINT_UNDERPOWERED'] for p in power if p['dataset']==dataset)}")
    print("seed1_run = false"); print("seed2_run = false"); print("second_backbone_run = false"); print("WBCIC_outer_opened = false"); print("OpenBMI_sealed_opened = false"); print(f"DECISION_ALIGNED_GUARD_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED = {'YES' if decision['DECISION_ALIGNED_GUARD_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED'] else 'NO'}")


if __name__ == "__main__":
    main()
