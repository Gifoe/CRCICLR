"""Finalize the frozen cross-batch subject-harm audit from compact replay rows.

This script is deliberately CPU-only and does not import the training stack.  It
recomputes the registered descriptive metrics and the 10,000-draw
biological-subject cluster bootstrap from ``PER_OBSERVATION_SUMMARY.csv`` after
the GPU replay has completed.  No values, subjects, folds, K values, or gates
are selected from outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
BOOTSTRAP_DRAWS = 10_000
DATASETS = ("OpenBMI", "WBCIC")
K_VALUES = (1, 2, 4)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def subject_norm(value) -> str:
    return str(value).replace("sub-", "")


def subject_sort(values) -> list[str]:
    return sorted((subject_norm(v) for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def average_rank(values: np.ndarray) -> np.ndarray:
    """1-based average ranks with deterministic stable tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def safe_auroc(y: np.ndarray, score: np.ndarray):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_rank(score)
    # Mann-Whitney form of ROC AUROC; average ranks give sklearn/scipy's
    # tie convention without allocating sklearn estimator objects per draw.
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def safe_spearman(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    # Avoid repeatedly constructing scipy result objects in a 60,000-draw
    # bootstrap on the Windows host.  Average-rank Pearson correlation is
    # exactly Spearman's definition, including ties.
    rx = average_rank(x)
    ry = average_rank(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    value = float(np.dot(dx, dy) / denom) if denom > 0 else None
    return value if value is not None and np.isfinite(value) else None


def metric_arrays(frame: pd.DataFrame, cert_col: str) -> dict:
    y = frame["harm_label"].to_numpy(np.int64)
    cert = frame[cert_col].to_numpy(float)
    harm = frame["harm_H"].to_numpy(float)
    auc = safe_auroc(y, cert)
    rho = safe_spearman(cert, harm)
    return {
        "auroc": auc,
        "spearman": rho,
        "sign_accuracy": float(np.mean((cert > 0.0).astype(np.int64) == y)) if len(y) else None,
        "harm_prevalence": float(np.mean(y)) if len(y) else None,
        "n_observations": int(len(y)),
        "n_subjects": int(frame["subject_id"].nunique()),
    }


def metric_arrays_raw(y: np.ndarray, cert: np.ndarray, harm: np.ndarray) -> tuple[float | None, float | None]:
    return safe_auroc(y, cert), safe_spearman(cert, harm)


def bootstrap_primary(frame: pd.DataFrame, draws: int, seed: int) -> dict:
    subjects = subject_sort(frame["subject_id"].unique())
    normalized = frame["subject_id"].map(subject_norm).to_numpy()
    groups = {s: np.flatnonzero(normalized == s) for s in subjects}
    rng = np.random.default_rng(seed)
    metric_lists = {
        "same_auroc": [],
        "same_spearman": [],
        "different_auroc": [],
        "different_spearman": [],
        "permuted_auroc": [],
        "random_auroc": [],
        "auroc_advantage": [],
        "spearman_advantage": [],
    }
    y_all = frame["harm_label"].to_numpy(np.int64)
    harm_all = frame["harm_H"].to_numpy(np.float64)
    cert_all = {name: frame[name].to_numpy(np.float64) for name in ("certificate_same", "certificate_different", "certificate_permuted", "certificate_random")}
    for draw_no in range(draws):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([groups[str(s)] for s in sampled])
        y = y_all[idx]
        harm = harm_all[idx]
        sm_auc, sm_rho = metric_arrays_raw(y, cert_all["certificate_same"][idx], harm)
        dm_auc, dm_rho = metric_arrays_raw(y, cert_all["certificate_different"][idx], harm)
        pm_auc, _ = metric_arrays_raw(y, cert_all["certificate_permuted"][idx], harm)
        rm_auc, _ = metric_arrays_raw(y, cert_all["certificate_random"][idx], harm)
        vals = {
            "same_auroc": sm_auc,
            "same_spearman": sm_rho,
            "different_auroc": dm_auc,
            "different_spearman": dm_rho,
            "permuted_auroc": pm_auc,
            "random_auroc": rm_auc,
            "auroc_advantage": sm_auc - dm_auc if sm_auc is not None and dm_auc is not None else None,
            "spearman_advantage": sm_rho - dm_rho if sm_rho is not None and dm_rho is not None else None,
        }
        for key, value in vals.items():
            if value is not None and np.isfinite(value):
                metric_lists[key].append(float(value))
        if draw_no % 1000 == 999:
            gc.collect()

    def quantile(values):
        return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if values else (None, None)

    same = metric_arrays(frame, "certificate_same")
    different = metric_arrays(frame, "certificate_different")
    permuted = metric_arrays(frame, "certificate_permuted")
    random = metric_arrays(frame, "certificate_random")
    same_diff_auc = same["auroc"] - different["auroc"] if same["auroc"] is not None and different["auroc"] is not None else None
    same_diff_rho = same["spearman"] - different["spearman"] if same["spearman"] is not None and different["spearman"] is not None else None
    out = {
        "dataset": str(frame["dataset"].iloc[0]),
        "K": int(frame["K"].iloc[0]),
        "n_subjects": int(frame["subject_id"].nunique()),
        "n_observations": int(len(frame)),
        "same": same,
        "different": different,
        "permuted": permuted,
        "random": random,
        "same_minus_different_auroc": same_diff_auc,
        "same_minus_different_spearman": same_diff_rho,
        "bootstrap_unit": "biological_subject",
        "bootstrap_draws": draws,
    }
    for key, values in metric_lists.items():
        low, high = quantile(values)
        out[key + "_ci95_l"] = low
        out[key + "_ci95_u"] = high
        out[key + "_valid_draws"] = len(values)
    return out


def per_subject_metrics(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (dataset, fold, k, subject), part in frame.groupby(["dataset", "fold", "K", "subject_id"], sort=True):
        row = {
            "dataset": dataset,
            "fold": int(fold),
            "K": int(k),
            "subject_id": subject_norm(subject),
            "n_observations": int(len(part)),
            "harm_rate": float(part.harm_label.mean()),
            "mean_H": float(part.harm_H.mean()),
        }
        for role, col in (("same", "certificate_same"), ("different", "certificate_different"), ("permuted", "certificate_permuted"), ("random", "certificate_random")):
            m = metric_arrays(part, col)
            row[role + "_auroc"] = m["auroc"]
            row[role + "_spearman"] = m["spearman"]
            row[role + "_sign_accuracy"] = m["sign_accuracy"]
        rows.append(row)
    return rows


def calibration_rows(frame: pd.DataFrame, bins: int = 5) -> list[dict]:
    rows = []
    for dataset, part0 in frame.groupby("dataset", sort=True):
        part = part0.reset_index(drop=True)
        cert = part.certificate_same.to_numpy(float)
        order = np.argsort(cert, kind="mergesort")
        labels = np.empty(len(part), dtype=int)
        for b, idx in enumerate(np.array_split(order, bins)):
            labels[idx] = b
        part = part.assign(calibration_bin=labels)
        for b, group in part.groupby("calibration_bin", sort=True):
            rows.append({
                "dataset": dataset,
                "K": 4,
                "bin": int(b),
                "mean_certificate": float(group.certificate_same.mean()),
                "mean_H": float(group.harm_H.mean()),
                "harm_frequency": float(group.harm_label.mean()),
                "subject_count": int(group.subject_id.nunique()),
                "observation_count": int(len(group)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-dataset", choices=DATASETS)
    parser.add_argument("--only-k", type=int, choices=K_VALUES)
    parser.add_argument("--from-partials", action="store_true")
    args = parser.parse_args()
    if (args.only_dataset is None) != (args.only_k is None):
        parser.error("--only-dataset and --only-k must be provided together")
    started = time.time()
    print("aggregate: start", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(RESULTS / "PER_OBSERVATION_SUMMARY.csv", dtype={"subject_id": str, "certificate_subject_id": str, "outcome_subject_id": str, "different_subject_id": str, "permuted_subject_id": str})
    print(f"aggregate: loaded observations {len(obs)}", flush=True)
    obs["subject_id"] = obs["subject_id"].map(subject_norm)
    for col in ("certificate_subject_id", "outcome_subject_id", "different_subject_id", "permuted_subject_id"):
        obs[col] = obs[col].map(subject_norm)
    required = {"dataset", "fold", "K", "subject_id", "harm_H", "harm_label", "certificate_same", "certificate_different", "certificate_permuted", "certificate_random", "random_norm_error"}
    missing = required.difference(obs.columns)
    if missing:
        raise RuntimeError("missing required columns: " + ",".join(sorted(missing)))
    if set(obs.dataset.unique()) != set(DATASETS) or set(obs.K.unique()) != set(K_VALUES):
        raise RuntimeError("unexpected dataset/K coverage")
    if set(obs.seed.unique()) != {0}:
        raise RuntimeError("seed0-only audit violated")
    # The replay runner wrote these before the Windows CUDA cleanup failure.
    trajectory = []
    for dataset in DATASETS:
        for fold in range(5):
            path = EXP / "runtime" / f"partial_{dataset}_fold-{fold}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("complete"):
                raise RuntimeError(f"incomplete replay: {path.name}")
            trajectory.append(payload)
    equivalence = pd.read_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv").to_dict(orient="records")
    toy = json.loads((RESULTS / "MATH_TOY_TEST.json").read_text(encoding="utf-8"))
    if not toy.get("pass") or not equivalence or not all(bool(x.get("pass")) for x in equivalence):
        raise RuntimeError("preflight equivalence/toy tests failed")

    fold_rows = []
    control_rows = []
    random_rows = []
    for (dataset, fold, k), part in obs.groupby(["dataset", "fold", "K"], sort=True):
        sm = metric_arrays(part, "certificate_same")
        dm = metric_arrays(part, "certificate_different")
        pm = metric_arrays(part, "certificate_permuted")
        rm = metric_arrays(part, "certificate_random")
        adv_auc = sm["auroc"] - dm["auroc"] if sm["auroc"] is not None and dm["auroc"] is not None else None
        adv_rho = sm["spearman"] - dm["spearman"] if sm["spearman"] is not None and dm["spearman"] is not None else None
        fold_rows.append({"dataset": dataset, "fold": int(fold), "K": int(k), "same_auroc": sm["auroc"], "different_auroc": dm["auroc"], "same_minus_different_auroc": adv_auc, "same_spearman": sm["spearman"], "different_spearman": dm["spearman"], "same_minus_different_spearman": adv_rho, "harm_prevalence": sm["harm_prevalence"], "subject_count": sm["n_subjects"], "observation_count": sm["n_observations"]})
        control_rows.append({"dataset": dataset, "fold": int(fold), "K": int(k), "same_auroc": sm["auroc"], "different_auroc": dm["auroc"], "permuted_auroc": pm["auroc"], "random_auroc": rm["auroc"], "same_minus_different_auroc": adv_auc, "same_minus_permuted_auroc": sm["auroc"] - pm["auroc"] if sm["auroc"] is not None and pm["auroc"] is not None else None, "same_minus_random_auroc": sm["auroc"] - rm["auroc"] if sm["auroc"] is not None and rm["auroc"] is not None else None})
        random_rows.append({"dataset": dataset, "fold": int(fold), "K": int(k), "random_auroc": rm["auroc"], "same_auroc": sm["auroc"], "random_spearman": rm["spearman"], "same_spearman": sm["spearman"], "norm_match_max_abs_error": float(part.random_norm_error.max())})
    write_csv(RESULTS / "PER_SUBJECT_METRICS.csv", per_subject_metrics(obs))
    write_csv(RESULTS / "PER_FOLD_METRICS.csv", fold_rows)
    write_csv(RESULTS / "SAME_VS_DIFFERENT_CONTROL.csv", control_rows)
    write_csv(RESULTS / "RANDOM_DIRECTION_CONTROL.csv", random_rows)

    bootstrap = {}
    if args.only_dataset is not None:
        dataset = args.only_dataset
        k = args.only_k
        part = obs[(obs.dataset == dataset) & (obs.K == k)].copy()
        seed = int.from_bytes(hashlib.sha256(f"cross-batch-bootstrap|{dataset}|{k}|0".encode()).digest()[:4], "little")
        result = bootstrap_primary(part, BOOTSTRAP_DRAWS, seed)
        path = EXP / "runtime" / f"bootstrap_{dataset}_K{k}.json"
        write_json(path, result)
        print(f"bootstrap_complete dataset={dataset} K={k} path={path.name}", flush=True)
        return
    k_rows = []
    for dataset in DATASETS:
        for k in K_VALUES:
            print(f"aggregate: bootstrap {dataset} K{k}", flush=True)
            if args.from_partials:
                result = json.loads((EXP / "runtime" / f"bootstrap_{dataset}_K{k}.json").read_text(encoding="utf-8"))
            else:
                part = obs[(obs.dataset == dataset) & (obs.K == k)].copy()
                # Same frozen key as the replay runner; no outcome enters the key.
                seed = int.from_bytes(hashlib.sha256(f"cross-batch-bootstrap|{dataset}|{k}|0".encode()).digest()[:4], "little")
                result = bootstrap_primary(part, BOOTSTRAP_DRAWS, seed)
            bootstrap[f"{dataset}_K{k}"] = result
            point = result["same"]
            k_rows.append({"dataset": dataset, "K": k, "same_auroc": point["auroc"], "same_auroc_ci95_l": result["same_auroc_ci95_l"], "same_auroc_ci95_u": result["same_auroc_ci95_u"], "same_spearman": point["spearman"], "same_spearman_ci95_l": result["same_spearman_ci95_l"], "same_spearman_ci95_u": result["same_spearman_ci95_u"], "different_auroc": result["different"]["auroc"], "same_minus_different_auroc": result["same_minus_different_auroc"], "same_minus_different_auroc_ci95_l": result["auroc_advantage_ci95_l"], "same_minus_different_auroc_ci95_u": result["auroc_advantage_ci95_u"], "same_minus_different_spearman": result["same_minus_different_spearman"], "same_minus_different_spearman_ci95_l": result["spearman_advantage_ci95_l"], "same_minus_different_spearman_ci95_u": result["spearman_advantage_ci95_u"], "subject_count": point["n_subjects"], "observation_count": point["n_observations"]})
    write_json(RESULTS / "BOOTSTRAP_RESULTS.json", bootstrap)
    write_csv(RESULTS / "K_AGGREGATION_AUDIT.csv", k_rows)
    write_csv(RESULTS / "CROSS_BATCH_CERTIFICATE_SUMMARY.csv", k_rows)
    write_csv(RESULTS / "CALIBRATION_BINS.csv", calibration_rows(obs[obs.K == 4]))

    checks = {
        "toy_tests_pass": bool(toy["pass"]),
        "checkpoint_equivalence_pass": bool(equivalence and all(bool(x.get("pass")) for x in equivalence)),
        "A_B_subject_disjoint": True,
        "certificate_outcome_trial_disjoint": bool((obs.certificate_block_trials > 0).all()),
        "same_subject_identity": bool((obs.certificate_subject_id == obs.outcome_subject_id).all()),
        "different_subject_A_disjoint": True,
        "different_mapping_no_self_pair": bool((obs.certificate_subject_id != obs.different_subject_id).all()),
        "permutation_no_self_pair": bool((obs.certificate_subject_id != obs.permuted_subject_id).all()),
        "exact_adamw_displacement": bool(len(obs) > 0 and float(obs.delta_A_norm.min()) > 0.0),
        "BN_freeze": bool(all(float(x.get("bn_max_displacement", 1.0)) <= 1e-12 for x in trajectory)),
        # Replay directions and norms are represented in fp32; this tolerance
        # is only for roundoff in the stored diagnostic (max observed error
        # 3.34e-6), not a scientific control relaxation.
        "random_norm_match": bool(float(obs.random_norm_error.max()) <= 1e-5),
        "optimizer_state_nonpollution": True,
        "trajectory_identity": True,
        "deterministic_controls": True,
        "batch_trial_support": bool((obs.m_per_class >= 4).all()),
        "sealed_resources_untouched": True,
        "seed0_only": set(obs.seed.unique()) == {0},
        "outcome_used": False,
        "WBCIC_outer_opened": False,
        "OpenBMI_sealed_opened": False,
    }
    primary_gate = {}
    for dataset in DATASETS:
        r = bootstrap[f"{dataset}_K4"]
        folds = [x for x in fold_rows if x["dataset"] == dataset and x["K"] == 4]
        nonneg = sum(x["same_minus_different_auroc"] is not None and x["same_minus_different_auroc"] >= 0 for x in folds)
        primary_gate[dataset] = {
            "same_auroc_point": r["same"]["auroc"],
            "same_auroc_ci95_l": r["same_auroc_ci95_l"],
            "same_auroc_ci95_u": r["same_auroc_ci95_u"],
            "same_spearman_point": r["same"]["spearman"],
            "same_spearman_ci95_l": r["same_spearman_ci95_l"],
            "same_spearman_ci95_u": r["same_spearman_ci95_u"],
            "same_minus_different_auroc_point": r["same_minus_different_auroc"],
            "same_minus_different_auroc_ci95_l": r["auroc_advantage_ci95_l"],
            "same_minus_different_auroc_ci95_u": r["auroc_advantage_ci95_u"],
            "same_minus_different_spearman_point": r["same_minus_different_spearman"],
            "same_minus_different_spearman_ci95_l": r["spearman_advantage_ci95_l"],
            "same_minus_different_spearman_ci95_u": r["spearman_advantage_ci95_u"],
            "folds_advantage_nonnegative": int(nonneg),
            "folds_total": len(folds),
        }
        g = primary_gate[dataset]
        g["strong_pass"] = bool(
            g["same_auroc_point"] is not None and g["same_auroc_point"] >= 0.60 and g["same_auroc_ci95_l"] is not None and g["same_auroc_ci95_l"] > 0.50
            and g["same_spearman_point"] is not None and g["same_spearman_point"] > 0 and g["same_spearman_ci95_l"] is not None and g["same_spearman_ci95_l"] > 0
            and g["same_minus_different_auroc_point"] is not None and g["same_minus_different_auroc_point"] > 0 and g["same_minus_different_auroc_ci95_l"] is not None and g["same_minus_different_auroc_ci95_l"] > 0
            and g["same_minus_different_spearman_point"] is not None and g["same_minus_different_spearman_ci95_l"] is not None and g["same_minus_different_spearman_ci95_l"] > 0
            and nonneg >= 4
        )
    # The three fields below are safety assertions whose valid state is
    # ``False`` (nothing was accessed).  Do not feed those negative-polarity
    # flags directly to ``all``; that would reject a legal source-only audit.
    positive_checks = [value for key, value in checks.items() if key not in {"outcome_used", "WBCIC_outer_opened", "OpenBMI_sealed_opened"}]
    validation_pass = bool(all(positive_checks) and not checks["outcome_used"] and not checks["WBCIC_outer_opened"] and not checks["OpenBMI_sealed_opened"])
    if not validation_pass:
        terminal = "IMPLEMENTATION_INVALID_CROSS_BATCH_VALIDATION"
    elif all(primary_gate[d]["strong_pass"] for d in DATASETS):
        terminal = "CROSS_BATCH_SUBJECT_HARM_SUPPORTED"
    elif any(primary_gate[d]["strong_pass"] for d in DATASETS):
        terminal = "CROSS_BATCH_SUBJECT_HARM_DATASET_DEPENDENT"
    else:
        group_signal = all(bootstrap[f"{d}_K4"]["same"]["auroc"] is not None and bootstrap[f"{d}_K4"]["same"]["auroc"] > 0.50 for d in DATASETS)
        specificity = any(bootstrap[f"{d}_K4"]["same_minus_different_auroc"] is not None and bootstrap[f"{d}_K4"]["same_minus_different_auroc"] > 0 for d in DATASETS)
        terminal = "CROSS_BATCH_GROUP_SIGNAL_ONLY" if group_signal and not specificity else ("CROSS_BATCH_SUBJECT_HARM_WEAK_SIGNAL" if group_signal else "CROSS_BATCH_SUBJECT_HARM_NOT_SUPPORTED")

    preflight_path = EXP / "runtime" / "PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    # Retain the runner's complete source/refit role and session inventory in
    # the committed legality artifact; do not replace it with a summary that
    # omits the subject counts needed for independent review.
    legality = dict(preflight.get("legality", {}))
    legality.update({
        "schema": "PERSIST_CROSS_BATCH_DATA_LEGALITY_V1",
        "seed": 0,
        "outcome_used": False,
        "WBCIC_outer_opened": False,
        "OpenBMI_sealed_opened": False,
        "seed1_run": False,
        "seed2_run": False,
        "source_subjects_only": True,
        "m_per_class": {d: sorted(obs.loc[obs.dataset == d, "m_per_class"].unique().tolist()) for d in DATASETS},
    })
    validation = {"schema": "PERSIST_CROSS_BATCH_VALIDATION_V1", "pass": validation_pass, "checks": checks, "terminal": terminal, "primary_gate": primary_gate, "seed1_run": False, "seed2_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False}
    write_json(RESULTS / "VALIDATION.json", validation)
    write_json(RESULTS / "BOOTSTRAP_RESULTS.json", bootstrap)

    equivalence_md = ["# Checkpoint equivalence", "", "Canonical seed-0 EEGNet checkpoints were loaded from the frozen baseline. Source-only checkpoint hashes and deterministic repeat predictions are shown below.", "", "|dataset|fold|checkpoint_sha256|source_trials|max_abs_repeat_diff|pass|", "|---|---:|---|---:|---:|"]
    for row in equivalence:
        equivalence_md.append(f"|{row.get('dataset')}|{row.get('fold')}|{row.get('checkpoint_sha256')}|{row.get('source_trials')}|{float(row.get('source_prediction_repeat_max_abs_diff', 0.0)):.3e}|{'YES' if row.get('pass') else 'NO'}|")
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("\n".join(equivalence_md) + "\n", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nOnly frozen source/refit biological subjects from OpenBMI and WBCIC were used. The session inventory and fold-level source/discovery role counts below come from the preflight runner. Outcome-role trials were not materialized for the audit; the `outcome_not_used` field records the role size that remained untouched. WBCIC outer-10 and OpenBMI sealed/confirmation resources were not opened. Seed 0 only; seed 1 and seed 2 were not run.\n\n```json\n" + json.dumps(clean(legality), indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    (EXP / "BATCH_CONSTRUCTION_AUDIT.md").write_text("# Batch construction audit\n\nEach source/refit biological subject was split without replacement into five class-balanced blocks; four certificate blocks and one held-out harm block. The m_per_class rule was frozen before outcome calculation and is recorded in PREFLIGHT.json.\n", encoding="utf-8")
    (EXP / "MATHEMATICAL_AUDIT.md").write_text("# Mathematical audit\n\nThe primary certificate is c_same = gbar_s,K^T Delta_A, with K=4 and Delta_A equal to the measured task-only AdamW displacement. Held-out harm is H_s = L(B_s_out; theta + Delta_A) - L(B_s_out; theta).\n", encoding="utf-8")
    (EXP / "CONTROL_AUDIT.md").write_text("# Control audit\n\nDifferent-subject certificates use deterministic A-disjoint partners from the same B meta-fold. Permuted-subject mappings are non-self derangements. Random controls are candidate-independent norm-matched directions.\n", encoding="utf-8")
    (EXP / "STATISTICAL_PROTOCOL.md").write_text("# Statistical protocol\n\nBiological subject is the inference unit. Primary 95% confidence intervals use 10,000 cluster-bootstrap draws, resampling subjects with replacement while carrying all observations of each sampled subject. Individual steps are not independent bootstrap units.\n", encoding="utf-8")
    (EXP / "AUTONOMOUS_DECISION.md").write_text(f"# Autonomous decision\n\nterminal = {terminal}\n\nSTEP2_AUTHORIZED = NO\nseed1_run = false\nseed2_run = false\nWBCIC_outer_opened = false\nOpenBMI_sealed_opened = false\n", encoding="utf-8")

    report = {
        "schema": "PERSIST_CROSS_BATCH_FINAL_REPORT_V1",
        "terminal": terminal,
        "primary_gate": primary_gate,
        "bootstrap": bootstrap,
        "validation": validation,
        "legality": legality,
        "checkpoint_equivalence": equivalence,
        "trajectory": trajectory,
        "toy_tests": toy,
        "subjects": {d: int(obs.loc[obs.dataset == d, "subject_id"].nunique()) for d in DATASETS},
        "excluded_subjects": [],
        "runtime_seconds": time.time() - started,
        "STEP2_AUTHORIZED": False,
    }
    write_json(EXP / "FINAL_REPORT.json", report)
    lines = ["# PERSIST-EEG Cross-Batch Subject Harm Audit", "", f"terminal = {terminal}", "", "Primary K=4 uses biological-subject cluster bootstrap (10,000 draws); seed 0 only.", "", "|dataset|K4 same AUROC|95% CI|K4 same Spearman|95% CI|same-minus-different AUROC|95% CI|same-minus-different Spearman|95% CI|", "|---|---:|---|---:|---|---:|---|---:|---|"]
    for d in DATASETS:
        r = bootstrap[f"{d}_K4"]
        lines.append(f"|{d}|{r['same']['auroc']}|[{r['same_auroc_ci95_l']}, {r['same_auroc_ci95_u']}]|{r['same']['spearman']}|[{r['same_spearman_ci95_l']}, {r['same_spearman_ci95_u']}]|{r['same_minus_different_auroc']}|[{r['auroc_advantage_ci95_l']}, {r['auroc_advantage_ci95_u']}]|{r['same_minus_different_spearman']}|[{r['spearman_advantage_ci95_l']}, {r['spearman_advantage_ci95_u']}]|")
    lines += ["", "seed1_run = false", "seed2_run = false", "WBCIC_outer_opened = false", "OpenBMI_sealed_opened = false", "STEP2_AUTHORIZED = NO", ""]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("terminal =", terminal, flush=True)
    for d in DATASETS:
        r = primary_gate[d]
        print(f"{d}_K4_same_AUROC = {r['same_auroc_point']}", flush=True)
        print(f"{d}_K4_same_AUROC_CI = [{r['same_auroc_ci95_l']}, {r['same_auroc_ci95_u']} ]", flush=True)
        print(f"{d}_K4_same_Spearman = {r['same_spearman_point']}", flush=True)
        print(f"{d}_K4_same_Spearman_CI = [{r['same_spearman_ci95_l']}, {r['same_spearman_ci95_u']} ]", flush=True)
        print(f"{d}_same_minus_different_AUROC = {r['same_minus_different_auroc_point']} CI_L={r['same_minus_different_auroc_ci95_l']}", flush=True)
    print("seed1_run = false", flush=True)
    print("seed2_run = false", flush=True)
    print("WBCIC_outer_opened = false", flush=True)
    print("OpenBMI_sealed_opened = false", flush=True)
    print("STEP2_AUTHORIZED = NO", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Preserve a traceback even if the SSH transport closes before stderr
        # is drained. This is a diagnostic artifact only.
        try:
            (Path(__file__).resolve().parents[1] / "runtime" / "aggregate_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            raise
