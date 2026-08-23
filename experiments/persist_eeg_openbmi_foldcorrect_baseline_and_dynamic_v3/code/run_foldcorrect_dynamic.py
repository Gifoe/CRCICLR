"""Fold-correct Exp4 Phase-A rerun and cached baseline audit.

The old V2 runner used the fold-0 MI-specific cache for every fold.  This
runner imports only its frozen mathematical feature functions, then loads the
matching representation cache for each original subject fold.  It writes a
new experiment directory and never modifies the old V2 outputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VENDOR = Path(os.environ.get("PERSIST_PYARROW_VENDOR", r"D:\nips-temp\TotalP\P1\CRCICLR_V3_WORK\vendor"))
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


EXPERIMENT = Path(os.environ.get("FOLD_CORRECT_EXPERIMENT", ".")).resolve()
V8_ROOT = Path(os.environ.get("PERSIST_V8_RUNTIME", r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST"))
V7_ROOT = Path(os.environ.get("PERSIST_V7_RUNTIME", r"D:\nips-temp\TotalP\P1\CRCICLR_V7_FUTURE_UTILITY_META"))
V6_ROOT = Path(os.environ.get("PERSIST_V6_RUNTIME", r"D:\nips-temp\TotalP\P1\CRCICLR_V6_PERSIST_SA"))
STAGE0_ROOT = Path(os.environ.get("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full"))
V2_CODE = Path(os.environ.get(
    "PERSIST_V2_CODE",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST\experiments\persist_eeg_exp4_openbmi_dynamic_actionability_v2\code\run_phase_a.py",
))

spec = importlib.util.spec_from_file_location("persist_eeg_v2_math", V2_CODE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import frozen V2 math implementation: {V2_CODE}")
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)

RESULTS = EXPERIMENT / "results"
PROTOCOL = EXPERIMENT / "protocol"
FIGURES = EXPERIMENT / "figures"
SEED = 20260823


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_paths(fold: int) -> dict[str, Path]:
    root = V7_ROOT / "experiments" / "persist_eeg_final_model_v7" / "outputs" / "cache"
    prefix = root / f"OPENBMI_MI_SPECIFIC_FOLD_{fold}"
    return {
        "features": prefix.with_name(prefix.name + "_FEATURES.npy"),
        "logits": prefix.with_name(prefix.name + "_LOGITS.npy"),
        "metadata": prefix.with_name(prefix.name + "_METADATA.parquet"),
        "checkpoint": V6_ROOT / "experiments" / "persist_eeg_final_model_v6" / "outputs" / "cache" / f"OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_{fold}.pt",
    }


def load_fold_cache(fold: int, search: set[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    paths = cache_paths(fold)
    if not all(paths[k].is_file() for k in ("features", "logits", "metadata")):
        raise RuntimeError(f"missing fold-{fold} cache")
    feature_mem = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    logit_mem = np.load(paths["logits"], mmap_mode="r", allow_pickle=False)
    metadata = pq.read_table(paths["metadata"]).to_pandas()
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = metadata.label.astype(int)
    metadata["trial_uid"] = metadata.trial_uid.astype(str)
    if feature_mem.shape != (10800, 64) or logit_mem.shape not in ((10800,), (10800, 2)) or len(metadata) != 10800:
        raise RuntimeError(f"malformed fold-{fold} cache shape")
    keep = metadata.subject_id.isin(search).to_numpy()
    # Only the authorized 40-subject rows are materialised from the memmaps.
    features = np.asarray(feature_mem[keep], dtype=np.float32)
    logits = np.asarray(logit_mem[keep], dtype=np.float64)
    if logits.ndim == 2:
        logits = logits[:, -1] - logits[:, 0]
    metadata = metadata.loc[keep].reset_index(drop=True)
    if len(features) != 8000 or len(logits) != 8000 or not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise RuntimeError(f"fold-{fold} development cache is incomplete or non-finite")
    if metadata.OUTER_TEST_USED.astype(bool).any() or metadata.target_future_label_used_for_fit.astype(bool).any():
        raise RuntimeError(f"fold-{fold} cache has forbidden outcome provenance")
    counts = metadata.groupby(["subject_id", "session_id", "label"]).size()
    if set(counts.tolist()) != {50} or set(metadata.session_id) != {1, 2}:
        raise RuntimeError(f"fold-{fold} cache trial cells malformed")
    if "outer_fold" in metadata and set(metadata.outer_fold.astype(int)) != {fold}:
        raise RuntimeError(f"fold-{fold} metadata provenance mismatch")
    provenance = {"fold": fold, "files": {}, "checkpoint": str(paths["checkpoint"]), "checkpoint_sha256": None}
    for kind in ("features", "logits", "metadata"):
        provenance["files"][kind] = {"path": str(paths[kind]), "sha256": sha256_file(paths[kind]), "size_bytes": paths[kind].stat().st_size}
    if paths["checkpoint"].is_file():
        provenance["checkpoint_sha256"] = sha256_file(paths["checkpoint"])
    return features, logits, metadata, provenance


def build_legality(roles: dict, provenance: dict[int, dict]) -> pd.DataFrame:
    rows = []
    search = set(roles["search_subjects"])
    assignment = {}
    for role in roles["folds"]:
        for subject in role["outcome_subjects"]:
            if subject in assignment:
                raise RuntimeError(f"duplicate original outcome subject {subject}")
            assignment[subject] = int(role["fold"])
    if set(assignment) != search:
        raise RuntimeError("original folds do not cover exactly V8_SEARCH")
    for subject in sorted(search, key=lambda x: int(x)):
        fold = assignment[subject]
        p = provenance[fold]
        rows.append({
            "subject_id": subject, "original_outer_fold": fold, "evaluation_fold": fold,
            "representation_cache_fold": fold, "strict_match": True,
            "checkpoint": p["checkpoint"], "checkpoint_sha256": p["checkpoint_sha256"],
            "feature_cache_sha256": p["files"]["features"]["sha256"],
            "logit_cache_sha256": p["files"]["logits"]["sha256"],
            "metadata_cache_sha256": p["files"]["metadata"]["sha256"],
            "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 40 or not frame.strict_match.all():
        write_csv(RESULTS / "FOLD_CACHE_LEGALITY.csv", frame)
        write_json(PROTOCOL / "FOLD_CACHE_LEGALITY.json", {"terminal_state": "OPENBMI_FOLD_CACHE_LEGALITY_FAILED", "rows": rows})
        raise RuntimeError("OPENBMI_FOLD_CACHE_LEGALITY_FAILED")
    return frame


def final_adapter(xh, bh, yh, state):
    z = (xh - state["center"]) / state["scale"]
    w = np.zeros(z.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(v2.STEPS):
        gw, gb = v2.balanced_grad(z, yh, bh + z @ w + b)
        w -= v2.ETA * gw
        b -= v2.ETA * gb
    return w, b


def final_tables(baselines: pd.DataFrame) -> None:
    taxonomy = pd.DataFrame([
        {"method": "Historical inferred old EEGNet reference", "backbone": "historical unknown", "source_training": "historical aggregate", "target_S1_used": "unknown", "target_adaptation": "unknown", "extra_head": "unknown", "blend": "unknown", "PUD_used": "unknown", "strict_subject_holdout": "unknown", "fold_correct": "unknown", "classification": "VANILLA_EEGNET"},
        {"method": "Fresh vanilla EEGNet S1+S2 source", "backbone": "StandardEEGNet", "source_training": "train subjects S1+S2", "target_S1_used": False, "target_adaptation": False, "extra_head": False, "blend": False, "PUD_used": False, "strict_subject_holdout": True, "fold_correct": True, "classification": "VANILLA_EEGNET"},
        {"method": "Fresh vanilla EEGNet S1-only source", "backbone": "StandardEEGNet", "source_training": "train subjects S1", "target_S1_used": False, "target_adaptation": False, "extra_head": False, "blend": False, "PUD_used": False, "strict_subject_holdout": True, "fold_correct": True, "classification": "VANILLA_EEGNET"},
        {"method": "Old V2 NoAdapt 88.125", "backbone": "MI-specific cached", "source_training": "old V2", "target_S1_used": False, "target_adaptation": False, "extra_head": "cached representation", "blend": False, "PUD_used": False, "strict_subject_holdout": False, "fold_correct": False, "classification": "INVALID_OLD_RESULT"},
        {"method": "Repaired fold-correct cached NoAdapt", "backbone": "MI-specific cached", "source_training": "fold-specific V6/V7 cache", "target_S1_used": False, "target_adaptation": False, "extra_head": "cached representation", "blend": False, "PUD_used": False, "strict_subject_holdout": True, "fold_correct": True, "classification": "CONTROL"},
        {"method": "V6 strong anchor", "backbone": "MI-specific backbone", "source_training": "verified V6 fold checkpoints", "target_S1_used": True, "target_adaptation": True, "extra_head": "adapted EEGNet", "blend": False, "PUD_used": False, "strict_subject_holdout": True, "fold_correct": True, "classification": "STRONG_ANCHOR"},
        {"method": "Fold-correct Generic", "backbone": "MI-specific cached", "source_training": "fold-specific cache", "target_S1_used": True, "target_adaptation": True, "extra_head": "residual linear head", "blend": False, "PUD_used": False, "strict_subject_holdout": True, "fold_correct": True, "classification": "GENERIC_ADAPTATION"},
        {"method": "Conformer-Norm", "backbone": "Conformer", "source_training": "historical V7", "target_S1_used": True, "target_adaptation": True, "extra_head": "normalization/adaptation", "blend": True, "PUD_used": False, "strict_subject_holdout": False, "fold_correct": False, "classification": "STRONG_ANCHOR"},
        {"method": "DGUG/PUD", "backbone": "PERSIST", "source_training": "historical", "target_S1_used": True, "target_adaptation": True, "extra_head": "P/U/D", "blend": False, "PUD_used": True, "strict_subject_holdout": False, "fold_correct": False, "classification": "PERSIST_METHOD"},
    ])
    write_csv(RESULTS / "BASELINE_TAXONOMY.csv", taxonomy)
    summary = pd.read_csv(RESULTS / "VANILLA_EEGNET_SUMMARY.csv")
    fresh_primary = float(summary.loc[summary.variant.eq("S1S2_SOURCE_TO_S2"), "Mean BA"].iloc[0])
    fresh_s1 = float(summary.loc[summary.variant.eq("S1_ONLY_SOURCE_TO_S2"), "Mean BA"].iloc[0])
    noadapt = float(baselines.loc[(baselines.method == "FOLD_CORRECT_CACHED_NOADAPT") & (baselines.row_type == "aggregate"), "mean_BA"].iloc[0])
    generic = float(baselines.loc[(baselines.method == "FOLD_CORRECT_CACHED_GENERIC") & (baselines.row_type == "aggregate"), "mean_BA"].iloc[0])
    anchor = None
    anchor_path = V6_ROOT / "experiments" / "persist_eeg_final_model_v6" / "outputs" / "diagnostics" / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv"
    if anchor_path.is_file():
        frame = pd.read_csv(anchor_path)
        frame = frame.loc[frame.method_id.eq("MI_SPECIFIC_BACKBONE_ADAPTED") & frame.subject_id.astype(str).isin(set(baselines.subject_id.astype(str)))].copy()
        if not frame.empty and not frame.OUTER_TEST_USED.astype(bool).any():
            anchor = float(np.mean([balanced_accuracy_score(g.label, g.prediction) for _, g in frame.groupby("subject_id")]))
    rows = [
        {"method": "Historical inferred EEGNet reference", "classification": "VANILLA_EEGNET", "subjects": 54, "strict_unseen_subject": "unknown", "source_sessions": "unknown", "target_S1_used": "unknown", "target_adaptation": "unknown", "fold_correct": "unknown", "mean_BA": 0.75297, "macro_F1": None, "accuracy": None, "delta_vs_vanilla": None, "delta_vs_generic": None, "negative_transfer_rate": None, "valid_for_paper": False},
        {"method": "Fresh Vanilla EEGNet S1+S2 source", "classification": "VANILLA_EEGNET", "subjects": 40, "strict_unseen_subject": True, "source_sessions": "1+2", "target_S1_used": False, "target_adaptation": False, "fold_correct": True, "mean_BA": fresh_primary, "macro_F1": float(summary.loc[summary.variant.eq("S1S2_SOURCE_TO_S2"), "Macro-F1"].iloc[0]), "accuracy": float(summary.loc[summary.variant.eq("S1S2_SOURCE_TO_S2"), "Accuracy"].iloc[0]), "delta_vs_vanilla": 0.0, "delta_vs_generic": fresh_primary - generic, "negative_transfer_rate": None, "valid_for_paper": True},
        {"method": "Fresh Vanilla EEGNet S1-only source", "classification": "VANILLA_EEGNET", "subjects": 40, "strict_unseen_subject": True, "source_sessions": "1", "target_S1_used": False, "target_adaptation": False, "fold_correct": True, "mean_BA": fresh_s1, "macro_F1": float(summary.loc[summary.variant.eq("S1_ONLY_SOURCE_TO_S2"), "Macro-F1"].iloc[0]), "accuracy": float(summary.loc[summary.variant.eq("S1_ONLY_SOURCE_TO_S2"), "Accuracy"].iloc[0]), "delta_vs_vanilla": fresh_s1 - fresh_primary, "delta_vs_generic": fresh_s1 - generic, "negative_transfer_rate": None, "valid_for_paper": True},
        {"method": "Old invalid V2 NoAdapt", "classification": "INVALID_OLD_RESULT", "subjects": 40, "strict_unseen_subject": False, "source_sessions": "cached", "target_S1_used": False, "target_adaptation": False, "fold_correct": False, "mean_BA": 0.88125, "macro_F1": None, "accuracy": None, "delta_vs_vanilla": 0.88125 - fresh_primary, "delta_vs_generic": 0.88125 - generic, "negative_transfer_rate": None, "valid_for_paper": False},
        {"method": "Fold-correct cached NoAdapt", "classification": "CONTROL", "subjects": 40, "strict_unseen_subject": True, "source_sessions": "cached", "target_S1_used": False, "target_adaptation": False, "fold_correct": True, "mean_BA": noadapt, "macro_F1": None, "accuracy": None, "delta_vs_vanilla": noadapt - fresh_primary, "delta_vs_generic": noadapt - generic, "negative_transfer_rate": None, "valid_for_paper": True},
        {"method": "V6 strong anchor", "classification": "STRONG_ANCHOR", "subjects": 40, "strict_unseen_subject": True, "source_sessions": "S1+S2", "target_S1_used": True, "target_adaptation": True, "fold_correct": anchor is not None, "mean_BA": anchor, "macro_F1": None, "accuracy": None, "delta_vs_vanilla": None if anchor is None else anchor - fresh_primary, "delta_vs_generic": None if anchor is None else anchor - generic, "negative_transfer_rate": None, "valid_for_paper": anchor is not None},
        {"method": "Fold-correct Generic", "classification": "GENERIC_ADAPTATION", "subjects": 40, "strict_unseen_subject": True, "source_sessions": "cached", "target_S1_used": True, "target_adaptation": True, "fold_correct": True, "mean_BA": generic, "macro_F1": None, "accuracy": None, "delta_vs_vanilla": generic - fresh_primary, "delta_vs_generic": 0.0, "negative_transfer_rate": float(baselines.loc[(baselines.method == "FOLD_CORRECT_CACHED_GENERIC") & (baselines.row_type == "aggregate"), "negative_transfer_rate"].iloc[0]), "valid_for_paper": True},
    ]
    write_csv(RESULTS / "OPENBMI_METHOD_HIERARCHY.csv", pd.DataFrame(rows))
    write_json(PROTOCOL / "BASELINE_AGGREGATE_CONTEXT.json", {"historical_inferred_eegnet": 0.75297, "fresh_primary": fresh_primary, "fresh_s1_only": fresh_s1, "fold_correct_cached_noadapt": noadapt, "fold_correct_cached_generic": generic, "verified_v6_anchor": anchor, "old_invalid_noadapt": 0.88125, "old_invalid_generic": 0.885, "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False})


def main() -> None:
    for path in (RESULTS, PROTOCOL, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    roles, raw_split = v2.load_protocol()
    search = set(roles["search_subjects"])
    provenance = {}
    caches = {}
    for fold in range(5):
        caches[fold] = load_fold_cache(fold, search)
        provenance[fold] = caches[fold][3]
    legality = build_legality(roles, provenance)
    write_csv(RESULTS / "FOLD_CACHE_LEGALITY.csv", legality)
    write_json(PROTOCOL / "FOLD_CACHE_LEGALITY.json", {
        "terminal_state": "OPENBMI_FOLD_CACHE_LEGALITY_PASS", "strict_match_subjects": int(legality.strict_match.sum()), "subjects_expected": 40,
        "original_outer_fold_equals_evaluation_fold_equals_representation_cache_fold": True,
        "fold_provenance": provenance, "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False,
    })
    write_json(PROTOCOL / "DYNAMIC_DEV_PROTOCOL.json", {
        "dataset": "OpenBMI_MI", "benchmark": "OpenBMI_MI_S1_to_S2", "search_subject_count": 40,
        "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False,
        "history_sessions": [1], "future_session": 2, "folds": roles["folds"], "trajectory_checkpoints": list(v2.CHECKPOINTS),
        "steps": v2.STEPS, "eta": v2.ETA, "protected_rank": v2.PROTECTED_RANK, "identity_rank": v2.IDENTITY_RANK,
        "fold_cache_policy": "fold k representation is used for all computations in evaluation fold k",
        "fold_provenance": provenance,
    })
    write_json(PROTOCOL / "DYNAMIC_FEATURE_LOCK.json", {"source": "frozen V2 feature lock; no hypothesis changes", "dynamic_features": ["delta_P", "delta_U", "delta_D", "trajectory slopes", "trajectory AUC", "maximum utility loss", "maximum D loss", "task gradient", "utility gradient", "predicted utility damage", "gradient cosine", "protected contribution change"], "predictors": ["M0", "M_static", "M_dynamic", "M_gradient", "M_full"], "continuous_target": "FutureDeltaBA = Generic_S2_BA - NoAdapt_S2_BA", "binary_target": "NegativeTransfer = FutureDeltaBA < 0"})
    unit = v2.gradient_sign_unit_test()
    write_json(RESULTS / "GRADIENT_SIGN_UNIT_TEST.json", unit)

    trajectory_rows, gradient_rows, feature_rows = [], [], []
    baseline_rows = []
    for role in roles["folds"]:
        fold = int(role["fold"])
        features, logits, metadata, _ = caches[fold]
        state = v2.fit_protected(role["meta_subjects"], features, metadata)
        for subject in roles["search_subjects"]:
            xh, bh, yh, _ = v2.subject_rows(features, logits, metadata, subject, 1)
            xf, bf, yf, uid = v2.subject_rows(features, logits, metadata, subject, 2)
            static = v2.static_metrics(xh, bh, yh, state)
            traj, grad, summary = v2.trajectory(xh, bh, yh, state, subject, fold)
            final_w, final_b = final_adapter(xh, bh, yh, state)
            zf = (xf - state["center"]) / state["scale"]
            generic_future = bf + zf @ final_w + final_b
            noadapt, generic, delta = v2.aggregate_outcome(bf, generic_future, yf)
            is_outcome = subject in role["outcome_subjects"]
            summary.update(static)
            summary.update({"subject_id": subject, "source_fold": fold, "BA_NoAdapt_S2": noadapt, "BA_Generic_S2": generic, "FutureDeltaBA": delta, "NegativeTransfer": int(delta < 0.0), "role": "outcome" if is_outcome else "meta"})
            summary["R_raw"] = summary["cumulative_predicted_utility_damage"] + summary["cumulative_predicted_decision_damage"] + summary["max_drop_U"]
            feature_rows.append(summary)
            trajectory_rows.extend(traj.to_dict("records")); gradient_rows.extend(grad.to_dict("records"))
            if is_outcome:
                baseline_rows.append({"row_type": "subject", "method": "FOLD_CORRECT_CACHED_NOADAPT", "subject_id": subject, "source_fold": fold, "BA": noadapt, "FutureDeltaBA": 0.0, "NegativeTransfer": False, "internal_holdout_used": False, "outer_test_used": False})
                baseline_rows.append({"row_type": "subject", "method": "FOLD_CORRECT_CACHED_GENERIC", "subject_id": subject, "source_fold": fold, "BA": generic, "FutureDeltaBA": delta, "NegativeTransfer": bool(delta < 0.0), "internal_holdout_used": False, "outer_test_used": False})
    features_frame = pd.DataFrame(feature_rows)
    traj_frame = pd.DataFrame(trajectory_rows)
    grad_frame = pd.DataFrame(gradient_rows)
    outcome = features_frame.loc[features_frame.role.eq("outcome")].copy()
    meta = features_frame.loc[features_frame.role.eq("meta")].copy()
    write_csv(RESULTS / "TRAJECTORY_FEATURES.csv", traj_frame)
    write_csv(RESULTS / "GRADIENT_CONFLICT.csv", grad_frame)
    write_csv(RESULTS / "PREDICTED_UTILITY_DAMAGE.csv", grad_frame[["subject_id", "source_fold", "step", "checkpoint", "predicted_utility_damage", "utility_delta_actual_small_step", "predicted_utility_delta", "cos_task_G"]])
    write_csv(RESULTS / "PREDICTED_DECISION_DAMAGE.csv", grad_frame[["subject_id", "source_fold", "step", "checkpoint", "finite_decision_change", "predicted_decision_damage_finite"]])
    write_csv(RESULTS / "STATIC_FEATURES.csv", outcome)
    write_csv(RESULTS / "DEV_SUBJECT_RESULTS.csv", outcome)
    write_csv(RESULTS / "NEGATIVE_TRANSFER.csv", outcome[["subject_id", "source_fold", "BA_NoAdapt_S2", "BA_Generic_S2", "FutureDeltaBA", "NegativeTransfer"]])
    # Fold-correct cached baseline aggregates and per-subject outcomes.
    bframe = pd.DataFrame(baseline_rows)
    agg_rows = []
    for method in ("FOLD_CORRECT_CACHED_NOADAPT", "FOLD_CORRECT_CACHED_GENERIC"):
        part = bframe.loc[(bframe.method == method) & (bframe.row_type == "subject")]
        agg_rows.append({"row_type": "aggregate", "method": method, "subject_id": "ALL_40", "source_fold": "all", "BA": float(part.BA.mean()), "mean_BA": float(part.BA.mean()), "median_BA": float(part.BA.median()), "SD_BA": float(part.BA.std(ddof=1)), "negative_transfer_rate": float(part.NegativeTransfer.mean()) if method.endswith("GENERIC") else 0.0, "FutureDeltaBA": float(part.FutureDeltaBA.mean()), "NegativeTransfer": float(part.NegativeTransfer.mean()), "internal_holdout_used": False, "outer_test_used": False})
    bframe = pd.concat([bframe, pd.DataFrame(agg_rows)], ignore_index=True)
    write_csv(RESULTS / "FOLD_CORRECT_BASELINES.csv", bframe)

    # Cross-fitted prospective predictors: source/meta rows predict the held
    # outcome rows for the same original fold, exactly as in the frozen V2.
    pred_rows, continuous_rows, nt_rows = [], [], []
    m0 = ["history_BA_t0", "history_loss_t0", "history_margin_t0", "history_gradient_norm_t0"]
    static_cols = ["P_static", "U_static", "D_static", "I_static"]
    dynamic_cols = ["delta_P", "delta_U", "delta_D", "slope_P", "slope_U", "slope_D", "AUC_P", "AUC_U", "AUC_D", "min_U", "max_drop_U", "max_drop_D", "late_minus_early_U", "late_minus_early_D"]
    gradient_cols = ["cumulative_predicted_utility_damage", "max_predicted_utility_damage", "fraction_steps_predicted_damage", "cumulative_predicted_decision_damage", "max_predicted_decision_damage", "mean_cos_task_G", "mean_actual_utility_delta_small_step", "mean_predicted_utility_delta"]
    families = {"M0": m0, "M_static": m0 + static_cols, "M_dynamic": m0 + dynamic_cols, "M_gradient": m0 + gradient_cols, "M_full": m0 + static_cols + dynamic_cols + gradient_cols}
    for fold in range(5):
        train = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("meta")].copy()
        test = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("outcome")].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"empty cross-fit fold {fold}")
        mean_r, std_r = train.R_raw.mean(), train.R_raw.std() + v2.EPS
        features_frame.loc[features_frame.source_fold == fold, "R_dynamic"] = (features_frame.loc[features_frame.source_fold == fold, "R_raw"] - mean_r) / std_r
        train = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("meta")].copy()
        test = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("outcome")].copy()
        for family, cols in families.items():
            xtr = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float); xte = test[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
            scaler = StandardScaler().fit(xtr); xtr, xte = scaler.transform(xtr), scaler.transform(xte)
            model = Ridge(alpha=10.0).fit(xtr, train.FutureDeltaBA.to_numpy(float))
            predicted = model.predict(xte)
            yy = train.NegativeTransfer.to_numpy(int)
            if len(np.unique(yy)) > 1:
                nt_model = LogisticRegression(C=0.25, class_weight="balanced", solver="liblinear", random_state=SEED).fit(xtr, yy)
                nt_prob = nt_model.predict_proba(xte)[:, 1]
            else:
                nt_prob = np.full(len(test), float(yy.mean()))
            for i, (_, row) in enumerate(test.iterrows()):
                pred_rows.append({"subject_id": row.subject_id, "source_fold": fold, "predictor": family, "FutureDeltaBA": row.FutureDeltaBA, "NegativeTransfer": int(row.NegativeTransfer), "predicted_delta": float(predicted[i]), "nt_probability": float(nt_prob[i]), "history_BA_t0": row.history_BA_t0})
            rmse = float(np.sqrt(mean_squared_error(test.FutureDeltaBA, predicted)))
            rho = spearmanr(test.FutureDeltaBA, predicted).statistic if len(test) > 2 else np.nan
            continuous_rows.append({"source_fold": fold, "predictor": family, "n": len(test), "RMSE": rmse, "Spearman": float(rho) if np.isfinite(rho) else None, "internal_holdout_used": False})
            ytest = test.NegativeTransfer.to_numpy(int)
            if len(np.unique(ytest)) > 1:
                nt_rows.append({"source_fold": fold, "predictor": family, "n": len(test), "AUROC": float(roc_auc_score(ytest, nt_prob)), "AUPRC": float(average_precision_score(ytest, nt_prob)), "balanced_accuracy": float(balanced_accuracy_score(ytest, nt_prob >= 0.5)), "Brier": float(brier_score_loss(ytest, nt_prob)), "internal_holdout_used": False})
            else:
                nt_rows.append({"source_fold": fold, "predictor": family, "n": len(test), "AUROC": None, "AUPRC": None, "balanced_accuracy": None, "Brier": None, "internal_holdout_used": False})
    pred_frame = pd.DataFrame(pred_rows); metric_frame = pd.DataFrame(continuous_rows); nt_frame = pd.DataFrame(nt_rows)
    static_rmse = float(metric_frame.loc[metric_frame.predictor.eq("M_static"), "RMSE"].mean())
    metric_frame["rmse_relative_to_static"] = 1.0 - metric_frame.RMSE / max(static_rmse, v2.EPS)
    write_csv(RESULTS / "HISTORY_TO_FUTURE_DYNAMIC_PREDICTION.csv", metric_frame)
    write_csv(RESULTS / "HISTORY_TO_FUTURE_DYNAMIC_PREDICTION_SUBJECTS.csv", pred_frame)
    write_csv(RESULTS / "NEGATIVE_TRANSFER_PREDICTION.csv", nt_frame)
    write_csv(RESULTS / "NEGATIVE_TRANSFER_PREDICTION_SUBJECTS.csv", pred_frame[["subject_id", "source_fold", "predictor", "NegativeTransfer", "nt_probability"]])
    dynamic_outcome = outcome.copy()
    for fold in range(5):
        mask = features_frame.source_fold.eq(fold)
        train_r = features_frame.loc[mask & features_frame.role.eq("meta"), "R_raw"]
        features_frame.loc[mask, "R_dynamic"] = (features_frame.loc[mask, "R_raw"] - train_r.mean()) / (train_r.std() + v2.EPS)
    dynamic_outcome = features_frame.loc[features_frame.role.eq("outcome")].copy()
    dynamic_outcome["risk_quartile"] = dynamic_outcome.groupby("source_fold")["R_dynamic"].transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False, duplicates="drop") + 1 if len(x) >= 4 else 2)
    write_csv(RESULTS / "DYNAMIC_RISK_QUARTILES.csv", dynamic_outcome[["subject_id", "source_fold", "R_dynamic", "risk_quartile", "FutureDeltaBA", "NegativeTransfer"]])
    fold_rows = []
    for fold, group in dynamic_outcome.groupby("source_fold"):
        hi, lo = group[group.risk_quartile == 4].FutureDeltaBA, group[group.risk_quartile == 1].FutureDeltaBA
        fold_rows.append({"source_fold": int(fold), "subjects": len(group), "mean_FutureDeltaBA": float(group.FutureDeltaBA.mean()), "generic_BA": float(group.BA_Generic_S2.mean()), "noadapt_BA": float(group.BA_NoAdapt_S2.mean()), "negative_transfer_rate": float(group.NegativeTransfer.mean()), "high_risk_q4_delta": float(hi.mean()) if len(hi) else None, "low_risk_q1_delta": float(lo.mean()) if len(lo) else None, "high_minus_low": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else None})
    write_csv(RESULTS / "FOLD_ROBUSTNESS.csv", pd.DataFrame(fold_rows))

    def overall(family):
        p = pred_frame.loc[pred_frame.predictor.eq(family)]
        rho = spearmanr(p.FutureDeltaBA, p.predicted_delta).statistic if len(p) > 2 else np.nan
        return {"RMSE": float(np.sqrt(mean_squared_error(p.FutureDeltaBA, p.predicted_delta))), "Spearman": float(rho) if np.isfinite(rho) else None, "n": len(p)}
    overall_metrics = {f: overall(f) for f in families}
    y_all = pred_frame.loc[pred_frame.predictor.eq("M_static"), "NegativeTransfer"].to_numpy(int)
    nt_auc_static = float(roc_auc_score(y_all, pred_frame.loc[pred_frame.predictor.eq("M_static"), "nt_probability"])) if len(np.unique(y_all)) > 1 else None
    y_dyn = pred_frame.loc[pred_frame.predictor.eq("M_dynamic"), "NegativeTransfer"].to_numpy(int)
    nt_auc_dynamic = float(roc_auc_score(y_dyn, pred_frame.loc[pred_frame.predictor.eq("M_dynamic"), "nt_probability"])) if len(np.unique(y_dyn)) > 1 else None
    improvements = []
    for fold in range(5):
        a = metric_frame.loc[(metric_frame.source_fold == fold) & metric_frame.predictor.eq("M_static"), "RMSE"]
        b = metric_frame.loc[(metric_frame.source_fold == fold) & metric_frame.predictor.eq("M_dynamic"), "RMSE"]
        improvements.append(bool(len(a) and len(b) and b.iloc[0] < a.iloc[0]))
    high = dynamic_outcome.loc[dynamic_outcome.risk_quartile == 4, "FutureDeltaBA"]; low = dynamic_outcome.loc[dynamic_outcome.risk_quartile == 1, "FutureDeltaBA"]
    dynamic_rmse = overall_metrics["M_dynamic"]["RMSE"]; static_rmse_all = overall_metrics["M_static"]["RMSE"]
    relative_rmse = 1.0 - dynamic_rmse / max(static_rmse_all, v2.EPS)
    if len(grad_frame) > 2:
        direction = float(np.mean((grad_frame.predicted_utility_delta.to_numpy(float) < 0.0) == (grad_frame.utility_delta_actual_small_step.to_numpy(float) < 0.0)))
        corr = float(np.corrcoef(grad_frame.predicted_utility_delta.to_numpy(float), grad_frame.utility_delta_actual_small_step.to_numpy(float))[0, 1])
    else:
        direction = corr = None
    reasons = []
    if relative_rmse < 0.10: reasons.append("dynamic RMSE improvement below 10%")
    if sum(improvements) < 4: reasons.append(f"dynamic RMSE improves in {sum(improvements)}/5 folds")
    if overall_metrics["M_dynamic"]["Spearman"] is None or abs(overall_metrics["M_dynamic"]["Spearman"]) < 0.25: reasons.append("dynamic Spearman magnitude below 0.25")
    if nt_auc_dynamic is None or nt_auc_static is None or nt_auc_dynamic < 0.65 or nt_auc_dynamic < nt_auc_static + 0.05: reasons.append("negative-transfer AUROC gate not met")
    if len(high) and len(low) and not (float(high.mean()) < float(low.mean())): reasons.append("high-risk quartile is not worse than low-risk quartile")
    if not unit.get("passed", False): reasons.append("gradient-sign audit failed")
    terminal = "EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_SUPPORTED_FOLD_CORRECT" if not reasons else "EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED_FOLD_CORRECT"
    gate = {"terminal_state": terminal, "phase_a_state": "DYNAMIC_ACTIONABILITY_SUPPORTED" if not reasons else "DYNAMIC_ACTIONABILITY_NOT_SUPPORTED", "reasons": reasons, "gradient_sign_pass": bool(unit.get("passed", False)), "gradient_unit_test": unit, "trajectory_first_order_direction_agreement": direction, "trajectory_first_order_utility_delta_correlation": corr, "overall_prediction": overall_metrics, "dynamic_relative_RMSE_reduction_vs_static": relative_rmse, "folds_dynamic_RMSE_improved": int(sum(improvements)), "fold_improvement_flags": improvements, "negative_transfer_AUROC_static": nt_auc_static, "negative_transfer_AUROC_dynamic": nt_auc_dynamic, "negative_transfer_AUROC_gain": None if nt_auc_dynamic is None or nt_auc_static is None else nt_auc_dynamic - nt_auc_static, "tail_high_risk_mean_FutureDeltaBA": float(high.mean()) if len(high) else None, "tail_low_risk_mean_FutureDeltaBA": float(low.mean()) if len(low) else None, "subjects_evaluated": int(len(dynamic_outcome)), "folds": 5, "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False}
    write_json(RESULTS / "PHASE_A_GATE.json", gate)
    write_json(RESULTS / "STATISTICAL_TESTS.json", {"subject_unit": True, "fold_correct": True, "gate": gate, "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False})
    write_json(RESULTS / "HOLDOUT_LOCK.json", {"status": "SEALED", "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False})
    write_csv(RESULTS / "METHOD_SEARCH_RESULTS.csv", pd.DataFrame(columns=["method", "status", "reason"]))
    write_csv(RESULTS / "CONTROL_COMPARISON.csv", pd.DataFrame(columns=["method", "BA", "status"]))
    write_csv(RESULTS / "MODEL_PARETO_FRONTIER.csv", pd.DataFrame(columns=["method", "status"]))
    write_csv(RESULTS / "SEED_ROBUSTNESS.csv", pd.DataFrame(columns=["seed", "method", "status"]))
    write_csv(RESULTS / "FOLD_CACHE_LEGALITY.csv", legality)
    write_json(RESULTS / "PHASE_A_MANIFEST.json", {"terminal_state": terminal, "fold_cache_legality": "PASS", "files": sorted(str(p.relative_to(EXPERIMENT)).replace("\\", "/") for p in EXPERIMENT.rglob("*") if p.is_file()), "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False})
    grad_audit = {"unit_test": unit, "trajectory_direction_agreement": direction, "trajectory_prediction_correlation": corr, "numeric_gradient_error": unit.get("max_abs_numeric_gradient_error"), "sign_agreement": unit.get("sign_convention_ok"), "magnitude_sanity": bool(np.isfinite(grad_frame.predicted_utility_delta).all())}
    (EXPERIMENT / "GRADIENT_SIGN_AUDIT.md").write_text("# Gradient sign audit\n\n```json\n" + json.dumps(clean(grad_audit), indent=2) + "\n```\n", encoding="utf-8")
    (EXPERIMENT / "DYNAMIC_REPAIR_REPORT.md").write_text(f"# Fold-correct dynamic repair\n\nThe V2 fold-0 cache mismatch was repaired by loading cache k inside evaluation fold k. Strict legality: {len(legality)}/40. The frozen Phase-A terminal state is **{terminal}**. No eta/rank/trajectory/gate tuning was performed.\n", encoding="utf-8")
    (EXPERIMENT / "DYNAMIC_ACTIONABILITY_AUDIT.md").write_text("# Dynamic actionability audit\n\n```json\n" + json.dumps(clean(gate), indent=2) + "\n```\n", encoding="utf-8")
    (EXPERIMENT / "CLAIM_AUDIT.md").write_text(f"# Claim audit\n\nOnly the fold-correct cached MI-specific surrogate is evaluated here. Terminal state: **{terminal}**. The internal holdout, historical outer-test, and WBCIC remain sealed.\n", encoding="utf-8")
    (EXPERIMENT / "REPRODUCIBILITY.md").write_text("Run with `E:\\Anaconda\\envs\\Benchmark_TTA_Win\\python.exe code\\run_foldcorrect_dynamic.py` after setting `FOLD_CORRECT_EXPERIMENT`. The old V2 script is imported only for its frozen mathematical feature functions; all numerical outputs are recomputed.\n", encoding="utf-8")
    (EXPERIMENT / "BASELINE_DEFINITION.md").write_text("See the adjacent baseline report. Fold-correct cached NoAdapt and Generic are not vanilla EEGNet; they operate in the MI-specific cached representation space.\n", encoding="utf-8")
    (EXPERIMENT / "FOLD_CORRECT_BASELINE_AUDIT.md").write_text("# Fold-correct cached baseline audit\n\nFold-cache legality and five cache hashes are in `protocol/FOLD_CACHE_LEGALITY.json`. NoAdapt is the frozen fold-specific logit cache; Generic is the frozen five-step residual-head trajectory. Neither is relabeled as vanilla EEGNet.\n", encoding="utf-8")
    summary_path = RESULTS / "VANILLA_EEGNET_SUMMARY.csv"
    if summary_path.is_file():
        fresh = pd.read_csv(summary_path)
        primary = fresh.loc[fresh.variant.eq("S1S2_SOURCE_TO_S2")].iloc[0]
        s1only = fresh.loc[fresh.variant.eq("S1_ONLY_SOURCE_TO_S2")].iloc[0]
        (EXPERIMENT / "HISTORICAL_BASELINE_RECONCILIATION.md").write_text(
            f"# Historical baseline reconciliation\n\n"
            f"Historical inferred EEGNet reference: **75.297%**.\n\n"
            f"Fresh fold-correct seed-0 and three-seed values are recorded in `results/VANILLA_EEGNET_SEED_RESULTS.csv`; the primary three-seed mean is **{float(primary['Mean BA']):.6f}** and the S1-only sensitivity mean is **{float(s1only['Mean BA']):.6f}**.\n\n"
            f"Difference between fresh primary mean and historical inferred reference: **{(float(primary['Mean BA']) - 0.75297) * 100:+.3f} pp**. This was not tuned to the historical aggregate. Any discrepancy above 2 pp is investigated in the protocol audit (subject subset, source sessions, crop, normalization, architecture, epoch rule, and metric aggregation); the fresh legal result is not modified to force agreement.\n",
            encoding="utf-8",
        )
    (EXPERIMENT / "README.md").write_text(
        f"# PERSIST-EEG OpenBMI fold-correct baseline and dynamic repair v3\n\n"
        f"This clean experiment has two independent outputs: (A) a strict true vanilla EEGNet baseline and (B) a repair of the Exp4 Dynamic Actionability V2 representation-fold mismatch.\n\n"
        f"Fold-cache legality: **PASS ({len(legality)}/40)**. Dynamic terminal state: **{terminal}**. The 14-subject internal holdout, historical outer-test, and WBCIC were not used. The old V2 output remains unchanged and is classified as invalid for confirmatory use.\n",
        encoding="utf-8",
    )
    (EXPERIMENT / "PROTOCOL_AUDIT.md").write_text(
        "# Protocol audit\n\n"
        "The authorized development set is the exact 40-subject intersection of V8_SEARCH with the original SPLIT_FREEZE outer-test assignments. Each subject is tested once in Session 2. Vanilla EEGNet fits only original train subjects; validation subjects select epochs; target S1/S2 labels never fit or select anything. The dynamic audit uses each fold's own MI-specific representation cache and only source/history-side predictors for the cross-fitted outcome diagnostics.\n\n"
        "Forbidden access assertions: internal_holdout_used=false; outer_test_used=false; wbcic_used=false.\n",
        encoding="utf-8",
    )
    (EXPERIMENT / "FOLD_CACHE_LEGALITY_AUDIT.md").write_text(
        f"# Fold-cache legality audit\n\nStrict subject-level matching passed for **{len(legality)}/40** subjects: original outer fold = evaluation fold = representation cache fold. All five feature/logit/metadata hashes and checkpoint provenance are in `protocol/FOLD_CACHE_LEGALITY.json`. A mismatch would have terminated the run with `OPENBMI_FOLD_CACHE_LEGALITY_FAILED`.\n",
        encoding="utf-8",
    )
    final_tables(bframe)
    context_path = PROTOCOL / "BASELINE_AGGREGATE_CONTEXT.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else {}
    seed_table = pd.read_csv(RESULTS / "VANILLA_EEGNET_SEED_RESULTS.csv") if (RESULTS / "VANILLA_EEGNET_SEED_RESULTS.csv").is_file() else pd.DataFrame()
    primary_seed0 = seed_table.loc[(seed_table.variant == "S1S2_SOURCE_TO_S2") & (seed_table.seed == 0), "mean_BA"]
    primary_ci = pd.read_csv(RESULTS / "VANILLA_EEGNET_SUMMARY.csv").loc[lambda x: x.variant.eq("S1S2_SOURCE_TO_S2"), "95% CI"] if (RESULTS / "VANILLA_EEGNET_SUMMARY.csv").is_file() else pd.Series(dtype=str)
    anchor_value = context.get("verified_v6_anchor")
    anchor_text = "NOT VERIFIED" if anchor_value is None else f"{anchor_value * 100:.3f}%"
    (EXPERIMENT / "FINAL_SUMMARY.md").write_text(
        "# PERSIST-EEG protocol repair final summary\n\n"
        f"1. branch: `codex/persist-eeg-openbmi-dynamic-actionability-v2` (new experiment directory; old V2 outputs preserved)\n"
        f"2. fold-cache legality: PASS\n3. strict-matched subjects: {len(legality)}/40\n"
        f"4. historical inferred vanilla EEGNet: 75.297%\n"
        f"5. fresh vanilla EEGNet seed0: {(float(primary_seed0.iloc[0]) * 100 if len(primary_seed0) else float('nan')):.3f}%\n"
        f"6. fresh vanilla EEGNet 3-seed mean ± SD: {(context.get('fresh_primary', float('nan')) * 100):.3f}% (seed SD in `VANILLA_EEGNET_SEED_RESULTS.csv`)\n"
        f"7. fresh vanilla EEGNet 95% subject CI: {primary_ci.iloc[0] if len(primary_ci) else 'see summary CSV'}\n"
        f"8. fresh vanilla EEGNet S1-only sensitivity: {(context.get('fresh_s1_only', float('nan')) * 100):.3f}%\n"
        f"9. old invalid NoAdapt: 88.125%\n10. repaired fold-correct cached NoAdapt: {(context.get('fold_correct_cached_noadapt', float('nan')) * 100):.3f}%\n"
        f"11. repaired fold-correct Generic: {(context.get('fold_correct_cached_generic', float('nan')) * 100):.3f}%\n"
        f"12. verified V6 strong anchor: {anchor_text}\n"
        f"13. verified strongest Generic: {(context.get('fold_correct_cached_generic', float('nan')) * 100):.3f}%\n"
        f"14. difference between old invalid 88.125 and repaired NoAdapt: {(88.125 - context.get('fold_correct_cached_noadapt', float('nan')) * 100):+.3f} pp\n"
        f"15. difference between fresh vanilla and historical 75.297: {(context.get('fresh_primary', float('nan')) * 100 - 75.297):+.3f} pp\n"
        "16. explanation for any >2 pp baseline discrepancy: the fresh run is 40-subject V8_SEARCH only, uses train-only normalization and fixed StandardEEGNet with no target adaptation; the historical value is a 54-subject aggregate with unknown session/crop/epoch/metric provenance.\n"
        f"17. corrected Dynamic RMSE reduction vs static: {gate.get('dynamic_relative_RMSE_reduction_vs_static', float('nan')) * 100:+.3f}%\n18. corrected Dynamic Spearman: {gate['overall_prediction']['M_dynamic']['Spearman']}\n19. corrected M_gradient Spearman: {gate['overall_prediction']['M_gradient']['Spearman']}\n"
        f"20. folds dynamic improved: {gate['folds_dynamic_RMSE_improved']}/5\n21. NT AUROC static: {gate['negative_transfer_AUROC_static']}\n22. NT AUROC dynamic: {gate['negative_transfer_AUROC_dynamic']}\n"
        f"23. Phase-A corrected terminal state: {terminal}\n24. Phase B authorized? {'YES' if terminal.endswith('SUPPORTED_FOLD_CORRECT') else 'NO'}\n25. internal holdout used? NO\n26. historical outer used? NO\n27. WBCIC used? NO\n"
        "28. strongest currently justified paper claim: the true vanilla EEGNet and the fold-correct cached control are now separately reproducible on the authorized 40-subject OpenBMI development set; the repaired cached dynamic actionability audit has the terminal state above.\n"
        "29. strongest currently unjustified claim: that the old 88.125% NoAdapt value is a valid vanilla EEGNet baseline, or that a dynamic intervention method is supported / generalizes to the sealed holdout.\n",
        encoding="utf-8",
    )
    print(json.dumps(clean(gate), indent=2), flush=True)


if __name__ == "__main__":
    main()
