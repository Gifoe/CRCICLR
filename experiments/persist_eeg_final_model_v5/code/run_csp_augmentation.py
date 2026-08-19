"""Add a subject-specific broad-band CSP expert to the local-history stack."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from common import CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, default_wbcic_repo, logit, stable_seed, ensure_directories, sha256_file, write_csv, write_json
from datasets import load_wbcic
from evaluation import summarize
from models import csp
from nested_cv import fold_assignment
from run_refit_disagreement import _fixed_outer
from run_reliability_stack import _build_features
from training import OOFResult


def _epoch_cache_root() -> Path:
    return (
        default_wbcic_repo()
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "cache"
        / "wbcic_epochs"
    )


def _covariances(path: Path, sos: np.ndarray, batch_size: int = 20) -> np.ndarray:
    epochs = np.load(path, mmap_mode="r", allow_pickle=False)
    if epochs.ndim != 3 or epochs.shape[1:] != (58, 1000):
        raise RuntimeError(f"Malformed WBCIC epoch cache: {path} {epochs.shape}")
    output = np.empty((len(epochs), 58, 58), dtype=np.float32)
    for start in range(0, len(epochs), batch_size):
        stop = min(start + batch_size, len(epochs))
        value = np.asarray(epochs[start:stop], dtype=np.float32)
        value = sosfiltfilt(sos, value, axis=-1).astype(np.float32)
        value -= value.mean(axis=-1, keepdims=True)
        covariance = np.einsum("nct,ndt->ncd", value, value, optimize=True)
        trace = np.trace(covariance, axis1=1, axis2=2)
        covariance /= np.maximum(trace[:, None, None], 1e-12)
        output[start:stop] = covariance
    return output


def _subject_csp(subject: str, cache_root: Path):
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=250.0, output="sos")
    covariances, labels, sessions = [], [], []
    for session in (0, 1, 2):
        epoch_path = cache_root / subject / f"ses-{session}_epochs.npy"
        label_path = cache_root / subject / f"ses-{session}_labels.npy"
        covariance = _covariances(epoch_path, sos)
        label = np.load(label_path, allow_pickle=False).astype(int)
        if len(covariance) != len(label) or set(label.tolist()) != {0, 1}:
            raise RuntimeError(f"Malformed labels for {subject}/session-{session}")
        covariances.append(covariance); labels.append(label); sessions.append(np.full(len(label), session, dtype=int))
    covariance = np.concatenate(covariances)
    label = np.concatenate(labels)
    session = np.concatenate(sessions)
    history = session < 2
    target = session == 2
    configurations = [
        {"pairs": pairs, "C": c_value}
        for pairs in (2, 3, 4)
        for c_value in (0.01, 0.1, 1.0)
    ]
    seed = stable_seed("V5_CSP", subject)
    selected, records = csp.select_history_configuration(
        covariance[history], label[history], session[history], configurations, seed
    )
    filters = csp.spatial_filters(covariance[history], label[history], int(selected["pairs"]))
    x_history = csp.features(covariance[history], filters)
    x_target = csp.features(covariance[target], filters)
    model = csp.build_head(float(selected["C"]), seed)
    model.fit(x_history, label[history])
    probability = model.predict_proba(x_target)[:, 1]
    rows = [
        {
            "subject_id": subject,
            **record,
            "selected": record["configuration"] == selected,
            "band_hz": "8-30",
            "target_prior_sessions_used": True,
            "target_S3_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        }
        for record in records
    ]
    return subject, probability, label[target], rows


def _extract_all(data, workers: int):
    cache_root = _epoch_cache_root()
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    subjects = sorted(np.unique(data.subjects).tolist())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        jobs = {executor.submit(_subject_csp, subject, cache_root): subject for subject in subjects}
        for future in as_completed(jobs):
            subject, probability, label, subject_rows = future.result()
            result[subject] = (probability, label)
            rows.extend(subject_rows)
            print(f"[CSP] completed {subject} ({len(result)}/{len(subjects)})", flush=True)
    aligned = np.full(len(data.labels), np.nan, dtype=float)
    for subject in subjects:
        mask = data.subjects == subject
        order = np.argsort(data.metadata.loc[mask, "trial_index_within_subject_session"].to_numpy(int))
        indices = np.flatnonzero(mask)[order]
        probability, label = result[subject]
        if len(indices) != len(probability) or not np.array_equal(data.labels[indices], label):
            raise RuntimeError(f"CSP S3 alignment mismatch for {subject}")
        aligned[indices] = probability
    if np.isnan(aligned).any():
        raise RuntimeError("Incomplete CSP predictions")
    return aligned, pd.DataFrame(rows)


def _fixed_result(data, method_id, probability):
    return OOFResult(method_id, probability, (probability >= 0.5).astype(int), fold_assignment(data), pd.DataFrame())


def run(workers: int) -> None:
    ensure_directories()
    data = load_wbcic()
    csp_probability, csp_selections = _extract_all(data, workers)
    simple, _, _ = _build_features(data)
    csp_features = np.column_stack(
        [
            logit(csp_probability),
            csp_probability,
            np.abs(csp_probability - 0.5),
            (csp_probability >= 0.5).astype(float),
        ]
    )
    augmented = np.column_stack([simple, csp_features]).astype(np.float32)
    jobs = [
        ("M13_SUBJECT_CSP_CONTROL", "subject_csp_control", _fixed_result(data, "M13_SUBJECT_CSP_CONTROL", csp_probability)),
        ("M13_CSP_AUGMENTED_TRAIN3", "csp_augmented_stack", _fixed_outer(data, "M13_CSP_AUGMENTED_TRAIN3", augmented, fit_scope="all", refit_nonoutcome=False, c_value=1.0)),
        ("M13_CSP_AUGMENTED_REFIT4", "csp_augmented_stack", _fixed_outer(data, "M13_CSP_AUGMENTED_REFIT4", augmented, fit_scope="all", refit_nonoutcome=True, c_value=1.0)),
    ]
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for method_id, family, result in jobs:
        row, subject, fold = summarize(data, method_id, result.prediction, result.probability, result.outer_fold, baseline="current")
        row.update({"architecture_family": family, "target_prior_sessions_used": True})
        rows.append(row); subjects.append(subject); folds.append(fold)
        if not result.selections.empty:
            selections.append(result.selections)
        predictions.append(
            pd.DataFrame(
                {
                    "dataset": data.dataset_id,
                    "trial_uid": data.trial_uid,
                    "subject_id": data.subjects,
                    "method_id": method_id,
                    "outer_fold": result.outer_fold,
                    "label": data.labels,
                    "B_STRONG_CURRENT_prediction": data.current_prediction,
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "target_prior_sessions_used": True,
                    "target_S3_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
        )
        print(json.dumps(row, indent=2), flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(["Delta_BA", "NLL"], ascending=[False, True])
    write_csv(LEADERBOARD / "WBCIC_CSP_AUGMENTATION.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_CSP_HISTORY_SELECTIONS.csv", csp_selections)
    write_csv(DIAGNOSTICS / "WBCIC_CSP_AUGMENTATION_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_CSP_AUGMENTATION_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_CSP_AUGMENTATION_SELECTIONS.csv", pd.concat(selections, ignore_index=True) if selections else pd.DataFrame())
    write_csv(DIAGNOSTICS / "WBCIC_CSP_AUGMENTATION_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_008.json",
        {
            "previous_failure": "The deep local-history stack reached +0.928 pp, leaving a small but real gap to the +1 pp target.",
            "hypothesis": "A subject-specific 8-30 Hz CSP expert captures spatial motor-imagery structure complementary to frozen deep representations.",
            "what_changed": "Added a classical CSP-logistic expert selected only by S1<->S2 validation and tested it as both a direct control and an extra fixed-stack input.",
            "grouped_result": best,
            "development_reuse_note": "Exploratory iteration after inspecting prior development OOF results; sealed outer data remained untouched.",
            "conclusion": "KEEP" if best["Delta_BA"] >= 0.01 else "ABANDON",
            "target_prior_sessions_used": True,
            "target_S3_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        },
    )
    write_json(
        PROTOCOL / "WBCIC_CSP_PROTOCOL_AUDIT.json",
        {
            "band_hz": [8.0, 30.0],
            "filter": "fourth-order Butterworth SOS, zero phase",
            "sampling_rate_hz": 250,
            "covariance": "per-trial trace normalized",
            "spatial_regularization": "1e-4 composite trace per channel",
            "selection": "target-subject S1->S2 and S2->S1 only",
            "target_batch_adaptation": False,
            "target_S3_labels_used_for_fit": False,
            "outer_split_lock_opened": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard.to_string(index=False), flush=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(max(1, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
