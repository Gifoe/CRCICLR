from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
)

from hsc_tta.contextual_risk.families import GRID, TPSFamily


ALPHA = 0.10
DELTA = 0.10
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20260806
PERMUTATION_REPETITIONS = 50
PERMUTATION_SEED = 20260807


class ScientificStop(RuntimeError):
    def __init__(self, verdict: str, reason: str, evidence_files: list[str]):
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason
        self.evidence_files = evidence_files


class TechnicalBlock(RuntimeError):
    def __init__(self, reason: str, evidence_files: list[str]):
        super().__init__(reason)
        self.reason = reason
        self.evidence_files = evidence_files


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fold_roles(evaluation_fold: int) -> dict[str, set[int]]:
    e = int(evaluation_fold)
    if e not in range(5):
        raise ValueError("evaluation_fold must be in [0,4]")
    return {
        "evaluation": {e},
        "calibration": {(e + 1) % 5, (e + 2) % 5},
        "development": {(e + 3) % 5, (e + 4) % 5},
    }


def role_for_screening_fold(screening_fold: int, evaluation_fold: int) -> str:
    value = int(screening_fold)
    roles = fold_roles(evaluation_fold)
    for role, folds in roles.items():
        if value in folds:
            return role
    raise AssertionError("rotation must partition five folds")


def higher_quantile(values: Iterable[int | float], probability: float) -> tuple[float, int, bool]:
    array = np.sort(np.asarray(list(values)))
    if len(array) == 0:
        raise ValueError("empty empirical quantile")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    k = max(1, math.ceil(probability * len(array)))
    return float(array[k - 1]), int(k), bool(k == len(array))


def subject_conformal_correction(values: Iterable[int], delta: float, K: int) -> tuple[int, int, int, bool]:
    array = np.sort(np.asarray(list(values), dtype=int))
    m = len(array)
    k = math.ceil((m + 1) * (1 - delta))
    if k > m:
        return int(K), int(m), int(k), True
    return int(array[k - 1]), int(m), int(k), False


def tps_sets(probabilities: np.ndarray) -> np.ndarray:
    sets, repairs = TPSFamily().build_sets(np.asarray(probabilities, dtype=float))
    if repairs != 0:
        raise ValueError(f"TPS required {repairs} monotonicity repairs")
    return sets


def inclusion_indices(sets: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    included = sets[np.arange(len(labels)), :, labels]
    if not included[:, -1].all():
        raise ValueError("full TPS index does not contain every label")
    return np.argmax(included, axis=1).astype(np.int16)


def smallest_index_for_risk(kappa: np.ndarray, alpha: float, K: int) -> int:
    values = np.asarray(kappa, dtype=int)
    for index in range(K + 1):
        if float(np.mean(values > index)) <= alpha:
            return index
    return K


def smallest_correction(kappa: np.ndarray, raw_index: int, alpha: float, K: int) -> int:
    for correction in range(K + 1):
        index = min(K, int(raw_index) + correction)
        if float(np.mean(np.asarray(kappa) > index)) <= alpha:
            return correction
    return K


def bootstrap_mean_ci(values: Iterable[float], repetitions: int = BOOTSTRAP_REPETITIONS,
                      seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def clopper_pearson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return math.nan
    if successes >= trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def top_label_ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    total = len(y)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    prediction = p.argmax(axis=1)
    clipped = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    one_hot = np.eye(n_classes, dtype=float)[y]
    recalls = recall_score(y, prediction, labels=np.arange(n_classes), average=None, zero_division=0)
    supports = np.bincount(y, minlength=n_classes)
    frequencies = np.bincount(prediction, minlength=n_classes) / len(prediction)
    present = np.flatnonzero(supports > 0)
    subject_balanced = float(recalls[present].mean()) if len(present) else math.nan
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=np.arange(n_classes), average="macro", zero_division=0)),
        "balanced_accuracy": subject_balanced,
        "cohen_kappa": float(cohen_kappa_score(y, prediction)),
        "nll": float(-np.log(clipped).mean()),
        "brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "top_label_ece": top_label_ece(p, y),
        "constant_prediction": bool(len(np.unique(prediction)) == 1),
        **{f"recall_{i}": float(recalls[i]) for i in range(n_classes)},
        **{f"support_{i}": int(supports[i]) for i in range(n_classes)},
        **{f"predicted_frequency_{i}": float(frequencies[i]) for i in range(n_classes)},
    }


def tps_sanity(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    sets = tps_sets(p)
    sizes = sets.sum(axis=2).mean(axis=0)
    coverage = sets[np.arange(len(y)), :, y].mean(axis=0)
    kappa = inclusion_indices(sets, y)
    return {
        "finite": bool(np.isfinite(p).all()),
        "nonnegative": bool((p >= 0).all()),
        "row_sum": bool(np.allclose(p.sum(axis=1), 1.0, atol=1e-5)),
        "labels_legal": bool(((y >= 0) & (y < p.shape[1])).all()),
        "nested": bool(np.all(sets[:, 1:, :] | ~sets[:, :-1, :])),
        "size_monotone": bool(np.all(np.diff(sizes) >= -1e-12)),
        "coverage_monotone": bool(np.all(np.diff(coverage) >= -1e-12)),
        "inclusion_index_legal": bool(((kappa >= 0) & (kappa < sets.shape[1])).all()),
        "full_index": bool(sets[:, -1, :].all()),
        "distinct_mean_sizes": int(len(np.unique(np.round(sizes, 12)))),
        "distinct_coverages": int(len(np.unique(np.round(coverage, 12)))),
        "nonfull_informative": bool(np.any((sizes[:-1] < p.shape[1]) & (coverage[:-1] > 0))),
        "K": int(sets.shape[1] - 1),
    }


@dataclass(frozen=True)
class BlockProtocol:
    hmc_block_epochs: int = 60
    hmc_tail_min_epochs: int = 30
    eeg_min_predictions: int = 8
    minimum_blocks_per_subject: int = 4


def build_canonical_blocks(metadata: pd.DataFrame, protocol: BlockProtocol = BlockProtocol()) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows: list[dict] = []
    sample_rows: list[dict] = []
    for (dataset, subject), subject_frame in metadata.groupby(["dataset", "subject_id"], sort=True):
        subject_frame = subject_frame.sort_values("chronological_index").copy()
        position = 0
        if dataset == "hmc":
            groups = []
            for recording, rec in subject_frame.groupby("recording_id", sort=False):
                rec = rec.sort_values("chronological_index")
                indices = rec["chronological_index"].to_numpy(dtype=int)
                starts = [0]
                if len(indices) > 1:
                    starts.extend((np.flatnonzero(np.diff(indices) != 1) + 1).tolist())
                starts.append(len(rec))
                for left, right in zip(starts[:-1], starts[1:]):
                    segment = rec.iloc[left:right]
                    for offset in range(0, len(segment), protocol.hmc_block_epochs):
                        candidate = segment.iloc[offset:offset + protocol.hmc_block_epochs]
                        if len(candidate) == protocol.hmc_block_epochs or len(candidate) >= protocol.hmc_tail_min_epochs:
                            groups.append((recording, candidate, ""))
                        else:
                            groups.append((recording, candidate, "tail_below_30"))
        elif dataset == "eegmmidb":
            groups = []
            for run_id, run in subject_frame.groupby("run_id", sort=False):
                run = run.sort_values("chronological_index")
                reason = "" if len(run) >= protocol.eeg_min_predictions else "run_below_8"
                groups.append((run.iloc[0]["recording_id"], run, reason))
        else:
            raise ValueError(f"unsupported dataset: {dataset}")

        for local_index, (recording, block, reason) in enumerate(groups):
            retained = reason == ""
            block_id = f"{dataset}:{subject}:b{local_index:04d}"
            n = int(len(block))
            _, rank, max_based = higher_quantile(np.arange(n), 1 - ALPHA)
            block_rows.append({
                "dataset": dataset,
                "subject_id": subject,
                "recording_id": str(recording),
                "screening_fold": int(block.iloc[0]["screening_fold"]),
                "block_id": block_id,
                "chronological_position": int(position),
                "original_run_id": int(block.iloc[0]["run_id"]),
                "start_sequence_position": int(block["chronological_index"].min()),
                "end_sequence_position": int(block["chronological_index"].max()),
                "number_of_valid_samples": n,
                "quantile_rank_k": rank,
                "quantile_is_max_based": max_based,
                "retained": retained,
                "exclusion_reason": reason,
            })
            if retained:
                for within, row in enumerate(block.itertuples(index=False)):
                    sample_rows.append({
                        "dataset": dataset,
                        "subject_id": subject,
                        "recording_id": str(recording),
                        "sample_id": row.sample_id,
                        "block_id": block_id,
                        "within_block_position": within,
                    })
                position += 1
    return pd.DataFrame(block_rows), pd.DataFrame(sample_rows)


def block_protocol_audit(blocks: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    method = cohorts[cohorts.master_cohort == "method_development"]
    retained = blocks[blocks.retained]
    for dataset in ("hmc", "eegmmidb"):
        expected = method[method.dataset == dataset].subject_id.nunique()
        frame = retained[retained.dataset == dataset]
        counts = frame.groupby("subject_id").size()
        valid_subjects = int((counts >= 4).sum())
        ratio = valid_subjects / expected if expected else 0.0
        max_rate = float(frame.quantile_is_max_based.mean()) if len(frame) else 1.0
        rows.append({
            "dataset": dataset,
            "method_development_subjects": expected,
            "retained_subjects": valid_subjects,
            "retention_rate": ratio,
            "canonical_blocks": int(frame[frame.subject_id.isin(counts[counts >= 4].index)].shape[0]),
            "max_based_rate": max_rate,
            "subjects_pass": bool(ratio >= 0.70),
            "blocks_pass": bool(frame[frame.subject_id.isin(counts[counts >= 4].index)].shape[0] >= 200),
            "quantile_pass": bool(max_rate <= 0.50),
        })
    result = pd.DataFrame(rows)
    result["pass"] = result[["subjects_pass", "blocks_pass", "quantile_pass"]].all(axis=1)
    return result


def rolling_weighted_risk(errors: np.ndarray, samples: np.ndarray, window: int) -> np.ndarray:
    errors = np.asarray(errors, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if len(errors) < window:
        return np.asarray([], dtype=float)
    return np.asarray([
        errors[i - window + 1:i + 1].sum() / samples[i - window + 1:i + 1].sum()
        for i in range(window - 1, len(errors))
    ])


def maximum_true_run(values: Iterable[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def dynamic_metrics(indices: np.ndarray, oracle_indices: np.ndarray, block_errors: np.ndarray,
                    block_samples: np.ndarray, alpha: float = ALPHA) -> dict:
    indices = np.asarray(indices, dtype=int)
    oracle = np.asarray(oracle_indices, dtype=int)
    risk = np.divide(block_errors, block_samples)
    changes = np.abs(np.diff(indices))
    cumulative_deficit = np.maximum(0.0, np.cumsum(block_errors) - alpha * np.cumsum(block_samples))
    shift_positions = np.flatnonzero(np.abs(np.diff(oracle)) >= 2) + 1
    lags = []
    for start in shift_positions:
        match = np.flatnonzero(np.abs(indices[start:] - oracle[start:]) <= 1)
        lags.append(int(match[0]) if len(match) else int(len(indices) - start + 1))
    rolling = rolling_weighted_risk(block_errors, block_samples, 4)
    return {
        "mean_rolling_4": float(rolling.mean()) if len(rolling) else math.nan,
        "q90_rolling_4": float(np.quantile(rolling, .9)) if len(rolling) else math.nan,
        "worst_rolling_4": float(rolling.max()) if len(rolling) else math.nan,
        "max_consecutive_high_risk": maximum_true_run(risk > alpha),
        "deficit_positive_rate": float(np.mean(cumulative_deficit > 0)),
        "maximum_deficit": float(cumulative_deficit.max(initial=0)),
        "final_deficit": float(cumulative_deficit[-1]) if len(cumulative_deficit) else 0.0,
        "mean_absolute_index_change": float(changes.mean()) if len(changes) else 0.0,
        "index_change_frequency": float(np.mean(changes > 0)) if len(changes) else 0.0,
        "maximum_index_change": int(changes.max(initial=0)),
        "median_adaptation_lag": float(np.median(lags)) if lags else math.nan,
        "overshoot": float(np.maximum(indices - oracle, 0).mean()),
        "undershoot": float(np.maximum(oracle - indices, 0).mean()),
    }


def causal_raw_indices(method: str, block_kappas: list[np.ndarray], raw_global: int, K: int,
                       hyperparameter: float | int | None = None, alpha: float = ALPHA) -> np.ndarray:
    output = []
    z = float(raw_global)
    for t, current in enumerate(block_kappas):
        if t == 0:
            raw = raw_global
        elif method == "EXPANDING_PREFIX_QUANTILE":
            raw = smallest_index_for_risk(np.concatenate(block_kappas[:t]), alpha, K)
        elif method == "SLIDING_WINDOW_QUANTILE":
            window = int(hyperparameter)
            raw = smallest_index_for_risk(np.concatenate(block_kappas[max(0, t-window):t]), alpha, K)
        elif method == "RISK_FEEDBACK_INDEX_CONTROL":
            raw = int(math.ceil(z))
        elif method == "TWO_TIMESCALE_FIXED":
            short = smallest_index_for_risk(block_kappas[t-1], alpha, K)
            long = smallest_index_for_risk(np.concatenate(block_kappas[:t]), alpha, K)
            lam = float(hyperparameter)
            raw = int(math.ceil(lam * short + (1 - lam) * long))
        else:
            raise ValueError(f"unknown causal method {method}")
        output.append(int(np.clip(raw, 0, K)))
        if method == "RISK_FEEDBACK_INDEX_CONTROL":
            risk_t = float(np.mean(current > int(math.ceil(z))))
            z = float(np.clip(z + float(hyperparameter) * (risk_t - alpha), 0, K))
    return np.asarray(output, dtype=int)


def certified_policy_indices(raw: np.ndarray, correction: int, certified_global: int, K: int) -> np.ndarray:
    result = np.clip(np.asarray(raw, dtype=int) + int(correction), 0, K)
    if len(result):
        result[0] = int(certified_global)
    return result
