"""Episodic low-rank history-conditioned hypernetwork adaptation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_bases.components import cosine, deterministic_history_halves
from common import CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V7_SEED, ensure_directories, logit, sigmoid, stable_seed, v6_outputs, write_csv, write_json
from protocol.datasets import load_fold


@dataclass
class Episode:
    subject: str
    context: np.ndarray
    persist: np.ndarray
    future_z: np.ndarray
    future_base: np.ndarray
    future_y: np.ndarray
    future_uid: np.ndarray


class LowRankHistoryHypernetwork(nn.Module):
    def __init__(self, context_dimension: int, feature_dimension: int, bases: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dimension, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.coefficient = nn.Linear(64, bases)
        self.bias = nn.Linear(64, 1)
        self.basis = nn.Parameter(torch.randn(bases, feature_dimension) * 0.01)

    def subject_update(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(context)
        alpha = torch.tanh(self.coefficient(hidden))
        weight = alpha @ self.basis
        bias = 0.25 * torch.tanh(self.bias(hidden)).squeeze(1)
        return weight, bias, alpha

    def forward(self, context: torch.Tensor, z: torch.Tensor, base: torch.Tensor, subject_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight, bias, alpha = self.subject_update(context)
        delta = torch.sum(z * weight[subject_index], dim=1) + bias[subject_index]
        return base + delta, alpha


def _cache(benchmark: str, fold: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    root = CACHE / f"{prefix}_CONFORMER_NORM_FOLD_{fold}"
    features = np.load(root.with_name(root.name + "_FEATURES.npy"), mmap_mode="r", allow_pickle=False)
    logits = np.load(root.with_name(root.name + "_LOGITS.npy"), mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(root.with_name(root.name + "_METADATA.parquet"))
    return np.asarray(features, dtype=np.float32), np.asarray(logits, dtype=np.float32), metadata


def _prototype_logit(train_z: np.ndarray, train_y: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    mean0 = train_z[train_y == 0].mean(axis=0)
    mean1 = train_z[train_y == 1].mean(axis=0)
    direction = mean1 - mean0
    midpoint = 0.5 * (mean0 + mean1)
    fit_score = ((train_z - midpoint) @ direction)[:, None]
    target_score = ((target_z - midpoint) @ direction)[:, None]
    model = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=2_000, random_state=V7_SEED)
    model.fit(fit_score, train_y)
    return np.asarray(model.decision_function(target_score), dtype=float)


def _ce(y: np.ndarray, value: np.ndarray) -> float:
    return float(log_loss(y, sigmoid(value), labels=[0, 1]))


def _context(z: np.ndarray, y: np.ndarray, base: np.ndarray, sessions: np.ndarray, base_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean0 = z[y == 0].mean(axis=0)
    mean1 = z[y == 1].mean(axis=0)
    direction = mean1 - mean0
    midpoint = 0.5 * (mean0 + mean1)
    spread = np.log(np.std(z, axis=0) + 1e-4)
    probability = sigmoid(base)
    entropy = -(probability * np.log(np.clip(probability, 1e-7, 1.0)) + (1 - probability) * np.log(np.clip(1 - probability, 1e-7, 1.0)))
    generic = np.concatenate([
        direction,
        midpoint,
        spread,
        np.asarray([
            _ce(y, base),
            balanced_accuracy_score(y, base >= 0.0),
            np.mean(np.abs(base)),
            np.std(np.abs(base)),
            np.mean(entropy),
            np.linalg.norm(direction),
            len(y) / 400.0,
            len(np.unique(sessions)),
        ], dtype=float),
    ])
    first, second = deterministic_history_halves(y, sessions)
    direction_first = z[first][y[first] == 1].mean(axis=0) - z[first][y[first] == 0].mean(axis=0)
    direction_second = z[second][y[second] == 1].mean(axis=0) - z[second][y[second] == 0].mean(axis=0)
    prototype = _prototype_logit(z, y, z)
    decision_dependence = float(np.mean((prototype >= 0.0) != (base >= 0.0)))
    cross_first = _prototype_logit(z[first], y[first], z[second])
    cross_second = _prototype_logit(z[second], y[second], z[first])
    transfer = 0.5 * (
        _ce(y[second], base[second]) - _ce(y[second], cross_first)
        + _ce(y[first], base[first]) - _ce(y[first], cross_second)
    )
    persist = np.asarray([
        cosine(direction_first, direction_second),
        0.0,
        decision_dependence,
        cosine(direction, base_weight),
        transfer,
    ], dtype=float)
    return generic.astype(np.float32), persist.astype(np.float32)


def _episodes(
    subjects: tuple[str, ...],
    data,
    features: np.ndarray,
    logits: np.ndarray,
    metadata: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    base_weight: np.ndarray,
) -> list[Episode]:
    result = []
    for subject in subjects:
        subject_mask = metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
        history = subject_mask & metadata.session_id.astype(int).isin(data.history_sessions).to_numpy()
        future = subject_mask & metadata.session_id.astype(int).eq(data.future_session).to_numpy()
        history_z = (features[history] - mean) / std
        future_z = (features[future] - mean) / std
        history_y = metadata.loc[history, "label"].to_numpy(int)
        history_sessions = metadata.loc[history, "session_id"].to_numpy(int)
        generic, persist = _context(history_z, history_y, logits[history], history_sessions, base_weight)
        result.append(Episode(
            subject=str(subject),
            context=generic,
            persist=persist,
            future_z=future_z.astype(np.float32),
            future_base=logits[future].astype(np.float32),
            future_y=metadata.loc[future, "label"].to_numpy(int),
            future_uid=metadata.loc[future, "trial_uid"].astype(str).to_numpy(),
        ))
    # Cross-fitted signed history-utility prior: each subject receives the mean
    # transfer evidence of the other meta subjects, never its future query.
    values = np.asarray([episode.persist[4] for episode in result], dtype=float)
    for index, episode in enumerate(result):
        episode.persist[1] = float((values.sum() - values[index]) / max(len(values) - 1, 1))
    return result


def _standardize_context(train: list[Episode], target: list[Episode], persist_mode: bool) -> tuple[np.ndarray, np.ndarray]:
    generic_train = np.stack([episode.context for episode in train])
    generic_target = np.stack([episode.context for episode in target])
    mean = generic_train.mean(axis=0)
    std = np.maximum(generic_train.std(axis=0), 1e-4)
    generic_train = (generic_train - mean) / std
    generic_target = (generic_target - mean) / std
    persist_train = np.stack([episode.persist for episode in train])
    persist_target = np.stack([episode.persist for episode in target])
    pmean = persist_train.mean(axis=0)
    pstd = np.maximum(persist_train.std(axis=0), 1e-4)
    if persist_mode:
        persist_train = (persist_train - pmean) / pstd
        persist_target = (persist_target - pmean) / pstd
    else:
        persist_train = np.zeros_like(persist_train)
        persist_target = np.zeros_like(persist_target)
    return np.column_stack([generic_train, persist_train]).astype(np.float32), np.column_stack([generic_target, persist_target]).astype(np.float32)


def _pack(episodes: list[Episode], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z, base, labels, subject = [], [], [], []
    for index, episode in enumerate(episodes):
        z.append(episode.future_z)
        base.append(episode.future_base)
        labels.append(episode.future_y)
        subject.append(np.full(len(episode.future_y), index, dtype=int))
    return (
        torch.as_tensor(np.concatenate(z), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(base), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(labels), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(subject), dtype=torch.long, device=device),
    )


def _train_model(train: list[Episode], target: list[Episode], persist_mode: bool, seed: int, device: torch.device):
    context_train, context_target = _standardize_context(train, target, persist_mode)
    context_tensor = torch.as_tensor(context_train, dtype=torch.float32, device=device)
    z, base, labels, subject = _pack(train, device)
    torch.manual_seed(seed)
    model = LowRankHistoryHypernetwork(context_train.shape[1], z.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    history = []
    for epoch in range(500):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, alpha = model(context_tensor, z, base, subject)
        loss = F.binary_cross_entropy_with_logits(prediction, labels)
        loss = loss + 1e-3 * torch.mean(torch.square(alpha)) + 1e-4 * torch.mean(torch.square(model.basis))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in {0, 99, 199, 299, 399, 499}:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    model.eval()
    target_context = torch.as_tensor(context_target, dtype=torch.float32, device=device)
    target_z, target_base, _, target_subject = _pack(target, device)
    with torch.inference_mode():
        prediction, alpha = model(target_context, target_z, target_base, target_subject)
    return prediction.cpu().numpy(), target_base.cpu().numpy(), alpha.cpu().numpy(), history, model


def _strong_anchor(benchmark: str) -> tuple[pd.DataFrame, str]:
    if benchmark == "openbmi":
        path = v6_outputs() / "diagnostics" / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv"
        method = "MI_SPECIFIC_BACKBONE_ADAPTED"
    else:
        path = v6_outputs() / "diagnostics" / "WBCIC_FUTURE_SESSION_POPULATION_PREDICTIONS.csv"
        method = "V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED"
    frame = pd.read_csv(path)
    return frame.loc[frame.method_id.astype(str).eq(method)].copy(), method


def _prediction_parts(episodes: list[Episode], logits_value: np.ndarray, method: str, benchmark_name: str, fold: int) -> list[pd.DataFrame]:
    parts = []
    cursor = 0
    for episode in episodes:
        stop = cursor + len(episode.future_y)
        probability = sigmoid(logits_value[cursor:stop])
        parts.append(pd.DataFrame({
            "benchmark": benchmark_name,
            "method_id": method,
            "trial_uid": episode.future_uid,
            "subject_id": episode.subject,
            "outer_fold": fold,
            "label": episode.future_y,
            "probability": probability,
            "prediction": probability >= 0.5,
            "target_history_labels_used": True,
            "target_future_labels_used_for_fit": False,
            "exploratory": True,
            "OUTER_TEST_USED": False,
        }))
        cursor = stop
    return parts


def _leaderboard(frame: pd.DataFrame, reference_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows = []
    for (method, subject), group in frame.groupby(["method_id", "subject_id"]):
        subject_rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "method_id": method,
            "subject_id": str(subject),
            "outer_fold": int(group.outer_fold.iloc[0]),
            "BA": balanced_accuracy_score(group.label, group.prediction),
            "NLL": log_loss(group.label, np.clip(group.probability, 1e-7, 1 - 1e-7), labels=[0, 1]),
        })
    subjects = pd.DataFrame(subject_rows)
    reference = subjects.loc[subjects.method_id.eq(reference_method)].set_index("subject_id").BA
    rows = []
    for method, group in subjects.groupby("method_id"):
        delta = group.BA.to_numpy(float) - reference.loc[group.subject_id].to_numpy(float)
        rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "method_id": method,
            "subjects": len(group),
            "mean_subject_BA": float(group.BA.mean()),
            "mean_subject_NLL": float(group.NLL.mean()),
            "reference_method_id": reference_method,
            "Delta_BA": float(delta.mean()),
            "positive_subject_fraction": float(np.mean(delta > 0)),
            "nonnegative_subject_fraction": float(np.mean(delta >= 0)),
            "worst_subject_delta": float(delta.min()),
            "target_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        })
    return pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False), subjects


def run_benchmark(benchmark: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor, anchor_method = _strong_anchor(benchmark)
    anchor_index = anchor.set_index("trial_uid")
    predictions = [anchor]
    audits = []
    for fold in range(5):
        data = load_fold(benchmark, fold)
        features, logits_array, metadata = _cache(benchmark, fold)
        meta_mask = metadata.subject_id.astype(str).isin(data.nonoutcome_subjects).to_numpy()
        mean = features[meta_mask].mean(axis=0)
        std = np.maximum(features[meta_mask].std(axis=0), 1e-4)
        standardized = (features[meta_mask] - mean) / std
        base_fit = LinearRegressionProxy.fit(standardized, logits_array[meta_mask])
        meta = _episodes(data.nonoutcome_subjects, data, features, logits_array, metadata, mean, std, base_fit)
        outcome = _episodes(data.outcome_subjects, data, features, logits_array, metadata, mean, std, base_fit)
        population_utility_prior = float(np.mean([episode.persist[4] for episode in meta]))
        for episode in outcome:
            episode.persist[1] = population_utility_prior
        for persist_mode, mode in ((False, "META_GENERIC_HYPER"), (True, "PERSIST_META_HYPER")):
            adapted, base, alpha, history, model = _train_model(
                meta, outcome, persist_mode,
                stable_seed(V7_SEED, benchmark, fold, "capacity-matched-hypernetwork"), device,
            )
            predictions.extend(_prediction_parts(outcome, adapted, mode, data.benchmark, fold))
            predictions.extend(_prediction_parts(outcome, base + 0.5 * (adapted - base), mode + "_HALF", data.benchmark, fold))
            uids = np.concatenate([episode.future_uid for episode in outcome])
            anchor_logit = logit(anchor_index.loc[uids, "probability"].to_numpy(float))
            predictions.extend(_prediction_parts(outcome, anchor_logit + 0.5 * (adapted - base), "ANCHOR_PLUS_" + mode, data.benchmark, fold))
            audits.append({
                "benchmark": data.benchmark,
                "outer_fold": fold,
                "mode": mode,
                "bases": 8,
                "history": history,
                "mean_abs_alpha": float(np.mean(np.abs(alpha))),
                "preserve_coefficient_fraction": float(np.mean(np.abs(alpha) < 0.1)),
                "target_future_labels_used_for_fit": False,
                "OUTER_TEST_USED": False,
            })
            checkpoint = CACHE / f"{'OPENBMI' if benchmark == 'openbmi' else 'WBCIC'}_{mode}_FOLD_{fold}.pt"
            torch.save({"model": model.state_dict(), "mode": mode, "fold": fold, "OUTER_TEST_USED": False}, checkpoint)
        print(f"[{benchmark} hypernetwork] fold={fold} complete", flush=True)
    frame = pd.concat(predictions, ignore_index=True)
    leaderboard, subjects = _leaderboard(frame, anchor_method)
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    write_csv(LEADERBOARD / f"{prefix}_HYPERNETWORK_META.csv", leaderboard)
    write_csv(DIAGNOSTICS / f"{prefix}_HYPERNETWORK_META_PREDICTIONS.csv", frame)
    write_csv(DIAGNOSTICS / f"{prefix}_HYPERNETWORK_META_SUBJECT_RESULTS.csv", subjects)
    write_json(DIAGNOSTICS / f"{prefix}_HYPERNETWORK_META_AUDIT.json", audits)
    print(leaderboard.to_string(index=False), flush=True)
    return {"leaderboard": leaderboard, "audits": audits}


class LinearRegressionProxy:
    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = np.column_stack([x, np.ones(len(x))])
        coefficient, *_ = np.linalg.lstsq(design, y, rcond=1e-4)
        return coefficient[:-1]


def run() -> None:
    ensure_directories()
    open_result = run_benchmark("openbmi")
    wbcic_result = run_benchmark("wbcic")
    write_json(PROTOCOL / "HYPERNETWORK_META_LEGALITY.json", {
        "meta_training": "non-outcome subject legal history-to-future episodes",
        "outcome_context": "history sessions only",
        "bases": 8,
        "generic_and_persist_capacity_matched": True,
        "outcome_future_labels_used_for_fit_or_selection": False,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / "ITERATION_004_HYPERNETWORK.md").write_text(
        "# Iteration 004 — low-rank history hypernetwork\n\n"
        "Eight globally shared feature-space bases are mixed from legal history context. "
        "META-GENERIC and PERSIST-Meta are capacity matched; PERSIST alone receives P/U/D/G/R.\n\n"
        "## OpenBMI\n\n```text\n" + open_result["leaderboard"].to_string(index=False) + "\n```\n\n"
        "## WBCIC\n\n```text\n" + wbcic_result["leaderboard"].to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
