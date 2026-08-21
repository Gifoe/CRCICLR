"""Query-trained low-rank adaptation bank with explicit coverage pressure.

The V8 search path receives only the materialized V8_SEARCH rows.  Every
transformation is fitted on legal history->future episodes from source-fold
non-outcome subjects and evaluated on source-fold outcome subjects.  Internal
V8 holdout rows are absent from this program by construction.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import (
    CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED,
    ensure_directories, logit, sigmoid, stable_seed, write_csv, write_json,
)
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


@dataclass
class Episode:
    subject_id: str
    source_fold: int
    context_raw: np.ndarray
    history_z: np.ndarray
    history_base: np.ndarray
    history_y: np.ndarray
    history_session: np.ndarray
    query_z: np.ndarray
    query_base: np.ndarray
    query_y: np.ndarray
    query_uid: np.ndarray


class QueryLowRankBank(nn.Module):
    def __init__(self, context_dim: int, feature_dim: int, experts: int, rank: int):
        super().__init__()
        self.experts = int(experts)
        self.rank = int(rank)
        self.population = nn.Parameter(torch.zeros(experts, feature_dim))
        self.population_bias = nn.Parameter(torch.zeros(experts))
        self.basis = nn.Parameter(torch.randn(experts, rank, feature_dim) * 0.02)
        hidden = max(16, min(48, 2 * context_dim))
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        self.coefficient = nn.Linear(hidden, experts * rank)
        self.context_bias = nn.Linear(hidden, experts)
        nn.init.normal_(self.population, std=0.005)
        nn.init.zeros_(self.population_bias)
        nn.init.zeros_(self.coefficient.weight)
        nn.init.zeros_(self.coefficient.bias)
        nn.init.zeros_(self.context_bias.weight)
        nn.init.zeros_(self.context_bias.bias)

    def forward(
        self,
        context: torch.Tensor,
        z: torch.Tensor,
        base: torch.Tensor,
        subject_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.context_encoder(context)
        coefficient = 1.5 * torch.tanh(self.coefficient(hidden).reshape(-1, self.experts, self.rank))
        subject_delta = torch.einsum("skr,krd->skd", coefficient, self.basis)
        weight = self.population.unsqueeze(0) + subject_delta
        bias = self.population_bias.unsqueeze(0) + 0.5 * torch.tanh(self.context_bias(hidden))
        residual = torch.einsum("nd,nkd->nk", z, weight[subject_index]) + bias[subject_index]
        return base[:, None] + residual, coefficient, residual


def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = []
    for label in (0.0, 1.0):
        mask = labels == label
        if not bool(mask.any()):
            raise RuntimeError("Episode lacks one class")
        values.append(F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction="mean"))
    return 0.5 * (values[0] + values[1])


def _context(z: np.ndarray, y: np.ndarray, base: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    mean0 = z[y == 0].mean(axis=0)
    mean1 = z[y == 1].mean(axis=0)
    direction = mean1 - mean0
    midpoint = 0.5 * (mean0 + mean1)
    spread = np.log(np.std(z, axis=0) + 1e-4)
    error = sigmoid(base) - y
    gradient = np.mean(error[:, None] * z, axis=0)
    latest = int(np.max(sessions))
    latest_mask = sessions == latest
    latest_direction = z[latest_mask & (y == 1)].mean(axis=0) - z[latest_mask & (y == 0)].mean(axis=0)
    if len(np.unique(sessions)) > 1:
        earliest = int(np.min(sessions))
        early_mask = sessions == earliest
        early_direction = z[early_mask & (y == 1)].mean(axis=0) - z[early_mask & (y == 0)].mean(axis=0)
        drift = latest_direction - early_direction
    else:
        drift = np.zeros_like(direction)
    probability = sigmoid(base)
    entropy = -(probability * np.log(np.clip(probability, 1e-7, 1.0)) + (1.0 - probability) * np.log(np.clip(1.0 - probability, 1e-7, 1.0)))
    scalars = np.asarray([
        balanced_accuracy_score(y, base >= 0.0),
        log_loss(y, probability, labels=[0, 1]),
        float(np.mean(np.abs(base))),
        float(np.std(base)),
        float(np.mean(entropy)),
        float(np.linalg.norm(direction)),
        float(np.linalg.norm(gradient)),
        float(np.linalg.norm(drift)),
        float(np.mean(base[y == 1]) - np.mean(base[y == 0])),
        float(len(y) / 400.0),
        float(len(np.unique(sessions)) / 2.0),
        1.0,
    ], dtype=np.float32)
    return np.concatenate((direction, midpoint, spread, gradient, latest_direction, drift, scalars)).astype(np.float32)


def _episode(fold_data, subject: str, mean: np.ndarray, std: np.ndarray) -> Episode:
    metadata = fold_data.metadata
    subject_mask = metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
    history = subject_mask & metadata.session_id.astype(int).isin(fold_data.protocol.history_sessions).to_numpy()
    future = subject_mask & metadata.session_id.astype(int).eq(fold_data.protocol.future_session).to_numpy()
    if not history.any() or not future.any():
        raise RuntimeError((fold_data.protocol.key, subject, "incomplete episode"))
    history_index = metadata.loc[history, "source_index"].to_numpy(int)
    future_index = metadata.loc[future, "source_index"].to_numpy(int)
    history_z = (np.asarray(fold_data.features[history_index], dtype=np.float32) - mean) / std
    future_z = (np.asarray(fold_data.features[future_index], dtype=np.float32) - mean) / std
    history_y = metadata.loc[history, "label"].to_numpy(int)
    history_base = np.asarray(fold_data.logits[history_index], dtype=np.float32)
    sessions = metadata.loc[history, "session_id"].to_numpy(int)
    return Episode(
        subject_id=str(subject),
        source_fold=int(fold_data.source_fold),
        context_raw=_context(history_z, history_y, history_base, sessions),
        history_z=history_z.astype(np.float32),
        history_base=history_base.astype(np.float32),
        history_y=history_y.astype(np.float32),
        history_session=sessions.astype(np.int64),
        query_z=future_z.astype(np.float32),
        query_base=np.asarray(fold_data.logits[future_index], dtype=np.float32),
        query_y=metadata.loc[future, "label"].to_numpy(np.float32),
        query_uid=metadata.loc[future, "trial_uid"].astype(str).to_numpy(),
    )


def _feature_statistics(fold_data, subjects: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    mask = fold_data.metadata.subject_id.astype(str).isin(subjects).to_numpy()
    indices = fold_data.metadata.loc[mask, "source_index"].to_numpy(int)
    values = np.asarray(fold_data.features[indices], dtype=np.float32)
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-4)


def _project_context(train: list[Episode], target: list[Episode], dimensions: int = 12) -> tuple[np.ndarray, np.ndarray, dict]:
    x = np.stack([episode.context_raw for episode in train]).astype(np.float64)
    target_x = np.stack([episode.context_raw for episode in target]).astype(np.float64)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-4)
    standardized = (x - mean) / std
    target_standardized = (target_x - mean) / std
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    components = int(min(dimensions, len(train) - 1, right.shape[0]))
    basis = right[:components]
    return (
        (standardized @ basis.T).astype(np.float32),
        (target_standardized @ basis.T).astype(np.float32),
        {
            "raw_dimension": int(x.shape[1]),
            "projected_dimension": components,
            "explained_variance_fraction": float(np.sum(singular[:components] ** 2) / max(np.sum(singular ** 2), 1e-12)),
        },
    )


def _pack(episodes: list[Episode], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[slice]]:
    z, base, y, subject, spans = [], [], [], [], []
    cursor = 0
    for index, episode in enumerate(episodes):
        count = len(episode.query_y)
        z.append(episode.query_z)
        base.append(episode.query_base)
        y.append(episode.query_y)
        subject.append(np.full(count, index, dtype=np.int64))
        spans.append(slice(cursor, cursor + count))
        cursor += count
    return (
        torch.as_tensor(np.concatenate(z), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(base), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(y), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(subject), dtype=torch.long, device=device),
        spans,
    )


def _loss_matrix(logits: torch.Tensor, labels: torch.Tensor, spans: list[slice]) -> torch.Tensor:
    rows = []
    for span in spans:
        rows.append(torch.stack([_balanced_bce(logits[span, k], labels[span]) for k in range(logits.shape[1])]))
    return torch.stack(rows)


def _train(
    train: list[Episode],
    target: list[Episode],
    experts: int,
    rank: int,
    epochs: int,
    tau: float,
    lambda_mean: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict, QueryLowRankBank, np.ndarray]:
    context_train, context_target, context_audit = _project_context(train, target)
    context_tensor = torch.as_tensor(context_train, dtype=torch.float32, device=device)
    z, base, labels, subject, spans = _pack(train, device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = QueryLowRankBank(context_train.shape[1], z.shape[1], experts, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    history = []
    log_k = float(np.log(experts))
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, coefficient, residual = model(context_tensor, z, base, subject)
        losses = _loss_matrix(prediction, labels, spans)
        coverage = (-tau * torch.logsumexp(-losses / tau, dim=1) + tau * log_k).mean()
        mean_loss = losses.mean()
        assignment = torch.softmax(-losses / tau, dim=1)
        balance = experts * torch.mean(torch.square(assignment.mean(dim=0) - 1.0 / experts))
        normalized_basis = F.normalize(model.basis.flatten(1), dim=1)
        gram = normalized_basis @ normalized_basis.T
        orthogonality = torch.mean(torch.square(gram - torch.eye(experts, device=device)))
        base_losses = torch.stack([_balanced_bce(base[span], labels[span]) for span in spans]).mean()
        competence = torch.relu(losses.mean(dim=0) - base_losses - 0.03).mean()
        regularity = torch.mean(torch.square(residual)) + torch.mean(torch.square(coefficient))
        objective = (
            coverage
            + lambda_mean * mean_loss
            + 0.08 * balance
            + 0.02 * competence
            + 0.003 * orthogonality
            + 0.0005 * regularity
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(objective.detach())
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
        if epoch in {0, 49, 99, 199, 399, epochs - 1}:
            history.append({
                "epoch": epoch + 1,
                "objective": value,
                "coverage_loss": float(coverage.detach()),
                "mean_loss": float(mean_loss.detach()),
                "usage_balance": float(balance.detach()),
                "competence_penalty": float(competence.detach()),
                "mean_abs_residual": float(torch.mean(torch.abs(residual)).detach()),
                "assignment": assignment.mean(dim=0).detach().cpu().numpy().tolist(),
            })
    model.load_state_dict(best_state)
    model.eval()
    target_context_tensor = torch.as_tensor(context_target, dtype=torch.float32, device=device)
    target_z, target_base, _, target_subject, _ = _pack(target, device)
    with torch.inference_mode():
        prediction, coefficient, residual = model(target_context_tensor, target_z, target_base, target_subject)
        train_prediction, _, _ = model(context_tensor, z, base, subject)
        train_losses = _loss_matrix(train_prediction, labels, spans)
        train_usage = torch.softmax(-train_losses / tau, dim=1).mean(dim=0).cpu().numpy()
    audit = {
        "best_objective": best_value,
        "history": history,
        "context": context_audit,
        "meta_assignment": train_usage.tolist(),
        "mean_abs_target_coefficient": float(np.mean(np.abs(coefficient.cpu().numpy()))),
        "mean_abs_target_residual": float(np.mean(np.abs(residual.cpu().numpy()))),
    }
    return prediction.cpu().numpy(), target_base.cpu().numpy(), audit, model, train_usage


def _parts(
    episodes: list[Episode],
    values: np.ndarray,
    family_id: str,
    method_id: str,
) -> list[pd.DataFrame]:
    result = []
    cursor = 0
    for episode in episodes:
        count = len(episode.query_y)
        probability = sigmoid(values[cursor:cursor + count])
        result.append(pd.DataFrame({
            "benchmark": family_id.split("__", 1)[0],
            "family_id": family_id,
            "method_id": method_id,
            "trial_uid": episode.query_uid,
            "subject_id": episode.subject_id,
            "source_fold": episode.source_fold,
            "label": episode.query_y.astype(int),
            "probability": probability,
            "prediction": (probability >= 0.5).astype(int),
            "target_history_labels_used": True,
            "target_future_labels_used_for_fit": False,
            "internal_holdout_used": False,
            "exploratory": True,
            "OUTER_TEST_USED": False,
        }))
        cursor += count
    return result


def _upsert(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    if path.is_file():
        existing = pd.read_csv(path)
        incoming_keys = set(tuple(map(str, row)) for row in frame[keys].to_numpy())
        keep = [tuple(map(str, row)) not in incoming_keys for row in existing[keys].to_numpy()]
        frame = pd.concat([existing.loc[keep], frame], ignore_index=True)
    write_csv(path, frame)


def run(
    benchmark: str,
    family: str,
    experts: int,
    rank: int,
    epochs: int,
    folds: tuple[int, ...],
    tau: float,
    lambda_mean: float,
) -> dict:
    ensure_directories()
    assert_search_only(tuple(), benchmark)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family_slug = f"LR_COVERAGE_{family.upper()}_K{experts}_R{rank}"
    benchmark_name = "OpenBMI_MI_S1_to_S2" if benchmark == "openbmi" else "WBCIC_S1S2_to_S3_authorized_development"
    family_id = f"{benchmark_name}__{family_slug}"
    baseline, baseline_source_method = baseline_predictions(benchmark)
    protocol = load_feature_fold(benchmark, 0, family).protocol
    baseline = baseline.loc[
        baseline.subject_id.astype(str).isin(protocol.search_subjects)
    ].copy()
    baseline["method_id"] = "B_STRONG_MATCHED_V7"
    baseline["family_id"] = family_id
    baseline["source_fold"] = baseline.outer_fold.astype(int)
    baseline["benchmark"] = benchmark_name
    baseline["internal_holdout_used"] = False
    predictions = [baseline]
    audits = []
    primary = [f"{family_slug}__E{index}_BLEND50" for index in range(experts)]
    for fold in folds:
        data = load_feature_fold(benchmark, fold, family)
        assert_search_only(list(data.meta_subjects) + list(data.search_outcome_subjects), benchmark)
        if not data.search_outcome_subjects:
            continue
        mean, std = _feature_statistics(data, data.meta_subjects)
        meta = [_episode(data, subject, mean, std) for subject in data.meta_subjects]
        outcome = [_episode(data, subject, mean, std) for subject in data.search_outcome_subjects]
        adapted, cached_base, audit, model, usage = _train(
            meta, outcome, experts, rank, epochs, tau, lambda_mean,
            stable_seed(V8_SEED, benchmark, family_slug, fold), device,
        )
        uids = np.concatenate([episode.query_uid for episode in outcome])
        baseline_index = baseline.set_index("trial_uid")
        locked_logit = logit(baseline_index.loc[uids, "probability"].to_numpy(float))
        for expert in range(experts):
            standalone = adapted[:, expert]
            predictions.extend(_parts(outcome, standalone, family_id, f"{family_slug}__E{expert}_STANDALONE"))
            predictions.extend(_parts(outcome, 0.5 * (locked_logit + standalone), family_id, primary[expert]))
        checkpoint = CACHE / f"{benchmark.upper()}_{family_slug}_FOLD_{fold}.pt"
        torch.save({
            "model": model.state_dict(),
            "family": family,
            "family_slug": family_slug,
            "benchmark": benchmark,
            "source_fold": fold,
            "meta_subjects": list(data.meta_subjects),
            "search_outcome_subjects": list(data.search_outcome_subjects),
            "experts": experts,
            "rank": rank,
            "context_fit": "V8_SEARCH meta subjects only",
            "internal_holdout_used": False,
            "OUTER_TEST_USED": False,
        }, checkpoint)
        audits.append({
            "benchmark": benchmark_name,
            "family_id": family_id,
            "source_fold": fold,
            "meta_subjects": len(meta),
            "search_outcome_subjects": len(outcome),
            "query_trials": int(sum(len(item.query_y) for item in meta)),
            "experts": experts,
            "rank": rank,
            "tau": tau,
            "lambda_mean": lambda_mean,
            "meta_usage": usage.tolist(),
            **audit,
            "internal_holdout_used": False,
            "OUTER_TEST_USED": False,
        })
        print(
            f"[{benchmark} {family_slug}] fold={fold} meta={len(meta)} outcome={len(outcome)} "
            f"loss={audit['best_objective']:.5f} usage={np.round(usage, 3).tolist()}",
            flush=True,
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(set(load_feature_fold(benchmark, fold, family).search_outcome_subjects) for fold in folds))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    expected = set(protocol.search_subjects) if folds == (0, 1, 2, 3, 4) else allowed
    if set(prediction_frame.loc[prediction_frame.method_id.eq("B_STRONG_MATCHED_V7"), "subject_id"].astype(str)) != expected:
        raise RuntimeError("Baseline/search outcome subject mismatch")
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "feature_family": family,
        "experts": experts,
        "rank": rank,
        "epochs": epochs,
        "tau": tau,
        "lambda_mean": lambda_mean,
        "folds": list(folds),
        "baseline_source_method": baseline_source_method,
        "training_objective": "normalized soft-min coverage + mean competence + usage balance + basis diversity",
    })
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    tag = f"{prefix}_{family_slug}"
    write_csv(DIAGNOSTICS / f"{tag}_SEARCH_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / f"{tag}_SUBJECT_RESULTS.csv", report["subjects"])
    write_json(DIAGNOSTICS / f"{tag}_TRAINING_AUDIT.json", audits)
    write_json(HEADROOM / f"{tag}_HEADROOM.json", summary)
    write_csv(HEADROOM / f"{tag}_SUBJECT_ORACLE.csv", report["oracle"])
    write_csv(HEADROOM / f"{tag}_EXPERT_COMPETENCE.csv", report["competence"])
    write_csv(HEADROOM / f"{tag}_EXPERT_DIVERSITY.csv", report["diversity"])
    write_csv(HEADROOM / f"{tag}_ORACLE_BY_FOLD.csv", report["folds"])
    _upsert(HEADROOM / "HEADROOM_FAMILY_TABLE.csv", pd.DataFrame([summary]), ["benchmark", "family_id"])
    _upsert(HEADROOM / "EXPERT_COMPETENCE.csv", report["competence"], ["benchmark", "family_id", "method_id"])
    _upsert(HEADROOM / "EXPERT_DIVERSITY.csv", report["diversity"], ["benchmark", "family_id", "expert_left", "expert_right"])
    _upsert(HEADROOM / "SUBJECT_ORACLE.csv", report["oracle"], ["benchmark", "family_id", "subject_id"])
    _upsert(HEADROOM / "ORACLE_BY_FOLD.csv", report["folds"], ["benchmark", "family_id", "source_fold"])
    write_json(PROTOCOL / f"{tag}_LEGALITY.json", {
        "partition": "V8_SEARCH only",
        "meta_training": "source-fold non-outcome V8_SEARCH subjects; legal history->future query",
        "search_evaluation": "source-fold outcome V8_SEARCH subjects; future labels scoring only",
        "internal_holdout_rows_passed_to_program": False,
        "outcome_future_labels_used_for_fit_or_selection": False,
        "oracle_outcomes_used_for_diagnostic_only": True,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    iteration_path = RESEARCH_LOG / f"ITERATION_{tag}.md"
    iteration_path.write_text(
        f"# {tag}\n\n"
        "Structural hypothesis: a query-trained low-rank classifier-adapter bank with a normalized "
        "soft-oracle coverage objective will create competent but complementary future-session experts.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--family", default="CONFORMER_NORM")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--lambda-mean", type=float, default=0.35)
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, args.family, args.experts, args.rank, args.epochs, folds, args.tau, args.lambda_mean)


if __name__ == "__main__":
    main()
