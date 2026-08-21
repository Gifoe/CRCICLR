"""Query-trained prototype/metric transport bank with session recency."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_banks.run_query_bank import Episode, _episode, _feature_statistics, _parts, _upsert
from common import CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED, ensure_directories, logit, stable_seed, write_csv, write_json
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


class MetricTransportBank(nn.Module):
    def __init__(self, feature_dim: int, experts: int, metric_dim: int, max_history_sessions: int):
        super().__init__()
        self.experts = experts
        self.metric_dim = metric_dim
        self.max_history_sessions = max_history_sessions
        projections = []
        generator = torch.Generator().manual_seed(98371)
        for _ in range(experts):
            random = torch.randn(feature_dim, metric_dim, generator=generator)
            q, _ = torch.linalg.qr(random, mode="reduced")
            projections.append(q.T)
        self.projection = nn.Parameter(torch.stack(projections))
        self.session_logits = nn.Parameter(torch.zeros(experts, max_history_sessions))
        if max_history_sessions > 1:
            with torch.no_grad():
                for expert in range(experts):
                    self.session_logits[expert] = torch.linspace(-1.0, 1.0, max_history_sessions) * (expert / max(experts - 1, 1))
        self.gain_raw = nn.Parameter(torch.linspace(-1.5, 0.5, experts))
        self.bias = nn.Parameter(torch.zeros(experts))
        self.base_calibration = nn.Parameter(torch.ones(experts))

    def episode_logits(self, episode: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        history_projection = torch.einsum("nd,kpd->nkp", episode["history_z"], self.projection)
        query_projection = torch.einsum("nd,kpd->nkp", episode["query_z"], self.projection)
        sessions = torch.unique(episode["history_session"], sorted=True)
        session_weight = torch.softmax(self.session_logits[:, :len(sessions)], dim=1)
        prototypes = []
        for label in (0.0, 1.0):
            by_session = []
            for session in sessions:
                mask = (episode["history_session"] == session) & (episode["history_y"] == label)
                by_session.append(history_projection[mask].mean(dim=0))
            stacked = torch.stack(by_session, dim=1)  # experts x sessions x metric_dim
            prototypes.append(torch.sum(stacked * session_weight[:, :, None], dim=1))
        distance0 = torch.mean(torch.square(query_projection - prototypes[0][None, :, :]), dim=2)
        distance1 = torch.mean(torch.square(query_projection - prototypes[1][None, :, :]), dim=2)
        metric_score = distance0 - distance1
        gain = F.softplus(self.gain_raw) + 0.05
        logits = self.base_calibration[None, :] * episode["query_base"][:, None] + gain[None, :] * metric_score + self.bias[None, :]
        return logits, metric_score


def _tensor_episode(episode: Episode, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history_z": torch.as_tensor(episode.history_z, dtype=torch.float32, device=device),
        "history_y": torch.as_tensor(episode.history_y, dtype=torch.float32, device=device),
        "history_session": torch.as_tensor(episode.history_session, dtype=torch.long, device=device),
        "query_z": torch.as_tensor(episode.query_z, dtype=torch.float32, device=device),
        "query_base": torch.as_tensor(episode.query_base, dtype=torch.float32, device=device),
        "query_y": torch.as_tensor(episode.query_y, dtype=torch.float32, device=device),
    }


def _balanced_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = []
    for expert in range(logits.shape[1]):
        per_class = []
        for label in (0.0, 1.0):
            mask = labels == label
            per_class.append(F.binary_cross_entropy_with_logits(logits[mask, expert], labels[mask]))
        result.append(0.5 * (per_class[0] + per_class[1]))
    return torch.stack(result)


def _train(
    train: list[Episode],
    target: list[Episode],
    experts: int,
    metric_dim: int,
    epochs: int,
    tau: float,
    lambda_mean: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict, MetricTransportBank]:
    train_tensors = [_tensor_episode(item, device) for item in train]
    target_tensors = [_tensor_episode(item, device) for item in target]
    max_sessions = max(len(np.unique(item.history_session)) for item in train)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MetricTransportBank(train[0].query_z.shape[1], experts, metric_dim, max_sessions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    log_k = float(np.log(experts))
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        base_losses = []
        scores = []
        for episode in train_tensors:
            logits, score = model.episode_logits(episode)
            losses.append(_balanced_losses(logits, episode["query_y"]))
            base_losses.append(_balanced_losses(episode["query_base"][:, None].expand(-1, experts), episode["query_y"]))
            scores.append(score)
        losses = torch.stack(losses)
        base_loss = torch.stack(base_losses).mean()
        coverage = (-tau * torch.logsumexp(-losses / tau, dim=1) + tau * log_k).mean()
        mean_loss = losses.mean()
        assignment = torch.softmax(-losses / tau, dim=1)
        balance = experts * torch.mean(torch.square(assignment.mean(dim=0) - 1.0 / experts))
        projection_gram = self_gram = model.projection @ model.projection.transpose(1, 2)
        row_orthogonality = torch.mean(torch.square(self_gram - torch.eye(metric_dim, device=device)[None]))
        flattened = F.normalize(model.projection.flatten(1), dim=1)
        cross_gram = flattened @ flattened.T
        expert_redundancy = torch.mean(torch.square(cross_gram - torch.eye(experts, device=device)))
        competence = torch.relu(losses.mean(dim=0) - base_loss - 0.04).mean()
        score_regularity = torch.mean(torch.square(torch.cat(scores, dim=0)))
        objective = coverage + lambda_mean * mean_loss + 0.10 * balance + 0.03 * competence + 0.005 * row_orthogonality + 0.003 * expert_redundancy + 0.0002 * score_regularity
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
                "assignment": assignment.mean(dim=0).detach().cpu().numpy().tolist(),
                "gain": (F.softplus(model.gain_raw) + 0.05).detach().cpu().numpy().tolist(),
                "session_weight": torch.softmax(model.session_logits, dim=1).detach().cpu().numpy().tolist(),
            })
    model.load_state_dict(best_state)
    model.eval()
    predictions, bases = [], []
    with torch.no_grad():
        for episode in target_tensors:
            logits, _ = model.episode_logits(episode)
            predictions.append(logits.cpu().numpy())
            bases.append(episode["query_base"].cpu().numpy())
    audit = {
        "best_objective": best_value,
        "history": history,
        "gain": (F.softplus(model.gain_raw) + 0.05).detach().cpu().numpy().tolist(),
        "base_calibration": model.base_calibration.detach().cpu().numpy().tolist(),
        "session_weight": torch.softmax(model.session_logits, dim=1).detach().cpu().numpy().tolist(),
    }
    return np.concatenate(predictions), np.concatenate(bases), audit, model


def run(benchmark: str, family: str, experts: int, metric_dim: int, epochs: int, folds: tuple[int, ...], tau: float, lambda_mean: float) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family_slug = f"METRIC_TRANSPORT_{family.upper()}_K{experts}_D{metric_dim}"
    benchmark_name = "OpenBMI_MI_S1_to_S2" if benchmark == "openbmi" else "WBCIC_S1S2_to_S3_authorized_development"
    family_id = f"{benchmark_name}__{family_slug}"
    baseline, baseline_source_method = baseline_predictions(benchmark)
    protocol = load_feature_fold(benchmark, 0, family).protocol
    baseline = baseline.loc[baseline.subject_id.astype(str).isin(protocol.search_subjects)].copy()
    baseline["method_id"] = "B_STRONG_MATCHED_V7"
    baseline["family_id"] = family_id
    baseline["source_fold"] = baseline.outer_fold.astype(int)
    baseline["benchmark"] = benchmark_name
    baseline["internal_holdout_used"] = False
    predictions = [baseline]
    primary = [f"{family_slug}__E{k}_RESIDUAL50" for k in range(experts)]
    audits = []
    for fold in folds:
        data = load_feature_fold(benchmark, fold, family)
        assert_search_only(list(data.meta_subjects) + list(data.search_outcome_subjects), benchmark)
        if not data.search_outcome_subjects:
            continue
        mean, std = _feature_statistics(data, data.meta_subjects)
        meta = [_episode(data, subject, mean, std) for subject in data.meta_subjects]
        outcome = [_episode(data, subject, mean, std) for subject in data.search_outcome_subjects]
        adapted, cached_base, audit, model = _train(
            meta, outcome, experts, metric_dim, epochs, tau, lambda_mean,
            stable_seed(V8_SEED, benchmark, family_slug, fold), device,
        )
        uids = np.concatenate([episode.query_uid for episode in outcome])
        locked = logit(baseline.set_index("trial_uid").loc[uids, "probability"].to_numpy(float))
        for expert in range(experts):
            residual = adapted[:, expert] - cached_base
            predictions.extend(_parts(outcome, adapted[:, expert], family_id, f"{family_slug}__E{expert}_STANDALONE"))
            predictions.extend(_parts(outcome, locked + 0.5 * residual, family_id, primary[expert]))
        torch.save({
            "model": model.state_dict(), "family_slug": family_slug, "source_fold": fold,
            "meta_subjects": list(data.meta_subjects), "search_outcome_subjects": list(data.search_outcome_subjects),
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        }, CACHE / f"{benchmark.upper()}_{family_slug}_FOLD_{fold}.pt")
        audits.append({
            "benchmark": benchmark_name, "family_id": family_id, "source_fold": fold,
            "meta_subjects": len(meta), "search_outcome_subjects": len(outcome), **audit,
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        })
        print(f"[{benchmark} {family_slug}] fold={fold} objective={audit['best_objective']:.5f} gain={np.round(audit['gain'], 3).tolist()}", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(set(load_feature_fold(benchmark, fold, family).search_outcome_subjects) for fold in folds))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "feature_family": family, "experts": experts, "metric_dimension": metric_dim,
        "epochs": epochs, "tau": tau, "lambda_mean": lambda_mean, "folds": list(folds),
        "baseline_source_method": baseline_source_method,
        "training_objective": "history-class prototype transport through query-trained metrics, learned recency, coverage, competence, and diversity",
        "deployment_transform": "locked strong anchor plus half learned metric residual",
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
        "partition": "V8_SEARCH only", "target_history": "class prototypes from legal sessions only",
        "metric_fit": "future-query loss on source-fold non-outcome V8_SEARCH subjects",
        "search_outcome_future_labels_used_for_fit_or_selection": False,
        "internal_holdout_used": False, "WBCIC_outer_split_opened": False, "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural hypothesis: learned class-conditional metric transport with legal-history recency can capture future-stable geometry missed by gradient and direct hypernetwork adapters.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--family", default="CONFORMER_NORM")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--metric-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--lambda-mean", type=float, default=0.35)
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, args.family, args.experts, args.metric_dim, args.epochs, folds, args.tau, args.lambda_mean)


if __name__ == "__main__":
    main()
