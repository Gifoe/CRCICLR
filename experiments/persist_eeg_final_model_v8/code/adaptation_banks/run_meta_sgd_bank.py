"""True query-trained feature-head Meta-SGD expert bank for V8 Phase A."""

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

from adaptation_banks.run_query_bank import (
    Episode, _episode, _feature_statistics, _parts, _upsert,
)
from common import CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED, ensure_directories, logit, stable_seed, write_csv, write_json
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


class MetaSGDBank(nn.Module):
    def __init__(self, feature_dim: int, experts: int):
        super().__init__()
        self.experts = experts
        self.initial_weight = nn.Parameter(torch.zeros(experts, feature_dim))
        self.initial_bias = nn.Parameter(torch.zeros(experts))
        initial_steps = torch.linspace(0.0, 2.5, experts).view(experts, 1).expand(experts, feature_dim)
        self.alpha_weight_raw = nn.Parameter(torch.atanh(torch.clamp(initial_steps / 4.0, -0.95, 0.95)))
        self.alpha_bias_raw = nn.Parameter(torch.atanh(torch.clamp(torch.linspace(0.0, 1.5, experts) / 4.0, -0.95, 0.95)))

    @property
    def alpha_weight(self) -> torch.Tensor:
        return 4.0 * torch.tanh(self.alpha_weight_raw)

    @property
    def alpha_bias(self) -> torch.Tensor:
        return 4.0 * torch.tanh(self.alpha_bias_raw)

    def episode_logits(self, episode: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_z = episode["history_z"]
        history_base = episode["history_base"]
        history_y = episode["history_y"]
        support = history_base[:, None] + history_z @ self.initial_weight.T + self.initial_bias
        error = torch.sigmoid(support) - history_y[:, None]
        weights = torch.zeros_like(history_y)
        for label in (0.0, 1.0):
            mask = history_y == label
            weights[mask] = 0.5 / torch.clamp(mask.sum(), min=1)
        weighted_error = error * weights[:, None]
        gradient_weight = torch.einsum("nk,nd->kd", weighted_error, history_z)
        gradient_bias = weighted_error.sum(dim=0)
        adapted_weight = self.initial_weight - self.alpha_weight * gradient_weight
        adapted_bias = self.initial_bias - self.alpha_bias * gradient_bias
        query = episode["query_base"][:, None] + episode["query_z"] @ adapted_weight.T + adapted_bias
        return query, adapted_weight, gradient_weight


def _tensor_episode(episode: Episode, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history_z": torch.as_tensor(episode.history_z, dtype=torch.float32, device=device),
        "history_base": torch.as_tensor(episode.history_base, dtype=torch.float32, device=device),
        "history_y": torch.as_tensor(episode.history_y, dtype=torch.float32, device=device),
        "query_z": torch.as_tensor(episode.query_z, dtype=torch.float32, device=device),
        "query_base": torch.as_tensor(episode.query_base, dtype=torch.float32, device=device),
        "query_y": torch.as_tensor(episode.query_y, dtype=torch.float32, device=device),
    }


def _balanced_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = []
    for expert in range(logits.shape[1]):
        terms = []
        for label in (0.0, 1.0):
            mask = labels == label
            terms.append(F.binary_cross_entropy_with_logits(logits[mask, expert], labels[mask]))
        values.append(0.5 * (terms[0] + terms[1]))
    return torch.stack(values)


def _train(
    train: list[Episode],
    target: list[Episode],
    experts: int,
    epochs: int,
    tau: float,
    lambda_mean: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict, MetaSGDBank]:
    train_tensors = [_tensor_episode(value, device) for value in train]
    target_tensors = [_tensor_episode(value, device) for value in target]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MetaSGDBank(train[0].query_z.shape[1], experts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-3)
    log_k = float(np.log(experts))
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        rows, base_rows, adapted_weights = [], [], []
        for episode in train_tensors:
            logits, adapted, _ = model.episode_logits(episode)
            rows.append(_balanced_losses(logits, episode["query_y"]))
            base_rows.append(_balanced_losses(episode["query_base"][:, None].expand(-1, experts), episode["query_y"]))
            adapted_weights.append(adapted)
        losses = torch.stack(rows)
        base_loss = torch.stack(base_rows).mean()
        coverage = (-tau * torch.logsumexp(-losses / tau, dim=1) + tau * log_k).mean()
        mean_loss = losses.mean()
        assignment = torch.softmax(-losses / tau, dim=1)
        balance = experts * torch.mean(torch.square(assignment.mean(dim=0) - 1.0 / experts))
        normalized = F.normalize(model.alpha_weight, dim=1)
        alpha_gram = normalized @ normalized.T
        diversity = torch.mean(torch.square(alpha_gram - torch.eye(experts, device=device)))
        competence = torch.relu(losses.mean(dim=0) - base_loss - 0.03).mean()
        adapted_norm = torch.mean(torch.square(torch.stack(adapted_weights)))
        objective = coverage + lambda_mean * mean_loss + 0.08 * balance + 0.02 * competence + 0.003 * diversity + 0.0005 * adapted_norm
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
                "alpha_mean": model.alpha_weight.mean(dim=1).detach().cpu().numpy().tolist(),
                "alpha_std": model.alpha_weight.std(dim=1).detach().cpu().numpy().tolist(),
            })
    model.load_state_dict(best_state)
    model.eval()
    values = []
    with torch.no_grad():
        for episode in target_tensors:
            logits, _, _ = model.episode_logits(episode)
            values.append(logits.cpu().numpy())
    audit = {
        "best_objective": best_value,
        "history": history,
        "alpha_mean": model.alpha_weight.mean(dim=1).detach().cpu().numpy().tolist(),
        "alpha_std": model.alpha_weight.std(dim=1).detach().cpu().numpy().tolist(),
        "initial_weight_norm": model.initial_weight.norm(dim=1).detach().cpu().numpy().tolist(),
    }
    return np.concatenate(values), audit, model


def run(benchmark: str, family: str, experts: int, epochs: int, folds: tuple[int, ...], tau: float, lambda_mean: float) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family_slug = f"META_SGD_{family.upper()}_K{experts}"
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
    primary = [f"{family_slug}__E{k}_BLEND50" for k in range(experts)]
    audits = []
    for fold in folds:
        data = load_feature_fold(benchmark, fold, family)
        assert_search_only(list(data.meta_subjects) + list(data.search_outcome_subjects), benchmark)
        if not data.search_outcome_subjects:
            continue
        mean, std = _feature_statistics(data, data.meta_subjects)
        meta = [_episode(data, subject, mean, std) for subject in data.meta_subjects]
        outcome = [_episode(data, subject, mean, std) for subject in data.search_outcome_subjects]
        adapted, audit, model = _train(
            meta, outcome, experts, epochs, tau, lambda_mean,
            stable_seed(V8_SEED, benchmark, family_slug, fold), device,
        )
        uids = np.concatenate([episode.query_uid for episode in outcome])
        locked = logit(baseline.set_index("trial_uid").loc[uids, "probability"].to_numpy(float))
        for expert in range(experts):
            predictions.extend(_parts(outcome, adapted[:, expert], family_id, f"{family_slug}__E{expert}_STANDALONE"))
            predictions.extend(_parts(outcome, 0.5 * (locked + adapted[:, expert]), family_id, primary[expert]))
        checkpoint = CACHE / f"{benchmark.upper()}_{family_slug}_FOLD_{fold}.pt"
        torch.save({
            "model": model.state_dict(), "family_slug": family_slug, "source_fold": fold,
            "meta_subjects": list(data.meta_subjects), "search_outcome_subjects": list(data.search_outcome_subjects),
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        }, checkpoint)
        audits.append({
            "benchmark": benchmark_name, "family_id": family_id, "source_fold": fold,
            "meta_subjects": len(meta), "search_outcome_subjects": len(outcome), **audit,
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        })
        print(f"[{benchmark} {family_slug}] fold={fold} objective={audit['best_objective']:.5f} alpha={np.round(audit['alpha_mean'], 3).tolist()}", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(set(load_feature_fold(benchmark, fold, family).search_outcome_subjects) for fold in folds))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "feature_family": family, "experts": experts, "epochs": epochs, "tau": tau,
        "lambda_mean": lambda_mean, "folds": list(folds), "baseline_source_method": baseline_source_method,
        "training_objective": "true differentiable history gradient step with learned signed per-feature step sizes and future-query coverage loss",
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
        "partition": "V8_SEARCH only", "adaptation": "one legal-history differentiable gradient step",
        "step_sizes": "query-trained on source-fold non-outcome V8_SEARCH subjects",
        "search_outcome_future_labels_used_for_fit_or_selection": False,
        "internal_holdout_used": False, "WBCIC_outer_split_opened": False, "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural hypothesis: a true future-query-trained Meta-SGD rule can turn legal history gradients into complementary future-session adaptation.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--family", default="CONFORMER_NORM")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--lambda-mean", type=float, default=0.35)
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, args.family, args.experts, args.epochs, folds, args.tau, args.lambda_mean)


if __name__ == "__main__":
    main()
