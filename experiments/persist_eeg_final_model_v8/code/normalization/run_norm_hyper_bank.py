"""Class-history-conditioned FiLM/normalization hypernetwork bank."""

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

from adaptation_banks.run_query_bank import Episode, _episode, _feature_statistics, _parts, _project_context, _upsert
from common import CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED, ensure_directories, logit, stable_seed, write_csv, write_json
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


class NormHyperBank(nn.Module):
    def __init__(self, context_dim: int, feature_dim: int, experts: int, rank: int):
        super().__init__()
        self.experts = experts
        self.rank = rank
        hidden = max(16, min(48, 2 * context_dim))
        self.encoder = nn.Sequential(nn.Linear(context_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.coefficient = nn.Linear(hidden, experts * rank)
        self.gamma_basis = nn.Parameter(torch.randn(experts, rank, feature_dim) * 0.02)
        self.beta_basis = nn.Parameter(torch.randn(experts, rank, feature_dim) * 0.02)
        self.residual_head = nn.Parameter(torch.randn(experts, feature_dim) * 0.01)
        self.residual_bias = nn.Parameter(torch.zeros(experts))
        initial_rho = torch.linspace(0.10, 0.75, experts)
        self.rho_raw = nn.Parameter(torch.logit(initial_rho))
        nn.init.zeros_(self.coefficient.weight)
        nn.init.zeros_(self.coefficient.bias)

    def episode_logits(self, context: torch.Tensor, episode: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(context[None])
        coefficient = 1.5 * torch.tanh(self.coefficient(hidden).reshape(self.experts, self.rank))
        gamma = 1.0 + 0.25 * torch.tanh(torch.einsum("kr,krd->kd", coefficient, self.gamma_basis))
        beta = 0.25 * torch.tanh(torch.einsum("kr,krd->kd", coefficient, self.beta_basis))
        history_mean = episode["history_z"].mean(dim=0)
        history_std = torch.clamp(episode["history_z"].std(dim=0, unbiased=False), min=0.20, max=5.0)
        normalized = (episode["query_z"] - history_mean) / history_std
        rho = torch.sigmoid(self.rho_raw)
        transported = (1.0 - rho[None, :, None]) * episode["query_z"][:, None, :] + rho[None, :, None] * normalized[:, None, :]
        transported = transported * gamma[None] + beta[None]
        residual = torch.einsum("nkd,kd->nk", transported, self.residual_head) + self.residual_bias
        return episode["query_base"][:, None] + residual, coefficient, rho


def _tensor_episode(episode: Episode, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history_z": torch.as_tensor(episode.history_z, dtype=torch.float32, device=device),
        "query_z": torch.as_tensor(episode.query_z, dtype=torch.float32, device=device),
        "query_base": torch.as_tensor(episode.query_base, dtype=torch.float32, device=device),
        "query_y": torch.as_tensor(episode.query_y, dtype=torch.float32, device=device),
    }


def _balanced_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = []
    for expert in range(logits.shape[1]):
        terms = []
        for label in (0.0, 1.0):
            mask = labels == label
            terms.append(F.binary_cross_entropy_with_logits(logits[mask, expert], labels[mask]))
        result.append(0.5 * (terms[0] + terms[1]))
    return torch.stack(result)


def _train(train: list[Episode], target: list[Episode], experts: int, rank: int, epochs: int, tau: float, lambda_mean: float, seed: int, device: torch.device):
    train_context, target_context, context_audit = _project_context(train, target)
    train_tensors = [_tensor_episode(item, device) for item in train]
    target_tensors = [_tensor_episode(item, device) for item in target]
    train_context_tensor = torch.as_tensor(train_context, dtype=torch.float32, device=device)
    target_context_tensor = torch.as_tensor(target_context, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = NormHyperBank(train_context.shape[1], train[0].query_z.shape[1], experts, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    log_k = float(np.log(experts))
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses, base_losses, coefficients = [], [], []
        for index, episode in enumerate(train_tensors):
            logits, coefficient, _ = model.episode_logits(train_context_tensor[index], episode)
            losses.append(_balanced_losses(logits, episode["query_y"]))
            base_losses.append(_balanced_losses(episode["query_base"][:, None].expand(-1, experts), episode["query_y"]))
            coefficients.append(coefficient)
        losses = torch.stack(losses)
        base_loss = torch.stack(base_losses).mean()
        coverage = (-tau * torch.logsumexp(-losses / tau, dim=1) + tau * log_k).mean()
        mean_loss = losses.mean()
        assignment = torch.softmax(-losses / tau, dim=1)
        balance = experts * torch.mean(torch.square(assignment.mean(dim=0) - 1.0 / experts))
        flattened = F.normalize(model.gamma_basis.flatten(1), dim=1)
        gram = flattened @ flattened.T
        redundancy = torch.mean(torch.square(gram - torch.eye(experts, device=device)))
        competence = torch.relu(losses.mean(dim=0) - base_loss - 0.03).mean()
        regularity = torch.mean(torch.square(torch.stack(coefficients))) + torch.mean(torch.square(model.residual_head))
        objective = coverage + lambda_mean * mean_loss + 0.08 * balance + 0.02 * competence + 0.004 * redundancy + 0.0005 * regularity
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(objective.detach())
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
        if epoch in {0, 49, 99, 199, 399, epochs - 1}:
            history.append({
                "epoch": epoch + 1, "objective": value, "coverage_loss": float(coverage.detach()),
                "mean_loss": float(mean_loss.detach()), "assignment": assignment.mean(dim=0).detach().cpu().numpy().tolist(),
                "rho": torch.sigmoid(model.rho_raw).detach().cpu().numpy().tolist(),
            })
    model.load_state_dict(best_state)
    model.eval()
    predictions, bases = [], []
    with torch.no_grad():
        for index, episode in enumerate(target_tensors):
            logits, _, _ = model.episode_logits(target_context_tensor[index], episode)
            predictions.append(logits.cpu().numpy())
            bases.append(episode["query_base"].cpu().numpy())
    audit = {
        "best_objective": best_value, "history": history, "context": context_audit,
        "rho": torch.sigmoid(model.rho_raw).detach().cpu().numpy().tolist(),
        "mean_abs_gamma_basis": float(model.gamma_basis.abs().mean().detach()),
        "mean_abs_beta_basis": float(model.beta_basis.abs().mean().detach()),
    }
    return np.concatenate(predictions), np.concatenate(bases), audit, model


def run(benchmark: str, family: str, experts: int, rank: int, epochs: int, folds: tuple[int, ...], tau: float, lambda_mean: float) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family_slug = f"NORM_HYPER_{family.upper()}_K{experts}_R{rank}"
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
        adapted, cached_base, audit, model = _train(meta, outcome, experts, rank, epochs, tau, lambda_mean, stable_seed(V8_SEED, benchmark, family_slug, fold), device)
        uids = np.concatenate([item.query_uid for item in outcome])
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
        print(f"[{benchmark} {family_slug}] fold={fold} objective={audit['best_objective']:.5f} rho={np.round(audit['rho'], 3).tolist()}", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(set(load_feature_fold(benchmark, fold, family).search_outcome_subjects) for fold in folds))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "feature_family": family, "experts": experts, "rank": rank, "epochs": epochs,
        "tau": tau, "lambda_mean": lambda_mean, "folds": list(folds), "baseline_source_method": baseline_source_method,
        "training_objective": "query-trained history-conditioned normalization/FiLM coefficients with coverage and competence",
        "deployment_transform": "locked strong anchor plus half learned normalization residual",
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
        "partition": "V8_SEARCH only", "generated_parameters": "low-rank FiLM gamma/beta only",
        "normalization_statistics": "legal target history only", "query_fit": "source-fold non-outcome future sessions",
        "search_outcome_future_labels_used_for_fit_or_selection": False, "internal_holdout_used": False,
        "WBCIC_outer_split_opened": False, "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural hypothesis: target-history normalization plus a constrained FiLM hypernetwork can correct session shift without generating an unconstrained classifier.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--family", default="MI_SPECIFIC")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--lambda-mean", type=float, default=0.35)
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, args.family, args.experts, args.rank, args.epochs, folds, args.tau, args.lambda_mean)


if __name__ == "__main__":
    main()
