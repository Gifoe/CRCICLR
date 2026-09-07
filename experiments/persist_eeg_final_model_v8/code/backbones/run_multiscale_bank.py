"""Multi-scale temporal-spatial TCN backbone with coverage-trained heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_banks.run_query_bank import Episode, _parts, _upsert
from common import CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED, ensure_directories, logit, stable_seed, v7_outputs, write_csv, write_json
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


class ResidualTCN(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, groups=channels, bias=False)
        self.point1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, groups=channels, bias=False)
        self.point2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.dropout(F.gelu(self.norm1(self.point1(self.conv1(x)))))
        value = self.dropout(self.norm2(self.point2(self.conv2(value))))
        return F.gelu(x + value)


class MultiScaleTCNBank(nn.Module):
    def __init__(self, eeg_channels: int, experts: int, dropout: float = 0.35, adapter_rank: int = 8):
        super().__init__()
        self.experts = experts
        self.adapter_rank = int(adapter_rank)
        self.temporal_short = nn.Conv2d(1, 8, (1, 15), padding="same", bias=False)
        self.temporal_mid = nn.Conv2d(1, 8, (1, 31), padding="same", bias=False)
        self.temporal_long = nn.Conv2d(1, 8, (1, 63), padding="same", bias=False)
        self.temporal_norm = nn.BatchNorm2d(24)
        self.spatial = nn.Conv2d(24, 48, (eeg_channels, 1), groups=24, bias=False)
        self.spatial_norm = nn.BatchNorm2d(48)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.depth = nn.Conv1d(48, 48, 15, padding=7, groups=48, bias=False)
        self.point = nn.Conv1d(48, 64, 1, bias=False)
        self.norm = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(5)
        self.tcn1 = ResidualTCN(64, 1, dropout)
        self.tcn2 = ResidualTCN(64, 2, dropout)
        self.attention = nn.Conv1d(64, 1, 1)
        self.embedding = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 64), nn.LayerNorm(64))
        self.population_head = nn.Linear(64, 2)
        self.adapters = nn.ModuleList([
            (
                nn.Sequential(
                    nn.LayerNorm(64),
                    nn.Linear(64, self.adapter_rank, bias=False),
                    nn.GELU(),
                    nn.Linear(self.adapter_rank, 64, bias=False),
                )
                if self.adapter_rank > 0
                else nn.Identity()
            )
            for _ in range(experts)
        ])
        self.heads = nn.ModuleList([nn.Linear(64, 2) for _ in range(experts)])
        if self.adapter_rank > 0:
            for adapter in self.adapters:
                nn.init.normal_(adapter[-1].weight, std=0.002)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        value = x[..., ::2]
        value = value - value.mean(dim=2, keepdim=True)
        scale = torch.sqrt(torch.clamp(torch.mean(torch.square(value), dim=(1, 2), keepdim=True), min=1e-8))
        return value / scale

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = self.normalize(x).unsqueeze(1)
        value = torch.cat((self.temporal_short(value), self.temporal_mid(value), self.temporal_long(value)), dim=1)
        value = self.dropout1(self.pool1(F.elu(self.spatial_norm(self.spatial(self.temporal_norm(value)))))).squeeze(2)
        value = self.pool2(F.gelu(self.norm(self.point(self.depth(value)))))
        value = self.tcn2(self.tcn1(value))
        attention = torch.softmax(self.attention(value), dim=2)
        mean = torch.sum(value * attention, dim=2)
        variance = torch.sum(torch.square(value - mean[:, :, None]) * attention, dim=2)
        return self.embedding(torch.cat((mean, torch.sqrt(torch.clamp(variance, min=1e-6))), dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.forward_features(x)
        expert_logits = []
        for adapter, head in zip(self.adapters, self.heads):
            expert_feature = (
                feature + 0.5 * torch.tanh(adapter(feature))
                if self.adapter_rank > 0
                else feature
            )
            expert_logits.append(head(expert_feature))
        return (
            torch.stack(expert_logits, dim=1),
            self.population_head(feature),
            feature,
        )


def _batch_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    sessions: torch.Tensor,
    future_session: int,
    tau: float,
    lambda_mean: float,
    query_competence: bool = False,
):
    count, experts, _ = logits.shape
    expanded = labels[:, None].expand(-1, experts)
    losses = F.cross_entropy(logits.reshape(count * experts, 2), expanded.reshape(-1), reduction="none").reshape(count, experts)
    query = sessions == future_session
    rows = []
    for subject in torch.unique(subjects[query]):
        mask = query & (subjects == subject)
        class_rows = []
        for label in (0, 1):
            cell = mask & (labels == label)
            if bool(cell.any()):
                class_rows.append(losses[cell].mean(dim=0))
        if len(class_rows) == 2:
            rows.append(0.5 * (class_rows[0] + class_rows[1]))
    if rows:
        subject_losses = torch.stack(rows)
        coverage = (-tau * torch.logsumexp(-subject_losses / tau, dim=1) + tau * np.log(experts)).mean()
        assignment = torch.softmax(-subject_losses / tau, dim=1)
        balance = experts * torch.mean(torch.square(assignment.mean(dim=0) - 1.0 / experts))
        query_mean = subject_losses.mean()
    else:
        coverage = losses.mean()
        balance = torch.zeros((), device=logits.device)
        assignment = torch.full((1, experts), 1.0 / experts, device=logits.device)
        query_mean = losses.mean()
    mean_loss = query_mean if query_competence else losses.mean()
    return coverage + lambda_mean * mean_loss + 0.05 * balance, coverage, mean_loss, balance, assignment


def _subject_balanced_ce(logits: torch.Tensor, labels: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
    """Equalize subjects and classes inside a stochastic training batch."""
    rows = []
    for subject in torch.unique(subjects):
        subject_mask = subjects == subject
        classes = []
        for label in (0, 1):
            cell = subject_mask & (labels == label)
            if bool(cell.any()):
                classes.append(F.cross_entropy(logits[cell].float(), labels[cell], reduction="mean"))
        if len(classes) == 2:
            rows.append(0.5 * (classes[0] + classes[1]))
    return torch.stack(rows).mean() if rows else F.cross_entropy(logits.float(), labels)


def _initialize_experts_from_population(model: MultiScaleTCNBank) -> None:
    with torch.no_grad():
        for expert_index, head in enumerate(model.heads):
            head.weight.copy_(model.population_head.weight)
            head.bias.copy_(model.population_head.bias)
            # The population warm start should preserve competence without
            # leaving a permutation-symmetric bank.  This fixed, seeded
            # perturbation lets the low-rank expert adapters receive distinct
            # coverage gradients from the first specialization step.
            head.weight.add_(0.002 * torch.randn_like(head.weight))
            head.bias.add_(0.001 * (expert_index - 0.5 * (model.experts - 1)))


def _train(
    model,
    raw_search: torch.Tensor,
    metadata: pd.DataFrame,
    train_subjects: tuple[str, ...],
    future_session: int,
    epochs: int,
    pretrain_epochs: int,
    training_mode: str,
    seed: int,
    device: torch.device,
    tau: float,
    lambda_mean: float,
):
    mask = metadata.subject_id.astype(str).isin(train_subjects).to_numpy()
    local_indices = np.flatnonzero(mask)
    labels_np = metadata.label.to_numpy(int)
    sessions_np = metadata.session_id.to_numpy(int)
    subject_codes, _ = pd.factorize(metadata.subject_id.astype(str), sort=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    index_tensor = torch.as_tensor(local_indices, dtype=torch.long, device=device)
    if training_mode == "staged":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(pretrain_epochs, 1), eta_min=1e-4,
        )
        for epoch in range(pretrain_epochs):
            model.train()
            permutation = index_tensor[torch.randperm(len(index_tensor), generator=generator, device=device)]
            total = 0.0
            steps = 0
            for start in range(0, len(permutation), 192):
                index = permutation[start:start + 192]
                cpu_index = index.detach().cpu().numpy()
                x = raw_search[index].float()
                y = torch.as_tensor(labels_np[cpu_index], dtype=torch.long, device=device)
                subject = torch.as_tensor(subject_codes[cpu_index], dtype=torch.long, device=device)
                shift = int(torch.randint(-12, 13, (1,), generator=generator, device=device))
                x = torch.roll(x, shift, dims=2)
                keep = (torch.rand((len(x), x.shape[1], 1), generator=generator, device=device) > 0.02).to(x.dtype)
                x = x * keep
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, population, _ = model(x)
                    objective = _subject_balanced_ce(population, y, subject)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(objective.detach())
                steps += 1
            scheduler.step()
            row = {
                "phase": "competence_pretrain",
                "epoch": epoch + 1,
                "objective": total / max(steps, 1),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(
                f"[multiscale pretrain] epoch={epoch + 1}/{pretrain_epochs} "
                f"objective={row['objective']:.4f}",
                flush=True,
            )
        _initialize_experts_from_population(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if training_mode == "staged" else 8e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=5e-5 if training_mode == "staged" else 1e-4)
    for epoch in range(epochs):
        model.train()
        permutation = index_tensor[torch.randperm(len(index_tensor), generator=generator, device=device)]
        totals = np.zeros(5, dtype=float)
        steps = 0
        usage = np.zeros(model.experts, dtype=float)
        for start in range(0, len(permutation), 192):
            index = permutation[start:start + 192]
            cpu_index = index.detach().cpu().numpy()
            x = raw_search[index].float()
            y = torch.as_tensor(labels_np[cpu_index], dtype=torch.long, device=device)
            session = torch.as_tensor(sessions_np[cpu_index], dtype=torch.long, device=device)
            subject = torch.as_tensor(subject_codes[cpu_index], dtype=torch.long, device=device)
            if model.training:
                shift = int(torch.randint(-12, 13, (1,), generator=generator, device=device))
                x = torch.roll(x, shift, dims=2)
                keep = (torch.rand((len(x), x.shape[1], 1), generator=generator, device=device) > 0.02).to(x.dtype)
                x = x * keep
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, population, _ = model(x)
                objective, coverage, mean_loss, balance, assignment = _batch_loss(
                    logits.float(), y, subject, session, future_session, tau,
                    lambda_mean, query_competence=training_mode == "staged",
                )
                population_loss = _subject_balanced_ce(population, y, subject)
                if training_mode == "staged":
                    objective = objective + 0.5 * population_loss
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [
                float(objective.detach()), float(coverage.detach()),
                float(mean_loss.detach()), float(balance.detach()),
                float(population_loss.detach()),
            ]
            usage += assignment.mean(dim=0).detach().cpu().numpy()
            steps += 1
        scheduler.step()
        row = {
            "phase": "coverage",
            "epoch": epoch + 1, "objective": totals[0] / steps, "coverage_loss": totals[1] / steps,
            "mean_loss": totals[2] / steps, "balance": totals[3] / steps,
            "population_loss": totals[4] / steps,
            "usage": (usage / steps).tolist(), "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(f"[multiscale train] epoch={epoch + 1}/{epochs} objective={row['objective']:.4f} usage={np.round(usage / steps, 3).tolist()}", flush=True)
    return history


def _extract(model, raw_search: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits = np.empty((len(raw_search), model.experts), dtype=np.float32)
    population = np.empty(len(raw_search), dtype=np.float32)
    features = np.empty((len(raw_search), 64), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(raw_search), 256):
            x = raw_search[start:start + 256].float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output, common, feature = model(x)
            logits[start:start + len(x)] = (output[:, :, 1] - output[:, :, 0]).float().cpu().numpy()
            population[start:start + len(x)] = (common[:, 1] - common[:, 0]).float().cpu().numpy()
            features[start:start + len(x)] = feature.float().cpu().numpy()
    return logits, population, features


def _history_head(history_z: np.ndarray, history_y: np.ndarray, query_z: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(history_z)
    classifier = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=V8_SEED)
    classifier.fit(scaler.transform(history_z), history_y)
    return np.asarray(classifier.decision_function(scaler.transform(query_z)), dtype=float)


def run(
    benchmark: str,
    experts: int,
    adapter_rank: int,
    epochs: int,
    pretrain_epochs: int,
    training_mode: str,
    folds: tuple[int, ...],
    tau: float,
    lambda_mean: float,
) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    benchmark_name = "OpenBMI_MI_S1_to_S2" if benchmark == "openbmi" else "WBCIC_S1S2_to_S3_authorized_development"
    family_slug = (
        (
            f"MULTISCALE_TCN_LR_STAGED_BANK_K{experts}_R{adapter_rank}"
            if adapter_rank > 0
            else f"MULTISCALE_TCN_STAGED_BANK_K{experts}"
        )
        if training_mode == "staged"
        else f"MULTISCALE_TCN_BANK_K{experts}"
    )
    family_id = f"{benchmark_name}__{family_slug}"
    protocol = load_feature_fold(benchmark, 0, "CONFORMER_NORM").protocol
    canonical = pd.read_parquet(CACHE / f"{prefix}_SEARCH_ROWS_FOLD_0.parquet").sort_values("source_index").reset_index(drop=True)
    raw_disk = np.load(v7_outputs() / "cache" / f"{prefix}_RAW_EPOCHS_FLOAT16.npy", mmap_mode="r", allow_pickle=False)
    raw_search = torch.as_tensor(np.asarray(raw_disk[canonical.source_index.to_numpy(int)], dtype=np.float16), device=device)
    canonical["local_index"] = np.arange(len(canonical), dtype=int)
    uid_to_local = canonical.set_index("trial_uid").local_index
    baseline, baseline_source_method = baseline_predictions(benchmark)
    baseline = baseline.loc[baseline.subject_id.astype(str).isin(protocol.search_subjects)].copy()
    baseline["method_id"] = "B_STRONG_MATCHED_V7"
    baseline["family_id"] = family_id
    baseline["source_fold"] = baseline.outer_fold.astype(int)
    baseline["benchmark"] = benchmark_name
    baseline["internal_holdout_used"] = False
    predictions = [baseline]
    primary = []
    if training_mode == "staged":
        primary.extend((f"{family_slug}__POPULATION_BLEND50", f"{family_slug}__FEATURE_HEAD_BLEND50"))
    for expert in range(experts):
        primary.extend((f"{family_slug}__E{expert}_GENERIC_BLEND50", f"{family_slug}__E{expert}_HISTORY_RESIDUAL50"))
    audits = []
    for fold in folds:
        data = load_feature_fold(benchmark, fold, "CONFORMER_NORM")
        assert_search_only(list(data.meta_subjects) + list(data.search_outcome_subjects), benchmark)
        if not data.search_outcome_subjects:
            continue
        seed = stable_seed(V8_SEED, benchmark, family_slug, fold)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = MultiScaleTCNBank(int(raw_search.shape[1]), experts, adapter_rank=adapter_rank).to(device)
        history = _train(
            model, raw_search, canonical, data.meta_subjects,
            data.protocol.future_session, epochs, pretrain_epochs,
            training_mode, seed, device, tau, lambda_mean,
        )
        logits_all, population_all, features_all = _extract(model, raw_search, device)
        for subject in data.search_outcome_subjects:
            rows = canonical.loc[canonical.subject_id.astype(str).eq(str(subject))]
            history_rows = rows.loc[rows.session_id.astype(int).isin(data.protocol.history_sessions)]
            future_rows = rows.loc[rows.session_id.astype(int).eq(data.protocol.future_session)]
            hi = history_rows.local_index.to_numpy(int)
            qi = future_rows.local_index.to_numpy(int)
            uid = future_rows.trial_uid.astype(str).to_numpy()
            y = future_rows.label.to_numpy(np.float32)
            locked = logit(baseline.set_index("trial_uid").loc[uid, "probability"].to_numpy(float))
            subject_head = _history_head(features_all[hi], history_rows.label.to_numpy(int), features_all[qi])
            shell = Episode(str(subject), fold, np.empty(0), np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0), np.empty((len(y), 0)), logits_all[qi, 0], y, uid)
            if training_mode == "staged":
                population = population_all[qi]
                predictions.extend(_parts([shell], population, family_id, f"{family_slug}__POPULATION_STANDALONE"))
                predictions.extend(_parts([shell], subject_head, family_id, f"{family_slug}__FEATURE_HEAD_STANDALONE"))
                predictions.extend(_parts([shell], 0.5 * (locked + population), family_id, f"{family_slug}__POPULATION_BLEND50"))
                predictions.extend(_parts([shell], 0.5 * (locked + subject_head), family_id, f"{family_slug}__FEATURE_HEAD_BLEND50"))
            for expert in range(experts):
                generic = logits_all[qi, expert]
                adapted = 0.5 * (generic + subject_head)
                predictions.extend(_parts([shell], generic, family_id, f"{family_slug}__E{expert}_STANDALONE"))
                predictions.extend(_parts([shell], 0.5 * (locked + generic), family_id, f"{family_slug}__E{expert}_GENERIC_BLEND50"))
                predictions.extend(_parts([shell], locked + 0.5 * (adapted - generic), family_id, f"{family_slug}__E{expert}_HISTORY_RESIDUAL50"))
        checkpoint = CACHE / f"{prefix}_{family_slug}_FOLD_{fold}.pt"
        torch.save({
            "model": model.state_dict(), "source_fold": fold, "experts": experts, "epochs": epochs,
            "adapter_rank": adapter_rank, "pretrain_epochs": pretrain_epochs, "training_mode": training_mode,
            "meta_subjects": list(data.meta_subjects), "search_outcome_subjects": list(data.search_outcome_subjects),
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        }, checkpoint)
        audits.append({
            "benchmark": benchmark_name, "family_id": family_id, "source_fold": fold, "history": history,
            "meta_subjects": len(data.meta_subjects), "search_outcome_subjects": len(data.search_outcome_subjects),
            "parameters": int(sum(value.numel() for value in model.parameters())),
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        })
        print(f"[{benchmark} {family_slug}] fold={fold} complete", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(set(load_feature_fold(benchmark, fold, "CONFORMER_NORM").search_outcome_subjects) for fold in folds))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "experts": experts, "adapter_rank": adapter_rank, "epochs": epochs, "pretrain_epochs": pretrain_epochs,
        "training_mode": training_mode, "tau": tau, "lambda_mean": lambda_mean,
        "folds": list(folds), "baseline_source_method": baseline_source_method,
        "training_objective": (
            "competence-first subject-balanced population pretraining followed by future-session subject-coverage specialization"
            if training_mode == "staged"
            else "future-session subject-coverage loss plus all-session competence on a multi-scale temporal-spatial TCN"
        ),
        "candidate_actions": "generic anchor blend and legal-history feature-head residual",
    })
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
        "raw_rows": "V8_SEARCH only", "population_fit": "source-fold non-outcome V8_SEARCH subjects",
        "coverage_query": "future sessions of population-fit subjects", "outcome_history_adaptation": "legal history only",
        "search_outcome_future_labels_used_for_fit_or_selection": False, "internal_holdout_used": False,
        "WBCIC_outer_split_opened": False, "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural hypothesis: a stronger multi-scale temporal-spatial TCN trained with subject-level future coverage can create competent complementary experts beyond frozen EEGNet/Conformer representations.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrain-epochs", type=int, default=24)
    parser.add_argument("--training-mode", choices=("joint", "staged"), default="staged")
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--lambda-mean", type=float, default=0.40)
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(
        args.benchmark, args.experts, args.adapter_rank, args.epochs, args.pretrain_epochs,
        args.training_mode, folds, args.tau, args.lambda_mean,
    )


if __name__ == "__main__":
    main()
