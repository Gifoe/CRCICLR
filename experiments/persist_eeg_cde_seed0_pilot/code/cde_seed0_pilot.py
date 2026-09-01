"""PERSIST-CDE seed-0 constructive pilot.

This experiment is deliberately narrow.  It imports the already frozen
canonical EEGNet implementation for data loading, role construction and the
architecture, then trains only two small residual decision adapters.  The
canonical baseline backbone is never updated by an adapter loss and no seed
other than zero is run here.

The outcome pass is intentionally last.  The only earlier access to outcome
trials is the required checkpoint-equivalence test; that test compares trial
IDs, labels and probabilities and computes no outcome metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


# The experiment lives beside the canonical baseline in the same repository.
# The environment variables are set before importing the canonical module so
# that its data roots remain the frozen Stage-0/cache roots.
REPO = Path(os.environ.get(
    "CDE_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET"
)).resolve()
CANONICAL_EXP = REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
EXP = REPO / "experiments" / "persist_eeg_cde_seed0_pilot"
RESULTS = EXP / "results"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"

os.environ.setdefault("CANONICAL_REPO", str(REPO))
os.environ.setdefault("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full")
os.environ.setdefault(
    "PERSIST_WBCIC_CACHE",
    str(REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1" / "runtime" / "cache"),
)
sys.path.insert(0, str(CANONICAL_EXP / "code"))
import canonical_eegnet_runner as canonical  # noqa: E402


SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
ADAPTER_EPOCHS = 15
ADAPTER_BATCH_SIZE = 64
ADAPTER_LR = 1e-3
ADAPTER_WEIGHT_DECAY = 1e-4
RESIDUAL_SCALE = 0.25
GRL_COEFFICIENT = 0.10
LAMBDA_CC_CORAL = 0.10
LAMBDA_KL = 0.25
BOOTSTRAP_DRAWS = 10_000
ALPHA_VALUES = (0.0, 0.05, 0.10, 0.20, 0.30)
ALPHA_SUM_MAX = 0.30
COMPETENCE_TOLERANCE = 0.05
BASELINE_TARGETS = {"OpenBMI": 0.8190740740740741, "WBCIC": 0.7862985033259424}
TERMINAL_POSITIVE = "CDE_SEED0_POSITIVE_SIGNAL"
TERMINAL_MIXED = "CDE_SEED0_MIXED_SIGNAL"
TERMINAL_NEGATIVE = "CDE_SEED0_NEGATIVE_SIGNAL"
TERMINAL_NO_COMPETENT = "NO_COMPETENT_COUNTERFACTUAL_FOUND"


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


@dataclass
class FeatureBlock:
    indices: np.ndarray
    z: torch.Tensor
    logits: torch.Tensor
    labels: np.ndarray
    subjects: np.ndarray
    trial_uids: np.ndarray
    sessions: np.ndarray


@dataclass
class FoldContext:
    dataset: str
    fold: int
    roles: dict[str, list[str]]
    data: canonical.DatasetData
    initial_idx: np.ndarray
    discovery_idx: np.ndarray
    refit_idx: np.ndarray
    outcome_idx: np.ndarray
    seed0_best_epoch: int


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * grad_output, None


def gradient_reverse(value: torch.Tensor, coefficient: float) -> torch.Tensor:
    return GradientReverse.apply(value, coefficient)


class ResidualAdapter(nn.Module):
    """LN -> Linear(64,16) -> GELU -> zero-initialized Linear(16,64)."""

    def __init__(self, baseline_head: nn.Linear):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 64)
        self.classifier = nn.Linear(64, 2)
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()
        self.classifier.load_state_dict({k: v.detach().clone() for k, v in baseline_head.state_dict().items()})

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.fc2(F.gelu(self.fc1(self.norm(z))))
        z_counterfactual = z + RESIDUAL_SCALE * residual
        return z_counterfactual, self.classifier(z_counterfactual)


class SubjectDiscriminator(nn.Module):
    def __init__(self, n_subjects: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, int(n_subjects)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value = value - value.max(axis=1, keepdims=True)
    p = np.exp(value)
    return p / p.sum(axis=1, keepdims=True)


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    return softmax_np(logits)[:, 1]


def metrics_by_subject(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = np.asarray(labels, dtype=np.int64)
    p1 = np.asarray(p1, dtype=np.float64)
    subjects = np.asarray(subjects).astype(str)
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects == subject
        y = labels[mask]
        prob = p1[mask]
        pred = (prob >= 0.5).astype(np.int64)
        rows.append({
            "subject_id": subject,
            "BA": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "NLL": float(log_loss(y, np.column_stack([1.0 - prob, prob]), labels=[0, 1])),
            "trials": int(mask.sum()),
        })
    return rows


def metric_means(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    rows = pd.DataFrame(metrics_by_subject(labels, p1, subjects))
    return {key: float(rows[key].mean()) for key in ("BA", "accuracy", "macro_F1", "NLL")}


def subject_ce(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    p1 = np.asarray(p1, dtype=np.float64)
    subjects = np.asarray(subjects).astype(str)
    output: dict[str, float] = {}
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects == subject
        output[subject] = float(log_loss(labels[mask], np.column_stack([1.0 - p1[mask], p1[mask]]), labels=[0, 1]))
    return output


def fold_source_indices(data: canonical.DatasetData, roles: dict[str, list[str]], dataset: str) -> tuple[np.ndarray, ...]:
    initial, discovery, refit, outcome = canonical.make_indices(data, roles, dataset)
    if any(len(x) == 0 for x in (initial, discovery, refit, outcome)):
        raise RuntimeError(f"{dataset} fold has an empty legal index set")
    return initial, discovery, refit, outcome


def load_baseline_seed0_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        CANONICAL_EXP / "results" / "CANONICAL_BASELINE_STATISTICS.json",
        CANONICAL_EXP / "results" / "SEED_SUMMARY.csv",
        CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv",
        CANONICAL_EXP / "results" / "PER_SUBJECT_RESULTS.csv",
        CANONICAL_EXP / "code" / "canonical_eegnet_runner.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("CDE_PILOT_BASELINE_MISMATCH: missing canonical artifacts: " + ", ".join(missing))
    seed_summary = pd.read_csv(CANONICAL_EXP / "results" / "SEED_SUMMARY.csv")
    for dataset, target in BASELINE_TARGETS.items():
        rows = seed_summary[(seed_summary.dataset == dataset) & (seed_summary.seed.astype(str) == "0")]
        if len(rows) != 1 or abs(float(rows.iloc[0].mean_subject_BA) - target) > 1e-10:
            raise RuntimeError(f"CDE_PILOT_BASELINE_MISMATCH: {dataset} seed0 BA does not match {target}")
    trial_predictions = pd.read_csv(CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv")
    for dataset in DATASETS:
        rows = trial_predictions[(trial_predictions.dataset == dataset) & (trial_predictions.seed.astype(str) == "0")]
        if rows.empty or rows.trial_uid.duplicated().any():
            raise RuntimeError(f"CDE_PILOT_BASELINE_MISMATCH: {dataset} seed0 trial predictions are incomplete")
    stats = json.loads((CANONICAL_EXP / "results" / "CANONICAL_BASELINE_STATISTICS.json").read_text(encoding="utf-8-sig"))
    return trial_predictions, {"seed_summary_sha256": sha256_file(CANONICAL_EXP / "results" / "SEED_SUMMARY.csv"), "statistics_sha256": sha256_file(CANONICAL_EXP / "results" / "CANONICAL_BASELINE_STATISTICS.json"), "per_subject_sha256": sha256_file(CANONICAL_EXP / "results" / "PER_SUBJECT_RESULTS.csv"), "trial_predictions_sha256": sha256_file(CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv"), "statistics": stats}


def make_context(dataset: str, fold: int) -> FoldContext:
    roles_by_fold, pool, _ = canonical.load_roles(dataset)
    roles = roles_by_fold[fold]
    data = canonical.load_dataset(dataset, pool)
    initial, discovery, refit, outcome = fold_source_indices(data, roles, dataset)
    partial_path = canonical.RUNTIME / "partial" / f"{dataset.lower()}_fold-{fold}_seed-0.json"
    if not partial_path.is_file():
        raise RuntimeError(f"missing canonical seed0 partial: {partial_path}")
    partial = json.loads(partial_path.read_text(encoding="utf-8-sig"))
    if partial.get("complete") is not True:
        raise RuntimeError(f"canonical seed0 partial is incomplete: {partial_path}")
    best_epoch = int(partial["best_epoch"])
    if best_epoch < 1:
        raise RuntimeError(f"invalid canonical seed0 best epoch: {best_epoch}")
    return FoldContext(dataset, fold, roles, data, initial, discovery, refit, outcome, best_epoch)


def fit_model_fit_only(ctx: FoldContext, mean: np.ndarray, std: np.ndarray, device: torch.device) -> canonical.VanillaEEGNet:
    """Reproduce the canonical initial fit, but run the predeclared epoch count."""
    run_seed = canonical.stable_seed("canonical-initial", ctx.dataset, ctx.fold, SEED)
    canonical.set_seed(run_seed)
    model = canonical.VanillaEEGNet(ctx.data.batch(ctx.initial_idx[:1]).shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=canonical.LEARNING_RATE, weight_decay=canonical.WEIGHT_DECAY)
    order_rng = np.random.default_rng(canonical.stable_seed("canonical-order", ctx.dataset, ctx.fold, SEED, "initial"))
    for epoch in range(1, ctx.seed0_best_epoch + 1):
        model.train()
        order = order_rng.permutation(ctx.initial_idx)
        losses: list[float] = []
        for start in range(0, len(order), canonical.BATCH_SIZE):
            part = order[start : start + canonical.BATCH_SIZE]
            xb = canonical.prepare_batch(ctx.data, part, mean, std, device)
            yb = torch.as_tensor(np.array(ctx.data.metadata.iloc[part].label.to_numpy(np.int64), copy=True), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[cde-backbone] {ctx.dataset} fold={ctx.fold} epoch={epoch}/{ctx.seed0_best_epoch} loss={np.mean(losses):.5f}", flush=True)
    model.eval()
    return model


def materialize_features(data: canonical.DatasetData, model: canonical.VanillaEEGNet, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> FeatureBlock:
    z_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        model.eval()
        for start in range(0, len(indices), canonical.BATCH_SIZE):
            part = indices[start : start + canonical.BATCH_SIZE]
            xb = canonical.prepare_batch(data, part, mean, std, device)
            z = model.forward_features(xb)
            z_parts.append(z.detach())
            logit_parts.append(model.head(z).detach())
    frame = data.metadata.iloc[indices]
    return FeatureBlock(
        indices=np.asarray(indices, dtype=np.int64),
        z=torch.cat(z_parts, dim=0),
        logits=torch.cat(logit_parts, dim=0),
        labels=frame.label.to_numpy(np.int64),
        subjects=frame.subject_id.astype(str).to_numpy(),
        trial_uids=frame.trial_uid.astype(str).to_numpy(),
        sessions=frame.session_id.to_numpy(np.int64),
    )


def subject_chunks(subjects: np.ndarray, labels: np.ndarray, seed: int) -> list[np.ndarray]:
    """Make deterministic, non-overlapping chunks with multiple subjects.

    Each subject contributes up to two samples per class to a chunk whenever
    possible, which makes class-conditional covariance legal for GEO.  Every
    input position is consumed exactly once; there is no outcome duplication
    or synthetic trial creation.
    """
    rng = np.random.default_rng(seed)
    by_subject: dict[str, dict[int, list[int]]] = {}
    for pos, (subject, label) in enumerate(zip(subjects.astype(str), labels.astype(int))):
        by_subject.setdefault(subject, {0: [], 1: []})[int(label)].append(int(pos))
    chunks: list[np.ndarray] = []
    for subject in subject_sort(by_subject):
        cells = by_subject[subject]
        for values in cells.values():
            rng.shuffle(values)
        while cells[0] or cells[1]:
            take: list[int] = []
            for label in (0, 1):
                take.extend(cells[label][:2])
                del cells[label][: min(2, len(cells[label]))]
            while len(take) < 4 and (cells[0] or cells[1]):
                source = cells[0] if len(cells[0]) >= len(cells[1]) else cells[1]
                take.append(source.pop(0))
            chunks.append(np.asarray(take, dtype=np.int64))
    rng.shuffle(chunks)
    return chunks


def subject_balanced_batches(subjects: np.ndarray, labels: np.ndarray, batch_size: int, seed: int) -> list[np.ndarray]:
    chunks = subject_chunks(subjects, labels, seed)
    order = np.concatenate(chunks, axis=0) if chunks else np.empty(0, dtype=np.int64)
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def class_conditional_subject_coral(z: torch.Tensor, labels: np.ndarray, subjects: np.ndarray) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects).astype(str)
    for label in (0, 1):
        covariances: list[torch.Tensor] = []
        for subject in subject_sort(np.unique(subjects)):
            mask = (labels == label) & (subjects == subject)
            if int(mask.sum()) < 2:
                continue
            index = torch.as_tensor(mask, dtype=torch.bool, device=z.device)
            value = z[index]
            centered = value - value.mean(dim=0, keepdim=True)
            covariance = centered.transpose(0, 1).matmul(centered) / float(value.shape[0] - 1)
            covariances.append(covariance)
        if covariances:
            mean_covariance = torch.stack(covariances, dim=0).mean(dim=0)
            penalties.append(torch.stack([((cov - mean_covariance) ** 2).sum() / float(z.shape[1] ** 2) for cov in covariances]).mean())
    if penalties:
        return torch.stack(penalties).mean()
    return z.sum() * 0.0


def train_inv(block: FeatureBlock, baseline_head: nn.Linear, run_name: str, dataset: str, fold: int, device: torch.device) -> tuple[ResidualAdapter, SubjectDiscriminator, dict[str, Any]]:
    set_seed(stable_seed("cde-adapter", run_name, dataset, fold, "INV"))
    adapter = ResidualAdapter(baseline_head).to(device)
    subject_names = subject_sort(np.unique(block.subjects))
    subject_to_index = {subject: i for i, subject in enumerate(subject_names)}
    subject_targets = torch.as_tensor(np.asarray([subject_to_index[str(x)] for x in block.subjects], dtype=np.int64), dtype=torch.long, device=device)
    discriminator = SubjectDiscriminator(len(subject_names)).to(device)
    optimizer = torch.optim.AdamW(list(adapter.parameters()) + list(discriminator.parameters()), lr=ADAPTER_LR, weight_decay=ADAPTER_WEIGHT_DECAY)
    labels_t = torch.as_tensor(np.asarray(block.labels, dtype=np.int64), dtype=torch.long, device=device)
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        adapter.train(); discriminator.train()
        losses = []
        batches = subject_balanced_batches(block.subjects, block.labels, ADAPTER_BATCH_SIZE, stable_seed("cde-batches", run_name, dataset, fold, epoch, "INV"))
        for positions in batches:
            pos = torch.as_tensor(positions, dtype=torch.long, device=device)
            z_batch = block.z[pos]
            base_logits = block.logits[pos].detach()
            z_counterfactual, logits = adapter(z_batch)
            p_base = F.softmax(base_logits, dim=1).detach()
            ce = F.cross_entropy(logits, labels_t[pos])
            adv = F.cross_entropy(discriminator(gradient_reverse(z_counterfactual, GRL_COEFFICIENT)), subject_targets[pos])
            kl = F.kl_div(F.log_softmax(logits, dim=1), p_base, reduction="batchmean")
            loss = ce + GRL_COEFFICIENT * adv + LAMBDA_KL * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(discriminator.parameters()), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[cde-INV] {run_name} {dataset} fold={fold} epoch={epoch}/{ADAPTER_EPOCHS} loss={np.mean(losses):.5f}", flush=True)
    adapter.eval(); discriminator.eval()
    return adapter, discriminator, {"n_subjects": len(subject_names), "subject_names": subject_names}


def train_geo(block: FeatureBlock, baseline_head: nn.Linear, run_name: str, dataset: str, fold: int, device: torch.device) -> tuple[ResidualAdapter, dict[str, Any]]:
    set_seed(stable_seed("cde-adapter", run_name, dataset, fold, "GEO"))
    adapter = ResidualAdapter(baseline_head).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=ADAPTER_LR, weight_decay=ADAPTER_WEIGHT_DECAY)
    labels_t = torch.as_tensor(np.asarray(block.labels, dtype=np.int64), dtype=torch.long, device=device)
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        adapter.train()
        losses = []
        batches = subject_balanced_batches(block.subjects, block.labels, ADAPTER_BATCH_SIZE, stable_seed("cde-batches", run_name, dataset, fold, epoch, "GEO"))
        for positions in batches:
            pos = torch.as_tensor(positions, dtype=torch.long, device=device)
            z_batch = block.z[pos]
            base_logits = block.logits[pos].detach()
            z_counterfactual, logits = adapter(z_batch)
            p_base = F.softmax(base_logits, dim=1).detach()
            ce = F.cross_entropy(logits, labels_t[pos])
            coral = class_conditional_subject_coral(z_counterfactual, block.labels[positions], block.subjects[positions])
            kl = F.kl_div(F.log_softmax(logits, dim=1), p_base, reduction="batchmean")
            loss = ce + LAMBDA_CC_CORAL * coral + LAMBDA_KL * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[cde-GEO] {run_name} {dataset} fold={fold} epoch={epoch}/{ADAPTER_EPOCHS} loss={np.mean(losses):.5f}", flush=True)
    adapter.eval()
    return adapter, {"coral": "class_conditional_subject_coral", "lambda": LAMBDA_CC_CORAL}


def adapter_logits(adapter: ResidualAdapter, block: FeatureBlock) -> np.ndarray:
    with torch.inference_mode():
        _, logits = adapter(block.z)
    return logits.detach().float().cpu().numpy()


def fusion_logits(base: np.ndarray, inv: np.ndarray, geo: np.ndarray, alpha_inv: float, alpha_geo: float) -> np.ndarray:
    return base + float(alpha_inv) * (inv - base) + float(alpha_geo) * (geo - base)


def candidate_grid() -> list[tuple[float, float]]:
    return [(ai, ag) for ai in ALPHA_VALUES for ag in ALPHA_VALUES if ai + ag <= ALPHA_SUM_MAX + 1e-12]


def choose_fusion(dataset: str, fold: int, labels: np.ndarray, subjects: np.ndarray, base_logits: np.ndarray, inv_logits: np.ndarray, geo_logits: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    base_p = probabilities_from_logits(base_logits)
    inv_p = probabilities_from_logits(inv_logits)
    geo_p = probabilities_from_logits(geo_logits)
    base_metrics = metric_means(labels, base_p, subjects)
    inv_metrics = metric_means(labels, inv_p, subjects)
    geo_metrics = metric_means(labels, geo_p, subjects)
    inv_incompetent = inv_metrics["BA"] < base_metrics["BA"] - COMPETENCE_TOLERANCE
    geo_incompetent = geo_metrics["BA"] < base_metrics["BA"] - COMPETENCE_TOLERANCE
    base_ce = subject_ce(labels, base_p, subjects)
    rows: list[dict[str, Any]] = []
    for alpha_inv, alpha_geo in candidate_grid():
        candidate = fusion_logits(base_logits, inv_logits, geo_logits, alpha_inv, alpha_geo)
        p = probabilities_from_logits(candidate)
        ce = subject_ce(labels, p, subjects)
        harm = np.asarray([max(ce[s] - base_ce[s], 0.0) for s in subject_sort(ce)], dtype=np.float64)
        cvar_count = max(1, int(math.ceil(0.25 * len(harm))))
        cvar_harm = float(np.sort(harm)[-cvar_count:].mean())
        mean_ce = float(np.mean(list(ce.values())))
        robust_loss = mean_ce + cvar_harm
        metric = metric_means(labels, p, subjects)
        eligible = not ((inv_incompetent and alpha_inv > 0) or (geo_incompetent and alpha_geo > 0))
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "alpha_inv": alpha_inv,
            "alpha_geo": alpha_geo,
            "mean_subject_BA": metric["BA"],
            "mean_CE": mean_ce,
            "CVaR25_harm": cvar_harm,
            "ROBUST_LOSS": robust_loss,
            "SE_subject": float(np.std(list(ce.values()), ddof=1) / math.sqrt(len(ce))) if len(ce) > 1 else 0.0,
            "eligible_after_competence_gate": eligible,
            "selected": False,
            "inv_incompetent": inv_incompetent,
            "geo_incompetent": geo_incompetent,
        })
    table = pd.DataFrame(rows)
    eligible_table = table[table.eligible_after_competence_gate].copy()
    if eligible_table.empty:
        selected_alpha = (0.0, 0.0)
        best_loss = float("nan")
        best_se = 0.0
        threshold = float("nan")
    else:
        best_idx = eligible_table.ROBUST_LOSS.astype(float).idxmin()
        best_row = table.loc[best_idx]
        best_loss = float(best_row.ROBUST_LOSS)
        best_se = float(best_row.SE_subject)
        threshold = best_loss + best_se
        one_se = eligible_table[eligible_table.ROBUST_LOSS <= threshold + 1e-12].copy()
        one_se["alpha_sum"] = one_se.alpha_inv + one_se.alpha_geo
        one_se = one_se.sort_values(["alpha_sum", "alpha_inv", "alpha_geo", "ROBUST_LOSS"], kind="stable")
        selected_row = one_se.iloc[0]
        selected_alpha = (float(selected_row.alpha_inv), float(selected_row.alpha_geo))
        selected_table_index = selected_row.name
        table.loc[selected_table_index, "selected"] = True
    if eligible_table.empty:
        table.loc[(table.alpha_inv == 0.0) & (table.alpha_geo == 0.0), "selected"] = True
    selected_row = table[table.selected].iloc[0]
    selection = {
        "dataset": dataset,
        "fold": fold,
        "alpha_inv": selected_alpha[0],
        "alpha_geo": selected_alpha[1],
        "baseline_discovery_BA": base_metrics["BA"],
        "inv_discovery_BA": inv_metrics["BA"],
        "geo_discovery_BA": geo_metrics["BA"],
        "inv_incompetent": bool(inv_incompetent),
        "geo_incompetent": bool(geo_incompetent),
        "best_robust_loss": best_loss,
        "SE_subject_best_candidate": best_se,
        "one_SE_threshold": threshold,
        "selected_robust_loss": float(selected_row.ROBUST_LOSS),
        "selected_mean_CE": float(selected_row.mean_CE),
        "selected_mean_subject_BA": float(selected_row.mean_subject_BA),
    }
    table["best_robust_loss"] = best_loss
    table["SE_subject_best_candidate"] = best_se
    table["one_SE_threshold"] = threshold
    table["baseline_discovery_BA"] = base_metrics["BA"]
    table["inv_discovery_BA"] = inv_metrics["BA"]
    table["geo_discovery_BA"] = geo_metrics["BA"]
    return selection, table, {"base": base_metrics, "inv": inv_metrics, "geo": geo_metrics}


def load_checkpoint(ctx: FoldContext, device: torch.device) -> tuple[canonical.VanillaEEGNet, np.ndarray, np.ndarray, Path]:
    path = canonical.RUNTIME / "checkpoints" / ctx.dataset / f"fold-{ctx.fold}" / "seed-0.pt"
    if not path.is_file():
        raise RuntimeError(f"missing canonical seed0 refit checkpoint: {path}")
    partial_path = canonical.RUNTIME / "partial" / f"{ctx.dataset.lower()}_fold-{ctx.fold}_seed-0.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8-sig"))
    state = torch.load(path, map_location=device)
    if partial.get("checkpoint_sha256") and partial["checkpoint_sha256"] != sha256_file(path):
        raise RuntimeError(f"canonical checkpoint hash mismatch: {path}")
    model = canonical.VanillaEEGNet(ctx.data.batch(ctx.outcome_idx[:1]).shape[1]).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    mean = np.asarray(state["normalizer_mean"], dtype=np.float32)
    std = np.asarray(state["normalizer_std"], dtype=np.float32)
    return model, mean, std, path


def checkpoint_equivalence(ctx: FoldContext, model: canonical.VanillaEEGNet, mean: np.ndarray, std: np.ndarray, canonical_predictions: pd.DataFrame, device: torch.device) -> tuple[FeatureBlock, dict[str, Any]]:
    # This is the mandated equivalence check.  It reads no outcome metric.
    block = materialize_features(ctx.data, model, ctx.outcome_idx, mean, std, device)
    expected = canonical_predictions[(canonical_predictions.dataset == ctx.dataset) & (canonical_predictions.seed.astype(str) == "0") & (canonical_predictions.fold == ctx.fold)].copy()
    expected["trial_uid"] = expected.trial_uid.astype(str)
    actual_uid = list(map(str, block.trial_uids))
    if set(actual_uid) != set(expected.trial_uid):
        raise RuntimeError(f"CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL: trial_uid mismatch {ctx.dataset} fold={ctx.fold}")
    expected = expected.set_index("trial_uid").loc[actual_uid]
    labels = expected.label.to_numpy(np.int64)
    if not np.array_equal(labels, block.labels):
        raise RuntimeError(f"CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL: label mismatch {ctx.dataset} fold={ctx.fold}")
    actual_probability = softmax_np(block.logits.detach().float().cpu().numpy())
    expected_probability = expected[["probability_class0", "probability_class1"]].to_numpy(np.float64)
    max_diff = float(np.max(np.abs(actual_probability - expected_probability)))
    actual_prediction = (actual_probability[:, 1] >= actual_probability[:, 0]).astype(np.int64)
    expected_prediction = expected.prediction.to_numpy(np.int64)
    if max_diff > 1e-5 or not np.array_equal(actual_prediction, expected_prediction):
        raise RuntimeError(f"CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL: probability/prediction mismatch {ctx.dataset} fold={ctx.fold} max_diff={max_diff}")
    return block, {"dataset": ctx.dataset, "fold": ctx.fold, "trial_count": len(block.indices), "max_probability_abs_diff": max_diff, "trial_uid_exact": True, "labels_exact": True, "predictions_exact": True, "pass": True}


def bootstrap_paired(delta: np.ndarray, dataset: str) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(stable_seed("cde-paired-bootstrap", dataset, SEED))
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return {"dataset": dataset, "n_subjects": int(len(values)), "mean_delta_BA": float(values.mean()), "mean_delta_pp": float(100.0 * values.mean()), "median_delta_BA": float(np.median(values)), "median_delta_pp": float(100.0 * np.median(values)), "positive_subject_fraction": float(np.mean(values > 0)), "nonnegative_subject_fraction": float(np.mean(values >= 0)), "paired_bootstrap_CI95_L": float(np.quantile(draws, 0.025)), "paired_bootstrap_CI95_U": float(np.quantile(draws, 0.975)), "paired_bootstrap_CI95_L_pp": float(100.0 * np.quantile(draws, 0.025)), "paired_bootstrap_CI95_U_pp": float(100.0 * np.quantile(draws, 0.975)), "bootstrap_draws": BOOTSTRAP_DRAWS}


def disagreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) != np.asarray(b)))


def rescue_corruption(base_pred: np.ndarray, aux_pred: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    base_pred = np.asarray(base_pred); aux_pred = np.asarray(aux_pred); labels = np.asarray(labels)
    wrong = base_pred != labels
    correct = ~wrong
    rescue = float(np.mean(aux_pred[wrong] == labels[wrong])) if np.any(wrong) else 0.0
    corruption = float(np.mean(aux_pred[correct] != labels[correct])) if np.any(correct) else 0.0
    return rescue, corruption


def run_fold(ctx: FoldContext, canonical_predictions: pd.DataFrame, selection_rows: list[pd.DataFrame], subject_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]], diagnostics_rows: list[dict[str, Any]], equivalence_rows: list[dict[str, Any]], bootstrap_subjects: dict[str, list[dict[str, Any]]], device: torch.device) -> None:
    print(f"[cde] starting {ctx.dataset} fold={ctx.fold} best_epoch={ctx.seed0_best_epoch}", flush=True)
    fit_mean, fit_std = canonical.compute_normalizer(ctx.data, ctx.initial_idx)
    development_backbone = fit_model_fit_only(ctx, fit_mean, fit_std, device)
    fit_block = materialize_features(ctx.data, development_backbone, ctx.initial_idx, fit_mean, fit_std, device)
    discovery_block = materialize_features(ctx.data, development_backbone, ctx.discovery_idx, fit_mean, fit_std, device)
    inv_dev, _, _ = train_inv(fit_block, development_backbone.head, "development", ctx.dataset, ctx.fold, device)
    geo_dev, _ = train_geo(fit_block, development_backbone.head, "development", ctx.dataset, ctx.fold, device)
    inv_dev_logits = adapter_logits(inv_dev, discovery_block)
    geo_dev_logits = adapter_logits(geo_dev, discovery_block)
    base_dev_logits = discovery_block.logits.detach().float().cpu().numpy()
    selection, selection_table, _ = choose_fusion(ctx.dataset, ctx.fold, discovery_block.labels, discovery_block.subjects, base_dev_logits, inv_dev_logits, geo_dev_logits)
    selection_rows.append(selection_table)
    print(f"[cde-selection] {ctx.dataset} fold={ctx.fold} alpha_inv={selection['alpha_inv']:.2f} alpha_geo={selection['alpha_geo']:.2f} inv_incompetent={selection['inv_incompetent']} geo_incompetent={selection['geo_incompetent']}", flush=True)
    del inv_dev, geo_dev, development_backbone, fit_block, discovery_block
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Equivalence is done before any final adapter training and does not score
    # outcome BA.  Its FeatureBlock is reused for the final outcome pass.
    canonical_model, refit_mean, refit_std, checkpoint_path = load_checkpoint(ctx, device)
    outcome_block, equivalence = checkpoint_equivalence(ctx, canonical_model, refit_mean, refit_std, canonical_predictions, device)
    equivalence["checkpoint_path"] = str(checkpoint_path)
    equivalence["checkpoint_sha256"] = sha256_file(checkpoint_path)
    equivalence_rows.append(equivalence)
    source_block = materialize_features(ctx.data, canonical_model, ctx.refit_idx, refit_mean, refit_std, device)
    inv_final, _, _ = train_inv(source_block, canonical_model.head, "final", ctx.dataset, ctx.fold, device)
    geo_final, _ = train_geo(source_block, canonical_model.head, "final", ctx.dataset, ctx.fold, device)
    inv_logits = adapter_logits(inv_final, outcome_block)
    geo_logits = adapter_logits(geo_final, outcome_block)
    base_logits = outcome_block.logits.detach().float().cpu().numpy()
    controls = {
        "B0_CANONICAL_SEED0": (0.0, 0.0, base_logits),
        "B1_INV_ONLY_DIAGNOSTIC": (1.0, 0.0, inv_logits),
        "B2_GEO_ONLY_DIAGNOSTIC": (0.0, 1.0, geo_logits),
        "B3_EQUAL_CONSERVATIVE": (0.10, 0.10, fusion_logits(base_logits, inv_logits, geo_logits, 0.10, 0.10)),
        "B4_SELECTED_CDE": (selection["alpha_inv"], selection["alpha_geo"], fusion_logits(base_logits, inv_logits, geo_logits, selection["alpha_inv"], selection["alpha_geo"])),
    }
    labels = outcome_block.labels
    subjects = outcome_block.subjects
    base_probability = probabilities_from_logits(base_logits)
    metric_cache: dict[str, list[dict[str, Any]]] = {}
    for control, (alpha_inv, alpha_geo, logits) in controls.items():
        p1 = probabilities_from_logits(logits)
        metrics = metrics_by_subject(labels, p1, subjects)
        metric_cache[control] = metrics
        for row in metrics:
            subject_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "control": control, "subject_id": row["subject_id"], "BA": row["BA"], "accuracy": row["accuracy"], "macro_F1": row["macro_F1"], "NLL": row["NLL"], "trials": row["trials"], "alpha_inv": alpha_inv, "alpha_geo": alpha_geo})
        fold_metric = metric_means(labels, p1, subjects)
        fold_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "control": control, "mean_subject_BA": fold_metric["BA"], "mean_accuracy": fold_metric["accuracy"], "mean_macro_F1": fold_metric["macro_F1"], "mean_NLL": fold_metric["NLL"], "n_subjects": int(len(set(subjects))), "alpha_inv": alpha_inv, "alpha_geo": alpha_geo})
        for i, (uid, subject, session, label, prob) in enumerate(zip(outcome_block.trial_uids, subjects, outcome_block.sessions, labels, p1)):
            trial_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "control": control, "subject_id": str(subject), "trial_uid": str(uid), "session": canonical.session_label(ctx.dataset, int(session)), "label": int(label), "probability_class0": float(1.0 - prob), "probability_class1": float(prob), "prediction": int(prob >= 0.5)})
    b0 = {row["subject_id"]: row for row in metric_cache["B0_CANONICAL_SEED0"]}
    b4 = {row["subject_id"]: row for row in metric_cache["B4_SELECTED_CDE"]}
    deltas: list[dict[str, Any]] = []
    for subject in subject_sort(b0):
        delta = float(b4[subject]["BA"] - b0[subject]["BA"])
        row = {"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "subject_id": subject, "canonical_BA": b0[subject]["BA"], "cde_BA": b4[subject]["BA"], "delta_BA": delta, "delta_pp": 100.0 * delta, "cde_alpha_inv": selection["alpha_inv"], "cde_alpha_geo": selection["alpha_geo"]}
        delta_rows.append(row); deltas.append(row)
    bootstrap_subjects.setdefault(ctx.dataset, []).extend(deltas)
    base_pred = (base_probability >= 0.5).astype(np.int64)
    inv_pred = (probabilities_from_logits(inv_logits) >= 0.5).astype(np.int64)
    geo_pred = (probabilities_from_logits(geo_logits) >= 0.5).astype(np.int64)
    cde_prob = probabilities_from_logits(controls["B4_SELECTED_CDE"][2])
    cde_pred = (cde_prob >= 0.5).astype(np.int64)
    inv_rescue, inv_corruption = rescue_corruption(base_pred, inv_pred, labels)
    geo_rescue, geo_corruption = rescue_corruption(base_pred, geo_pred, labels)
    cde_rescue, cde_corruption = rescue_corruption(base_pred, cde_pred, labels)
    inv_metric = metric_means(labels, probabilities_from_logits(inv_logits), subjects)
    geo_metric = metric_means(labels, probabilities_from_logits(geo_logits), subjects)
    cde_metric = metric_means(labels, cde_prob, subjects)
    diagnostics_rows.append({
        "dataset": ctx.dataset,
        "fold": ctx.fold,
        "seed": SEED,
        "alpha_inv": selection["alpha_inv"],
        "alpha_geo": selection["alpha_geo"],
        "inv_incompetent": selection["inv_incompetent"],
        "geo_incompetent": selection["geo_incompetent"],
        "inv_BA": inv_metric["BA"],
        "inv_NLL": inv_metric["NLL"],
        "geo_BA": geo_metric["BA"],
        "geo_NLL": geo_metric["NLL"],
        "cde_BA": cde_metric["BA"],
        "cde_NLL": cde_metric["NLL"],
        "inv_vs_base_disagreement_rate": disagreement(base_pred, inv_pred),
        "geo_vs_base_disagreement_rate": disagreement(base_pred, geo_pred),
        "inv_vs_geo_disagreement_rate": disagreement(inv_pred, geo_pred),
        "cde_vs_base_disagreement_rate": disagreement(base_pred, cde_pred),
        "inv_rescue_rate_on_base_errors": inv_rescue,
        "geo_rescue_rate_on_base_errors": geo_rescue,
        "cde_rescue_rate_on_base_errors": cde_rescue,
        "inv_corruption_rate_on_base_correct": inv_corruption,
        "geo_corruption_rate_on_base_correct": geo_corruption,
        "cde_corruption_rate_on_base_correct": cde_corruption,
    })
    del inv_final, geo_final, source_block, outcome_block, canonical_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[cde] completed {ctx.dataset} fold={ctx.fold}", flush=True)


def classification_from_deltas(summary: dict[str, dict[str, Any]]) -> str:
    open_delta = float(summary["OpenBMI"]["delta_pp"])
    wbcic_delta = float(summary["WBCIC"]["delta_pp"])
    if open_delta > 0 and wbcic_delta > 0:
        return TERMINAL_POSITIVE
    if ((open_delta > 0 > wbcic_delta) or (wbcic_delta > 0 > open_delta)) and abs(open_delta if open_delta < 0 else wbcic_delta) < 0.5:
        return TERMINAL_MIXED
    return TERMINAL_NEGATIVE


def write_reports(canonical_audit: dict[str, Any], summary: pd.DataFrame, diagnostics: pd.DataFrame, selections: pd.DataFrame, equivalence: list[dict[str, Any]], paired: dict[str, Any], all_two_incompetent: bool, runtime_terminal: str) -> None:
    summary_map = {str(row.dataset): row.to_dict() for _, row in summary.iterrows()}
    pilot_terminal = classification_from_deltas(summary_map)
    terminal = TERMINAL_NO_COMPETENT if all_two_incompetent else pilot_terminal
    recommend = "YES" if pilot_terminal == TERMINAL_POSITIVE else "NO"
    branch_diversity = bool((diagnostics[["inv_vs_base_disagreement_rate", "geo_vs_base_disagreement_rate"]].to_numpy() > 0).any()) and not all_two_incompetent
    equal_b3 = summary.set_index("dataset")["B3_equal_conservative_BA"].to_dict()
    lines = [
        "# PERSIST-CDE SEED-0 PILOT", "", "Competence-Preserving Counterfactual Decision Ensemble (PERSIST-CDE) was evaluated once with seed=0 under the canonical outer evaluator.", "", "## Primary comparison", "", "| Dataset | Canonical seed0 | CDE seed0 | Delta pp | paired 95% CI |", "|---|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        row = summary_map[dataset]
        lines.append(f"| {dataset} | {row['canonical_seed0_BA']:.6f} | {row['cde_seed0_BA']:.6f} | {row['delta_pp']:+.3f} | [{row['paired_CI95_L_pp']:+.3f}, {row['paired_CI95_U_pp']:+.3f}] |")
    lines += ["", "## Selected alpha per fold", "", "| Dataset | Fold | alpha_inv | alpha_geo | INV discovery BA | GEO discovery BA |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in selections[selections.selected == True].sort_values(["dataset", "fold"]).iterrows():  # noqa: E712
        lines.append(f"| {row.dataset} | {int(row.fold)} | {row.alpha_inv:.2f} | {row.alpha_geo:.2f} | {row.inv_discovery_BA:.6f} | {row.geo_discovery_BA:.6f} |")
    lines += ["", "The selected-alpha table above records the selected candidate BA; branch BA and all outcome diagnostics are in `BRANCH_DIAGNOSTICS.csv`.", "", "## Outcome mechanism diagnostics", "", "| Dataset | INV BA | GEO BA | INV→base disagreement | GEO→base disagreement | INV→GEO disagreement | CDE rescue | CDE corruption |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in diagnostics.sort_values(["dataset", "fold"]).iterrows():
        lines.append(f"| {row.dataset} fold {int(row.fold)} | {row.inv_BA:.6f} | {row.geo_BA:.6f} | {row.inv_vs_base_disagreement_rate:.4f} | {row.geo_vs_base_disagreement_rate:.4f} | {row.inv_vs_geo_disagreement_rate:.4f} | {row.cde_rescue_rate_on_base_errors:.4f} | {row.cde_corruption_rate_on_base_correct:.4f} |")
    lines += ["", "## Direct answers", ""]
    q = {
        "Q1_OpenBMI_exceeds_matched_seed0": bool(summary_map["OpenBMI"]["delta_pp"] > 0),
        "Q2_WBCIC_exceeds_matched_seed0": bool(summary_map["WBCIC"]["delta_pp"] > 0),
        "Q3_deltas_same_direction": bool((summary_map["OpenBMI"]["delta_pp"] > 0 and summary_map["WBCIC"]["delta_pp"] > 0) or (summary_map["OpenBMI"]["delta_pp"] < 0 and summary_map["WBCIC"]["delta_pp"] < 0)),
        "Q4_competent_auxiliary_decision_diversity": branch_diversity,
        "Q5_improvement_over_equal_conservative_fusion": bool(any(summary_map[d]["cde_seed0_BA"] > float(equal_b3[d]) for d in DATASETS)),
        "Q6_outcome_driven_model_selection": False,
        "Q7_sealed_outer_untouched": True,
        "Q8_recommend_run_seed1_seed2": recommend == "YES",
    }
    q_text = [
        f"1. OpenBMI exceeds matched canonical seed0: {'YES' if q['Q1_OpenBMI_exceeds_matched_seed0'] else 'NO'}.",
        f"2. WBCIC exceeds matched canonical seed0: {'YES' if q['Q2_WBCIC_exceeds_matched_seed0'] else 'NO'}.",
        f"3. The two deltas have the same strict direction: {'YES' if q['Q3_deltas_same_direction'] else 'NO'}.",
        f"4. Competent auxiliary decision diversity exists: {'YES' if q['Q4_competent_auxiliary_decision_diversity'] else 'NO'}.",
        f"5. CDE improves over equal conservative fusion on at least one dataset: {'YES' if q['Q5_improvement_over_equal_conservative_fusion'] else 'NO'}.",
        "6. Any outcome-driven model selection: NO.",
        "7. Sealed outer cohorts untouched: YES.",
        f"8. Run seed1/seed2: {recommend} (predeclared rule: only a two-dataset positive seed-0 signal earns YES).",
    ]
    lines += q_text + ["", f"pilot_classification = {pilot_terminal}", f"method_terminal = {terminal}", "", "The seed-0 pilot is not a final multi-seed paper claim. No seed1/seed2 run was started automatically.", "", f"terminal = {terminal}"]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "title": "PERSIST-CDE SEED-0 PILOT",
        "terminal": terminal,
        "pilot_classification": pilot_terminal,
        "recommend_run_seed1_seed2": recommend,
        "summary": clean(summary.to_dict(orient="records")),
        "selected_alpha_per_fold": clean(selections[selections.selected == True].sort_values(["dataset", "fold"]).to_dict(orient="records")),  # noqa: E712
        "branch_diagnostics": clean(diagnostics.to_dict(orient="records")),
        "paired_bootstrap": paired,
        "checkpoint_equivalence": equivalence,
        "canonical_baseline_audit": canonical_audit,
        "outer_status": {"WBCIC_outer_10_accessed": False, "OpenBMI_sealed_holdout_accessed": False},
        "runtime_terminal": runtime_terminal,
        "questions": q,
    }
    write_json(EXP / "FINAL_REPORT.json", report)


def write_legality(contexts: list[FoldContext]) -> None:
    lines = ["# DATA LEGALITY AUDIT", "", "## PERSIST-CDE SEED-0 PILOT", "", "PASS: the pilot uses only the frozen canonical roles and authorized development subjects.", "", "- OpenBMI: 54 Stage-0-frozen subjects, S1+S2 source and S2 outcome, five frozen folds.", "- WBCIC: 41 subjects from `DEVELOPMENT_SCOPE_LOCK.json`, S1+S2 source and S3 outcome, five frozen folds.", "- `outer_subject_ids_present=false` was asserted from the WBCIC scope lock; the sealed outer 10 were not enumerated or opened.", "- No outcome subject history was used for adaptation, normalization, epoch selection or fusion selection.", "- The only pre-final outcome access was the mandated checkpoint-equivalence comparison of IDs, labels and probabilities; no outcome metric entered selection.", "", "Observed fold role counts:", ""]
    for ctx in contexts:
        lines.append(f"- {ctx.dataset} fold {ctx.fold}: model_fit={len(ctx.roles['model_fit'])}, discovery={len(ctx.roles['discovery'])}, outcome={len(ctx.roles['outcome'])}; initial_rows={len(ctx.initial_idx)}, discovery_rows={len(ctx.discovery_idx)}, refit_rows={len(ctx.refit_idx)}, outcome_rows={len(ctx.outcome_idx)}")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if tuple(args.datasets) != DATASETS:
        # The predeclared pilot is exactly OpenBMI + WBCIC.  A partial run is
        # not a scientific result and is refused rather than silently emitted.
        raise RuntimeError("PERSIST-CDE pilot requires exactly --datasets OpenBMI WBCIC")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    for path in (RESULTS, PROTOCOL, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)
    if not (EXP / "METHOD.md").is_file() or not (EXP / "PROTOCOL_LOCK.json").is_file():
        raise RuntimeError("predeclared METHOD.md and PROTOCOL_LOCK.json must exist before training")
    canonical_predictions, canonical_audit = load_baseline_seed0_audit()
    contexts = [make_context(dataset, fold) for dataset in DATASETS for fold in FOLDS]
    write_legality(contexts)
    selection_tables: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    bootstrap_subjects: dict[str, list[dict[str, Any]]] = {}
    for ctx in contexts:
        run_fold(ctx, canonical_predictions, selection_tables, subject_rows, delta_rows, trial_rows, fold_rows, diagnostics_rows, equivalence_rows, bootstrap_subjects, device)
    selection_frame = pd.concat(selection_tables, ignore_index=True)
    subject_frame = pd.DataFrame(subject_rows)
    delta_frame = pd.DataFrame(delta_rows)
    trial_frame = pd.DataFrame(trial_rows)
    fold_frame = pd.DataFrame(fold_rows)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    write_csv(RESULTS / "FUSION_SELECTION.csv", selection_frame.sort_values(["dataset", "fold", "alpha_inv", "alpha_geo"]))
    write_csv(RESULTS / "PER_SUBJECT_RESULTS.csv", subject_frame.sort_values(["dataset", "fold", "control", "subject_id"]))
    write_csv(RESULTS / "PER_SUBJECT_DELTA.csv", delta_frame.sort_values(["dataset", "fold", "subject_id"]))
    write_csv(RESULTS / "TRIAL_PREDICTIONS.csv", trial_frame.sort_values(["dataset", "fold", "control", "subject_id", "trial_uid"]))
    write_csv(RESULTS / "PER_FOLD_RESULTS.csv", fold_frame.sort_values(["dataset", "fold", "control"]))
    write_csv(RESULTS / "BRANCH_DIAGNOSTICS.csv", diagnostics_frame.sort_values(["dataset", "fold"]))
    paired: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        ds_delta = pd.DataFrame(bootstrap_subjects[dataset])
        paired[dataset] = bootstrap_paired(ds_delta.delta_BA.to_numpy(float), dataset)
        # Aggregate over biological subjects, not by unweighted fold means.
        # The frozen canonical seed-0 reference uses this same subject-level
        # aggregation and folds have unequal subject counts.
        ds_subject = subject_frame[subject_frame.dataset == dataset]
        def ba(control: str) -> float:
            return float(ds_subject[ds_subject.control == control].BA.mean())
        canonical_ba = ba("B0_CANONICAL_SEED0")
        cde_ba = ba("B4_SELECTED_CDE")
        b3_ba = ba("B3_EQUAL_CONSERVATIVE")
        summary_rows.append({"dataset": dataset, "seed": SEED, "canonical_seed0_BA": canonical_ba, "cde_seed0_BA": cde_ba, "B3_equal_conservative_BA": b3_ba, "delta_BA": cde_ba - canonical_ba, "delta_pp": 100.0 * (cde_ba - canonical_ba), "paired_CI95_L": paired[dataset]["paired_bootstrap_CI95_L"], "paired_CI95_U": paired[dataset]["paired_bootstrap_CI95_U"], "paired_CI95_L_pp": paired[dataset]["paired_bootstrap_CI95_L_pp"], "paired_CI95_U_pp": paired[dataset]["paired_bootstrap_CI95_U_pp"], "median_delta_pp": paired[dataset]["median_delta_pp"], "positive_subject_fraction": paired[dataset]["positive_subject_fraction"], "nonnegative_subject_fraction": paired[dataset]["nonnegative_subject_fraction"], "INV_BA": float(diagnostics_frame[diagnostics_frame.dataset == dataset].inv_BA.mean()), "GEO_BA": float(diagnostics_frame[diagnostics_frame.dataset == dataset].geo_BA.mean()), "selected_alpha_inv_mean": float(diagnostics_frame[diagnostics_frame.dataset == dataset].alpha_inv.mean()), "selected_alpha_geo_mean": float(diagnostics_frame[diagnostics_frame.dataset == dataset].alpha_geo.mean())})
    summary_frame = pd.DataFrame(summary_rows)
    write_csv(RESULTS / "PILOT_SUMMARY.csv", summary_frame)
    write_json(RESULTS / "PAIRED_BOOTSTRAP.json", paired)
    write_json(PROTOCOL / "CHECKPOINT_EQUIVALENCE.json", {"pass": True, "rows": equivalence_rows})
    eq_lines = ["# CHECKPOINT EQUIVALENCE", "", "The canonical seed-0 refit checkpoint was loaded without modification. Before final adapter training, outcome trial IDs, labels, predictions and probabilities were compared to the canonical seed-0 trial table; no outcome metric was computed or used for selection.", "", "| Dataset | Fold | Trials | Max probability abs diff | IDs | Labels | Predictions |", "|---|---:|---:|---:|---|---|---|"]
    for row in equivalence_rows:
        eq_lines.append(f"| {row['dataset']} | {row['fold']} | {row['trial_count']} | {row['max_probability_abs_diff']:.3e} | PASS | PASS | PASS |")
    eq_lines += ["", "checkpoint_equivalence = PASS", "terminal on failure would have been `CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL`."]
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("\n".join(eq_lines) + "\n", encoding="utf-8")
    all_two_incompetent = bool(selection_frame.groupby(["dataset", "fold"])[["inv_incompetent", "geo_incompetent"]].first().all(axis=1).all())
    runtime_terminal = TERMINAL_POSITIVE if all(float(x) > 0 for x in summary_frame.delta_pp) else TERMINAL_NEGATIVE
    write_reports(canonical_audit, summary_frame, diagnostics_frame, selection_frame, equivalence_rows, paired, all_two_incompetent, runtime_terminal)
    write_json(RUNTIME / "PILOT_RUN.exit.json", {"complete": True, "exit_code": 0, "terminal": classification_from_deltas({r.dataset: r.to_dict() for _, r in summary_frame.iterrows()})})
    print("branch = codex/persist-eeg-cde-seed0-pilot", flush=True)
    print("commit SHA = recorded after artifact commit", flush=True)
    print(f"terminal = {TERMINAL_NO_COMPETENT if all_two_incompetent else classification_from_deltas({r.dataset: r.to_dict() for _, r in summary_frame.iterrows()})}", flush=True)
    for _, row in summary_frame.iterrows():
        ds = row.dataset
        selected = selection_frame[(selection_frame.dataset == ds) & (selection_frame.selected == True)].sort_values("fold")  # noqa: E712
        print(f"{ds}: canonical seed0 BA={row.canonical_seed0_BA:.9f} CDE seed0 BA={row.cde_seed0_BA:.9f} delta_pp={row.delta_pp:+.4f} paired_CI=[{row.paired_CI95_L_pp:+.4f},{row.paired_CI95_U_pp:+.4f}] INV BA={row.INV_BA:.9f} GEO BA={row.GEO_BA:.9f} selected_alpha_per_fold=" + ",".join(f"({x.alpha_inv:.2f},{x.alpha_geo:.2f})" for _, x in selected), flush=True)
    print("sealed status = WBCIC_outer_10_untouched; OpenBMI_sealed_holdout_untouched", flush=True)
    print("checkpoint equivalence status = PASS", flush=True)
    print(f"recommend_run_seed1_seed2 = {'YES' if classification_from_deltas({r.dataset: r.to_dict() for _, r in summary_frame.iterrows()}) == TERMINAL_POSITIVE else 'NO'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        if "CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL" in str(exc):
            print("terminal = CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL", flush=True)
        elif "CDE_PILOT_BASELINE_MISMATCH" in str(exc):
            print("terminal = CDE_PILOT_BASELINE_MISMATCH", flush=True)
        raise
