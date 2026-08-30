"""Core implementation of PERSIST-RE.

The source experiment operates on the verified clean-room ATCNet feature
archives produced by the frozen specialist protocol.  Keeping the expensive
EEG encoder frozen makes the block-coordinate and gradient-quarantine
semantics directly testable and leaves no runtime data in the repository.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


# Resolve the repository from the checked-out experiment rather than baking in
# the server's drive letter.  ``PERSIST_RE_REPO`` remains an explicit override
# for server jobs whose representations live in a separate checkout.
_DEFAULT_REPO = Path(__file__).resolve().parents[3]
REPO = Path(os.environ.get("PERSIST_RE_REPO", str(_DEFAULT_REPO))).resolve()
EXP = REPO / "experiments" / "persist_eeg_persist_re_final"
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
OLD_REP_ROOT = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "runtime" / "specialist_representations"
STAGE1_REP_ROOT = REPO / "experiments" / "persist_eeg_scst_utility_stage1" / "runtime" / "representations"

DATASETS = ("OpenBMI", "WBCIC")
FOLDS = tuple(range(5))
SEEDS = tuple(range(3))
METHODS = ("ERM", "SubjectBalancedERM", "Mixup", "GroupDRO", "ProspectiveOnly", "RandomEffectOnly", "AdversarialMixed", "PERSIST-RE")
SEARCH_RANKS = (1, 2, 4)
SEARCH_LAMBDA_R = (1e-3, 1e-2)
SEARCH_LAMBDA_P = (0.5, 1.0)
# Thirty epochs matches the frozen specialist budget and gives the low-rate
# final feature block enough updates to move off the identity initialization.
EPOCHS = 30
FIT_VERSION = "v2_epoch30"
LR_HEAD = 1e-4
LR_FEATURE = 1e-5
LR_RE = 1e-3
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 3.0
GAMMA_A = 1.0
LAMBDA_Q = 1.0


def clean(value):
    if isinstance(value, Mapping):
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
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def load_rep(dataset: str, fold: int, seed: int, role: str, backbone: str = "ATCNet") -> dict[str, np.ndarray]:
    candidates = [OLD_REP_ROOT / backbone, STAGE1_REP_ROOT / backbone]
    path = next((root / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz" for root in candidates if (root / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz").is_file()), None)
    if path is None:
        raise FileNotFoundError(f"authorized source representation missing for backbone={backbone}, dataset={dataset}, fold={fold}, seed={seed}")
    with np.load(path, allow_pickle=True) as z:
        out = {key: z[key] for key in z.files}
    out["features"] = np.asarray(out["features"], np.float32)
    out["labels"] = np.asarray(out["labels"], np.int64)
    out["subjects"] = np.asarray(out["subjects"]).astype("U")
    out["sessions"] = np.asarray(out["sessions"], np.int64)
    out["indices"] = np.asarray(out["indices"], np.int64)
    if out["features"].ndim != 2 or out["features"].shape[1] < 1:
        raise RuntimeError(f"unexpected clean-room representation shape: {path} {out['features'].shape}")
    if len(np.unique(out["indices"])) != len(out["indices"]):
        raise RuntimeError(f"duplicate source indices in {path}")
    if not np.isfinite(out["features"]).all():
        raise RuntimeError(f"non-finite source features in {path}")
    return out


@dataclass
class FoldData:
    model_fit: dict[str, np.ndarray]
    validation: dict[str, np.ndarray]
    outcome: dict[str, np.ndarray]

    @property
    def subjects(self) -> list[str]:
        return subject_sort(np.unique(self.model_fit["subjects"]))

    @property
    def dimension(self) -> int:
        return int(self.model_fit["features"].shape[1])


def load_fold(dataset: str, fold: int, seed: int, backbone: str = "ATCNet") -> FoldData:
    return FoldData(*(load_rep(dataset, fold, seed, role, backbone) for role in ("model_fit", "validation", "outcome")))


def concat_rep(*reps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = ("indices", "features", "labels", "subjects", "sessions")
    out = {key: np.concatenate([rep[key] for rep in reps]) for key in keys}
    if len(np.unique(out["indices"])) != len(out["indices"]):
        raise RuntimeError("source partitions overlap")
    return out


def subject_index(subjects: Iterable[object]) -> tuple[list[str], dict[str, int]]:
    ordered = subject_sort(np.unique(np.asarray(list(subjects)).astype(str)))
    return ordered, {s: i for i, s in enumerate(ordered)}


def subject_ids(rep: dict[str, np.ndarray], mapping: dict[str, int]) -> np.ndarray:
    return np.asarray([mapping[str(s)] for s in rep["subjects"]], dtype=np.int64)


def partition_subjects(subjects: Iterable[object], epoch: int, seed: int, pseudo_fraction: float = 0.25) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic subject-level episode split with rotating pseudo-unseen roles."""
    ordered = subject_sort(subjects)
    if len(ordered) < 2:
        raise ValueError("at least two subjects are required")
    # Hash ordering plus a cyclic offset makes every subject pseudo-unseen over
    # a complete schedule while remaining independent of trial order.
    hashed = sorted(ordered, key=lambda s: (stable_seed("episode", seed, s), s))
    shift = (int(epoch) * max(1, round(len(ordered) * pseudo_fraction))) % len(ordered)
    rotated = hashed[shift:] + hashed[:shift]
    n_pseudo = max(1, min(len(ordered) - 1, round(len(ordered) * pseudo_fraction)))
    pseudo = tuple(subject_sort(rotated[:n_pseudo]))
    context = tuple(subject_sort(rotated[n_pseudo:]))
    return context, pseudo


def episode_schedule(subjects: Iterable[object], epochs: int, seed: int) -> dict[str, int]:
    counts = {s: 0 for s in subject_sort(subjects)}
    for epoch in range(int(epochs)):
        _, pseudo = partition_subjects(subjects, epoch, seed)
        for s in pseudo:
            counts[s] += 1
    return counts


def per_subject_class_ce(logits: torch.Tensor, labels: torch.Tensor, subjects: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Class-balanced loss inside each subject, then subject-balanced mean."""
    losses: list[torch.Tensor] = []
    by_subject: dict[int, torch.Tensor] = {}
    for sid in torch.unique(subjects, sorted=True):
        mask_s = subjects == sid
        class_losses = []
        for cls in torch.unique(labels[mask_s], sorted=True):
            class_losses.append(F.cross_entropy(logits[mask_s & (labels == cls)], labels[mask_s & (labels == cls)]))
        if not class_losses:
            continue
        value = torch.stack(class_losses).mean()
        losses.append(value)
        by_subject[int(sid.detach().cpu())] = value
    if not losses:
        raise ValueError("empty subject loss")
    return torch.stack(losses).mean(), by_subject


def subject_loss_vector(logits: torch.Tensor, labels: torch.Tensor, subjects: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vals = []
    ids = []
    for sid in torch.unique(subjects, sorted=True):
        mask = subjects == sid
        vals.append(F.cross_entropy(logits[mask], labels[mask]))
        ids.append(sid)
    return torch.stack(vals), torch.stack(ids)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coefficient * grad, None


def grad_reverse(x: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, coefficient)


class PERSISTRE(nn.Module):
    """Population predictor plus a centered, low-rank decision random effect."""

    def __init__(self, dimension: int, n_subjects: int, rank: int, classes: int = 2):
        super().__init__()
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.classes = int(classes)
        self.feature_block = nn.Linear(dimension, dimension)
        with torch.no_grad():
            self.feature_block.weight.copy_(torch.eye(dimension))
            self.feature_block.bias.zero_()
        self.layer_norm = nn.LayerNorm(dimension)
        self.population_head = nn.Linear(dimension, classes)
        self.U = nn.Parameter(torch.randn(dimension, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(classes, rank) * 0.02)
        self.subject_embedding = nn.Parameter(torch.randn(n_subjects, rank) * 0.02)
        self.subject_intercept = nn.Parameter(torch.randn(n_subjects, classes) * 0.02)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.feature_block(features))

    def centered_effects(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.subject_embedding - self.subject_embedding.mean(0, keepdim=True), self.subject_intercept - self.subject_intercept.mean(0, keepdim=True)

    def random_effect(self, z: torch.Tensor, subjects: torch.Tensor, use_random_effect: bool = True) -> torch.Tensor:
        if not use_random_effect:
            return torch.zeros((z.shape[0], self.classes), dtype=z.dtype, device=z.device)
        e, a = self.centered_effects()
        # z is detached exactly at the branch input: no representation gradient
        # can flow through U^T z into feature_block or layer_norm.
        projected = torch.matmul(z.detach(), self.U)
        slope = projected * e[subjects]
        return torch.matmul(slope, self.B.t()) + a[subjects]

    def forward(self, features: torch.Tensor, subjects: torch.Tensor | None = None, use_random_effect: bool = False):
        z = self.encode(features)
        population = self.population_head(z)
        if subjects is None or not use_random_effect:
            effect = torch.zeros_like(population)
        else:
            effect = self.random_effect(z, subjects, True)
        return population, effect, z


def initialize_population_head(model: PERSISTRE, features: np.ndarray, labels: np.ndarray, device: torch.device) -> None:
    """Deterministic ridge initialization shared by every matched method."""
    with torch.no_grad():
        x = model.encode(torch.as_tensor(features, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float64)
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    target = np.eye(model.classes, dtype=np.float64)[np.asarray(labels, np.int64)]
    alpha = 1e-2
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    beta = np.linalg.solve(gram, design.T @ target)
    with torch.no_grad():
        model.population_head.weight.copy_(torch.as_tensor(beta[:-1].T, dtype=torch.float32, device=device))
        model.population_head.bias.copy_(torch.as_tensor(beta[-1], dtype=torch.float32, device=device))


def optimizer_groups(model: PERSISTRE) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    shared = torch.optim.AdamW([
        {"params": list(model.feature_block.parameters()) + list(model.layer_norm.parameters()), "lr": LR_FEATURE},
        {"params": model.population_head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    random_effect = torch.optim.AdamW([model.U, model.B, model.subject_embedding, model.subject_intercept], lr=LR_RE, weight_decay=WEIGHT_DECAY)
    return shared, random_effect


def set_requires(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(value)


def regularizer(model: PERSISTRE) -> torch.Tensor:
    return (model.subject_embedding.square().mean() + GAMMA_A * model.subject_intercept.square().mean())


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def fit_model(method: str, train_rep: dict[str, np.ndarray], rank: int, lambda_r: float, lambda_p: float, seed: int, epochs: int = EPOCHS, device: torch.device | None = None) -> tuple[PERSISTRE, dict[str, object]]:
    if method not in METHODS:
        raise KeyError(method)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects, mapping = subject_index(train_rep["subjects"])
    sid = subject_ids(train_rep, mapping)
    features = torch.as_tensor(train_rep["features"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(train_rep["labels"], dtype=torch.long, device=device)
    subject_tensor = torch.as_tensor(sid, dtype=torch.long, device=device)
    set_seed(stable_seed("persist-re-fit", method, rank, lambda_r, lambda_p, seed, tuple(subjects)))
    model = PERSISTRE(features.shape[1], len(subjects), rank).to(device)
    initialize_population_head(model, train_rep["features"], train_rep["labels"], device)
    shared_opt, re_opt = optimizer_groups(model)
    adversary = nn.Linear(features.shape[1], len(subjects)).to(device) if method == "AdversarialMixed" else None
    adv_opt = torch.optim.AdamW(adversary.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY) if adversary is not None else None
    rng = np.random.default_rng(stable_seed("persist-re-mixup", method, rank, lambda_r, lambda_p, seed))
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        context, pseudo = partition_subjects(subjects, epoch, stable_seed("partition", seed))
        context_ids = torch.as_tensor([mapping[s] for s in context], dtype=torch.long, device=device)
        pseudo_ids = torch.as_tensor([mapping[s] for s in pseudo], dtype=torch.long, device=device)
        context_mask = torch.isin(subject_tensor, context_ids)
        pseudo_mask = torch.isin(subject_tensor, pseudo_ids)
        # Random-effect block-coordinate step.  Population logits and z are
        # detached; only U, B, e, and a are permitted to update.
        re_loss_value = 0.0
        if method in {"PERSIST-RE", "RandomEffectOnly", "AdversarialMixed"}:
            set_requires(model.feature_block, False); set_requires(model.layer_norm, False); set_requires(model.population_head, False)
            for p in (model.U, model.B, model.subject_embedding, model.subject_intercept): p.requires_grad_(True)
            re_opt.zero_grad(set_to_none=True)
            p_det, _, z_det = model(features[context_mask], None, False)
            re = model.random_effect(z_det.detach(), subject_tensor[context_mask], True)
            re_loss, _ = per_subject_class_ce(p_det.detach() + re, labels[context_mask], subject_tensor[context_mask])
            loss_re = re_loss + float(lambda_r) * regularizer(model)
            loss_re.backward(); torch.nn.utils.clip_grad_norm_([model.U, model.B, model.subject_embedding, model.subject_intercept], GRAD_CLIP); re_opt.step()
            re_loss_value = float(loss_re.detach().cpu())
        # Shared population block-coordinate step.
        set_requires(model.feature_block, True); set_requires(model.layer_norm, True); set_requires(model.population_head, True)
        for p in (model.U, model.B, model.subject_embedding, model.subject_intercept): p.requires_grad_(False)
        shared_opt.zero_grad(set_to_none=True)
        population, effect, z = model(features, subject_tensor, method in {"PERSIST-RE", "RandomEffectOnly", "AdversarialMixed"})
        if method == "ERM":
            shared_loss = F.cross_entropy(population, labels)
        elif method == "SubjectBalancedERM":
            shared_loss, _ = per_subject_class_ce(population, labels, subject_tensor)
        elif method == "Mixup":
            perm = torch.as_tensor(rng.permutation(len(features)), dtype=torch.long, device=device)
            weight = float(rng.beta(0.4, 0.4))
            mixed = weight * features + (1.0 - weight) * features[perm]
            target = weight * F.one_hot(labels, 2).float() + (1.0 - weight) * F.one_hot(labels[perm], 2).float()
            mixed_logits, _, _ = model(mixed, None, False)
            shared_loss = soft_cross_entropy(mixed_logits, target)
        elif method == "GroupDRO":
            values, ids = subject_loss_vector(population, labels, subject_tensor)
            weights = torch.softmax(values.detach() * 5.0, dim=0)
            shared_loss = (values * weights).sum()
        elif method == "ProspectiveOnly":
            context_loss, _ = per_subject_class_ce(population[context_mask], labels[context_mask], subject_tensor[context_mask])
            pseudo_loss, _ = per_subject_class_ce(population[pseudo_mask], labels[pseudo_mask], subject_tensor[pseudo_mask])
            shared_loss = context_loss + pseudo_loss
        elif method == "RandomEffectOnly":
            mixed_loss, _ = per_subject_class_ce(population[context_mask] + effect[context_mask].detach(), labels[context_mask], subject_tensor[context_mask])
            population_loss, _ = per_subject_class_ce(population[context_mask], labels[context_mask], subject_tensor[context_mask])
            shared_loss = mixed_loss + float(lambda_p) * population_loss
        else:
            mixed_loss, _ = per_subject_class_ce(population[context_mask] + effect[context_mask].detach(), labels[context_mask], subject_tensor[context_mask])
            population_context, _ = per_subject_class_ce(population[context_mask], labels[context_mask], subject_tensor[context_mask])
            pseudo_population, _ = per_subject_class_ce(population[pseudo_mask], labels[pseudo_mask], subject_tensor[pseudo_mask])
            shared_loss = mixed_loss + float(lambda_p) * population_context + LAMBDA_Q * pseudo_population
            if method == "AdversarialMixed" and adversary is not None:
                if adv_opt is not None:
                    adv_opt.zero_grad(set_to_none=True)
                    adv_logits = adversary(z.detach())
                    adv_loss = F.cross_entropy(adv_logits, subject_tensor)
                    adv_loss.backward(); adv_opt.step()
                shared_loss = shared_loss + 0.1 * F.cross_entropy(adversary(grad_reverse(z, 1.0)), subject_tensor)
        shared_loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.feature_block.parameters()) + list(model.layer_norm.parameters()) + list(model.population_head.parameters()), GRAD_CLIP); shared_opt.step()
        history.append({"epoch": epoch + 1, "loss_shared": float(shared_loss.detach().cpu()), "loss_re": re_loss_value, "context_subjects": len(context), "pseudo_subjects": len(pseudo)})
    model.eval()
    diagnostics = {
        "subject_count": len(subjects),
        "episode_schedule": episode_schedule(subjects, epochs, stable_seed("partition", seed)),
        "center_e_norm": float(model.centered_effects()[0].detach().norm().cpu()),
        "center_a_norm": float(model.centered_effects()[1].detach().norm().cpu()),
        "center_e_mean_norm": float(model.centered_effects()[0].detach().mean(0).norm().cpu()),
        "center_a_mean_norm": float(model.centered_effects()[1].detach().mean(0).norm().cpu()),
        "random_effect_variance": float(model.centered_effects()[1].detach().var().cpu() + model.centered_effects()[0].detach().var().cpu()),
        "random_effect_parameter_norm": float(torch.cat([model.U.detach().flatten(), model.B.detach().flatten(), model.subject_embedding.detach().flatten(), model.subject_intercept.detach().flatten()]).norm().cpu()),
        "history": history,
    }
    return model, diagnostics


@torch.no_grad()
def predict(model: PERSISTRE, rep: dict[str, np.ndarray], mapping: dict[str, int], device: torch.device) -> dict[str, np.ndarray]:
    features = torch.as_tensor(rep["features"], dtype=torch.float32, device=device)
    # This inference path deliberately never accepts or looks up a subject ID.
    population, effect, _ = model(features, None, False)
    return {"population_logits": population.detach().cpu().numpy().astype(np.float32), "random_effect": np.zeros_like(population.detach().cpu().numpy(), dtype=np.float32), "labels": rep["labels"], "subjects": rep["subjects"]}


def metric_rows(dataset: str, fold: int, seed: int, method: str, pred: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    logits = pred["population_logits"]
    labels = pred["labels"]
    subjects = pred["subjects"].astype(str)
    probabilities = np.exp(logits - logits.max(1, keepdims=True)); probabilities /= probabilities.sum(1, keepdims=True)
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects == subject
        rows.append({
            "dataset": dataset, "fold": fold, "seed": seed, "method": method, "subject_id": subject,
            "BA": float(balanced_accuracy_score(labels[mask], logits[mask].argmax(1))),
            "macro_F1": float(f1_score(labels[mask], logits[mask].argmax(1), average="macro", zero_division=0)),
            "NLL": float(log_loss(labels[mask], probabilities[mask], labels=[0, 1])),
            "trials": int(mask.sum()), "inference_random_effect": False,
        })
    return rows


def model_parameter_groups(model: PERSISTRE) -> dict[str, set[str]]:
    return {
        "shared": {"feature_block.weight", "feature_block.bias", "layer_norm.weight", "layer_norm.bias", "population_head.weight", "population_head.bias"},
        "random_effect": {"U", "B", "subject_embedding", "subject_intercept"},
    }
