"""Auditable, source-only core for PERSIST-PDA.

The implementation intentionally works on the already frozen CleanRoom
representations.  The population logits and feature encoder are never
updated.  Subject adapters are fitted from historical blocks only; future
blocks are passed to metric functions but never to fitting or Fisher pooling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import f1_score


REPO = Path(os.environ.get("PDA_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC"))
EXP = REPO / "experiments" / "persist_eeg_persistent_decision_adapter_final"
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"  # never committed; contains only transient fit caches
OLD_REP_ROOT = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "runtime" / "specialist_representations"

DATASETS = ("OpenBMI", "WBCIC")
FOLDS = tuple(range(5))
SEEDS = tuple(range(3))
RANKS = (1, 2, 4)
LAMBDA_X = (0.5, 1.0)
LAMBDA_PRECISION = (1e-3, 1e-2)
LAMBDA_T = 1.0
CLASSES = 2
N_BOOTSTRAP = 10_000
FORBIDDEN_FUTURE_TOKENS = ("utility_metrics", "utility_units", "sealed", "outer", "session-2-utility")


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
        x = float(value)
        return x if math.isfinite(x) else None
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


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def load_rep(dataset: str, fold: int, seed: int, role: str, backbone: str = "ATCNet") -> dict[str, np.ndarray]:
    """Load only the authorized frozen representation archive."""
    path = OLD_REP_ROOT / backbone / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_FUTURE_TOKENS):
        raise PermissionError(f"future/sealed resource is not available to source PDA: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing authorized source representation: {path}")
    with np.load(path, allow_pickle=True) as z:
        out = {key: z[key] for key in z.files}
    required = {"indices", "features", "logits", "labels", "subjects", "sessions"}
    missing = required.difference(out)
    if missing:
        raise RuntimeError(f"{path} missing keys {sorted(missing)}")
    out["features"] = np.asarray(out["features"], np.float64)
    out["logits"] = np.asarray(out["logits"], np.float64)
    out["labels"] = np.asarray(out["labels"], np.int64)
    out["subjects"] = np.asarray(out["subjects"]).astype("U")
    out["sessions"] = np.asarray(out["sessions"], np.int64)
    out["indices"] = np.asarray(out["indices"], np.int64)
    if out["features"].ndim != 2 or out["logits"].shape != (len(out["labels"]), CLASSES):
        raise RuntimeError(f"unexpected archive shape: {path}")
    if len(np.unique(out["indices"])) != len(out["indices"]):
        raise RuntimeError(f"duplicate source indices: {path}")
    if not np.isfinite(out["features"]).all() or not np.isfinite(out["logits"]).all():
        raise RuntimeError(f"non-finite source archive: {path}")
    return out


@dataclass(frozen=True)
class Fold:
    model_fit: dict[str, np.ndarray]
    validation: dict[str, np.ndarray]
    outcome: dict[str, np.ndarray]


def load_fold(dataset: str, fold: int, seed: int, backbone: str = "ATCNet") -> Fold:
    return Fold(*(load_rep(dataset, fold, seed, role, backbone) for role in ("model_fit", "validation", "outcome")))


def layer_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5)


@dataclass(frozen=True)
class SubjectTransition:
    subject: str
    history_blocks: tuple[dict[str, np.ndarray], ...]
    future: dict[str, np.ndarray]


def _subset(rep: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in rep.items()}


def make_transitions(rep: dict[str, np.ndarray], n_blocks: int = 2) -> list[SubjectTransition]:
    """Use the earliest session as historical and the latest as future.

    With one session, deterministic contiguous temporal blocks are used for
    historical fitting and the last block is evaluated as the prospective
    block.  For the source archives every subject has two natural sessions,
    so the first session is split into two contiguous historical blocks and the
    second session is untouched future evaluation.
    """
    sessions = sorted(int(x) for x in np.unique(rep["sessions"]))
    if not sessions:
        return []
    subjects = subject_sort(np.unique(rep["subjects"]))
    out: list[SubjectTransition] = []
    for subject in subjects:
        sm = rep["subjects"] == subject
        ss = sessions[-1] if len(sessions) > 1 else sessions[0]
        hs = sessions[:-1] if len(sessions) > 1 else [sessions[0]]
        hist_mask = sm & np.isin(rep["sessions"], hs)
        future_mask = sm & (rep["sessions"] == ss)
        # If there is only one session there is no genuinely future session;
        # use the final temporal block solely for diagnostic tests.  The source
        # archives take the natural two-session branch above.
        hist_indices = np.flatnonzero(hist_mask)
        hist_indices = hist_indices[np.argsort(rep["indices"][hist_indices], kind="stable")]
        if len(hist_indices) < 2:
            continue
        blocks_idx = np.array_split(hist_indices, max(2, min(int(n_blocks), len(hist_indices))))
        blocks = tuple(_subset(rep, np.isin(np.arange(len(rep["labels"])), idx)) for idx in blocks_idx if len(idx))
        future = _subset(rep, future_mask)
        if not len(future["labels"]):
            continue
        out.append(SubjectTransition(subject, blocks, future))
    return out


def fit_ridge_correction(rep: dict[str, np.ndarray], ridge: float, target_scale: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a stable linear decision correction on one historical block.

    This is the deterministic quadratic approximation used to initialize the
    low-rank adapter.  It is not a changed population classifier: only the
    subject-specific correction is fitted.  The target is the class-balanced
    desired logit minus the frozen population logit.
    """
    x = layer_norm(rep["features"])
    n, d = x.shape
    design = np.column_stack([x, np.ones(n)])
    desired = np.zeros((n, CLASSES), np.float64)
    desired[np.arange(n), rep["labels"]] = target_scale
    residual = desired - rep["logits"]
    reg = np.eye(d + 1, dtype=np.float64) * float(ridge)
    reg[-1, -1] = float(ridge) * 0.25
    lhs = design.T @ design + reg
    coef = np.linalg.solve(lhs, design.T @ residual)
    return coef[:d].T, coef[d], np.diag(design.T @ design)[:d]


@dataclass(frozen=True)
class Basis:
    U: np.ndarray  # d x r
    V: np.ndarray  # C x r
    design: np.ndarray  # (C*d) x r, vec(V_j U_j^T)


def _pad_columns(base: np.ndarray, width: int, seed: int) -> np.ndarray:
    """Deterministically pad a column matrix; QR avoids random state."""
    rows = base.shape[0]
    cols = [base[:, i] for i in range(base.shape[1])]
    rng = np.random.default_rng(seed)
    while len(cols) < width:
        v = rng.normal(size=rows)
        for q in cols:
            v = v - float(v @ q) * q
        norm = float(np.linalg.norm(v))
        if norm < 1e-10:
            v = np.zeros(rows); v[len(cols) % rows] = 1.0
            for q in cols:
                v = v - float(v @ q) * q
            norm = float(np.linalg.norm(v))
        cols.append(v / max(norm, 1e-12))
    return np.column_stack(cols[:width])


def fit_shared_basis(training_rep: dict[str, np.ndarray], rank: int, ridge: float) -> Basis:
    """Estimate shared U,V from model-fit historical labels only."""
    transitions = make_transitions(training_rep)
    matrices = []
    for tr in transitions:
        for block in tr.history_blocks:
            m, _, _ = fit_ridge_correction(block, ridge)
            matrices.append(m)
    if not matrices:
        raise RuntimeError("no historical blocks for shared basis")
    stack = np.concatenate(matrices, axis=0)
    _, _, vh = np.linalg.svd(stack, full_matrices=False)
    d = training_rep["features"].shape[1]
    k = min(int(rank), d)
    u = vh[:k].T
    if k < rank:
        # deterministic orthogonal completion in feature space
        u = _pad_columns(u, int(rank), stable_seed("U", rank, d))
    mean_m = np.mean(np.stack(matrices), axis=0)
    left, _, _ = np.linalg.svd(mean_m, full_matrices=False)
    v = _pad_columns(left[:, : min(left.shape[1], rank)], int(rank), stable_seed("V", rank))
    design = np.column_stack([np.outer(v[:, j], u[:, j]).reshape(-1) for j in range(int(rank))])
    return Basis(u.astype(np.float64), v.astype(np.float64), design.astype(np.float64))


def decompose(matrix: np.ndarray, basis: Basis) -> np.ndarray:
    a, *_ = np.linalg.lstsq(basis.design, np.asarray(matrix).reshape(-1), rcond=None)
    return np.asarray(a, np.float64)


def reconstruct(a: np.ndarray, c: np.ndarray, basis: Basis, x: np.ndarray) -> np.ndarray:
    x = layer_norm(x)
    responses = x @ basis.U
    return (responses * np.asarray(a)[None, :]) @ basis.V.T + np.asarray(c)[None, :]


def fit_block_adapter(block: dict[str, np.ndarray], basis: Basis, ridge: float) -> dict[str, np.ndarray | float]:
    matrix, intercept, fisher_x = fit_ridge_correction(block, ridge)
    a = decompose(matrix, basis)
    # Diagonal Fisher approximation for each code.  The frozen population
    # probabilities provide the local curvature; no held/future labels enter.
    logits = block["logits"] + reconstruct(a, intercept, basis, block["features"])
    p = softmax(logits, axis=1)
    curvature = np.mean(p * (1.0 - p), axis=1)
    response = layer_norm(block["features"]) @ basis.U
    fisher_a = np.maximum(np.mean(curvature[:, None] * response**2, axis=0) * len(block["labels"]), 1e-8)
    fisher_c = np.maximum(np.mean(p * (1.0 - p), axis=0) * len(block["labels"]), 1e-8)
    return {"a": a, "c": np.asarray(intercept, np.float64), "fisher_a": fisher_a, "fisher_c": fisher_c,
            "n": float(len(block["labels"])), "matrix_norm": float(np.linalg.norm(matrix))}


def precision_pool(parts: list[dict[str, np.ndarray | float]], lambda_precision: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not parts:
        raise ValueError("cannot pool empty adapter list")
    fa = np.sum([np.asarray(p["fisher_a"]) for p in parts], axis=0)
    fc = np.sum([np.asarray(p["fisher_c"]) for p in parts], axis=0)
    a = np.sum([np.asarray(p["fisher_a"]) * np.asarray(p["a"]) for p in parts], axis=0) / (float(lambda_precision) + fa)
    c = np.sum([np.asarray(p["fisher_c"]) * np.asarray(p["c"]) for p in parts], axis=0) / (float(lambda_precision) + fc)
    return a, c, fa, fc


def center_transient_components(session_codes: np.ndarray, persistent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reparameterize residuals so transient components have zero mean."""
    session_codes = np.asarray(session_codes, np.float64)
    persistent = np.asarray(persistent, np.float64)
    residual = session_codes - persistent[None, ...]
    residual = residual - residual.mean(axis=0, keepdims=True)
    return persistent[None, ...] + residual, residual


def unknown_subject_params(basis: Basis) -> dict[str, np.ndarray]:
    """Population-only fallback when no biological subject ID is known."""
    return {"a": np.zeros(basis.U.shape[1], np.float64), "c": np.zeros(CLASSES, np.float64)}


def fit_subject_methods(tr: SubjectTransition, basis: Basis, ridge: float, lambda_x: float, lambda_precision: float) -> dict[str, dict[str, object]]:
    parts = [fit_block_adapter(block, basis, ridge) for block in tr.history_blocks]
    # Explicit leave-one-historical-block-out estimates.  The held block is
    # never included in its own persistent estimate.
    loo = []
    for k in range(len(parts)):
        others = [p for j, p in enumerate(parts) if j != k]
        pa, pc, _, _ = precision_pool(others, lambda_precision)
        loo.append((pa, pc))
    pa_raw, pc_raw, fa, fc = precision_pool(parts, lambda_precision)
    w = float(lambda_x) / (1.0 + float(lambda_x))
    # Cross-fitted objective surrogate: each block code is shrunk toward the
    # estimate formed without that block.  lambda_T=1 fixed shrinks transient
    # deviations but does not alter the population model.
    adjusted = []
    for k, p in enumerate(parts):
        pa, pc = loo[k]
        ak = (1.0 - w) * np.asarray(p["a"]) + w * pa
        ck = (1.0 - w) * np.asarray(p["c"]) + w * pc
        adjusted.append({**p, "a": ak, "c": ck})
    pa_full, pc_full, fa_full, fc_full = precision_pool(adjusted, lambda_precision)
    centered_a, transient_a = center_transient_components(np.stack([np.asarray(p["a"]) for p in adjusted]), pa_full)
    centered_c, transient_c = center_transient_components(np.stack([np.asarray(p["c"]) for p in adjusted]), pc_full)
    mean_a = np.mean([np.asarray(p["a"]) for p in parts], axis=0)
    mean_c = np.mean([np.asarray(p["c"]) for p in parts], axis=0)
    # Ordinary pooled adapter is fitted on all historical samples, not an
    # average of blocks.  It is the matched B2 baseline.
    all_rep = {k: np.concatenate([b[k] for b in tr.history_blocks]) for k in tr.history_blocks[0]}
    pooled = fit_block_adapter(all_rep, basis, ridge)
    single = parts[-1]
    return {
        "population": unknown_subject_params(basis),
        "intercept_only": {"a": np.zeros(basis.U.shape[1]), "c": pooled["c"]},
        "ordinary_adapter": {"a": pooled["a"], "c": pooled["c"]},
        "single_session": {"a": single["a"], "c": single["c"]},
        "mean_pooled": {"a": mean_a, "c": mean_c},
        "no_crossfit": {"a": pa_raw, "c": pc_raw},
        "crossfit_mean": {"a": np.mean([x[0] for x in loo], axis=0), "c": np.mean([x[1] for x in loo], axis=0)},
        "full_pda": {"a": pa_full, "c": pc_full},
        "centered_session_codes": {"a": centered_a, "c": centered_c},
        "transient_codes": {"a": transient_a, "c": transient_c},
        "parts": parts,
        "loo": loo,
        "fisher_a": fa_full,
        "fisher_c": fc_full,
        "persistent_norm": float(np.linalg.norm(pa_full)),
        "transient_norm": float(np.mean(np.linalg.norm(transient_a, axis=1))),
        "persistent_transient_ratio": float(np.linalg.norm(pa_full) / max(np.mean(np.linalg.norm(transient_a, axis=1)), 1e-12)),
        "transient_mean_norm": float(np.linalg.norm(transient_a.mean(axis=0))),
        "crossfit_gain": float(np.mean([metric_for_rep(b, np.asarray(loo[k][0]), np.asarray(loo[k][1]), basis)[0] - metric_for_rep(b, np.zeros(basis.U.shape[1]), np.zeros(CLASSES), basis)[0] for k, b in enumerate(tr.history_blocks)])),
    }


def predict_rep(rep: dict[str, np.ndarray], params: Mapping[str, np.ndarray], basis: Basis) -> np.ndarray:
    delta = reconstruct(np.asarray(params["a"]), np.asarray(params["c"]), basis, rep["features"])
    return rep["logits"] + delta


def metric_for_rep(rep: dict[str, np.ndarray], a: np.ndarray, c: np.ndarray, basis: Basis) -> tuple[float, float]:
    pred = (rep["logits"] + reconstruct(a, c, basis, rep["features"])).argmax(axis=1)
    y = np.asarray(rep["labels"], np.int64)
    recalls = [float(np.mean(pred[y == cls] == cls)) for cls in range(CLASSES) if np.any(y == cls)]
    ba = float(np.mean(recalls)) if recalls else float("nan")
    # Explicit labels keep the macro-F1 definition stable for a one-class
    # temporal block while the prospective sessions contain both classes.
    f1 = float(f1_score(y, pred, labels=list(range(CLASSES)), average="macro", zero_division=0))
    return ba, f1


def evaluate_transition(tr: SubjectTransition, methods: dict[str, dict[str, object]], basis: Basis, fold: int, seed: int, dataset: str, role: str, recipe_id: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    for method, params in methods.items():
        if method in {"parts", "loo", "fisher_a", "fisher_c", "persistent_norm", "transient_norm", "persistent_transient_ratio", "transient_mean_norm", "crossfit_gain", "centered_session_codes", "transient_codes"}:
            continue
        ba, f1 = metric_for_rep(tr.future, np.asarray(params["a"]), np.asarray(params["c"]), basis)
        rows.append({"dataset": dataset, "role": role, "fold": fold, "seed": seed, "subject": tr.subject, "method": method, "recipe": recipe_id, "BA": ba, "macro_F1": f1, "future_session_used_for_fit": False, "future_labels_used_for_fit": False})
    # Mechanism controls.  Wrong adapters are norm matched; shuffled is a
    # deterministic permutation at the caller's subject-set level.
    return rows, []


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    values = np.asarray(values, np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    samples = values[rng.integers(0, len(values), size=(int(draws), len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_subject_bootstrap(frame: pd.DataFrame, left: str, right: str, dataset: str, seed_tag: str) -> dict[str, object]:
    a = frame[(frame.dataset == dataset) & (frame.method == left)].groupby("subject", as_index=True).BA.mean()
    b = frame[(frame.dataset == dataset) & (frame.method == right)].groupby("subject", as_index=True).BA.mean()
    common = a.index.intersection(b.index)
    delta = (a.loc[common] - b.loc[common]).to_numpy(np.float64)
    mean, lo, hi = bootstrap_ci(delta, stable_seed(seed_tag, dataset, left, right))
    return {"dataset": dataset, "comparison": f"{left}-{right}", "delta_BA": mean, "CI95_L": lo, "CI95_U": hi, "subjects": int(len(common)), "nonnegative_fraction": float(np.mean(delta >= 0)) if len(delta) else float("nan")}


def transition_hash(rep: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("indices", "subjects", "sessions"):
        h.update(np.asarray(rep[key]).tobytes())
    return h.hexdigest()


def population_checkpoint_id(rep: dict[str, np.ndarray]) -> str:
    """Stable identifier for the frozen population representation/checkpoint."""
    h = hashlib.sha256()
    for key in ("features", "logits", "indices", "subjects", "sessions"):
        value = np.asarray(rep[key])
        h.update(key.encode("utf-8")); h.update(str(value.shape).encode("ascii")); h.update(value.tobytes())
    return h.hexdigest()


def control_assignments(subjects: Iterable[object], norms: Mapping[object, float]) -> tuple[dict[str, str], dict[str, str]]:
    """Return deterministic norm-matched wrong and cyclic shuffled mappings."""
    ids = subject_sort(subjects)
    wrong: dict[str, str] = {}
    shuffled: dict[str, str] = {}
    for i, subject in enumerate(ids):
        candidates = [x for x in ids if x != subject]
        wrong[subject] = min(candidates, key=lambda x: (abs(float(norms[x]) - float(norms[subject])), x)) if candidates else subject
        shuffled[subject] = ids[(i + 1) % len(ids)] if ids else subject
    return wrong, shuffled


def assert_future_resource_locked(lock_path: Path) -> None:
    """Fail closed unless an explicit source-passed lock authorizes future use."""
    if not lock_path.is_file():
        raise PermissionError(f"future resource lock missing: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("status") != "AUTHORIZED" or payload.get("source_gate_pass") is not True:
        raise PermissionError("future resource remains sealed until source gate passes")
