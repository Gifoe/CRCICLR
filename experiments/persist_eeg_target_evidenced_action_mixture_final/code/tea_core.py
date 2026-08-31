"""TEA-EEG source-only implementation.

This module deliberately operates on frozen prediction caches.  It contains
no test-time optimizer and never accepts labels while constructing a target
context.  The WBCIC helper is explicitly marked as a matched proxy because
the repository has no per-sample S0/S1 action bank.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

SEED = 20260819
SPLIT_SALT = "PERSIST_EEG_POLICY_V2_20260819"
HOLDOUT_THRESHOLD = 0.25
SAFE_ACTIONS = ("amplify", "geometry")
FULL_ACTIONS = ("amplify", "geometry", "erase")
ID_COLUMNS = ["fold", "seed", "router_fold", "manifest_index", "subject", "session"]


def stable_unit(*parts: object) -> float:
    text = ":".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") / 2**64


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean(value)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, frame: pd.DataFrame | list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame([clean(dict(x)) for x in frame])
    tmp = path.with_suffix(path.suffix + ".part")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _subject_key(value: object) -> str:
    text = str(value)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def split_subjects(cache_root: Path) -> dict[str, Any]:
    """Create the old identity-only split without scanning labels/outcomes."""
    features = cache_root / "OOF_ROUTER_FEATURES.parquet"
    identity = pd.read_parquet(features, columns=ID_COLUMNS)
    if len(identity) != 40800:
        raise RuntimeError(f"expected 40800 source rows, found {len(identity)}")
    subjects = sorted({_subject_key(x) for x in identity.subject.unique()}, key=lambda x: int(x))
    if len(subjects) != 52:
        raise RuntimeError(f"expected 52 subjects, found {len(subjects)}")
    assignments: list[dict[str, Any]] = []
    exploration: list[str] = []
    holdout: list[str] = []
    for subject in subjects:
        unit = stable_unit(SPLIT_SALT, subject)
        pool = "DEVELOPMENT_HOLDOUT" if unit < HOLDOUT_THRESHOLD else "EXPLORATION_POOL"
        (holdout if pool == "DEVELOPMENT_HOLDOUT" else exploration).append(subject)
        assignments.append(
            {
                "subject_id": subject,
                "sha256": canonical_hash({"salt": SPLIT_SALT, "subject_id": subject}),
                "hash_unit_interval": unit,
                "pool": pool,
            }
        )
    if (len(exploration), len(holdout)) != (40, 12):
        raise RuntimeError("frozen hash split changed")
    ordered = sorted(exploration, key=lambda s: stable_unit(SPLIT_SALT, "cv", s))
    fold_map = {s: i % 5 for i, s in enumerate(ordered)}
    for row in assignments:
        row["exploration_cv_fold"] = fold_map.get(row["subject_id"])
    return {
        "status": "AUTONOMOUS_RESEARCH_SPLIT_FROZEN",
        "algorithm": "SHA256(salt:canonical_subject_id), first 64 bits mapped to [0,1)",
        "salt": SPLIT_SALT,
        "development_holdout_rule": f"hash_unit_interval < {HOLDOUT_THRESHOLD}",
        "assignment_hash": canonical_hash(assignments),
        "counts": {"all_train_subjects": 52, "exploration_pool": 40, "development_holdout": 12},
        "assignments": assignments,
        "source_identity_rows": len(identity),
        "source_sha256": {str(cache_root / n): sha256_file(cache_root / n) for n in (
            "OOF_ROUTER_FEATURES.parquet", "OOF_BASE_LOGITS.parquet",
            "OOF_COUNTERFACTUAL_LOGITS.parquet", "OOF_GEOMETRY_FEATURES.parquet")},
        "labels_scanned_for_split": False,
        "intervention_outcomes_scanned_for_split": False,
        "DEVELOPMENT_HOLDOUT_OPENED": False,
        "OUTER_TEST_USED": False,
    }


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _read_openbmi(cache_root: Path, subjects: Sequence[str]) -> pd.DataFrame:
    paths = {
        "features": cache_root / "OOF_ROUTER_FEATURES.parquet",
        "base": cache_root / "OOF_BASE_LOGITS.parquet",
        "counterfactual": cache_root / "OOF_COUNTERFACTUAL_LOGITS.parquet",
        "geometry": cache_root / "OOF_GEOMETRY_FEATURES.parquet",
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, filters=[("subject", "in", list(subjects))])
        frame = frame.sort_values(ID_COLUMNS).reset_index(drop=True)
        observed = {_subject_key(x) for x in frame.subject.unique()}
        if not observed.issubset(set(subjects)):
            raise RuntimeError("parquet predicate admitted an excluded subject")
        frames[name] = frame
    ref = frames["features"][ID_COLUMNS]
    for name, frame in frames.items():
        if not ref.equals(frame[ID_COLUMNS]):
            raise RuntimeError(f"identity mismatch in {name}")
    raw = frames["features"].copy()
    for name in ("base", "counterfactual", "geometry"):
        extra = [c for c in frames[name].columns if c not in ID_COLUMNS + ["label"]]
        raw = pd.concat([raw, frames[name][extra]], axis=1)
    d = pd.DataFrame(
        {
            "fold": raw.fold.astype(int), "seed": raw.seed.astype(int),
            "router_fold": raw.router_fold.astype(int), "manifest_index": raw.manifest_index.astype(int),
            "subject": raw.subject.map(_subject_key), "session": raw.session.astype(str),
            "y": raw.label.astype(int),
        }
    )
    pairs = {
        "keep": ("keep_logit_0", "keep_logit_1"),
        "erase": ("erase_logit_0", "erase_logit_1"),
        "amplify": ("amplify_logit_0", "amplify_logit_1"),
        "geometry": ("geometry_logit_0", "geometry_logit_1"),
    }
    for action, (left, right) in pairs.items():
        margin = raw[right].to_numpy(float) - raw[left].to_numpy(float)
        p = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
        d[f"margin_{action}"] = margin
        d[f"p_{action}"] = p
        d[f"pred_{action}"] = (margin >= 0).astype(np.int8)
        d[f"confidence_{action}"] = np.maximum(p, 1 - p)
        d[f"entropy_{action}"] = _entropy(p)
    base_correct = d.pred_keep.to_numpy() == d.y.to_numpy()
    d["baseline_error"] = (~base_correct).astype(np.int8)
    for action in FULL_ACTIONS:
        action_correct = d[f"pred_{action}"].to_numpy() == d.y.to_numpy()
        d[f"effect_{action}"] = action_correct.astype(np.int8) - base_correct.astype(np.int8)
        ce_keep = -np.log(np.where(d.y.to_numpy() == 1, d.p_keep, 1 - d.p_keep).clip(1e-9, 1))
        ce_action = -np.log(np.where(d.y.to_numpy() == 1, d[f"p_{action}"], 1 - d[f"p_{action}"]).clip(1e-9, 1))
        d[f"dce_{action}"] = ce_keep - ce_action
        d[f"flip_{action}"] = (d[f"pred_{action}"] != d.pred_keep).astype(np.int8)
        d[f"delta_margin_{action}"] = d[f"margin_{action}"] - d.margin_keep
        d[f"delta_probability_{action}"] = d[f"p_{action}"] - d.p_keep
        d[f"confidence_change_{action}"] = d[f"confidence_{action}"] - d.confidence_keep
    d["action_disagreement_count"] = d[[f"flip_{a}" for a in FULL_ACTIONS]].sum(axis=1)
    d["action_vote_fraction"] = d[[f"pred_{a}" for a in ("keep",) + FULL_ACTIONS]].mean(axis=1)
    d["action_margin_mean"] = d[[f"margin_{a}" for a in ("keep",) + FULL_ACTIONS]].mean(axis=1)
    d["action_margin_std"] = d[[f"margin_{a}" for a in ("keep",) + FULL_ACTIONS]].std(axis=1, ddof=0)
    # Frozen source-side features are legal only after a name-prefix audit.
    for column in frames["features"].columns:
        if column in ID_COLUMNS + ["label"]:
            continue
        d[f"source_{column}"] = pd.to_numeric(frames["features"][column], errors="coerce").to_numpy(float)
    d = add_cross_run_features(d)
    d["pool"] = "UNKNOWN"
    return d.reset_index(drop=True)


def add_cross_run_features(d: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-run features; caller must isolate a pool before calling."""
    out = d.copy()
    for action in ("keep",) + FULL_ACTIONS:
        group = out.groupby("manifest_index")[f"margin_{action}"]
        count = group.transform("size").to_numpy(float)
        if np.any(count < 2):
            # WBCIC matched proxies have one row per trial; mark unavailable.
            out[f"other_run_mean_margin_{action}"] = np.nan
            out[f"other_run_std_margin_{action}"] = np.nan
            out[f"other_run_vote_{action}"] = np.nan
            continue
        margin = out[f"margin_{action}"].to_numpy(float)
        total = group.transform("sum").to_numpy(float)
        sq = out.assign(_sq=margin**2).groupby("manifest_index")._sq.transform("sum").to_numpy(float)
        mean = (total - margin) / (count - 1)
        var = np.maximum((sq - margin**2) / (count - 1) - mean**2, 0.0)
        votes = out.groupby("manifest_index")[f"pred_{action}"].transform("sum").to_numpy(float)
        out[f"other_run_mean_margin_{action}"] = mean
        out[f"other_run_std_margin_{action}"] = np.sqrt(var)
        out[f"other_run_vote_{action}"] = (votes - out[f"pred_{action}"].to_numpy()) / (count - 1)
    other = out["other_run_vote_keep"]
    out["other_run_base_majority"] = (other >= 0.5).astype(float)
    out["other_run_base_disagrees"] = (out.other_run_base_majority != out.pred_keep).astype(float)
    out["other_run_vote_strength"] = np.abs(other - 0.5) * 2
    out["target_vs_other_margin"] = out.margin_keep - out.other_run_mean_margin_keep
    return out


def add_temporal_blocks(d: pd.DataFrame, n_blocks: int = 5) -> pd.DataFrame:
    out = d.copy()
    out["target_block"] = out.groupby(["subject", "session"], sort=False)["manifest_index"].transform(
        lambda s: pd.factorize(s, sort=True)[0] * n_blocks // max(1, s.nunique())
    ).astype(int)
    return out


CONTEXT_COLUMNS = [
    "margin_keep", "margin_amplify", "margin_geometry", "margin_erase",
    "p_keep", "p_amplify", "p_geometry", "p_erase",
    "confidence_keep", "confidence_amplify", "confidence_geometry", "confidence_erase",
    "entropy_keep", "entropy_amplify", "entropy_geometry", "entropy_erase",
    "other_run_mean_margin_keep", "other_run_mean_margin_amplify",
    "other_run_mean_margin_geometry", "other_run_mean_margin_erase",
    "other_run_vote_keep", "other_run_vote_amplify", "other_run_vote_geometry", "other_run_vote_erase",
]


def build_target_context(d: pd.DataFrame, actions: Sequence[str] = SAFE_ACTIONS) -> tuple[pd.DataFrame, list[str]]:
    """Build context only from the other four deterministic temporal blocks."""
    out = add_temporal_blocks(d)
    columns = [c for c in CONTEXT_COLUMNS if c in out.columns]
    context_value_columns = columns + [f"pred_{a}" for a in ("keep",) + FULL_ACTIONS if f"pred_{a}" in out.columns]
    rows: list[dict[str, Any]] = []
    for (subject, session, block), indices in out.groupby(["subject", "session", "target_block"]).groups.items():
        other = out.loc[
            out.subject.eq(subject) & out.session.eq(session) & out.target_block.ne(block), context_value_columns
        ]
        values = np.nan_to_num(other.to_numpy(float), nan=0.0)
        means = values.mean(axis=0) if len(values) else np.zeros(len(columns))
        variances = values.var(axis=0) if len(values) else np.zeros(len(columns))
        # Covariance eigenvalues are an unlabeled distribution summary.
        if len(values) > 2 and values.shape[1] > 1:
            cov = np.cov(values, rowvar=False)
            eigen = np.sort(np.linalg.eigvalsh(np.nan_to_num(cov)))[-3:]
            eigen = np.pad(eigen, (3 - len(eigen), 0))
        else:
            eigen = np.zeros(3)
        action_means = [float(other[f"p_{a}"].mean()) if len(other) else 0.5 for a in ("keep",) + FULL_ACTIONS]
        agreement = []
        for a in ("keep",) + FULL_ACTIONS:
            agreement.append(float(np.mean(np.asarray(other[f"pred_{a}"]) == np.asarray(other.pred_keep)))) if len(other) else agreement.append(0.0)
        row: dict[str, Any] = {"subject": subject, "session": session, "target_block": int(block), "context_n": int(len(other))}
        row.update({f"ctx_mean_{i}": float(x) for i, x in enumerate(means)})
        row.update({f"ctx_var_{i}": float(x) for i, x in enumerate(variances)})
        row.update({f"ctx_eigen_{i}": float(x) for i, x in enumerate(eigen)})
        row.update({f"ctx_predprop_{i}": float(x) for i, x in enumerate(action_means)})
        row.update({f"ctx_agreement_{i}": float(x) for i, x in enumerate(agreement)})
        row["ctx_cross_run_agreement"] = float(np.nanmean(other.get("other_run_vote_keep", pd.Series(dtype=float)))) if len(other) else 0.0
        rows.append(row)
    context = pd.DataFrame(rows)
    return out.merge(context, on=["subject", "session", "target_block"], how="left", validate="many_to_one"), [c for c in context.columns if c.startswith("ctx_") or c == "context_n"]


class DeepSetsContextEncoder:
    """Small deterministic phi/rho encoder; fitting uses unlabeled context only."""

    def __init__(self, width: int, seed: int = SEED):
        if width not in (32, 64):
            raise ValueError("width must be 32 or 64")
        self.width = width
        self.seed = seed
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.w_phi: np.ndarray | None = None
        self.b_phi: np.ndarray | None = None
        self.w_rho: np.ndarray | None = None
        self.b_rho: np.ndarray | None = None

    def fit(self, context: np.ndarray) -> "DeepSetsContextEncoder":
        context = np.nan_to_num(np.asarray(context, dtype=float), nan=0.0)
        self.mean = context.mean(axis=0)
        self.scale = context.std(axis=0) + 1e-6
        rng = np.random.default_rng(self.seed + self.width)
        self.w_phi = rng.normal(0, 1 / np.sqrt(context.shape[1]), (context.shape[1], self.width))
        self.b_phi = rng.normal(0, 0.05, self.width)
        self.w_rho = rng.normal(0, 1 / np.sqrt(self.width), (self.width, self.width))
        self.b_rho = rng.normal(0, 0.05, self.width)
        return self

    def transform(self, context: np.ndarray) -> np.ndarray:
        if self.mean is None or self.w_phi is None or self.w_rho is None:
            raise RuntimeError("context encoder is not fitted")
        x = (np.nan_to_num(np.asarray(context, dtype=float), nan=0.0) - self.mean) / self.scale
        return np.tanh(np.tanh(x @ self.w_phi + self.b_phi) @ self.w_rho + self.b_rho)


def legal_sample_columns(d: pd.DataFrame) -> list[str]:
    forbidden = ("effect_", "dce_", "y", "baseline_error", "subject", "session", "manifest", "fold", "seed", "router_fold", "target_block", "context_n")
    candidates = [c for c in d.columns if c.startswith(("margin_", "p_", "confidence_", "entropy_", "flip_", "delta_", "action_", "other_run_", "target_vs_other_", "source_"))]
    return [c for c in candidates if not any(token in c for token in forbidden)]


def paired_bootstrap(values: np.ndarray, seed_offset: int = 0, draws: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(SEED + seed_offset)
    sample = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def realized_prediction(d: pd.DataFrame, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = d.pred_keep.to_numpy(int).copy()
    effect = np.zeros(len(d), dtype=float)
    for action in FULL_ACTIONS:
        mask = selected == action
        pred[mask] = d.loc[mask, f"pred_{action}"].to_numpy(int)
        effect[mask] = d.loc[mask, f"effect_{action}"].to_numpy(float)
    return pred, effect


def policy_metrics(
    d: pd.DataFrame,
    selected: np.ndarray,
    prediction: np.ndarray | None = None,
    seed_offset: int = 0,
    bootstrap_draws: int = 5000,
) -> dict[str, Any]:
    pred, effect = realized_prediction(d, selected) if prediction is None else (prediction, np.zeros(len(d)))
    run_rows: list[dict[str, Any]] = []
    for (fold, seed, subject), indices in d.groupby(["fold", "seed", "subject"], sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        run_rows.append({"subject": str(subject), "fold": int(fold), "seed": int(seed), "delta": float(balanced_accuracy_score(d.y.to_numpy()[idx], pred[idx]) - balanced_accuracy_score(d.y.to_numpy()[idx], d.pred_keep.to_numpy()[idx]))})
    run = pd.DataFrame(run_rows)
    if run.empty:
        # WBCIC matched proxy has one pseudo-run per subject.
        for subject, indices in d.groupby("subject", sort=False).groups.items():
            idx = np.asarray(list(indices), dtype=int)
            run_rows.append({"subject": str(subject), "fold": 0, "seed": 0, "delta": float(balanced_accuracy_score(d.y.to_numpy()[idx], pred[idx]) - balanced_accuracy_score(d.y.to_numpy()[idx], d.pred_keep.to_numpy()[idx]))})
        run = pd.DataFrame(run_rows)
    subject = run.groupby("subject", as_index=False).delta.mean()
    vals = subject.delta.to_numpy(float)
    mean, ci_l, ci_u = paired_bootstrap(vals, seed_offset, draws=bootstrap_draws)
    intervened = selected != "keep"
    if prediction is not None:
        # Mixture intervention effects are defined by the selected conservative action.
        _, effect = realized_prediction(d, selected)
    return {
        "mean_subject_delta_BA": mean,
        "bootstrap_CI95_L": ci_l,
        "bootstrap_CI95_U": ci_u,
        "positive_subject_fraction": float(np.mean(vals > 0)) if len(vals) else 0.0,
        "nonnegative_subject_fraction": float(np.mean(vals >= 0)) if len(vals) else 0.0,
        "positive_run_fraction": float(np.mean(run.delta > 0)) if len(run) else 0.0,
        "action_rate": float(np.mean(intervened)) if len(intervened) else 0.0,
        "unsafe_intervention_rate": float(np.mean(effect[intervened] < 0)) if intervened.any() else 0.0,
        "rescue_precision": float(np.mean(effect[intervened] > 0)) if intervened.any() else 0.0,
        "mean_effect_when_intervene": float(np.mean(effect[intervened])) if intervened.any() else 0.0,
        "subjects": int(len(subject)),
        "runs": int(len(run)),
        "OUTER_TEST_USED": False,
        "subject_deltas": vals.tolist(),
    }


def old_select(d: pd.DataFrame, actions: Sequence[str]) -> np.ndarray:
    selected = np.full(len(d), "keep", dtype=object)
    gate = d.other_run_base_disagrees.to_numpy(dtype=bool)
    for action in actions:
        available = d[f"pred_{action}"].to_numpy() != d.pred_keep.to_numpy()
        take = gate & (selected == "keep") & available
        selected[take] = action
    return selected


def oracle_selected(d: pd.DataFrame, actions: Sequence[str]) -> np.ndarray:
    selected = np.full(len(d), "keep", dtype=object)
    for action in actions:
        take = (selected == "keep") & (d[f"effect_{action}"].to_numpy() > 0)
        selected[take] = action
    return selected


def apply_mixture(d: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, beta: float, kappa: float, actions: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen TEA equation; returns probability, selected action, weights."""
    g = np.asarray(mu, dtype=float) - float(kappa) * np.asarray(sigma, dtype=float)
    positive = np.maximum(g, 0.0)
    weights = np.c_[np.ones(len(d)), np.exp(float(beta) * positive)]
    probs = np.stack([d.p_keep.to_numpy(float)] + [d[f"p_{a}"].to_numpy(float) for a in actions], axis=1)
    final = np.sum(weights * probs, axis=1) / np.sum(weights, axis=1)
    all_nonpositive = np.all(g <= 0, axis=1)
    final[all_nonpositive] = d.p_keep.to_numpy(float)[all_nonpositive]
    selected = np.full(len(d), "keep", dtype=object)
    best = np.argmax(np.where(g > 0, g, -np.inf), axis=1)
    active = ~all_nonpositive
    if len(actions):
        selected[active] = np.asarray(actions, dtype=object)[best[active]]
    return final, selected, weights


@dataclass
class FoldPredictions:
    mu: np.ndarray
    sigma: np.ndarray
    context_encoded: np.ndarray


def fit_oof_regret(
    d: pd.DataFrame,
    actions: Sequence[str],
    width: int,
    use_context: bool = True,
    n_bootstrap_models: int = 3,
) -> FoldPredictions:
    """Subject-disjoint OOF regret predictions; no validation labels are used."""
    subjects = np.array(sorted(d.subject.unique(), key=lambda x: str(x)))
    fold_map = {s: i % 5 for i, s in enumerate(subjects)}
    cv = d.subject.map(fold_map).to_numpy(int)
    sample_columns = legal_sample_columns(d)
    sample = np.nan_to_num(d[sample_columns].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    context_columns = [c for c in d.columns if c.startswith("ctx_") or c == "context_n"]
    context = np.nan_to_num(d[context_columns].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0) if context_columns else np.zeros((len(d), 1))
    mu = np.zeros((len(d), len(actions)), dtype=float)
    sigma = np.zeros_like(mu)
    enc_all = np.zeros((len(d), width), dtype=float)
    for fold in range(5):
        train = cv != fold
        valid = cv == fold
        encoder = DeepSetsContextEncoder(width, seed=SEED + fold).fit(context[train])
        h_train = encoder.transform(context[train])
        h_valid = encoder.transform(context[valid])
        enc_all[valid] = h_valid
        x_train = np.c_[sample[train], h_train] if use_context else sample[train]
        x_valid = np.c_[sample[valid], h_valid] if use_context else sample[valid]
        for j, action in enumerate(actions):
            target = d[f"dce_{action}"].to_numpy(float)
            preds: list[np.ndarray] = []
            train_indices = np.flatnonzero(train)
            for rep in range(n_bootstrap_models):
                rng = np.random.default_rng(SEED + width * 100 + fold * 10 + j * 3 + rep)
                draw = rng.choice(train_indices, size=len(train_indices), replace=True)
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    HistGradientBoostingRegressor(
                        max_iter=80, max_leaf_nodes=15, learning_rate=0.08,
                        l2_regularization=5.0, random_state=SEED + rep,
                    ),
                )
                model.fit(x_train[draw - train_indices.min()] if False else x_train[np.searchsorted(train_indices, draw)], target[draw])
                preds.append(model.predict(x_valid))
            pred = np.stack(preds, axis=1)
            mu[valid, j] = pred.mean(axis=1)
            residual = target[train] - np.mean(np.stack([p for p in preds], axis=1), axis=1).mean()
            residual_scale = float(np.std(target[train]))
            sigma[valid, j] = np.maximum(pred.std(axis=1), 0.15 * residual_scale)
    return FoldPredictions(mu, sigma, enc_all)


def fit_oof_error_router(d: pd.DataFrame, actions: Sequence[str]) -> np.ndarray:
    subjects = np.array(sorted(d.subject.unique(), key=lambda x: str(x)))
    fold_map = {s: i % 5 for i, s in enumerate(subjects)}
    cv = d.subject.map(fold_map).to_numpy(int)
    cols = legal_sample_columns(d)
    x = np.nan_to_num(d[cols].to_numpy(float), nan=0.0)
    score = np.zeros(len(d), dtype=float)
    selected = np.full(len(d), "keep", dtype=object)
    train_gain = np.zeros((5, len(actions)))
    for fold in range(5):
        tr, va = cv != fold, cv == fold
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.03, max_iter=500, class_weight="balanced", random_state=SEED + fold))
        model.fit(x[tr], d.baseline_error.to_numpy(int)[tr])
        score[va] = model.predict_proba(x[va])[:, 1]
        for j, action in enumerate(actions):
            train_gain[fold, j] = float(d.loc[tr, f"dce_{action}"].mean())
        gate = score[va] > 0.5
        for j, action in enumerate(actions):
            available = d.loc[va, f"pred_{action}"].to_numpy() != d.loc[va, "pred_keep"].to_numpy()
            take = gate & (selected[va] == "keep") & available
            selected[np.flatnonzero(va)[take]] = action
    return selected


def summarize_target_context(d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in d.columns if c.startswith("ctx_") or c == "context_n"]
    if not cols:
        return pd.DataFrame()
    return d.groupby(["subject", "session", "target_block"], as_index=False)[cols].first()
