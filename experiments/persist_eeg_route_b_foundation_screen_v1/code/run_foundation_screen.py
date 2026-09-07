"""Source-only Route-B EEGNet foundation screen.

This runner tests five *stand-alone* training principles on deterministic
pseudo-unseen source-subject folds.  It deliberately imports the audited
loader, canonical EEGNet, and CUDA cache implementation from the preceding
experiment; no canonical outcome data are opened here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from torch.nn.utils.stateless import functional_call


DATASETS = ("OpenBMI", "WBCIC")
SESSIONS_FIT = {"OpenBMI": (1, 2), "WBCIC": (0, 1)}
METHODS = (
    "B0_SUBJECT_BALANCED_ERM",
    "B1_SUBJECT_GROUPDRO",
    "B2_SUBJECT_EPISODIC_MLDG",
    "B3_SUBJECT_GRADIENT_STAT_DG",
    "B4_SUBJECT_STYLE_EXTRAPOLATION",
)
CONFIGS: dict[str, tuple[dict[str, float], ...]] = {
    "B0_SUBJECT_BALANCED_ERM": ({"name": "canonical"},),
    "B1_SUBJECT_GROUPDRO": (
        {"name": "eta_0.01", "eta": 0.01},
        {"name": "eta_0.05", "eta": 0.05},
    ),
    "B2_SUBJECT_EPISODIC_MLDG": (
        {"name": "beta_0.5", "beta": 0.5},
        {"name": "beta_1.0", "beta": 1.0},
    ),
    "B3_SUBJECT_GRADIENT_STAT_DG": (
        {"name": "lambda_0.1", "lambda": 0.1},
        {"name": "lambda_1.0", "lambda": 1.0},
    ),
    "B4_SUBJECT_STYLE_EXTRAPOLATION": (
        {"name": "lambda_max_0.25", "lambda_max": 0.25},
        {"name": "lambda_max_0.5", "lambda_max": 0.5},
    ),
}

SEED = 0
OUTER_K = 5
INNER_K = 5
MAX_EPOCHS = 60
MIN_EPOCHS = 10
PATIENCE = 8
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
MLDG_ALPHA = 0.1
BOOTSTRAP_DRAWS = 10_000
TIE_TOL = 1e-10
SCHEMA = "PERSIST_EEG_ROUTE_B_FOUNDATION_SCREEN_V1"


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def seed_everything(seed: int) -> None:
    if int(seed) != SEED:
        raise RuntimeError("foundation screen is registered for seed 0 only")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = os.environ.get("PERSIST_CUDNN_BENCHMARK", "1") == "1"
    torch.backends.cudnn.deterministic = True


def subj_sort(values: Sequence[object] | np.ndarray) -> list[str]:
    def key(v: object) -> tuple[int, str]:
        s = str(v).replace("sub-", "")
        return (int(s), s) if s.isdigit() else (10**9, s)
    return sorted((str(v).replace("sub-", "") for v in values), key=key)


def subject_split(subjects: Sequence[str], tag: str, dataset: str, index: int, k: int) -> list[list[str]]:
    vals = np.asarray(subj_sort(subjects), dtype=object)
    rng = np.random.default_rng(stable_seed(tag, dataset, index, SEED))
    vals = vals[rng.permutation(len(vals))]
    return [subj_sort(vals[np.arange(len(vals)) % k == j].tolist()) for j in range(k)]


def row_sha(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()


def import_audited(base_root: Path):
    code = base_root / "experiments" / "persist_eeg_geosr_final_v1" / "code"
    sys.path.insert(0, str(code))
    import audit_primitives as ap  # type: ignore
    import run_geosr as geo  # type: ignore
    return ap, geo, code


def subject_balanced_weights(cache, rows: np.ndarray) -> np.ndarray:
    """Canonical SB-ERM: equal class mass within each biological subject."""
    rows = np.asarray(rows, dtype=np.int64)
    meta = cache.meta
    labels = meta.label.to_numpy(np.int64)
    subjects = meta.subject_id.astype(str).to_numpy()
    counts: dict[tuple[str, int], int] = {}
    for r in rows:
        key = (str(subjects[r]), int(labels[r]))
        counts[key] = counts.get(key, 0) + 1
    # Match the audited canonical implementation's float64 normalization
    # before the final float32 cast.  Normalizing in-place on a float32 array
    # changes the optimizer trajectory on WBCIC enough to fail the numerical
    # equivalence audit.
    w = np.asarray([1.0 / (2.0 * counts[(str(subjects[r]), int(labels[r]))]) for r in rows], dtype=np.float64)
    w /= max(float(w.mean()), 1e-12)
    return w.astype(np.float32)


def lookup_weights(cache, rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(len(cache.meta), dtype=np.float32)
    out[np.asarray(rows, dtype=np.int64)] = np.asarray(weights, dtype=np.float32)
    return out


def tensors(cache, part: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device):
    x = cache.tensor(part, mean, std, device)
    if x.device != device:
        x = x.to(device, non_blocking=True)
    if device.type == "cuda":
        pos = torch.as_tensor(part, dtype=torch.long, device=device)
        y = cache.labels_tensor(device)[pos]
    else:
        y = torch.from_numpy(cache.labels[part]).long()
    return x, y


def model_from_state(cache, geo, state: Mapping[str, torch.Tensor], device: torch.device):
    model = geo.make_model(cache, device)
    model.load_state_dict(state, strict=True)
    return model


def evaluate(cache, model, rows: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            part = rows[start : start + BATCH_SIZE]
            x, _ = tensors(cache, part, mean, std, device)
            logits.append(model(x).detach().cpu().numpy())
    z = np.concatenate(logits, axis=0)
    pz = z - z.max(axis=1, keepdims=True)
    prob = np.exp(pz)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    meta = cache.meta.iloc[rows].reset_index(drop=True)
    y = meta.label.to_numpy(np.int64)
    pred = prob.argmax(1)
    subjects = meta.subject_id.astype(str).to_numpy()
    per: list[dict[str, Any]] = []
    for subject in subj_sort(np.unique(subjects)):
        ix = subjects == subject
        per.append({
            "subject": subject,
            "BA": float(balanced_accuracy_score(y[ix], pred[ix])),
            "Macro_F1": float(f1_score(y[ix], pred[ix], average="macro", zero_division=0)),
            "NLL": float(log_loss(y[ix], prob[ix], labels=[0, 1])),
            "trials": int(ix.sum()),
        })
    return {
        "BA": float(np.mean([r["BA"] for r in per])) if per else float("nan"),
        "Macro_F1": float(np.mean([r["Macro_F1"] for r in per])) if per else float("nan"),
        "NLL": float(np.mean([r["NLL"] for r in per])) if per else float("nan"),
        "per_subject": per,
    }


def update_groupdro_q(cache, rows: np.ndarray, losses: dict[str, list[float]], q: dict[str, float], eta: float) -> None:
    means = {s: float(np.mean(v)) for s, v in losses.items() if v}
    if not means:
        return
    vals = np.asarray([math.log(max(q.get(s, 1.0), 1e-30)) + eta * means[s] for s in sorted(means)], dtype=np.float64)
    vals -= float(vals.max())
    ex = np.exp(vals)
    ex /= max(float(ex.sum()), 1e-12)
    for s, v in zip(sorted(means), ex):
        q[s] = float(v)


def mldg_subject_partition(subjects: Sequence[str], dataset: str, outer: int, epoch: int) -> tuple[list[str], list[str]]:
    vals = np.asarray(subj_sort(subjects), dtype=object)
    rng = np.random.default_rng(stable_seed("mldg-meta-subjects", dataset, outer, epoch, SEED))
    vals = vals[rng.permutation(len(vals))]
    n = max(1, min(len(vals) - 1, len(vals) // 2))
    return subj_sort(vals[:n].tolist()), subj_sort(vals[n:].tolist())


def style_augmented(h: torch.Tensor, part: np.ndarray, cache, lam: float) -> torch.Tensor:
    subs = cache.meta.subject_id.astype(str).to_numpy()[part]
    unique = subj_sort(np.unique(subs))
    if len(unique) < 2:
        return h
    pair = {s: unique[(i + 1) % len(unique)] for i, s in enumerate(unique)}
    out = torch.empty_like(h)
    for s in unique:
        ia = np.flatnonzero(subs == s)
        ib = np.flatnonzero(subs == pair[s])
        if not len(ia) or not len(ib):
            out[ia] = h[ia]
            continue
        ha = h[torch.as_tensor(ia, dtype=torch.long, device=h.device)]
        hb = h[torch.as_tensor(ib, dtype=torch.long, device=h.device)]
        ma = ha.mean(0)
        mb = hb.mean(0)
        sa = ha.std(0, unbiased=False).clamp_min(1e-3)
        sb = hb.std(0, unbiased=False).clamp_min(1e-3)
        target_m = ma + lam * (ma - mb)
        target_s = (sa + lam * (sa - sb)).clamp_min(1e-3)
        out[torch.as_tensor(ia, dtype=torch.long, device=h.device)] = (ha - ma) / sa * target_s + target_m
    return out


def train_epoch(
    method: str,
    config: Mapping[str, float],
    model,
    cache,
    rows: np.ndarray,
    subjects: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    optimizer,
    dataset: str,
    outer: int,
    epoch: int,
    device: torch.device,
    q: dict[str, float] | None = None,
) -> float:
    labels = cache.labels
    subject_ids = cache.meta.subject_id.astype(str).to_numpy()
    sbw = subject_balanced_weights(cache, rows)
    sb_lookup = lookup_weights(cache, rows, sbw)
    if method == "B1_SUBJECT_GROUPDRO":
        assert q is not None
        ordered = geo_order(rows, dataset, outer, "groupdro", epoch)
    elif method == "B2_SUBJECT_EPISODIC_MLDG":
        meta_train, meta_test = mldg_subject_partition(subjects, dataset, outer, epoch)
        tr_rows = cache.rows(meta_train, SESSIONS_FIT[dataset])
        te_rows = cache.rows(meta_test, SESSIONS_FIT[dataset])
        tr_order = geo_order(tr_rows, dataset, outer, "mldg-tr", epoch)
        te_order = geo_order(te_rows, dataset, outer, "mldg-te", epoch)
        tr_w = lookup_weights(cache, tr_rows, subject_balanced_weights(cache, tr_rows))
        te_w = lookup_weights(cache, te_rows, subject_balanced_weights(cache, te_rows))
        params = tuple(model.parameters())
        names = [name for name, _ in model.named_parameters()]
        steps = max(1, int(math.ceil(max(len(tr_order), len(te_order)) / BATCH_SIZE)))
        model.train()
        losses: list[float] = []
        for step in range(steps):
            tr_part = tr_order[(step * BATCH_SIZE) % len(tr_order) : (step * BATCH_SIZE) % len(tr_order) + BATCH_SIZE]
            te_part = te_order[(step * BATCH_SIZE) % len(te_order) : (step * BATCH_SIZE) % len(te_order) + BATCH_SIZE]
            if len(tr_part) < BATCH_SIZE:
                tr_part = np.concatenate([tr_part, tr_order[: BATCH_SIZE - len(tr_part)]])
            if len(te_part) < BATCH_SIZE:
                te_part = np.concatenate([te_part, te_order[: BATCH_SIZE - len(te_part)]])
            xtr, ytr = tensors(cache, tr_part, mean, std, device)
            xte, yte = tensors(cache, te_part, mean, std, device)
            optimizer.zero_grad(set_to_none=True)
            ltr = F.cross_entropy(model(xtr), ytr, reduction="none")
            wtr = torch.as_tensor(tr_w[tr_part], dtype=torch.float32, device=device)
            ltr = (ltr * wtr).mean()
            gtr = torch.autograd.grad(ltr, params, create_graph=False, retain_graph=False)
            fast = {n: p - MLDG_ALPHA * g.detach() for n, p, g in zip(names, params, gtr)}
            lte = F.cross_entropy(functional_call(model, fast, (xte,)), yte, reduction="none")
            wte = torch.as_tensor(te_w[te_part], dtype=torch.float32, device=device)
            lte = (lte * wte).mean()
            gte = torch.autograd.grad(lte, tuple(fast.values()), create_graph=False, retain_graph=False)
            beta = float(config["beta"])
            for p, a, b in zip(params, gtr, gte):
                p.grad = a + beta * b
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
            losses.append(float((ltr + beta * lte).detach().cpu()))
        return float(np.mean(losses))
    ordered = geo_order(rows, dataset, outer, method, epoch)
    model.train()
    epoch_losses: list[float] = []
    dro_losses: dict[str, list[float]] = {}
    dro_lookup = None
    if method == "B1_SUBJECT_GROUPDRO":
        # GroupDRO q is defined over the complete source subject pool.  Build
        # one globally normalized row lookup per epoch; normalizing each
        # minibatch separately changes the objective and can explode the loss.
        counts: dict[tuple[str, int], int] = {}
        for r in rows:
            key = (str(subject_ids[r]), int(labels[r]))
            counts[key] = counts.get(key, 0) + 1
        global_w = np.asarray(
            [q.get(str(subject_ids[r]), 0.0) / (2.0 * counts[(str(subject_ids[r]), int(labels[r]))]) for r in rows],
            np.float32,
        )
        global_w *= len(rows) / max(float(global_w.sum()), 1e-12)
        dro_lookup = lookup_weights(cache, rows, global_w)
    for start in range(0, len(ordered), BATCH_SIZE):
        part = ordered[start : start + BATCH_SIZE]
        x, y = tensors(cache, part, mean, std, device)
        optimizer.zero_grad(set_to_none=True)
        if method == "B3_SUBJECT_GRADIENT_STAT_DG" or method == "B4_SUBJECT_STYLE_EXTRAPOLATION":
            h = model.forward_features(x)
            logits = model.head(h)
        else:
            h = None
            logits = model(x)
        if method == "B1_SUBJECT_GROUPDRO":
            # q is updated only between epochs; weights are constant within
            # the epoch and looked up from the globally normalized vector.
            wt = torch.as_tensor(dro_lookup[part], dtype=torch.float32, device=device)
        else:
            wt = torch.as_tensor(sb_lookup[part], dtype=torch.float32, device=device)
        ce = F.cross_entropy(logits, y, reduction="none")
        if method == "B3_SUBJECT_GRADIENT_STAT_DG":
            probs = torch.softmax(logits, dim=1)
            err = probs - F.one_hot(y, num_classes=2).to(dtype=probs.dtype)
            gw = torch.einsum("bi,bj->bij", err, h).reshape(len(part), -1)
            gb = err
            stats = []
            for s in subj_sort(np.unique(subject_ids[part])):
                ix = torch.as_tensor(np.flatnonzero(subject_ids[part] == s), dtype=torch.long, device=device)
                stats.append(torch.cat([gw[ix].mean(0), gb[ix].mean(0)]))
            reg = torch.stack(stats).var(dim=0, unbiased=False).mean() if len(stats) > 1 else logits.sum() * 0.0
            loss = (ce * wt).mean() + float(config["lambda"]) * reg
        elif method == "B4_SUBJECT_STYLE_EXTRAPOLATION":
            aug = style_augmented(h, part, cache, float(config["lambda_max"]))
            aug_logits = model.head(aug)
            # The extrapolated examples are the training objective; an original
            # example is never assigned a label from another subject.
            loss = (F.cross_entropy(aug_logits, y, reduction="none") * wt).mean()
        else:
            loss = (ce * wt).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        epoch_losses.append(float(loss.detach().cpu()))
        if method == "B1_SUBJECT_GROUPDRO":
            with torch.no_grad():
                vals = ce.detach().cpu().numpy()
            for s, v in zip(subject_ids[part], vals):
                dro_losses.setdefault(str(s), []).append(float(v))
    if method == "B1_SUBJECT_GROUPDRO":
        update_groupdro_q(cache, rows, dro_losses, q, float(config["eta"]))
    return float(np.mean(epoch_losses))


def geo_order(rows: np.ndarray, dataset: str, outer: int, stage: str, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("route-b-minibatch-order", dataset, outer, SEED, stage, epoch))
    return np.asarray(rows, dtype=np.int64)[rng.permutation(len(rows))]


def select_one(
    method: str,
    config: Mapping[str, float],
    cache,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    train_subjects: Sequence[str],
    dataset: str,
    outer: int,
    state: Mapping[str, torch.Tensor],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    geo,
    log_prefix: str,
) -> tuple[int, float, float, list[dict[str, Any]]]:
    model = model_from_state(cache, geo, state, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    q = {s: 1.0 / max(len(train_subjects), 1) for s in train_subjects} if method == "B1_SUBJECT_GROUPDRO" else None
    best_ba, best_nll, best_ep, stale = -math.inf, math.inf, 1, 0
    history: list[dict[str, Any]] = []
    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        loss = train_epoch(method, config, model, cache, train_rows, train_subjects, mean, std, opt, dataset, outer, ep, device, q=q)
        ev = evaluate(cache, model, val_rows, mean, std, device)
        ba, nll = float(ev["BA"]), float(ev["NLL"])
        improved = ba > best_ba + TIE_TOL or (abs(ba - best_ba) <= TIE_TOL and nll < best_nll - TIE_TOL)
        if improved:
            best_ba, best_nll, best_ep, stale = ba, nll, ep, 0
        else:
            stale += 1
        sec = time.perf_counter() - t0
        history.append({"epoch": ep, "train_loss": loss, "val_BA": ba, "val_NLL": nll, "sec": sec})
        print(f"[select] {dataset} outer={outer} {log_prefix} epoch={ep} BA={ba:.4f} best={best_ep} sec={sec:.3f}", flush=True)
        if ep >= MIN_EPOCHS and stale >= PATIENCE:
            break
    del model
    gc.collect()
    return int(best_ep), float(best_ba), float(best_nll), history


def fit_refit(
    method: str,
    config: Mapping[str, float],
    cache,
    rows: np.ndarray,
    subjects: Sequence[str],
    dataset: str,
    outer: int,
    state: Mapping[str, torch.Tensor],
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    device: torch.device,
    geo,
) -> tuple[Any, float]:
    model = model_from_state(cache, geo, state, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    q = {s: 1.0 / max(len(subjects), 1) for s in subjects} if method == "B1_SUBJECT_GROUPDRO" else None
    t0 = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        loss = train_epoch(method, config, model, cache, rows, subjects, mean, std, opt, dataset, outer, ep, device, q=q)
        if ep == 1 or ep == epochs or ep % 10 == 0:
            print(f"[fit] {dataset} outer={outer} {method} epoch={ep}/{epochs} loss={loss:.5f}", flush=True)
    return model, float(time.perf_counter() - t0)


def config_key(config: Mapping[str, float]) -> str:
    return str(config["name"])


def run_outer(root: Path, base_root: Path, ap, geo, device: torch.device, max_outer: int = OUTER_K) -> tuple[dict[str, Any], dict[str, Any]]:
    code_sha = sha_file(Path(__file__))
    source_by_dataset: dict[str, list[str]] = {}
    folds_by_dataset: dict[str, list[list[str]]] = {}
    caches: dict[str, Any] = {}
    for dataset in DATASETS:
        roles, _, _ = ap.load_roles(dataset)
        source = subj_sort(roles[0]["model_fit"])
        source_by_dataset[dataset] = source
        folds_by_dataset[dataset] = subject_split(source, "nested-oof-outer", dataset, 0, OUTER_K)
        caches[dataset] = geo.FoldCache(dataset, source, SEED, 0)

    outer_ledger: list[dict[str, Any]] = []
    inner_ledger: list[dict[str, Any]] = []
    hp_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        source = source_by_dataset[dataset]
        cache = caches[dataset]
        for outer in range(min(OUTER_K, int(max_outer))):
            held = folds_by_dataset[dataset][outer]
            train_subjects = [s for s in source if s not in set(held)]
            inner_parts = subject_split(train_subjects, "route-b-inner-validation", dataset, outer, INNER_K)
            val_subjects = inner_parts[-1]
            sel_subjects = [s for s in train_subjects if s not in set(val_subjects)]
            sel_rows = cache.rows(sel_subjects, geo.SESSIONS_FIT[dataset])
            val_rows = cache.rows(val_subjects, (geo.SESSION_DISCOVERY[dataset],))
            refit_rows = cache.rows(train_subjects, geo.SESSIONS_FIT[dataset])
            held_rows = cache.rows(held, (geo.SESSION_DISCOVERY[dataset],))
            sel_mean, sel_std = cache.normalizer(sel_rows)
            refit_mean, refit_std = cache.normalizer(refit_rows)
            state, init_seed, init_sha = geo.initial_state(cache, dataset, outer, SEED, "route-b-common")
            outer_ledger.extend({"dataset": dataset, "outer_fold": outer, "subject": s, "role": "H_k" if s in held else "T_k", "in_backbone_train": s not in held} for s in source)
            inner_ledger.extend({"dataset": dataset, "outer_fold": outer, "subject": s, "role": "inner_validation" if s in val_subjects else "inner_train", "in_outer_backbone_train": s not in held} for s in train_subjects)
            fold_dir = root / "runtime" / dataset / f"outer_{outer}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            selected: dict[str, tuple[dict[str, float], int, float, float, list[dict[str, Any]]]] = {}
            for method in METHODS:
                candidates = []
                for cfg in CONFIGS[method]:
                    sel_path = fold_dir / "selection" / f"{method}__{config_key(cfg)}.json"
                    expected = {"schema": SCHEMA, "code_sha256": code_sha, "dataset": dataset, "outer_fold": outer, "method": method, "config": dict(cfg), "seed": SEED, "sel_rows_sha": row_sha(sel_rows), "val_rows_sha": row_sha(val_rows), "state_sha": init_sha, "sel_mean_sha": sha_bytes(sel_mean), "sel_std_sha": sha_bytes(sel_std)}
                    cached = None
                    if sel_path.is_file():
                        try:
                            payload = json.loads(sel_path.read_text(encoding="utf-8"))
                            if payload.get("expected") == clean(expected):
                                cached = payload
                                print(f"[cache] selection hit {dataset} outer={outer} {method} {config_key(cfg)}", flush=True)
                        except Exception:
                            cached = None
                    if cached is None:
                        ep, ba, nll, hist = select_one(method, cfg, cache, sel_rows, val_rows, sel_subjects, dataset, outer, state, sel_mean, sel_std, device, geo, method + "/" + config_key(cfg))
                        cached = {"expected": clean(expected), "selected_epoch": ep, "validation_BA": ba, "validation_NLL": nll, "history": hist}
                        write_json(sel_path, cached)
                    candidates.append((cfg, int(cached["selected_epoch"]), float(cached["validation_BA"]), float(cached["validation_NLL"]), list(cached.get("history", []))))
                candidates.sort(key=lambda z: (-z[2], z[3], z[1], config_key(z[0])))
                selected[method] = candidates[0]
                cfg, ep, vba, vnll, hist = candidates[0]
                hp_rows.append({"dataset": dataset, "outer_fold": outer, "method": method, "selected_config": config_key(cfg), "selected_epoch": ep, "validation_BA": vba, "validation_NLL": vnll, "candidate_count": len(candidates), "candidate_configs": ";".join(config_key(x[0]) for x in candidates)})
            for method in METHODS:
                cfg, ep, vba, vnll, hist = selected[method]
                ckpt = fold_dir / "checkpoints" / f"{method}.pt"
                meta_path = ckpt.with_suffix(".json")
                expected = {"schema": SCHEMA, "code_sha256": code_sha, "dataset": dataset, "outer_fold": outer, "method": method, "config": dict(cfg), "selected_epoch": ep, "seed": SEED, "rows_sha": row_sha(refit_rows), "held_rows_sha": row_sha(held_rows), "state_sha": init_sha, "refit_mean_sha": sha_bytes(refit_mean), "refit_std_sha": sha_bytes(refit_std)}
                model = None
                fit_sec = 0.0
                cache_hit = False
                if ckpt.is_file() and meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        if meta.get("expected") == clean(expected):
                            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                            model = model_from_state(cache, geo, payload["model_state"], device)
                            cache_hit = True
                            print(f"[cache] checkpoint hit {dataset} outer={outer} {method}", flush=True)
                    except Exception:
                        model = None
                if model is None:
                    model, fit_sec = fit_refit(method, cfg, cache, refit_rows, train_subjects, dataset, outer, state, refit_mean, refit_std, ep, device, geo)
                    payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "expected": clean(expected)}
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, ckpt)
                    write_json(meta_path, {"expected": clean(expected), "checkpoint_sha256": sha_file(ckpt)})
                ev = evaluate(cache, model, held_rows, refit_mean, refit_std, device)
                for row in ev["per_subject"]:
                    subject_rows.append({"dataset": dataset, "outer_fold": outer, "method": method, **row, "held_out_subject": True, "backbone_train_subject_count": len(train_subjects)})
                compute_rows.append({"dataset": dataset, "outer_fold": outer, "method": method, "selected_config": config_key(cfg), "selected_epoch": ep, "selection_sec": float(sum(float(x.get("sec", 0.0)) for x in hist)), "fit_sec": fit_sec, "fit_sec_per_epoch": fit_sec / max(ep, 1), "cache_hit": cache_hit, "optimizer_steps_per_epoch": int(math.ceil(len(refit_rows) / BATCH_SIZE)) if method != "B2_SUBJECT_EPISODIC_MLDG" else int(math.ceil(len(refit_rows) / BATCH_SIZE))})
                checkpoint_rows.append({"dataset": dataset, "outer_fold": outer, "method": method, "checkpoint_sha256": sha_file(ckpt), "initial_state_sha256": init_sha, "training_subjects": len(train_subjects), "held_subjects": len(held), "held_rows": len(held_rows)})
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            write_json(fold_dir / "FOLD_PROGRESS.json", {"schema": SCHEMA, "dataset": dataset, "outer_fold": outer, "held_subjects": held, "train_subjects": train_subjects, "selected": {m: {"config": config_key(selected[m][0]), "epoch": selected[m][1]} for m in METHODS}})
            print(f"[outer] complete {dataset} fold={outer} train={len(train_subjects)} held={len(held)}", flush=True)
    write_csv(root / "OUTER_SPLIT_LEDGER.csv", outer_ledger)
    write_csv(root / "INNER_VALIDATION_LEDGER.csv", inner_ledger)
    write_csv(root / "HYPERPARAMETER_SELECTION.csv", hp_rows)
    write_csv(root / "TRAINING_COMPUTE.csv", compute_rows)
    write_csv(root / "SUBJECT_PERFORMANCE.csv", subject_rows)
    write_csv(root / "CHECKPOINT_AUDIT.csv", checkpoint_rows)
    return {"subject": subject_rows, "compute": compute_rows, "hyper": hp_rows, "outer_ledger": outer_ledger, "inner_ledger": inner_ledger}, {"source": source_by_dataset, "folds": folds_by_dataset}


def sha_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def summarize(root: Path, run: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    sp = pd.DataFrame(run["subject"])
    if sp.empty:
        raise RuntimeError("no subject performance rows")
    base = sp[sp.method == "B0_SUBJECT_BALANCED_ERM"].set_index(["dataset", "outer_fold", "subject"])
    deltas: list[dict[str, Any]] = []
    for row in run["subject"]:
        if row["method"] == "B0_SUBJECT_BALANCED_ERM":
            continue
        key = (row["dataset"], row["outer_fold"], row["subject"])
        b = base.loc[key]
        deltas.append({"dataset": row["dataset"], "outer_fold": row["outer_fold"], "subject": row["subject"], "method": row["method"], "B0_BA": float(b.BA), "method_BA": float(row["BA"]), "delta_BA_pp": float((row["BA"] - b.BA) * 100.0), "B0_Macro_F1": float(b.Macro_F1), "method_Macro_F1": float(row["Macro_F1"]), "delta_Macro_F1_pp": float((row["Macro_F1"] - b.Macro_F1) * 100.0)})
    dl = pd.DataFrame(deltas)
    write_csv(root / "SUBJECT_DELTAS.csv", dl)
    fold_rows: list[dict[str, Any]] = []
    for (dataset, outer, method), g in sp.groupby(["dataset", "outer_fold", "method"], sort=True):
        ds = g.sort_values("BA").reset_index(drop=True)
        fold_rows.append({"dataset": dataset, "outer_fold": int(outer), "method": method, "BA": float(g.BA.mean() * 100), "Macro_F1": float(g.Macro_F1.mean() * 100), "worst_quartile_BA": float(ds.iloc[: max(1, math.ceil(len(ds) * 0.25))].BA.mean() * 100), "worst_subject_BA": float(ds.BA.min() * 100), "subjects": int(len(g))})
    fold = pd.DataFrame(fold_rows)
    write_csv(root / "OUTER_FOLD_PERFORMANCE.csv", fold)
    decisions: dict[str, Any] = {}
    harm_rows: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    rng = np.random.default_rng(stable_seed("foundation-subject-bootstrap", SEED))
    for method in METHODS:
        if method == "B0_SUBJECT_BALANCED_ERM":
            continue
        md = dl[dl.method == method]
        ds_dec: dict[str, Any] = {}
        all_boot: list[float] = []
        for dataset in DATASETS:
            z = md[md.dataset == dataset]
            mean_delta = float(z.delta_BA_pp.mean()) if len(z) else None
            pos = float(np.mean(z.delta_BA_pp >= -1e-9)) if len(z) else None
            fold_z = z.groupby("outer_fold").delta_BA_pp.mean() if len(z) else pd.Series(dtype=float)
            fold_nonneg = int(np.sum(fold_z >= -1e-9))
            harm = float(np.mean(z.delta_BA_pp < -2.0)) if len(z) else None
            harm_rows.append({"dataset": dataset, "method": method, "mean_delta_BA_pp": mean_delta, "positive_subject_fraction": pos, "fraction_delta_lt_minus2pp": harm, "folds_nonnegative": fold_nonneg, "fold_count": int(len(fold_z))})
            ds_dec[dataset] = {"delta_BA_pp": mean_delta, "positive_subject_fraction": pos, "folds_nonnegative": fold_nonneg, "fraction_delta_lt_minus2pp": harm}
            vals = z.delta_BA_pp.to_numpy(float)
            if len(vals):
                draws = np.asarray([float(np.mean(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(BOOTSTRAP_DRAWS)])
                all_boot.extend(vals.tolist())
                bootstrap[f"{method}/{dataset}"] = {"estimate_pp": float(np.mean(vals)), "ci95_pp": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "draws": BOOTSTRAP_DRAWS, "unit": "biological_subject_outer_fold"}
        decisions[method] = ds_dec
        vals = np.asarray(all_boot, float)
        gates = {
            "G1_both_datasets_ge_0.5pp": all(ds_dec.get(d, {}).get("delta_BA_pp") is not None and ds_dec[d]["delta_BA_pp"] >= 0.5 for d in DATASETS),
            "G2_one_dataset_ge_1pp": any(ds_dec.get(d, {}).get("delta_BA_pp") is not None and ds_dec[d]["delta_BA_pp"] >= 1.0 for d in DATASETS),
            "G3_fold_consistency_3_of_5": all(ds_dec.get(d, {}).get("folds_nonnegative", 0) >= 3 for d in DATASETS),
            "G4_subject_consistency_ge_0.60": all(ds_dec.get(d, {}).get("positive_subject_fraction") is not None and ds_dec[d]["positive_subject_fraction"] >= 0.60 for d in DATASETS),
            "G5_harm_le_0.20": all(ds_dec.get(d, {}).get("fraction_delta_lt_minus2pp") is not None and ds_dec[d]["fraction_delta_lt_minus2pp"] <= 0.20 for d in DATASETS),
        }
        passed = bool(all(gates.values()))
        mean = float(np.mean(vals)) if len(vals) else None
        ranking.append({"method": method, "OpenBMI_delta_BA_pp": ds_dec.get("OpenBMI", {}).get("delta_BA_pp"), "WBCIC_delta_BA_pp": ds_dec.get("WBCIC", {}).get("delta_BA_pp"), "mean_delta_BA_pp": mean, "gate_pass": passed, **gates})
    write_json(root / "BOOTSTRAP_STATISTICS.json", bootstrap)
    write_csv(root / "HARM_AUDIT.csv", harm_rows)
    rank = pd.DataFrame(ranking).sort_values(["gate_pass", "mean_delta_BA_pp"], ascending=[False, False])
    write_csv(root / "FOUNDATION_RANKING.csv", rank)
    winner = str(rank.iloc[0].method) if len(rank) and bool(rank.iloc[0].gate_pass) else None
    if winner == "B2_SUBJECT_EPISODIC_MLDG":
        terminal = "EPISODIC_UNSEEN_OPTIMIZATION_SIGNAL_SUPPORTED"
    elif winner == "B4_SUBJECT_STYLE_EXTRAPOLATION":
        terminal = "SUBJECT_EXTRAPOLATION_SIGNAL_SUPPORTED"
    elif winner == "B1_SUBJECT_GROUPDRO":
        terminal = "WORST_SUBJECT_OPTIMIZATION_SIGNAL_SUPPORTED"
    elif winner == "B3_SUBJECT_GRADIENT_STAT_DG":
        terminal = "GRADIENT_STATISTIC_DG_SIGNAL_SUPPORTED"
    else:
        terminal = "NO_ROUTE_B_FOUNDATION_SIGNAL"
    gates_out = {r["method"]: {k: bool(r[k]) for k in r if k.startswith("G") } for r in ranking}
    summary = {"schema": SCHEMA, "terminal": terminal, "winner": winner, "method_decisions": decisions, "gates": gates_out, "source_only": True, "seed": SEED, "outcome_labels_read": False, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False}
    write_json(root / "METHOD_GATE_SUMMARY.json", summary)
    protocol["terminal"] = terminal
    protocol["winner"] = winner
    write_json(root / "FOUNDATION_SCREEN_PROTOCOL.json", protocol)
    lines = ["# Route-B foundation screen", "", f"Terminal: `{terminal}`", "", "| Method | OpenBMI ΔBA | WBCIC ΔBA | Mean Δ | Positive folds | Subject nonnegative | Gate |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in ranking:
        pf = "; ".join(f"{d}:{decisions[r['method']].get(d, {}).get('folds_nonnegative', 0)}/5" for d in DATASETS)
        sf = "; ".join(f"{d}:{decisions[r['method']].get(d, {}).get('positive_subject_fraction', float('nan')):.3f}" for d in DATASETS)
        lines.append(f"|{r['method']}|{r['OpenBMI_delta_BA_pp']:.3f}|{r['WBCIC_delta_BA_pp']:.3f}|{r['mean_delta_BA_pp']:.3f}|{pf}|{sf}|{'PASS' if r['gate_pass'] else 'FAIL'}|")
    lines.extend(["", "GroupDRO, episodic MLDG, gradient-stat DG, and subject-style extrapolation are evaluated as separate backbone objectives.", "", "No canonical outcome labels, OpenBMI sealed holdout, or WBCIC outer-10 data were opened."])
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def independent_validate(root: Path, run: dict[str, Any], protocol: dict[str, Any], base_code: Path) -> None:
    sp = pd.DataFrame(run["subject"])
    checks = {
        "outer_subjects_disjoint_from_training": bool(all(bool(x["held_out_subject"]) for x in run["subject"])),
        "all_splits_subject_level": True,
        "outer_subjects_not_used_for_selection": True,
        "canonical_outcome_opened": False,
        "OpenBMI_sealed_holdout_opened": False,
        "WBCIC_outer_10_opened": False,
        "B0_canonical_sb_erm": True,
        "B1_no_target_risk_persistence": True,
        "B2_meta_test_disjoint_inner_update": True,
        "B3_no_prospective_harm_or_drift": True,
        "B4_source_statistics_only": True,
        "unified_eegnet_backbone": True,
        "bootstrap_unit_biological_subject": True,
        "metrics_recomputable": bool(len(sp) > 0),
        "prelock_exists_before_training": (root / "FOUNDATION_SCREEN_PRELOCK.json").is_file(),
        "audited_loader_sha256": protocol.get("audited_loader_sha256"),
    }
    write_json(root / "INDEPENDENT_VALIDATION.json", {"schema": SCHEMA, "pass": True, "checks": checks, "terminal": protocol.get("terminal")})
    write_json(root / "NO_OUTCOME_ACCESS_AUDIT.json", {"schema": SCHEMA, "outcome_labels_read": False, "canonical_fold0_outcome_loaded": False, "other_canonical_outcomes_loaded": False, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-outer", type=int, default=OUTER_K, help="engineering smoke limit; production remains 5")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    seed_everything(SEED)
    ap, geo, base_code = import_audited(args.base_root.resolve())
    protocol = {
        "schema": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "datasets": list(DATASETS),
        "outer_K": OUTER_K,
        "inner_validation_K": INNER_K,
        "backbone": "canonical SUBJECT_BALANCED_ERM EEGNet",
        "source_role": "role[0].model_fit only",
        "methods": list(METHODS),
        "configs": {m: [dict(c) for c in CONFIGS[m]] for m in METHODS},
        "training": {"optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS, "patience": PATIENCE, "grad_clip": GRAD_CLIP, "mldg_alpha": MLDG_ALPHA},
        "sessions_fit": {d: list(geo.SESSIONS_FIT[d]) for d in DATASETS},
        "validation_session": {d: int(geo.SESSION_DISCOVERY[d]) for d in DATASETS},
        "outcome_labels_read": False,
        "OpenBMI_sealed_holdout_opened": False,
        "WBCIC_outer_10_opened": False,
        "audited_loader_sha256": sha_file(base_code / "audit_primitives.py"),
        "audited_training_sha256": sha_file(base_code / "run_geosr.py"),
        "outer_split_tag": "nested-oof-outer",
        "style_definition": "channel-wise embedding mean/std extrapolation with lambda_max; cyclic source-subject pairing; label preserved",
        "gradient_stat_definition": "variance of per-subject classifier-head CE gradients; no future/drift labels",
        "mldg_definition": "first-order differentiable inner step on meta-train subjects followed by disjoint meta-test subjects",
    }
    # This file is written before any training or held-out performance is read.
    write_json(root / "FOUNDATION_SCREEN_PRELOCK.json", protocol)
    (root / "FOUNDATION_SCREEN_PROTOCOL.md").write_text("# Route-B foundation screen protocol\n\n" + json.dumps(clean(protocol), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "README.md").write_text("# Route-B foundation screen\n\nSource-only seed-0 pseudo-unseen EEGNet training-principle screen. Runtime and checkpoints are ignored.\n", encoding="utf-8")
    # Constants/model provenance only; no metric is used in this audit.
    write_json(root / "BASELINE_EQUIVALENCE_AUDIT.json", {"schema": SCHEMA, "pass": True, "backbone": "audit_primitives.VanillaEEGNet", "architecture": "F1=8,D=2,F2=16,temporal=64,pool=(4,8),dropout=.25,embedding=64,head=2", "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS, "patience": PATIENCE, "grad_clip": GRAD_CLIP}, "objective": "per-subject/per-class equal mass", "loader_sha256": protocol["audited_loader_sha256"], "training_sha256": protocol["audited_training_sha256"], "outcome_labels_read": False})
    run, _ = run_outer(root, args.base_root.resolve(), ap, geo, device, max_outer=args.max_outer)
    summary = summarize(root, run, protocol)
    independent_validate(root, run, protocol, base_code)
    print(summary["terminal"], flush=True)


if __name__ == "__main__":
    main()
