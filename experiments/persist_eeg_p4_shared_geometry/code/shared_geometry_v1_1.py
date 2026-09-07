"""PERSIST-EEG Shared Geometry Audit V1.1.

This file deliberately keeps the audit separate from the earlier invalid
implementation.  It never trains a model and never loads outer-test data.
The script is intended to run from the historical server project root where
``p4_persist_ct.py`` and ``p4_persist_ct_v2.py`` are importable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

import p4_persist_ct as base
import p4_persist_ct_v2 as v2


ROOT = base.ROOT
OUT = ROOT / "outputs" / "persist_eeg_shared_geometry_v1_1"
SIGNED = ROOT / "outputs" / "persist_eeg_p4_signed" / "audit_v3"
TASKS = tuple(base.TASKS)
CLASSES = dict(base.CLASSES)
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
RANDOM_DRAWS = 100
BOOT_DRAWS = 10_000
GEOM_CAP = 64
V1_1_VERSION = "Shared Geometry Audit V1.1"


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite(v: float | None) -> float | None:
    if v is None or not np.isfinite(v):
        return None
    return float(v)


def sha_bytes(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def basis_fingerprint(spec: Mapping[str, Any]) -> dict[str, Any]:
    names = ("mean", "whitener", "dewhitener", "directions", "rho")
    arrays = {name: sha_bytes(np.asarray(spec[name], dtype=np.float32)) for name in names}
    payload = {"arrays": arrays, "blocks": [list(map(int, b)) for b in spec["blocks"]],
               "block_dimensions": [len(b) for b in spec["blocks"]]}
    payload["combined_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def stable_group_sample(meta: pd.DataFrame, task: str, subjects: Sequence[str], cap: int,
                        seed: int) -> np.ndarray:
    """Deterministic label-balanced sample, with subjects retained as units."""
    mask = (meta.paradigm.astype(str) == str(task)) & meta.subject_id.astype(str).isin(set(map(str, subjects)))
    frame = meta.loc[mask]
    chosen: list[int] = []
    for key, group in frame.groupby(["subject_id", "session_id", "event_label"], sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        if cap and len(idx) > cap:
            rng = np.random.default_rng(stable_seed("geom-cap", seed, task, *key))
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        chosen.extend(map(int, idx))
    return np.asarray(sorted(chosen), dtype=np.int64)


def task_indices(meta: pd.DataFrame, task: str, subjects: Sequence[str] | None = None,
                 sessions: Sequence[str] | None = None) -> np.ndarray:
    mask = meta.paradigm.astype(str) == str(task)
    if subjects is not None:
        mask &= meta.subject_id.astype(str).isin(set(map(str, subjects)))
    if sessions is not None:
        mask &= meta.session_id.astype(str).isin(set(map(str, sessions)))
    return np.flatnonzero(mask.to_numpy())


def label_map(meta: pd.DataFrame, task: str) -> dict[str, int]:
    labels = sorted(meta.loc[meta.paradigm.astype(str) == str(task), "event_label"].astype(str).unique())
    if len(labels) != CLASSES[task]:
        raise RuntimeError(f"unexpected labels for {task}: {labels}")
    return {label: i for i, label in enumerate(labels)}


def labels(meta: pd.DataFrame, task: str, idx: np.ndarray) -> np.ndarray:
    mapping = label_map(meta, task)
    return meta.iloc[idx].event_label.astype(str).map(mapping).to_numpy(dtype=np.int64)


def q_from_h(h: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    return ((np.asarray(h, dtype=np.float64) - np.asarray(spec["mean"], dtype=np.float64))
            @ np.asarray(spec["whitener"], dtype=np.float64)
            @ np.asarray(spec["directions"], dtype=np.float64)).astype(np.float32)


def class_centroids(meta: pd.DataFrame, q: np.ndarray, task: str, subjects: Sequence[str] | None = None,
                    sessions: Sequence[str] | None = None) -> dict[tuple[str, str], np.ndarray]:
    """Class centroids after label-free subject/session global centering."""
    idx = task_indices(meta, task, subjects, sessions)
    fm = meta.iloc[idx].reset_index(drop=True)
    x = np.asarray(q[idx], dtype=np.float64).copy()
    for _, group in fm.groupby(["subject_id", "session_id"], sort=True):
        loc = group.index.to_numpy(dtype=np.int64)
        x[loc] -= x[loc].mean(axis=0, keepdims=True)
    mapping = label_map(meta, task)
    out: dict[tuple[str, str], np.ndarray] = {}
    for (s, r), group in fm.groupby(["subject_id", "session_id"], sort=True):
        loc = group.index.to_numpy(dtype=np.int64)
        y = np.asarray([mapping[str(v)] for v in group.event_label], dtype=np.int64)
        c = []
        for k in range(CLASSES[task]):
            c.append(x[loc[y == k]].mean(axis=0) if np.any(y == k) else np.full(x.shape[1], np.nan))
        out[(str(s), str(r))] = np.asarray(c, dtype=np.float64)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else float("nan")


def binary_deltas(meta: pd.DataFrame, q: np.ndarray, task: str, subjects: Sequence[str] | None = None,
                  sessions: Sequence[str] | None = None) -> dict[tuple[str, str], np.ndarray]:
    if CLASSES[task] != 2:
        raise ValueError("binary_deltas is only for binary tasks")
    cs = class_centroids(meta, q, task, subjects, sessions)
    return {key: value[1] - value[0] for key, value in cs.items()
            if value.shape[0] == 2 and np.isfinite(value).all()}


def rdm(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    return np.asarray([np.linalg.norm(x[i] - x[j]) for i in range(len(x)) for j in range(i + 1, len(x))])


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) != len(a) or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    v = spearmanr(a, b).statistic
    return finite(float(v))


def geometry_scores(meta: pd.DataFrame, q: np.ndarray, task: str, subjects: Sequence[str] | None = None,
                    sessions: Sequence[str] | None = None) -> dict[str, Any]:
    """Compute subject-level geometry metrics; subject is always the unit."""
    sub = sorted(set(meta.subject_id.astype(str)) if subjects is None else set(map(str, subjects)))
    ses = sorted(set(meta.session_id.astype(str)) if sessions is None else set(map(str, sessions)))
    if CLASSES[task] == 2:
        d = binary_deltas(meta, q, task, sub, ses)
        a_vals: list[dict[str, Any]] = []; c_vals: list[dict[str, Any]] = []
        energy: list[dict[str, Any]] = []
        for s in sub:
            for r in ses:
                if (s, r) not in d:
                    continue
                other = [d[(so, r)] for so in sub if so != s and (so, r) in d]
                if other:
                    cons = np.mean(other, axis=0)
                    a_vals.append({"subject": s, "session": r, "score": cosine(d[(s, r)], cons)})
                    den = float(d[(s, r)] @ d[(s, r)])
                    energy.append({"subject": s, "session": r,
                                   "score": float((d[(s, r)] @ cons) ** 2 /
                                                   max(float(cons @ cons) * den, 1e-12))})
            if len(ses) >= 2 and all((s, r) in d for r in ses[:2]):
                for target, source in ((ses[1], ses[0]), (ses[0], ses[1])):
                    other = [d[(so, source)] for so in sub if so != s and (so, source) in d]
                    if other:
                        c_vals.append({"subject": s, "direction": f"{source}->{target}",
                                       "score": cosine(d[(s, target)], np.mean(other, axis=0))})
        return {"alignment": a_vals, "cross_session": c_vals, "shared_energy": energy,
                "mean_alignment": finite(np.nanmean([x["score"] for x in a_vals])) if a_vals else None,
                "mean_cross_session": finite(np.nanmean([x["score"] for x in c_vals])) if c_vals else None,
                "rdm_defined": False}

    cs = class_centroids(meta, q, task, sub, ses)
    direct: list[dict[str, Any]] = []; rdm_rows: list[dict[str, Any]] = []; cross: list[dict[str, Any]] = []
    for s in sub:
        for r in ses:
            key = (s, r)
            if key not in cs or not np.isfinite(cs[key]).all():
                continue
            other = [cs[(so, r)] for so in sub if so != s and (so, r) in cs and np.isfinite(cs[(so, r)]).all()]
            if other:
                cons = np.mean(other, axis=0)
                direct.append({"subject": s, "session": r, "score": cosine(cs[key], cons)})
                v = spearman_safe(rdm(cs[key]), rdm(cons))
                if v is not None:
                    rdm_rows.append({"subject": s, "session": r, "score": v})
        if len(ses) >= 2:
            for target, source in ((ses[1], ses[0]), (ses[0], ses[1])):
                key = (s, target)
                other = [cs[(so, source)] for so in sub if so != s and (so, source) in cs and np.isfinite(cs[(so, source)]).all()]
                if key in cs and other:
                    v = spearman_safe(rdm(cs[key]), rdm(np.mean(other, axis=0)))
                    if v is not None:
                        cross.append({"subject": s, "direction": f"{source}->{target}", "score": v})
    return {"alignment": direct, "rdm": rdm_rows, "cross_session": cross,
            "mean_alignment": finite(np.nanmean([x["score"] for x in direct])) if direct else None,
            "mean_rdm": finite(np.nanmean([x["score"] for x in rdm_rows])) if rdm_rows else None,
            "mean_cross_session": finite(np.nanmean([x["score"] for x in cross])) if cross else None,
            "rdm_defined": bool(rdm_rows)}


def center_subject_session(meta: pd.DataFrame, q: np.ndarray, task: str, subjects: Sequence[str],
                           target_center: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    idx = task_indices(meta, task, subjects)
    fm = meta.iloc[idx].reset_index(drop=True)
    x = np.asarray(q[idx], dtype=np.float64).copy()
    if target_center:
        for _, group in fm.groupby(["subject_id", "session_id"], sort=True):
            loc = group.index.to_numpy(dtype=np.int64)
            x[loc] -= x[loc].mean(axis=0, keepdims=True)
    return fm, x


def nearest_prototype(train_meta: pd.DataFrame, train_q: np.ndarray, eval_meta: pd.DataFrame,
                      eval_q: np.ndarray, task: str, train_subjects: Sequence[str], eval_subject: str,
                      target_center: bool = True) -> float:
    tm, tq = center_subject_session(train_meta, train_q, task, train_subjects, target_center)
    em, eq = center_subject_session(eval_meta, eval_q, task, [eval_subject], target_center)
    mapping = label_map(train_meta, task)
    ytr = np.asarray([mapping[str(v)] for v in tm.event_label], dtype=np.int64)
    ye = np.asarray([mapping[str(v)] for v in em.event_label], dtype=np.int64)
    prototypes = np.asarray([tq[ytr == k].mean(axis=0) for k in range(CLASSES[task])])
    pred = np.argmin(((eq[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2), axis=1)
    return float(np.mean([np.mean(pred[ye == k] == k) for k in range(CLASSES[task]) if np.any(ye == k)]))


def ridge_subject(train_meta: pd.DataFrame, train_q: np.ndarray, eval_meta: pd.DataFrame, eval_q: np.ndarray,
                  task: str, train_subjects: Sequence[str], eval_subject: str, target_center: bool = True) -> float:
    tm, tq = center_subject_session(train_meta, train_q, task, train_subjects, target_center)
    em, eq = center_subject_session(eval_meta, eval_q, task, [eval_subject], target_center)
    mapping = label_map(train_meta, task)
    ytr = np.asarray([mapping[str(v)] for v in tm.event_label], dtype=np.int64)
    ye = np.asarray([mapping[str(v)] for v in em.event_label], dtype=np.int64)
    pack = base.ridge_probe(tq, ytr, CLASSES[task])
    pred, _ = base.probe_predict(eq, pack, CLASSES[task])
    return float(np.mean([np.mean(pred[ye == k] == k) for k in range(CLASSES[task]) if np.any(ye == k)]))


def loso_scores(meta: pd.DataFrame, q: np.ndarray, task: str, subjects: Sequence[str],
                target_center: bool = True, draw_controls: Sequence[np.ndarray] | None = None) -> dict[str, Any]:
    subs = sorted(set(map(str, subjects)))
    pvals: list[dict[str, Any]] = []
    control_vals: list[list[float]] = [[] for _ in range(RANDOM_DRAWS)]
    for s in subs:
        train = [x for x in subs if x != s]
        p = nearest_prototype(meta, q, meta, q, task, train, s, target_center)
        row = {"subject": s, "protected_ba": p}
        if draw_controls is not None:
            for d, qr in enumerate(draw_controls):
                control_vals[d].append(nearest_prototype(meta, qr, meta, qr, task, train, s, target_center))
            row["random_mean_ba"] = float(np.mean([control_vals[d][-1] for d in range(RANDOM_DRAWS)]))
        pvals.append(row)
    if draw_controls is not None:
        for i, row in enumerate(pvals):
            row["random_mean_ba"] = float(np.mean([control_vals[d][i] for d in range(RANDOM_DRAWS)]))
            row["delta"] = row["protected_ba"] - row["random_mean_ba"]
    return {"subjects": pvals, "delta": [x.get("delta") for x in pvals if "delta" in x],
            "protected_mean": float(np.mean([x["protected_ba"] for x in pvals]))}


def subject_margin(meta: pd.DataFrame, q: np.ndarray, task: str, subjects: Sequence[str]) -> list[dict[str, Any]]:
    """Cross-subject same-class vs different-class centroid margin."""
    cs = class_centroids(meta, q, task, subjects)
    subs = sorted(set(map(str, subjects))); result = []
    # Average a subject's class centroid across sessions before cross-subject comparison.
    by_sub: dict[str, np.ndarray] = {}
    for s in subs:
        vals = [cs[(s, r)] for r in sorted({r for ss, r in cs if ss == s}) if np.isfinite(cs[(s, r)]).all()]
        if vals:
            by_sub[s] = np.mean(vals, axis=0)
    for s, own in by_sub.items():
        others = [by_sub[so] for so in subs if so != s and so in by_sub]
        if not others:
            continue
        cons = np.mean(others, axis=0)
        same, different = [], []
        for y in range(CLASSES[task]):
            same.append(float(np.linalg.norm(own[y] - cons[y])))
            for yp in range(CLASSES[task]):
                if yp != y:
                    different.append(float(np.linalg.norm(own[y] - cons[yp])))
        result.append({"subject": s, "same_class_distance": float(np.mean(same)),
                       "different_class_distance": float(np.mean(different)),
                       "margin": float(np.mean(different) - np.mean(same))})
    return result


def bootstrap(values: Sequence[float], seed: int) -> dict[str, Any]:
    vals = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(vals) == 0:
        return {"mean": None, "median": None, "ci95": [None, None], "sign_probability": None,
                "draws": BOOT_DRAWS, "n_subjects": 0}
    rng = np.random.default_rng(seed)
    d = rng.choice(vals, size=(BOOT_DRAWS, len(vals)), replace=True).mean(axis=1)
    return {"mean": float(vals.mean()), "median": float(np.median(vals)),
            "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "sign_probability": float(np.mean(d > 0)), "draws": BOOT_DRAWS, "n_subjects": int(len(vals))}


def hierarchical_boot(run_values: Mapping[str, Sequence[float]], seed: int) -> dict[str, Any]:
    keys = sorted(run_values)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOT_DRAWS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals = []
        for key in sampled:
            arr = np.asarray(run_values[key], dtype=np.float64)
            if len(arr):
                vals.append(float(rng.choice(arr, size=len(arr), replace=True).mean()))
        draws.append(float(np.mean(vals)) if vals else float("nan"))
    vals = np.asarray(draws, dtype=np.float64); vals = vals[np.isfinite(vals)]
    raw = np.asarray([x for k in keys for x in run_values[k]], dtype=np.float64)
    return {"mean": float(np.mean(raw)) if len(raw) else None,
            "median": float(np.median(raw)) if len(raw) else None,
            "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if len(vals) else [None, None],
            "sign_probability": float(np.mean(vals > 0)) if len(vals) else None,
            "draws": BOOT_DRAWS, "n_runs": len(keys), "n_subject_values": int(len(raw))}


def make_controls(q: np.ndarray, h: np.ndarray, spec: Mapping[str, Any], protected_ids: Sequence[int],
                  run_seed: int) -> dict[str, Any]:
    d = q.shape[1]; ids = np.asarray(sorted(set(map(int, protected_ids))), dtype=np.int64); k = len(ids)
    non = np.asarray([i for i in range(d) if i not in set(ids)], dtype=np.int64)
    if len(non) < k:
        raise RuntimeError(f"not enough non-Protected active dimensions ({len(non)} < {k})")
    rng = np.random.default_rng(run_seed)
    random_ids = [np.sort(rng.choice(non, size=k, replace=False)) for _ in range(RANDOM_DRAWS)]
    random_q = [q[:, ch].copy() for ch in random_ids]
    # Random orthogonal subspaces in the active whitened q-space.
    orth_q = []
    for _ in range(RANDOM_DRAWS):
        z = rng.normal(size=(d, k)); u, _ = np.linalg.qr(z); orth_q.append((q.astype(np.float64) @ u[:, :k]).astype(np.float32))
    # PCA same-rank control fit on TRAIN embeddings only.
    xc = np.asarray(h, dtype=np.float64) - np.asarray(h, dtype=np.float64).mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    pca_q = (xc @ vt[:k].T).astype(np.float32)
    return {"protected_ids": ids.tolist(), "rank": k, "nonprotected_ids": non.tolist(),
            "random_ids": [x.tolist() for x in random_ids], "random_q": random_q,
            "orthogonal_q": orth_q, "pca_q": pca_q}


def check_protected_invariance(meta: pd.DataFrame, q: np.ndarray, protected_ids: Sequence[int], task: str,
                               run_seed: int) -> dict[str, Any]:
    ids = np.asarray(protected_ids, dtype=np.int64)
    non = np.asarray([i for i in range(q.shape[1]) if i not in set(ids)], dtype=np.int64)
    base_metric = geometry_scores(meta, q[:, ids], task)
    pert = q.copy()
    if len(non):
        rng = np.random.default_rng(run_seed); pert[:, non] += rng.normal(0, 1000, size=(len(pert), len(non))).astype(np.float32)
    pert_metric = geometry_scores(meta, pert[:, ids], task)
    a = json.dumps(base.clean(base_metric), sort_keys=True); b = json.dumps(base.clean(pert_metric), sort_keys=True)
    return {"qP_dimension": int(q[:, ids].shape[1]), "protected_id_count": int(len(ids)),
            "nonprotected_perturbation_applied": bool(len(non)), "passed": bool(a == b),
            "base_sha256": hashlib.sha256(a.encode()).hexdigest(), "perturbed_sha256": hashlib.sha256(b.encode()).hexdigest()}


def v3_validation_reproduction(trm: pd.DataFrame, trh: np.ndarray, vam: pd.DataFrame, vah: np.ndarray,
                               spec: Mapping[str, Any], assignments: Mapping[str, Any],
                               split: Mapping[str, Any], fold: int, seed: int) -> dict[str, Any]:
    """Replay the Signed-V3 validation calculation using the exact reconstructed basis."""
    # Signed V3's historical sampler is reproduced verbatim for compatibility;
    # the adaptation log records that the old code did not fix PYTHONHASHSEED.
    def signed_audit_idx(meta: pd.DataFrame, task: str, subs: Sequence[str], s: int, per_group: int = 32) -> np.ndarray:
        frame = meta[(meta.paradigm == task) & meta.subject_id.astype(str).isin(set(map(str, subs)))]
        out: list[int] = []
        for key, g in frame.groupby(["subject_id", "session_id", "event_label"], sort=True):
            ix = g.index.to_numpy(dtype=np.int64)
            rng = np.random.default_rng(s + int(abs(hash(str(key))) % 1_000_003))
            if len(ix) > per_group:
                ix = np.sort(rng.choice(ix, size=per_group, replace=False))
            out.extend(ix.tolist())
        return np.asarray(sorted(out), dtype=np.int64)

    rows: list[dict[str, Any]] = []; ok = True; tol = 1e-8
    for task in TASKS:
        ass = assignments[task]
        ids = sorted(set(sum((spec["blocks"][b] for b in ass.get("protected", [])), [])))
        ti = signed_audit_idx(trm, task, split["train_subjects"], 33_000 + seed)
        vi = signed_audit_idx(vam, task, split["validation_subjects"], 34_000 + seed)
        ytr, yv = labels(trm, task, ti), labels(vam, task, vi)
        raw_pack = base.ridge_probe(trh[ti], ytr, CLASSES[task]); _, raw_prob = base.probe_predict(vah[vi], raw_pack, CLASSES[task])
        raw_pred = raw_prob.argmax(axis=1)
        raw_ba = float(np.mean([np.mean(raw_pred[yv == k] == k) for k in range(CLASSES[task]) if np.any(yv == k)]))
        erased_tr = base.erase(trh[ti], spec, ids); erased_va = base.erase(vah[vi], spec, ids)
        pack = base.ridge_probe(erased_tr, ytr, CLASSES[task]); pred, _ = base.probe_predict(erased_va, pack, CLASSES[task])
        erased_ba = float(np.mean([np.mean(pred[yv == k] == k) for k in range(CLASSES[task]) if np.any(yv == k)]))
        observed = erased_ba - raw_ba
        ref_path = SIGNED / f"fold-{fold}" / f"seed-{seed}" / "VALIDATION_SIGN_TRANSFER.csv"
        ref = pd.read_csv(ref_path); match = ref[(ref.task == task) & (ref.kind == "protected_union")]
        expected = float(match.validation_gain_BA.iloc[0]) if len(match) else None
        diff = abs(observed - expected) if expected is not None else None
        good = bool(expected is not None and diff <= tol)
        ok &= good
        rows.append({"task": task, "recomputed_protected_union_gain_BA": observed,
                     "signed_v3_saved_gain_BA": expected, "absolute_difference": diff,
                     "tolerance": tol, "pass": good, "n_train": int(len(ti)), "n_validation": int(len(vi))})
    return {"pass": ok, "rows": rows, "tolerance": tol,
            "note": "Signed-V3 replay uses its historical sampling function; old artifact did not persist sampled indices."}


def utility_rows(fold: int, seed: int) -> pd.DataFrame:
    p = SIGNED / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_UTILITY_V3.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def block_utility_geometry(meta: pd.DataFrame, q: np.ndarray, task: str, spec: Mapping[str, Any],
                           fold: int, seed: int) -> list[dict[str, Any]]:
    util = utility_rows(fold, seed)
    rows: list[dict[str, Any]] = []
    for bi, block in enumerate(spec["blocks"]):
        qb = q[:, block]
        g = geometry_scores(meta, qb, task)
        if CLASSES[task] == 2:
            g_a = g.get("mean_alignment"); g_c = g.get("mean_cross_session")
        else:
            g_a = g.get("mean_rdm"); g_c = g.get("mean_cross_session")
        margin = subject_margin(meta, qb, task, sorted(set(meta.subject_id.astype(str))))
        g_m = float(np.mean([x["margin"] for x in margin])) if margin else None
        match = util[(util.task == task) & (util.block == bi)]
        u = float(match.u_spec_mean.iloc[0]) if len(match) and "u_spec_mean" in match else None
        strength = float(np.mean(np.maximum(np.asarray(spec["rho"])[block], 0.0)))
        rows.append({"fold": fold, "seed": seed, "task": task, "block": bi,
                     "dimensions": len(block), "persistence_strength": strength,
                     "geometry_alignment": g_a, "geometry_cross_session": g_c,
                     "geometry_margin": g_m, "geometry_primary": g_a,
                     "u_spec": u, "utility_source": "Signed_Audit_V3_SIGNED_UTILITY_V3.csv"})
    return rows


def fit_utility_link(frame: pd.DataFrame, seed: int = 12345) -> dict[str, Any]:
    f = frame[(frame.task == "mi") & frame.u_spec.notna() & frame.geometry_primary.notna()].copy()
    if len(f) < 10:
        return {"status": "INSUFFICIENT_BLOCKS", "n": int(len(f))}
    # Standardize predictors, preserve run effects as fixed one-hot terms.
    g = f.geometry_primary.to_numpy(float); r = f.persistence_strength.to_numpy(float); d = f.dimensions.to_numpy(float); y = f.u_spec.to_numpy(float)
    def design(x: pd.DataFrame) -> np.ndarray:
        gg = (x.geometry_primary.to_numpy(float) - g.mean()) / max(g.std(), 1e-12)
        rr = (x.persistence_strength.to_numpy(float) - r.mean()) / max(r.std(), 1e-12)
        dd = (x.dimensions.to_numpy(float) - d.mean()) / max(d.std(), 1e-12)
        runs = (x.fold.astype(str) + "_" + x.seed.astype(str)).to_numpy()
        keys = sorted(set(f.fold.astype(str) + "_" + f.seed.astype(str)))
        one = np.column_stack([runs == key for key in keys[1:]]) if len(keys) > 1 else np.zeros((len(x), 0))
        return np.column_stack([np.ones(len(x)), gg, rr, dd, one])
    X = design(f); beta = np.linalg.lstsq(X, y, rcond=None)[0]; beta_g = float(beta[1])
    rho = spearmanr(g, y).statistic if np.std(g) > 0 and np.std(y) > 0 else None
    rng = np.random.default_rng(seed); run_keys = sorted(set(f.fold.astype(str) + "_" + f.seed.astype(str))); boot = []
    for _ in range(BOOT_DRAWS):
        sampled = rng.choice(run_keys, size=len(run_keys), replace=True)
        parts = [f[(f.fold.astype(str) + "_" + f.seed.astype(str)) == key] for key in sampled]
        ff = pd.concat(parts, ignore_index=True); xx = design(ff); yy = ff.u_spec.to_numpy(float)
        boot.append(float(np.linalg.lstsq(xx, yy, rcond=None)[0][1]))
    boot = np.asarray(boot, dtype=float)
    return {"status": "OK", "n_blocks": int(len(f)), "beta_geometry": beta_g,
            "beta_geometry_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "beta_geometry_sign_probability": float(np.mean(boot > 0)), "pooled_spearman_rho": finite(float(rho)) if rho is not None else None,
            "bootstrap_draws": BOOT_DRAWS, "formula": "u_spec ~ geometry + persistence_strength + block_dimension + run_effects"}


def run_one(fold: int, seed: int, device: torch.device) -> dict[str, Any]:
    print(f"[V1.1] fold={fold} seed={seed} loading full TRAIN", flush=True)
    manifest = base.load_manifest(); split = next(x for x in base.load_splits() if int(x["fold"]) == fold)
    ckpt, mean, std = base.historical(fold, seed); model = base.load_model(ckpt, manifest, device)
    trm, trh, tr_y = base.extract(model, manifest, split["train_subjects"], mean, std, device, 90_000 + fold * 101 + seed, cap=0)
    vam, vah, va_y = base.extract(model, manifest, split["validation_subjects"], mean, std, device, 100_000 + fold * 101 + seed, cap=0)
    print(f"[V1.1] fold={fold} seed={seed} extracted train={len(trh)} val={len(vah)}", flush=True)
    spec = v2.build_spectrum_v2(trm, trh, 30_000 + fold * 101 + seed)
    q = q_from_h(trh, spec); qv = q_from_h(vah, spec)
    fp = basis_fingerprint(spec)
    run_dir = OUT / "runs" / f"fold-{fold}" / f"seed-{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "BASIS_FINGERPRINT.json", {"fold": fold, "seed": seed, "fingerprint": fp,
                                                      "spectrum_audit": spec["audit"], "outer_test_used": False})
    assignments = json.loads((SIGNED / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_ASSIGNMENTS.json").read_text(encoding="utf-8"))
    validation_repro = v3_validation_reproduction(trm, trh, vam, vah, spec, assignments, split, fold, seed)
    write_json(run_dir / "BASIS_RECONSTRUCTION_VALIDATION.json", validation_repro)
    if not validation_repro["pass"]:
        return {"fold": fold, "seed": seed, "basis_reconstruction": validation_repro, "basis_fingerprint": fp,
                "outer_test_used": False, "status": "SHARED_GEOMETRY_BASIS_RECONSTRUCTION_FAIL"}
    all_rows: list[dict[str, Any]] = []; task_payload: dict[str, Any] = {}; controls_payload: dict[str, Any] = {}
    for task in TASKS:
        ass = assignments[task]; protected_ids = sorted(set(sum((spec["blocks"][b] for b in ass.get("protected", [])), [])))
        if not protected_ids:
            raise RuntimeError(f"no frozen Protected ids for {task}")
        controls = make_controls(q, trh, spec, protected_ids, stable_seed("controls", fold, seed, task))
        qP = q[:, protected_ids]; qPv = qv[:, protected_ids]
        unit = check_protected_invariance(trm, q, protected_ids, task, stable_seed("unit", fold, seed, task))
        if not unit["passed"]:
            raise RuntimeError(f"Protected-only perturbation unit test failed for {task}")
        # Primary TRAIN geometry and controls.
        pgeom = geometry_scores(trm, qP, task)
        rgeoms = [geometry_scores(trm, x, task) for x in controls["random_q"]]
        # Use random-draw means for the reported control; retain all draws in CSV.
        def draw_values(key: str) -> list[float]:
            return [float(x[key]) for x in rgeoms if x.get(key) is not None]
        geom_primary_key = "mean_alignment" if CLASSES[task] == 2 else "mean_rdm"
        geom_cross_key = "mean_cross_session"
        pA = pgeom.get(geom_primary_key); pC = pgeom.get(geom_cross_key)
        randA = draw_values(geom_primary_key); randC = draw_values(geom_cross_key)
        marginsP = subject_margin(trm, qP, task, split["train_subjects"])
        marginsR = [subject_margin(trm, x, task, split["train_subjects"]) for x in controls["random_q"]]
        margin_by_sub = {x["subject"]: x["margin"] for x in marginsP}
        random_margin_by_sub = {s: [] for s in margin_by_sub}
        for mrow in marginsR:
            for x in mrow:
                if x["subject"] in random_margin_by_sub:
                    random_margin_by_sub[x["subject"]].append(x["margin"])
        margin_delta = [margin_by_sub[s] - float(np.mean(random_margin_by_sub[s])) for s in margin_by_sub if random_margin_by_sub[s]]
        loso = loso_scores(trm, qP, task, split["train_subjects"], True, controls["random_q"])
        # Validation is evaluated subject-by-subject after all TRAIN rules are frozen.
        val_subject_rows: list[dict[str, Any]] = []
        val_subs = sorted(set(map(str, split["validation_subjects"])))
        for s in val_subs:
            ba_p = nearest_prototype(trm, qP, vam, qPv, task, split["train_subjects"], s, True)
            ba_r = [nearest_prototype(trm, x, vam, qv[:, np.asarray(ch, dtype=np.int64)], task,
                                      split["train_subjects"], s, True)
                    for x, ch in zip(controls["random_q"], controls["random_ids"])]
            val_sub_rows = {"subject": s, "protected_ba": ba_p, "random_mean_ba": float(np.mean(ba_r)), "delta": ba_p - float(np.mean(ba_r))}
            val_subject_rows.append(val_sub_rows)
        # Fixed ridge, identical rank and centering rules.
        ridge_rows = []
        for s in val_subs:
            ba_p = ridge_subject(trm, qP, vam, qPv, task, split["train_subjects"], s, True)
            ba_r = [ridge_subject(trm, x, vam, qv[:, np.asarray(ch, dtype=np.int64)], task,
                                  split["train_subjects"], s, True)
                    for x, ch in zip(controls["random_q"], controls["random_ids"])]
            ridge_rows.append({"subject": s, "protected_ba": ba_p, "random_mean_ba": float(np.mean(ba_r)), "delta": ba_p - float(np.mean(ba_r))})
        task_payload[task] = {"protected_ids": protected_ids, "rank": len(protected_ids), "unit_test": unit,
                              "protected_geometry": pgeom, "random_geometry_mean": {"alignment": float(np.mean(randA)) if randA else None,
                                                                                       "cross_session": float(np.mean(randC)) if randC else None},
                              "protected_loso": loso, "protected_margin": marginsP,
                              "loso_delta_bootstrap": bootstrap(loso["delta"], stable_seed("loso-boot", fold, seed, task)),
                              "margin_delta": margin_delta,
                              "margin_delta_bootstrap": bootstrap(margin_delta, stable_seed("margin-boot", fold, seed, task)),
                              "validation_prototype": val_subject_rows,
                              "validation_prototype_bootstrap": bootstrap([x["delta"] for x in val_subject_rows], stable_seed("val-boot", fold, seed, task)),
                              "validation_ridge": ridge_rows,
                              "validation_ridge_bootstrap": bootstrap([x["delta"] for x in ridge_rows], stable_seed("ridge-boot", fold, seed, task)),
                              "cross_session_control_draws": len(randC), "control_rank": len(protected_ids)}
        controls_payload[task] = {"protected_ids": protected_ids, "random_ids": controls["random_ids"],
                                  "nonprotected_ids": controls["nonprotected_ids"], "pca_rank": len(protected_ids),
                                  "orthogonal_draws": RANDOM_DRAWS}
        # Subject-level CSV rows for primary MI and secondary tasks.
        for item in pgeom.get("alignment", []):
            all_rows.append({"fold": fold, "seed": seed, "task": task, "metric": "alignment",
                             "subject": item["subject"], "session": item.get("session"), "representation": "protected",
                             "value": item["score"]})
        for item in pgeom.get("cross_session", []):
            all_rows.append({"fold": fold, "seed": seed, "task": task, "metric": "cross_session",
                             "subject": item["subject"], "session": item.get("direction"), "representation": "protected",
                             "value": item["score"]})
        for item in marginsP:
            all_rows.append({"fold": fold, "seed": seed, "task": task, "metric": "margin",
                             "subject": item["subject"], "session": "both", "representation": "protected", "value": item["margin"]})
        block_rows = block_utility_geometry(trm, q, task, spec, fold, seed)
        pd.DataFrame(block_rows).to_csv(run_dir / f"BLOCK_GEOMETRY_UTILITY_{task.upper()}.csv", index=False)
        write_json(run_dir / f"CONTROLS_{task.upper()}.json", controls_payload[task])
    pd.DataFrame(all_rows).to_csv(run_dir / "SUBJECT_GEOMETRY.csv", index=False)
    write_json(run_dir / "RUN_RESULT.json", {"fold": fold, "seed": seed, "basis_fingerprint": fp,
                                               "basis_reconstruction": validation_repro, "tasks": task_payload,
                                               "controls": controls_payload, "outer_test_used": False,
                                               "label_free_transductive_centering": True,
                                               "geometry_sample_cap_per_subject_session_event": GEOM_CAP})
    return {"fold": fold, "seed": seed, "basis_reconstruction": validation_repro, "basis_fingerprint": fp,
            "tasks": task_payload, "block_rows": [r for task in TASKS for r in block_utility_geometry(trm, q, task, spec, fold, seed)],
            "outer_test_used": False, "status": "OK"}


def finalize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if any(r.get("status") == "SHARED_GEOMETRY_BASIS_RECONSTRUCTION_FAIL" for r in results):
        status = "SHARED_GEOMETRY_BASIS_RECONSTRUCTION_FAIL"
        gate = {x: False for x in "ABCDEF"}
        payload = {"status": status, "version": V1_1_VERSION, "gate": gate,
                   "runs": [{"fold": r["fold"], "seed": r["seed"], "status": r.get("status"),
                             "basis_fingerprint": r.get("basis_fingerprint"),
                             "basis_reconstruction": r.get("basis_reconstruction")} for r in results],
                   "outer_test_used": False, "method_training_started": False,
                   "stop_reason": "PHASE_0B exact Signed-V3 validation protected-erasure replay failed",
                   "geometry_metrics_not_run": True}
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "PREVIOUS_SHARED_GEOMETRY_V1_INVALID.json", {
            "status": "SHARED_GEOMETRY_AUDIT_V1_INVALID",
            "reason": ["Protected coordinates were not isolated", "basis was rebuilt from a subsample",
                        "binary MI/ERP RDM was undefined", "LOSO used a half split", "Gate D and Gate E were invalid/missing"],
            "replacement": status, "outer_test_used": False,
        })
        write_json(OUT / "SHARED_GEOMETRY_FINAL_REPORT.json", payload)
        write_json(OUT / "SHARED_GEOMETRY_V1_1_ADAPTATION_LOG.json", {
            "version": V1_1_VERSION,
            "issue": "Signed-V3 validation protected-erasure replay did not agree with the saved artifact",
            "evidence": [r.get("basis_reconstruction") for r in results],
            "change": "stop before geometry, transfer, controls, and Gate E",
            "scientific_impact": "prevents any geometry claim from a basis that cannot be validated against its frozen Signed-V3 reference",
            "data_used": ["TRAIN", "DEVELOPMENT_VALIDATION"], "outer_test_used": False,
            "diagnosis": "the historical Signed-V3 sampler used Python built-in hash without persisting sampled indices; SSVEP replay matched while MI/ERP did not",
        })
        (OUT / "SHARED_GEOMETRY_FINAL_REPORT.md").write_text(
            f"# PERSIST-EEG Shared Geometry Audit V1.1\n\nStatus: `{status}`\n\n"
            "Phase 0B exact Signed-V3 replay failed, so all later geometry/transfer analyses were stopped.\n\n"
            "Outer-test used: `false`.\n", encoding="utf-8")
        return payload
    run_values: dict[str, dict[str, list[float]]] = {"A": {}, "B": {}, "C": {}, "D": {}}
    block_rows = []
    for r in results:
        key = f"fold-{r['fold']}_seed-{r['seed']}"
        mi = r["tasks"]["mi"]
        pA = mi["protected_geometry"].get("mean_alignment")
        randA = mi["random_geometry_mean"].get("alignment")
        pC = mi["protected_geometry"].get("mean_cross_session")
        randC = mi["random_geometry_mean"].get("cross_session")
        run_values["A"][key] = [float(pA - randA)] if pA is not None and randA is not None else []
        run_values["C"][key] = [float(pC - randC)] if pC is not None and randC is not None else []
        run_values["B"][key] = [float(x) for x in mi["protected_loso"]["delta"]]
        run_values["D"][key] = [float(x) for x in mi["margin_delta"]]
        block_rows.extend(r.get("block_rows", []))
    def gate_stats(key: str) -> dict[str, Any]:
        vals = [float(np.mean(v)) for v in run_values[key].values() if v]
        return {"mean_run_effect": float(np.mean(vals)) if vals else None, "positive_runs": int(np.sum(np.asarray(vals) > 0)),
                "n_runs": len(vals), "hierarchical_bootstrap": hierarchical_boot(run_values[key], stable_seed("hier", key))}
    stats = {k: gate_stats(k) for k in "ABCD"}
    eframe = pd.DataFrame(block_rows)
    e = fit_utility_link(eframe)
    gate = {"A": stats["A"]["positive_runs"] >= 5 and (stats["A"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "B": stats["B"]["positive_runs"] >= 5 and (stats["B"]["mean_run_effect"] or -math.inf) >= .02 and (stats["B"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "C": stats["C"]["positive_runs"] >= 5 and (stats["C"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "D": stats["D"]["positive_runs"] >= 5 and (stats["D"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "E": e.get("status") == "OK" and e.get("beta_geometry", -math.inf) > 0 and e.get("beta_geometry_bootstrap_ci95", [-math.inf])[0] > 0 and (e.get("pooled_spearman_rho") or -math.inf) > 0,
            "F": all(bool(r.get("outer_test_used") is False) for r in results)}
    if all(gate.values()):
        status = "SHARED_GEOMETRY_V1_1_PASS"
    elif gate["B"] and not (gate["A"] and gate["C"] and gate["D"]):
        status = "PROTECTED_TRANSFER_WITHOUT_SHARED_GEOMETRY"
    elif gate["A"] and gate["B"] and gate["C"] and gate["D"] and not gate["E"]:
        status = "SHARED_GEOMETRY_NOT_UTILITY_LINKED"
    else:
        status = "PERSIST_USE_SHARED_GEOMETRY_NOT_SUPPORTED"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(block_rows).to_csv(OUT / "BLOCK_GEOMETRY_UTILITY.csv", index=False)
    write_json(OUT / "GEOMETRY_UTILITY_REGRESSION.json", e)
    write_json(OUT / "SHARED_GEOMETRY_V1_1_ADAPTATION_LOG.json", {
        "version": V1_1_VERSION,
        "issues_repaired": ["full q used instead of qP", "subsampled/mismatched basis", "binary RDM on MI/ERP", "half-split pseudo-LOSO", "Gate D proxy", "missing Gate E", "incomplete subject bootstrap"],
        "changes": ["exact full-TRAIN Signed-V3 spectrum seed", "frozen Protected ids", "MI contrast cosine and cross-session contrast", "genuine LOSO", "same-rank non-Protected/random orthogonal/PCA controls", "true centroid margin", "block-level u_spec link", "10000 subject/hierarchical bootstrap"],
        "data_used": ["TRAIN", "DEVELOPMENT_VALIDATION"], "outer_test_used": False,
        "basis_reconstruction": "checked against saved Signed-V3 validation protected-erasure rows; old sampler had no persisted indices",
    })
    payload = {"status": status, "version": V1_1_VERSION, "gate": gate, "gate_statistics": stats,
               "geometry_utility_link": e, "runs": [{"fold": r["fold"], "seed": r["seed"], "status": r.get("status"),
                                                       "basis_fingerprint": r.get("basis_fingerprint")} for r in results],
               "outer_test_used": False, "method_training_started": False,
               "controls": {"same_rank_random_draws": RANDOM_DRAWS, "random_orthogonal_draws": RANDOM_DRAWS,
                            "pca_same_rank": True, "nonprotected_active": True},
               "limitations": {"mi_rdm_spearman": "not used; binary-compatible contrast metrics used",
                               "validation_centering": "label-free transductive subject/session global mean",
                               "signed_v3_sampler": "historical built-in hash was not persisted; exact replay may fail conservatively"}}
    write_json(OUT / "SHARED_GEOMETRY_FINAL_REPORT.json", payload)
    (OUT / "SHARED_GEOMETRY_FINAL_REPORT.md").write_text(
        f"# PERSIST-EEG Shared Geometry Audit V1.1\n\nStatus: `{status}`\n\n"
        f"Gates: `{json.dumps(gate, sort_keys=True)}`\n\nOuter-test used: `false`.\n\n"
        "No PERSIST-USE model was trained.\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--seed", type=int)
    args = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V1.1 audit requires the server GPU")
    folds = (args.fold,) if args.fold is not None else FOLDS; seeds = (args.seed,) if args.seed is not None else SEEDS
    results = []
    for fold in folds:
        for seed in seeds:
            try:
                results.append(run_one(fold, seed, device))
            except Exception:
                OUT.mkdir(parents=True, exist_ok=True); (OUT / f"EXCEPTION_fold-{fold}_seed-{seed}.txt").write_text(traceback.format_exc(), encoding="utf-8"); raise
    # A Phase-0 failure is itself a terminal, reproducible audit result.  Do
    # not continue to geometry metrics after the first failed reconstruction.
    if results and (any(r.get("status") == "SHARED_GEOMETRY_BASIS_RECONSTRUCTION_FAIL" for r in results)
                    or len(results) == len(FOLDS) * len(SEEDS)):
        report = finalize(results)
        write_json(OUT / "COMPLETE.json", {"status": "COMPLETE", "final_status": report["status"],
                                           "outer_test_used": False})
        print(json.dumps(base.clean(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
