"""Shared Geometry Audit V1.2 using the canonical Signed-V3.1 artifacts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import p4_persist_ct as base
import shared_geometry_v1_1 as g


ROOT = base.ROOT
V31 = ROOT / "outputs" / "persist_eeg_p4_signed_v3_1"
OUT = ROOT / "outputs" / "persist_eeg_shared_geometry_v1_2"
TASKS = tuple(base.TASKS)
CLASSES = dict(base.CLASSES)
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
DRAW = 100
BOOT = 10_000
GEOMETRY_CAP = 32
VALIDATION_CAP = 32


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_spec(run_dir: Path) -> dict[str, Any]:
    z = np.load(run_dir / "spectrum" / "PERSISTENCE_SPECTRUM.npz", allow_pickle=False)
    return {"mean": z["mean"], "whitener": z["whitener"], "dewhitener": z["dewhitener"],
            "directions": z["directions"], "rho": z["rho"],
            "blocks": json.loads(str(z["blocks_json"].item())),
            "audit": json.loads(str(z["audit_json"].item()))}


def load_run_assignment(fold: int, seed: int) -> dict[str, Any]:
    p = V31 / "runs" / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_ASSIGNMENTS_V3_1.json"
    return json.loads(p.read_text(encoding="utf-8"))


def q_features(h: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return ((np.asarray(h, dtype=np.float64) - spec["mean"]) @ spec["whitener"] @ spec["directions"]).astype(np.float32)


def fsha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_uint64(*parts: Any) -> int:
    """Process/machine/date independent seed for V1.2 sampling."""
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def capped_indices(meta: pd.DataFrame, task: str, subjects: Sequence[str], fold: int,
                   seed: int, purpose: str, cap: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Stable subject/session/event cap; subjects remain the statistical unit."""
    paradigms = np.asarray(meta.paradigm.astype(str).to_numpy(), dtype=object)
    subject_values = np.asarray(meta.subject_id.astype(str).to_numpy(), dtype=object)
    session_values = np.asarray(meta.session_id.astype(str).to_numpy(), dtype=object)
    event_values = np.asarray(meta.event_label.astype(str).to_numpy(), dtype=object)
    allowed = set(map(str, subjects))
    mask = (paradigms == str(task)) & np.isin(subject_values, list(allowed))
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    global_values = meta["global_index"].to_numpy() if "global_index" in meta else np.arange(len(meta))
    keys = sorted(set(zip(subject_values[mask].tolist(), session_values[mask].tolist(), event_values[mask].tolist())))
    for subject, session, event in keys:
        idx = np.flatnonzero(mask & (subject_values == subject) & (session_values == session) & (event_values == event)).astype(np.int64)
        before = len(idx)
        if cap and len(idx) > cap:
            rng = np.random.default_rng(stable_uint64(fold, seed, task, subject, session, event, purpose))
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        selected.extend(map(int, idx))
        for row in idx:
            records.append({"frame_index": int(row), "global_index": int(global_values[int(row)]),
                            "task": str(task), "subject_id": str(subject), "session_id": str(session),
                            "event_label": str(event), "fold": int(fold), "seed": int(seed),
                            "purpose": str(purpose), "group_count_before": int(before),
                            "selected_count": int(len(idx))})
    out = np.asarray(sorted(selected), dtype=np.int64)
    return out, pd.DataFrame(records).sort_values("frame_index").reset_index(drop=True)


def save_cap(run_dir: Path, task: str, purpose: str, idx: np.ndarray, metadata: pd.DataFrame) -> None:
    d = run_dir / "sampling" / "geometry" / str(task)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{purpose}_indices.npy", np.asarray(idx, dtype=np.int64))
    metadata.to_csv(d / f"{purpose}_metadata.csv", index=False)


def plain_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """Avoid pandas/pyarrow string-array row-slice failures in transfer code."""
    cols = [c for c in ("subject_id", "session_id", "paradigm", "event_label") if c in meta.columns]
    out = pd.DataFrame({c: np.asarray(meta[c].astype(str).to_numpy(), dtype=object) for c in cols})
    return out.reset_index(drop=True)


@torch.inference_mode()
def extract_selected(model: torch.nn.Module, manifest: pd.DataFrame, selected: Mapping[str, np.ndarray],
                     mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[pd.DataFrame, np.ndarray]:
    """Encode only the deterministic geometry cap, never the full trial table."""
    maps = base.label_maps(manifest)
    metadata, embeddings = [], []
    for task in TASKS:
        idx = np.asarray(selected[task], dtype=np.int64)
        ds = base.TrialDataset(manifest, idx, mean, std, maps[task])
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
        bh, bi = [], []
        for x, _, gi in loader:
            bh.append(model.encoder(x.to(device, non_blocking=True)).float().cpu().numpy())
            bi.append(gi.numpy())
        joined = np.concatenate(bi).astype(np.int64)
        frame = manifest[["subject_id", "session_id", "paradigm", "event_label"]].iloc[joined].copy().reset_index(drop=True)
        frame["global_index"] = joined
        metadata.append(frame); embeddings.append(np.concatenate(bh).astype(np.float32))
    return plain_metadata(pd.concat(metadata, ignore_index=True)), np.concatenate(embeddings)


def permuted_meta(meta: pd.DataFrame, task: str, seed: int) -> pd.DataFrame:
    out = plain_metadata(meta)
    paradigm = np.asarray(out.paradigm.astype(str).to_numpy(), dtype=object)
    subjects = np.asarray(out.subject_id.astype(str).to_numpy(), dtype=object)
    sessions = np.asarray(out.session_id.astype(str).to_numpy(), dtype=object)
    events = np.asarray(out.event_label.astype(str).to_numpy(), dtype=object)
    mask = paradigm == task
    rng = np.random.default_rng(g.stable_seed("label-permutation", seed, task))
    keys = sorted(set(zip(subjects[mask].tolist(), sessions[mask].tolist())))
    for subject, session in keys:
        idx = np.flatnonzero(mask & (subjects == subject) & (sessions == session))
        vals = events[idx].copy()
        rng.shuffle(vals)
        events[idx] = vals
    out["event_label"] = events
    return out


def geometry_from_centroids(cs: dict[tuple[str, str], np.ndarray], task: str) -> dict[str, Any]:
    """Same geometry definitions as V1.1, but reuse centroids for 100 controls."""
    subs = sorted(set(s for s, _ in cs)); ses = sorted(set(r for _, r in cs))
    if CLASSES[task] == 2:
        d = {k: v[1] - v[0] for k, v in cs.items() if np.isfinite(v).all()}
        align = []; cross = []; energy = []
        for s in subs:
            for r in ses:
                if (s, r) not in d: continue
                other = [d[(so, r)] for so in subs if so != s and (so, r) in d]
                if other:
                    c = np.mean(other, axis=0); align.append({"subject": s, "session": r, "score": g.cosine(d[(s, r)], c)})
                    den = float(d[(s, r)] @ d[(s, r)]); energy.append({"subject": s, "session": r, "score": float((d[(s, r)] @ c) ** 2 / max(float(c @ c) * den, 1e-12))})
            if len(ses) >= 2 and all((s, r) in d for r in ses[:2]):
                for target, source in ((ses[1], ses[0]), (ses[0], ses[1])):
                    other = [d[(so, source)] for so in subs if so != s and (so, source) in d]
                    if other: cross.append({"subject": s, "direction": f"{source}->{target}", "score": g.cosine(d[(s, target)], np.mean(other, axis=0))})
        return {"alignment": align, "cross_session": cross, "shared_energy": energy, "mean_alignment": float(np.mean([x["score"] for x in align])) if align else None, "mean_cross_session": float(np.mean([x["score"] for x in cross])) if cross else None, "rdm_defined": False}
    direct = []; rdm_rows = []; cross = []
    for s in subs:
        for r in ses:
            if (s, r) not in cs or not np.isfinite(cs[(s, r)]).all(): continue
            other = [cs[(so, r)] for so in subs if so != s and (so, r) in cs and np.isfinite(cs[(so, r)]).all()]
            if other:
                c = np.mean(other, axis=0); direct.append({"subject": s, "session": r, "score": g.cosine(cs[(s, r)], c)})
                z = g.spearman_safe(g.rdm(cs[(s, r)]), g.rdm(c))
                if z is not None: rdm_rows.append({"subject": s, "session": r, "score": z})
        if len(ses) >= 2:
            for target, source in ((ses[1], ses[0]), (ses[0], ses[1])):
                other = [cs[(so, source)] for so in subs if so != s and (so, source) in cs and np.isfinite(cs[(so, source)]).all()]
                if (s, target) in cs and other:
                    z = g.spearman_safe(g.rdm(cs[(s, target)]), g.rdm(np.mean(other, axis=0)))
                    if z is not None: cross.append({"subject": s, "direction": f"{source}->{target}", "score": z})
    return {"alignment": direct, "rdm": rdm_rows, "cross_session": cross, "mean_alignment": float(np.mean([x["score"] for x in direct])) if direct else None, "mean_rdm": float(np.mean([x["score"] for x in rdm_rows])) if rdm_rows else None, "mean_cross_session": float(np.mean([x["score"] for x in cross])) if cross else None, "rdm_defined": bool(rdm_rows)}


def margin_from_centroids(cs: dict[tuple[str, str], np.ndarray], task: str, subjects: Sequence[str]) -> list[dict[str, Any]]:
    by_sub: dict[str, np.ndarray] = {}
    for s in sorted(set(map(str, subjects))):
        vals = [v for (ss, _), v in cs.items() if ss == s and np.isfinite(v).all()]
        if vals: by_sub[s] = np.mean(vals, axis=0)
    out = []
    for s, own in by_sub.items():
        other = [by_sub[so] for so in by_sub if so != s]
        if not other: continue
        cons = np.mean(other, axis=0); same = []; diff = []
        for y in range(CLASSES[task]):
            same.append(float(np.linalg.norm(own[y] - cons[y])))
            diff.extend(float(np.linalg.norm(own[y] - cons[yp])) for yp in range(CLASSES[task]) if yp != y)
        out.append({"subject": s, "same_class_distance": float(np.mean(same)), "different_class_distance": float(np.mean(diff)), "margin": float(np.mean(diff) - np.mean(same))})
    return out


def fast_geometry(meta: pd.DataFrame, q: np.ndarray, task: str,
                  subjects: Sequence[str]) -> tuple[dict[str, Any], dict[tuple[str, str], np.ndarray]]:
    """Geometry from class centroids; avoids rescanning trials for controls."""
    cs = class_centroids_numpy(meta, q, task, subjects)
    return geometry_from_centroids(cs, task), cs


def class_centroids_numpy(meta: pd.DataFrame, q: np.ndarray, task: str,
                          subjects: Sequence[str]) -> dict[tuple[str, str], np.ndarray]:
    """Arrow-independent equivalent of V1.1 class_centroids."""
    subjects_arr = np.asarray(meta.subject_id.astype(str).to_numpy(), dtype=object)
    sessions_arr = np.asarray(meta.session_id.astype(str).to_numpy(), dtype=object)
    paradigms_arr = np.asarray(meta.paradigm.astype(str).to_numpy(), dtype=object)
    events_arr = np.asarray(meta.event_label.astype(str).to_numpy(), dtype=object)
    allowed = set(map(str, subjects)); mask = (paradigms_arr == str(task)) & np.isin(subjects_arr, list(allowed))
    labels = sorted(set(events_arr[mask].tolist()))
    if len(labels) != CLASSES[task]:
        raise RuntimeError(f"unexpected labels for {task}: {labels}")
    mapping = {label: i for i, label in enumerate(labels)}
    out: dict[tuple[str, str], np.ndarray] = {}
    keys = sorted(set(zip(subjects_arr[mask].tolist(), sessions_arr[mask].tolist())))
    x = np.asarray(q)
    for subject, session in keys:
        loc = np.flatnonzero(mask & (subjects_arr == subject) & (sessions_arr == session))
        group_x = np.asarray(x[loc], dtype=np.float64)
        centered = group_x - group_x.mean(axis=0, keepdims=True)
        y = np.asarray([mapping[str(v)] for v in events_arr[loc]], dtype=np.int64)
        centroids = [centered[y == k].mean(axis=0) if np.any(y == k) else np.full(x.shape[1], np.nan)
                     for k in range(CLASSES[task])]
        out[(str(subject), str(session))] = np.asarray(centroids, dtype=np.float64)
    return out


def label_map_numpy(meta: pd.DataFrame, task: str) -> dict[str, int]:
    paradigms = np.asarray(meta.paradigm.astype(str).to_numpy(), dtype=object)
    events = np.asarray(meta.event_label.astype(str).to_numpy(), dtype=object)
    labels = sorted(set(events[paradigms == str(task)].tolist()))
    if len(labels) != CLASSES[task]:
        raise RuntimeError(f"unexpected labels for {task}: {labels}")
    return {label: i for i, label in enumerate(labels)}


def protected_invariance_test(meta: pd.DataFrame, q: np.ndarray, protected_ids: Sequence[int], task: str,
                              run_seed: int) -> dict[str, Any]:
    ids = np.asarray(protected_ids, dtype=np.int64)
    non = np.asarray([i for i in range(q.shape[1]) if i not in set(ids)], dtype=np.int64)
    base_metric = fast_geometry(meta, q[:, ids], task, sorted(set(meta.subject_id.astype(str))))[0]
    pert = np.asarray(q, dtype=np.float32).copy()
    if len(non):
        rng = np.random.default_rng(run_seed); pert[:, non] += rng.normal(0, 1000, size=(len(pert), len(non))).astype(np.float32)
    pert_metric = fast_geometry(meta, pert[:, ids], task, sorted(set(meta.subject_id.astype(str))))[0]
    a = json.dumps(base.clean(base_metric), sort_keys=True); b = json.dumps(base.clean(pert_metric), sort_keys=True)
    return {"qP_dimension": int(len(ids)), "protected_id_count": int(len(ids)),
            "nonprotected_perturbation_applied": bool(len(non)), "passed": bool(a == b),
            "base_sha256": hashlib.sha256(a.encode()).hexdigest(),
            "perturbed_sha256": hashlib.sha256(b.encode()).hexdigest()}


def batch_class_centroids_numpy(meta: pd.DataFrame, representations: Sequence[np.ndarray], task: str,
                                subjects: Sequence[str]) -> tuple[list[dict[tuple[str, str], np.ndarray]], np.ndarray]:
    """Compute all matched-control centroids in one bounded NumPy pass."""
    subjects_arr = np.asarray(meta.subject_id.astype(str).to_numpy(), dtype=object)
    sessions_arr = np.asarray(meta.session_id.astype(str).to_numpy(), dtype=object)
    paradigms_arr = np.asarray(meta.paradigm.astype(str).to_numpy(), dtype=object)
    events_arr = np.asarray(meta.event_label.astype(str).to_numpy(), dtype=object)
    allowed = set(map(str, subjects)); mask = (paradigms_arr == str(task)) & np.isin(subjects_arr, list(allowed))
    labels = sorted(set(events_arr[mask].tolist()))
    if len(labels) != CLASSES[task]:
        raise RuntimeError(f"unexpected labels for {task}: {labels}")
    mapping = {label: i for i, label in enumerate(labels)}
    keys = sorted(set(zip(subjects_arr[mask].tolist(), sessions_arr[mask].tolist())))
    stack = np.stack([np.asarray(x, dtype=np.float32) for x in representations], axis=0)
    cent_stack = np.full((len(representations), len(keys), CLASSES[task], stack.shape[2]), np.nan, dtype=np.float64)
    for gi, (subject, session) in enumerate(keys):
        loc = np.flatnonzero(mask & (subjects_arr == subject) & (sessions_arr == session))
        group_x = stack[:, loc, :].astype(np.float64)
        centered = group_x - group_x.mean(axis=1, keepdims=True)
        y = np.asarray([mapping[str(v)] for v in events_arr[loc]], dtype=np.int64)
        for k in range(CLASSES[task]):
            if np.any(y == k):
                cent_stack[:, gi, k, :] = centered[:, y == k, :].mean(axis=1)
    outputs: list[dict[tuple[str, str], np.ndarray]] = []
    for ri in range(len(representations)):
        outputs.append({(str(subject), str(session)): cent_stack[ri, gi]
                        for gi, (subject, session) in enumerate(keys)})
    return outputs, cent_stack


def batch_fast_geometry(meta: pd.DataFrame, representations: Sequence[np.ndarray], task: str,
                        subjects: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[tuple[str, str], np.ndarray]]]:
    centroids, _ = batch_class_centroids_numpy(meta, representations, task, subjects)
    return [geometry_from_centroids(cs, task) for cs in centroids], centroids


def _center_representation(meta: pd.DataFrame, q: np.ndarray, target_center: bool) -> np.ndarray:
    x = np.asarray(q, dtype=np.float64).copy()
    if target_center:
        subjects = np.asarray(meta.subject_id.astype(str).to_numpy(), dtype=object)
        sessions = np.asarray(meta.session_id.astype(str).to_numpy(), dtype=object)
        for subject, session in sorted(set(zip(subjects.tolist(), sessions.tolist()))):
            loc = np.flatnonzero((subjects == subject) & (sessions == session))
            x[loc] -= x[loc].mean(axis=0, keepdims=True)
    return x


def loso_scores_fast(meta: pd.DataFrame, q: np.ndarray, task: str, subjects: Sequence[str],
                     target_center: bool, draw_controls: Sequence[np.ndarray]) -> dict[str, Any]:
    """True subject-disjoint LOSO with the same nearest-prototype protocol.

    The implementation precomputes each representation's subject/session
    centering once, then evaluates every held-out subject exactly once. This
    is numerically equivalent to the historical helper but avoids 100 repeated
    dataframe scans per subject.
    """
    frame = meta.reset_index(drop=True).copy()
    subs = sorted(set(map(str, subjects)))
    frame["_subject"] = frame.subject_id.astype(str)
    mapping = label_map_numpy(frame, task)
    y = frame.event_label.astype(str).map(mapping).to_numpy(dtype=np.int64)
    subject_locs = {s: np.flatnonzero(frame["_subject"].to_numpy() == s) for s in subs}
    representations = [np.asarray(q)] + [np.asarray(x) for x in draw_controls]
    all_ba = np.full((len(representations), len(subs)), np.nan, dtype=np.float64)
    for ri, rep in enumerate(representations):
        x = _center_representation(frame, rep, target_center)
        for si, subject in enumerate(subs):
            test = subject_locs[subject]
            train = np.concatenate([subject_locs[s] for s in subs if s != subject])
            protos = np.asarray([x[train][y[train] == k].mean(axis=0) for k in range(CLASSES[task])])
            pred = np.argmin(((x[test, None, :] - protos[None, :, :]) ** 2).sum(axis=2), axis=1)
            per_class = [np.mean(pred[y[test] == k] == k) for k in range(CLASSES[task]) if np.any(y[test] == k)]
            all_ba[ri, si] = float(np.mean(per_class)) if per_class else np.nan
    rows = []
    for si, subject in enumerate(subs):
        random_mean = float(np.nanmean(all_ba[1:, si])) if len(draw_controls) else None
        row = {"subject": subject, "protected_ba": float(all_ba[0, si])}
        if random_mean is not None:
            row["random_mean_ba"] = random_mean
            row["delta"] = row["protected_ba"] - random_mean
        rows.append(row)
    delta = [float(x["delta"]) for x in rows if "delta" in x and np.isfinite(x["delta"])]
    return {"subjects": rows, "delta": delta,
            "protected_mean": float(np.nanmean(all_ba[0])) if len(all_ba) else None,
            "target_center": bool(target_center), "n_controls": int(len(draw_controls))}


def _balanced_accuracy(pred: np.ndarray, y: np.ndarray, classes: int) -> float:
    vals = [np.mean(pred[y == k] == k) for k in range(classes) if np.any(y == k)]
    return float(np.mean(vals)) if vals else float("nan")


def transfer_prototype_aligned(tm: pd.DataFrame, em: pd.DataFrame, ytr: np.ndarray, yev: np.ndarray,
                                train_subjects: Sequence[str], eval_subjects: Sequence[str],
                                target_center: bool, train_reps: Sequence[np.ndarray],
                                eval_reps: Sequence[np.ndarray], classes: int) -> list[dict[str, Any]]:
    """Internal aligned-representation implementation for prototype transfer."""
    subjects = sorted(set(map(str, eval_subjects)))
    locs = {s: np.flatnonzero(em.subject_id.astype(str).to_numpy() == s) for s in subjects}
    outputs = [[ ] for _ in subjects]
    for tr, ev in zip(train_reps, eval_reps):
        xt = _center_representation(tm, tr, target_center); xe = _center_representation(em, ev, target_center)
        protos = np.asarray([xt[ytr == k].mean(axis=0) for k in range(classes)])
        for si, subject in enumerate(subjects):
            loc = locs[subject]; pred = np.argmin(((xe[loc, None, :] - protos[None, :, :]) ** 2).sum(axis=2), axis=1)
            outputs[si].append(_balanced_accuracy(pred, yev[loc], protos.shape[0]))
    return [{"subject": s, "protected_ba": float(vals[0]), "random_mean_ba": float(np.mean(vals[1:])),
             "delta": float(vals[0] - np.mean(vals[1:]))} for s, vals in zip(subjects, outputs)]


def transfer_ridge_aligned(tm: pd.DataFrame, em: pd.DataFrame, ytr: np.ndarray, yev: np.ndarray,
                           eval_subjects: Sequence[str], target_center: bool,
                           train_reps: Sequence[np.ndarray], eval_reps: Sequence[np.ndarray],
                           classes: int) -> list[dict[str, Any]]:
    subjects = sorted(set(map(str, eval_subjects)))
    locs = {s: np.flatnonzero(em.subject_id.astype(str).to_numpy() == s) for s in subjects}
    outputs = [[] for _ in subjects]
    for tr, ev in zip(train_reps, eval_reps):
        xt = _center_representation(tm, tr, target_center); xe = _center_representation(em, ev, target_center)
        pack = base.ridge_probe(xt, ytr, classes)
        for si, subject in enumerate(subjects):
            loc = locs[subject]; pred, _ = base.probe_predict(xe[loc], pack, classes)
            outputs[si].append(_balanced_accuracy(pred, yev[loc], classes))
    return [{"subject": s, "protected_ba": float(vals[0]), "random_mean_ba": float(np.mean(vals[1:])),
             "delta": float(vals[0] - np.mean(vals[1:]))} for s, vals in zip(subjects, outputs)]


def _metric_rows(geom: dict[str, Any], metric: str) -> dict[tuple[str, str], float]:
    rows = geom.get(metric, [])
    key_name = "direction" if metric == "cross_session" else "session"
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        value = row.get("score")
        if value is not None and np.isfinite(value):
            out[(str(row["subject"]), str(row.get(key_name, "both")))] = float(value)
    return out


def geometry_contrasts(protected: dict[str, Any], controls: Sequence[dict[str, Any]],
                      task: str, fold: int, seed: int, representation: str = "random") -> list[dict[str, Any]]:
    """Per-subject/session protected-minus-matched-control contrasts."""
    rows: list[dict[str, Any]] = []
    for metric in ("alignment", "rdm", "cross_session"):
        p = _metric_rows(protected, metric)
        if not p:
            continue
        controls_by_key = [_metric_rows(x, metric) for x in controls]
        for key, value in sorted(p.items()):
            null = [c[key] for c in controls_by_key if key in c and np.isfinite(c[key])]
            if not null:
                continue
            rows.append({"fold": int(fold), "seed": int(seed), "task": str(task), "metric": metric,
                         "subject": key[0], "session_or_direction": key[1],
                         "protected": float(value), "random_mean": float(np.mean(null)),
                         "delta": float(value - np.mean(null)), "representation": representation,
                         "n_controls": int(len(null))})
    return rows


def fit_utility_link_numpy(frame: pd.DataFrame, seed: int = 12345) -> dict[str, Any]:
    """Arrow-independent run-clustered utility regression.

    This is the same prespecified model as V1.1, but builds all design
    matrices from NumPy arrays.  The server's pandas/pyarrow combination can
    crash during repeated ``pd.concat`` in the 10,000-draw bootstrap.
    """
    if frame.empty:
        return {"status": "INSUFFICIENT_BLOCKS", "n": 0}
    task = np.asarray(frame["task"].astype(str).to_numpy(), dtype=object)
    cols = {name: np.asarray(frame[name].to_numpy(dtype=float), dtype=np.float64)
            for name in ("geometry_primary", "persistence_strength", "dimensions", "u_spec")}
    fold = np.asarray(frame["fold"].to_numpy(dtype=np.int64)); run_seed = np.asarray(frame["seed"].to_numpy(dtype=np.int64))
    valid = (task == "mi")
    for arr in cols.values(): valid &= np.isfinite(arr)
    if int(valid.sum()) < 10:
        return {"status": "INSUFFICIENT_BLOCKS", "n": int(valid.sum())}
    gval, rval, dval, y = (cols["geometry_primary"][valid], cols["persistence_strength"][valid],
                           cols["dimensions"][valid], cols["u_spec"][valid])
    runs = np.asarray([f"{f}_{s}" for f, s in zip(fold[valid], run_seed[valid])], dtype=object)
    run_keys = sorted(set(runs.tolist())); run_index = {key: i for i, key in enumerate(run_keys)}
    ri = np.asarray([run_index[x] for x in runs], dtype=np.int64); n_runs = len(run_keys)
    def zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / max(float(x.std()), 1e-12)
    gg, rr, dd = zscore(gval), zscore(rval), zscore(dval)
    one = np.column_stack([ri == k for k in range(1, n_runs)]) if n_runs > 1 else np.zeros((len(y), 0))
    X = np.column_stack([np.ones(len(y)), gg, rr, dd, one])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_g = float(beta[1])
    group_xtx = []; group_xty = []
    for k in range(n_runs):
        ix = np.flatnonzero(ri == k); xk = X[ix]
        group_xtx.append(xk.T @ xk); group_xty.append(xk.T @ y[ix])
    rng = np.random.default_rng(seed); boot = np.empty(BOOT, dtype=np.float64)
    group_xtx_arr = np.asarray(group_xtx); group_xty_arr = np.asarray(group_xty)
    for b in range(BOOT):
        counts = np.bincount(rng.integers(0, n_runs, size=n_runs), minlength=n_runs).astype(np.float64)
        xtx = np.tensordot(counts, group_xtx_arr, axes=(0, 0)); xty = np.tensordot(counts, group_xty_arr, axes=(0, 0))
        boot[b] = float((np.linalg.pinv(xtx, rcond=1e-10) @ xty)[1])
    rho = g.spearman_safe(gval, y)
    return {"status": "OK", "n_blocks": int(len(y)), "beta_geometry": beta_g,
            "beta_geometry_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "beta_geometry_sign_probability": float(np.mean(boot > 0)), "pooled_spearman_rho": rho,
            "bootstrap_draws": BOOT, "formula": "u_spec ~ geometry + persistence_strength + block_dimension + run_effects",
            "implementation": "numpy_run_clustered_bootstrap"}


def run_one(fold: int, seed: int, device: torch.device) -> dict[str, Any]:
    manifest = base.load_manifest(); split = next(x for x in base.load_splits() if int(x["fold"]) == fold)
    ckpt, mean, std = base.historical(fold, seed); model = base.load_model(ckpt, manifest, device)
    run_dir = OUT / "runs" / f"fold-{fold}" / f"seed-{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    train_selected: dict[str, np.ndarray] = {}; validation_selected: dict[str, np.ndarray] = {}
    for task in TASKS:
        train_selected[task], train_records = capped_indices(manifest, task, split["train_subjects"], fold, seed, "train_geometry", GEOMETRY_CAP)
        validation_selected[task], validation_records = capped_indices(manifest, task, split["validation_subjects"], fold, seed, "validation_geometry", VALIDATION_CAP)
        save_cap(run_dir, task, "train_geometry", train_selected[task], train_records)
        save_cap(run_dir, task, "validation_geometry", validation_selected[task], validation_records)
    print(f"[SG-V1.2] fold={fold} seed={seed} extracting deterministic TRAIN/VALIDATION caps", flush=True)
    trm, trh = extract_selected(model, manifest, train_selected, mean, std, device)
    vam, vah = extract_selected(model, manifest, validation_selected, mean, std, device)
    # The historical encoder and manifest are no longer needed after capped
    # embeddings are materialized; release them before control geometry.
    del model, manifest, train_selected, validation_selected
    gc.collect(); torch.cuda.empty_cache()
    canonical = V31 / "runs" / f"fold-{fold}" / f"seed-{seed}" / "spectrum" / "PERSISTENCE_SPECTRUM.npz"
    spec = load_spec(V31 / "runs" / f"fold-{fold}" / f"seed-{seed}")
    saved_fp = json.loads((V31 / "runs" / f"fold-{fold}" / f"seed-{seed}" / "spectrum" / "PERSISTENCE_SPECTRUM_FINGERPRINT.json").read_text(encoding="utf-8"))
    assignments = load_run_assignment(fold, seed)
    q = q_features(trh, spec); qv = q_features(vah, spec)
    task_results: dict[str, Any] = {}; subject_rows: list[dict[str, Any]] = []; contrast_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []; pca_rows: list[dict[str, Any]] = []; non_rows: list[dict[str, Any]] = []
    orth_rows: list[dict[str, Any]] = []; perm_rows: list[dict[str, Any]] = []; block_rows: list[dict[str, Any]] = []
    run_block_utility = pd.read_csv(V31 / "runs" / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_UTILITY_V3_1.csv")
    for task in TASKS:
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} start", flush=True)
        train_pos = np.flatnonzero(np.asarray(trm.paradigm.astype(str).to_numpy(), dtype=object) == task)
        validation_pos = np.flatnonzero(np.asarray(vam.paradigm.astype(str).to_numpy(), dtype=object) == task)
        tm = plain_metadata(trm.iloc[train_pos]); tq = q[train_pos]; th = trh[train_pos]
        vm = plain_metadata(vam.iloc[validation_pos]); vq = qv[validation_pos]
        train_subjects = sorted(set(map(str, split["train_subjects"])))
        validation_subjects = sorted(set(map(str, split["validation_subjects"])))
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} subset", flush=True)
        protected_blocks = assignments[task]["protected"]
        protected_ids = sorted(set(sum((spec["blocks"][b] for b in protected_blocks), [])))
        if not protected_ids:
            if task == "mi":
                raise RuntimeError(f"V3.1 has no Protected ids for primary MI fold={fold} seed={seed}")
            task_results[task] = {"status": "NO_PROTECTED_ASSIGNMENT", "protected_blocks": [], "protected_ids": [],
                                  "rank": 0, "unit_test": {"passed": True, "not_applicable": True},
                                  "random_draws": 0}
            print(f"[SG-V1.2] fold={fold} seed={seed} task={task} no Protected assignment; secondary report only", flush=True)
            continue
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} ids={protected_ids}", flush=True)
        ctrl = g.make_controls(tq, th, spec, protected_ids, stable_uint64("v1.2-controls", fold, seed, task))
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} controls", flush=True)
        qP = tq[:, protected_ids]; qPv = vq[:, protected_ids]
        unit = protected_invariance_test(tm, tq, protected_ids, task, stable_uint64("v1.2-unit", fold, seed, task))
        if not unit["passed"]:
            raise RuntimeError(f"Protected perturbation unit test failed for {task}")
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} unit", flush=True)
        pgeom, pcs = fast_geometry(tm, qP, task, train_subjects)
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} pgeom", flush=True)
        rgeom, random_centroids = batch_fast_geometry(tm, ctrl["random_q"], task, train_subjects)
        margin_r = [margin_from_centroids(rcs, task, train_subjects) for rcs in random_centroids]
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} rgeom", flush=True)
        orthogonal_q = ctrl.pop("orthogonal_q")
        ogeom: list[dict[str, Any]] = []
        for start in range(0, len(orthogonal_q), 10):
            chunk_geom, _ = batch_fast_geometry(tm, orthogonal_q[start:start + 10], task, train_subjects)
            ogeom.extend(chunk_geom)
            del chunk_geom
            gc.collect()
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} ogeom", flush=True)
        pkey = "mean_alignment" if CLASSES[task] == 2 else "mean_rdm"
        ckey = "mean_cross_session"
        random_primary = [x[pkey] for x in rgeom if x.get(pkey) is not None]
        random_cross = [x[ckey] for x in rgeom if x.get(ckey) is not None]
        orth_primary = [x[pkey] for x in ogeom if x.get(pkey) is not None]
        orth_cross = [x[ckey] for x in ogeom if x.get(ckey) is not None]
        margin_p = margin_from_centroids(pcs, task, train_subjects)
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} margin", flush=True)
        rmargin = {str(x["subject"]): [] for x in margin_p}
        pmargin = {str(x["subject"]): float(x["margin"]) for x in margin_p}
        for null_block in margin_r:
            for x in null_block:
                if str(x["subject"]) in rmargin: rmargin[str(x["subject"])].append(float(x["margin"]))
        margin_delta = [pmargin[s] - float(np.mean(rmargin[s])) for s in pmargin if rmargin[s]]
        loso = loso_scores_fast(tm, qP, task, train_subjects, True, ctrl["random_q"])
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} loso", flush=True)
        loso_ind = loso_scores_fast(tm, qP, task, train_subjects, False, ctrl["random_q"])
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} losoi", flush=True)
        mapping = label_map_numpy(tm, task)
        ytr = tm.event_label.astype(str).map(mapping).to_numpy(dtype=np.int64)
        yv = vm.event_label.astype(str).map(mapping).to_numpy(dtype=np.int64)
        train_reps = [qP] + list(ctrl["random_q"])
        eval_reps = [qPv] + [vq[:, np.asarray(ch, dtype=np.int64)] for ch in ctrl["random_ids"]]
        val_proto = transfer_prototype_aligned(tm, vm, ytr, yv, train_subjects, validation_subjects, True, train_reps, eval_reps, CLASSES[task])
        val_proto_ind = transfer_prototype_aligned(tm, vm, ytr, yv, train_subjects, validation_subjects, False, train_reps, eval_reps, CLASSES[task])
        val_ridge = transfer_ridge_aligned(tm, vm, ytr, yv, validation_subjects, True, train_reps, eval_reps, CLASSES[task])
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} transfer", flush=True)
        # PCA and non-Protected controls are reported even though the primary
        # gates use the matched random coordinate null.
        pca_geom = fast_geometry(tm, ctrl["pca_q"], task, train_subjects)[0]
        non_q = tq[:, np.asarray(ctrl["nonprotected_ids"], dtype=np.int64)]
        non_geom = fast_geometry(tm, non_q, task, train_subjects)[0]
        perm_meta = permuted_meta(tm, task, stable_uint64("perm", fold, seed, task))
        perm_geom = fast_geometry(perm_meta, qP, task, train_subjects)[0]
        contrasts = geometry_contrasts(pgeom, rgeom, task, fold, seed)
        contrast_rows.extend(contrasts)
        task_results[task] = {
            "protected_blocks": protected_blocks, "protected_ids": protected_ids, "rank": len(protected_ids), "unit_test": unit,
            "protected_geometry": pgeom, "random_geometry_mean": {"primary": float(np.mean(random_primary)) if random_primary else None, "cross_session": float(np.mean(random_cross)) if random_cross else None},
            "orthogonal_geometry_mean": {"primary": float(np.mean(orth_primary)) if orth_primary else None, "cross_session": float(np.mean(orth_cross)) if orth_cross else None},
            "pca_geometry": pca_geom, "nonprotected_geometry": non_geom, "label_permutation_geometry": perm_geom,
            "protected_margin": margin_p, "margin_delta": margin_delta, "margin_delta_bootstrap": g.bootstrap(margin_delta, g.stable_seed("margin-boot", fold, seed, task)),
            "loso": loso, "loso_inductive": loso_ind, "loso_delta_bootstrap": g.bootstrap(loso["delta"], g.stable_seed("loso-boot", fold, seed, task)),
            "validation_prototype": val_proto, "validation_prototype_inductive": val_proto_ind, "validation_ridge": val_ridge,
            "validation_delta_bootstrap": g.bootstrap([x["delta"] for x in val_proto], g.stable_seed("val-boot", fold, seed, task)),
            "validation_inductive_delta_bootstrap": g.bootstrap([x["delta"] for x in val_proto_ind], g.stable_seed("val-ind-boot", fold, seed, task)),
            "validation_ridge_delta_bootstrap": g.bootstrap([x["delta"] for x in val_ridge], g.stable_seed("ridge-boot", fold, seed, task)),
            "random_draws": DRAW,
        }
        for row in pgeom.get("alignment", []): subject_rows.append({"fold": fold, "seed": seed, "task": task, "metric": "alignment", "subject": row["subject"], "session": row.get("session"), "representation": "protected", "value": row["score"]})
        for row in pgeom.get("rdm", []): subject_rows.append({"fold": fold, "seed": seed, "task": task, "metric": "rdm", "subject": row["subject"], "session": row.get("session"), "representation": "protected", "value": row["score"]})
        for row in pgeom.get("cross_session", []): subject_rows.append({"fold": fold, "seed": seed, "task": task, "metric": "cross_session", "subject": row["subject"], "session": row.get("direction"), "representation": "protected", "value": row["score"]})
        for row in margin_p: subject_rows.append({"fold": fold, "seed": seed, "task": task, "metric": "margin", "subject": row["subject"], "session": "both", "representation": "protected", "value": row["margin"]})
        random_rows.append({"fold": fold, "seed": seed, "task": task, "rank": len(protected_ids), "draws": DRAW, "primary_mean": float(np.mean(random_primary)) if random_primary else None, "cross_session_mean": float(np.mean(random_cross)) if random_cross else None, "margin_mean": float(np.mean([x["margin"] for block in margin_r for x in block])) if margin_r else None})
        pca_rows.append({"fold": fold, "seed": seed, "task": task, "rank": len(protected_ids), "primary": pca_geom.get(pkey), "cross_session": pca_geom.get(ckey)})
        non_rows.append({"fold": fold, "seed": seed, "task": task, "dimensions": len(ctrl["nonprotected_ids"]), "primary": non_geom.get(pkey), "cross_session": non_geom.get(ckey)})
        orth_rows.append({"fold": fold, "seed": seed, "task": task, "rank": len(protected_ids), "draws": DRAW, "primary_mean": float(np.mean(orth_primary)) if orth_primary else None, "cross_session_mean": float(np.mean(orth_cross)) if orth_cross else None})
        perm_rows.append({"fold": fold, "seed": seed, "task": task, "primary": perm_geom.get(pkey), "cross_session": perm_geom.get(ckey)})
        for _, ur in run_block_utility[run_block_utility.task == task].iterrows():
            bi = int(ur.block); qb = tq[:, spec["blocks"][bi]]; bg = fast_geometry(tm, qb, task, train_subjects)[0]; primary = bg.get(pkey)
            block_rows.append({"fold": fold, "seed": seed, "task": task, "block": bi, "dimensions": len(spec["blocks"][bi]),
                               "persistence_strength": float(np.mean(np.maximum(np.asarray(spec["rho"])[spec["blocks"][bi]], 0.0))),
                               "geometry_primary": primary, "geometry_cross_session": bg.get(ckey),
                               "u_abs": float(ur.u_abs_mean), "u_spec": float(ur.u_spec_mean)})
        print(f"[SG-V1.2] fold={fold} seed={seed} task={task} done", flush=True)
        del ctrl, orthogonal_q, rgeom, random_centroids, ogeom, margin_r, tq, th, vq, qP, qPv, train_reps, eval_reps
        gc.collect()
    print(f"[SG-V1.2] fold={fold} seed={seed} building result", flush=True)
    result = {"fold": fold, "seed": seed, "canonical_spectrum_npz_sha256": fsha(canonical), "canonical_spectrum_fingerprint": saved_fp,
              "tasks": task_results, "subject_rows": subject_rows, "contrast_rows": contrast_rows,
              "random_rows": random_rows, "pca_rows": pca_rows, "orthogonal_rows": orth_rows,
              "non_rows": non_rows, "perm_rows": perm_rows, "block_rows": block_rows, "outer_test_used": False,
              "label_free_transductive_centering": True, "spectrum_rebuilt": False,
              "sampling": {"train_geometry_cap_per_subject_session_event": GEOMETRY_CAP,
                           "validation_geometry_cap_per_subject_session_event": VALIDATION_CAP,
                           "seed_formula": "sha256(fold|seed|task|subject|session|event|purpose)"}}
    write_json(run_dir / "RUN_RESULT_V1_2.json", result)
    print(f"[SG-V1.2] fold={fold} seed={seed} result json written", flush=True)
    pd.DataFrame(subject_rows).to_csv(run_dir / "SUBJECT_GEOMETRY.csv", index=False)
    pd.DataFrame(random_rows).to_csv(run_dir / "RANDOM_SAME_RANK.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(run_dir / "PCA_SAME_RANK.csv", index=False)
    pd.DataFrame(orth_rows).to_csv(run_dir / "ORTHOGONAL_SUBSPACE.csv", index=False)
    pd.DataFrame(non_rows).to_csv(run_dir / "NONPROTECTED.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(run_dir / "LABEL_PERMUTATION.csv", index=False)
    pd.DataFrame(block_rows).to_csv(run_dir / "BLOCK_GEOMETRY_UTILITY.csv", index=False)
    print(f"[SG-V1.2] fold={fold} seed={seed} files written", flush=True)
    return result


def finalize(results: list[dict[str, Any]]) -> dict[str, Any]:
    run_values = {k: {} for k in "ABCD"}; all_subject = []; all_contrasts = []; all_random = []; all_pca = []; all_orth = []; all_non = []; all_perm = []; all_blocks = []
    for r in results:
        key = f"fold-{r['fold']}_seed-{r['seed']}"; mi = r["tasks"]["mi"]
        cr = [x for x in r.get("contrast_rows", []) if x.get("task") == "mi"]
        def unique_subject_values(metric: str) -> list[float]:
            by_subject: dict[str, list[float]] = {}
            for row in cr:
                if row.get("metric") != metric or not np.isfinite(row.get("delta", np.nan)):
                    continue
                by_subject.setdefault(str(row["subject"]), []).append(float(row["delta"]))
            return [float(np.mean(values)) for values in by_subject.values() if values]
        a = unique_subject_values("alignment")
        c = unique_subject_values("cross_session")
        d = list(map(float, mi["margin_delta"]))
        run_values["A"][key] = a; run_values["C"][key] = c
        run_values["B"][key] = list(map(float, mi["loso"]["delta"])); run_values["D"][key] = d
        all_subject.extend(r["subject_rows"]); all_contrasts.extend(r.get("contrast_rows", [])); all_random.extend(r["random_rows"]); all_pca.extend(r["pca_rows"]); all_orth.extend(r.get("orthogonal_rows", [])); all_non.extend(r["non_rows"]); all_perm.extend(r["perm_rows"]); all_blocks.extend(r["block_rows"])
    stats = {}
    for key, values in run_values.items():
        vals = [float(np.mean(x)) for x in values.values() if x]
        stats[key] = {"mean_run_effect": float(np.mean(vals)) if vals else None, "positive_runs": int(sum(x > 0 for x in vals)), "n_runs": len(vals), "hierarchical_bootstrap": g.hierarchical_boot(values, g.stable_seed("hier-v1.2", key))}
    block_frame = pd.DataFrame(all_blocks); link = fit_utility_link_numpy(block_frame)
    gate = {"A": stats["A"]["positive_runs"] >= 5 and (stats["A"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "B": stats["B"]["positive_runs"] >= 5 and (stats["B"]["mean_run_effect"] or -math.inf) >= .02 and (stats["B"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "C": stats["C"]["positive_runs"] >= 5 and (stats["C"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "D": stats["D"]["positive_runs"] >= 5 and (stats["D"]["hierarchical_bootstrap"]["ci95"][0] or -math.inf) > 0,
            "E": link.get("status") == "OK" and link.get("beta_geometry", -math.inf) > 0 and link.get("beta_geometry_bootstrap_ci95", [-math.inf])[0] > 0 and (link.get("pooled_spearman_rho") or -math.inf) > 0,
        "F": all(r.get("outer_test_used") is False and r.get("spectrum_rebuilt") is False and all(t.get("unit_test", {}).get("passed", True) for t in r.get("tasks", {}).values()) for r in results)}
    if all(gate.values()): status = "SHARED_GEOMETRY_V1_2_PASS"
    elif gate["B"] and not (gate["A"] and gate["C"] and gate["D"]): status = "PROTECTED_TRANSFER_WITHOUT_SHARED_GEOMETRY"
    elif gate["A"] and gate["C"] and gate["D"] and not gate["B"]: status = "SHARED_GEOMETRY_NOT_FUNCTIONALLY_TRANSFERABLE"
    elif gate["A"] and gate["B"] and gate["C"] and gate["D"] and not gate["E"]: status = "SHARED_GEOMETRY_NOT_UTILITY_LINKED"
    else: status = "PERSIST_USE_SHARED_GEOMETRY_NOT_SUPPORTED"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_subject).to_csv(OUT / "SUBJECT_CLASS_CONTRASTS.csv", index=False)
    pd.DataFrame(all_contrasts).to_csv(OUT / "SUBJECT_GEOMETRY_CONTRASTS.csv", index=False)
    pd.DataFrame([x for x in all_subject if x["metric"] == "rdm"]).to_csv(OUT / "CROSS_SUBJECT_RDM.csv", index=False)
    pd.DataFrame([x for x in all_subject if x["metric"] == "cross_session"]).to_csv(OUT / "CROSS_SESSION_RDM.csv", index=False)
    pd.DataFrame([x for x in all_subject if x["metric"] == "margin"]).to_csv(OUT / "GEOMETRY_MARGIN.csv", index=False)
    loso_rows = [{"fold": r["fold"], "seed": r["seed"], "task": t, **x} for r in results for t in TASKS if "loso" in r["tasks"].get(t, {}) for x in r["tasks"][t]["loso"]["subjects"]]
    ridge_rows = [{"fold": r["fold"], "seed": r["seed"], "task": t, **x} for r in results for t in TASKS if "validation_ridge" in r["tasks"].get(t, {}) for x in r["tasks"][t]["validation_ridge"]]
    pd.DataFrame(loso_rows).to_csv(OUT / "LOSO_PROTOTYPE_RESULTS.csv", index=False); pd.DataFrame(ridge_rows).to_csv(OUT / "RIDGE_TRANSFER_RESULTS.csv", index=False)
    pd.DataFrame(all_random).to_csv(OUT / "RANDOM_SAME_RANK.csv", index=False); pd.DataFrame(all_pca).to_csv(OUT / "PCA_SAME_RANK.csv", index=False); pd.DataFrame(all_orth).to_csv(OUT / "ORTHOGONAL_SUBSPACE.csv", index=False); pd.DataFrame(all_non).to_csv(OUT / "NONPROTECTED.csv", index=False); pd.DataFrame(all_perm).to_csv(OUT / "LABEL_PERMUTATION.csv", index=False); block_frame.to_csv(OUT / "BLOCK_GEOMETRY_UTILITY.csv", index=False)
    write_json(OUT / "GEOMETRY_UTILITY_REGRESSION.json", link); write_json(OUT / "SHARED_GEOMETRY_V1_2_ADAPTATION_LOG.json", {"version": "Shared Geometry V1.2", "canonical_source": "Signed Audit V3.1", "spectrum_rebuilt": False, "protected_redefined": False, "issues_repaired": ["full q to qP leakage", "basis remapping", "binary MI RDM", "half-split LOSO", "invalid margin proxy", "missing utility link"], "data_used": ["TRAIN", "DEVELOPMENT_VALIDATION"], "outer_test_used": False})
    payload = {"status": status, "version": "Shared Geometry V1.2", "gate": gate, "gate_statistics": stats, "geometry_utility_link": link,
               "runs": [{"fold": r["fold"], "seed": r["seed"], "canonical_spectrum_npz_sha256": r["canonical_spectrum_npz_sha256"], "outer_test_used": False} for r in results],
               "outer_test_used": False, "method_training_started": False, "canonical_spectrum_source": "Signed_V3.1"}
    write_json(OUT / "SHARED_GEOMETRY_FINAL_REPORT.json", payload)
    (OUT / "SHARED_GEOMETRY_FINAL_REPORT.md").write_text(f"# PERSIST-EEG Shared Geometry Audit V1.2\n\nStatus: `{status}`\n\nGates: `{json.dumps(gate, sort_keys=True)}`\n\nCanonical Signed-V3.1 basis loaded; no spectrum rebuilt.\n\nGeometry/LOSO caps are deterministic subject/session/event caps (`{GEOMETRY_CAP}` train, `{VALIDATION_CAP}` validation); subject remains the statistical unit.\n\nOuter-test used: `false`.\n\nNo PERSIST-USE model was trained.\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--seed", type=int); ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()
    if args.finalize_only:
        runs = []
        for fold in FOLDS:
            for seed in SEEDS:
                p = OUT / "runs" / f"fold-{fold}" / f"seed-{seed}" / "RUN_RESULT_V1_2.json"
                if not p.exists():
                    raise FileNotFoundError(str(p))
                runs.append(json.loads(p.read_text(encoding="utf-8")))
        print(json.dumps(base.clean(finalize(runs)), indent=2), flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Shared Geometry V1.2 requires server GPU")
    prereq = json.loads((V31 / "SIGNED_V3_1_FINAL_REPORT.json").read_text(encoding="utf-8"))
    if prereq.get("status") != "PERSISTENCE_UTILITY_ASSIGNMENT_REPRODUCIBLE": raise RuntimeError("V3.1 prerequisite did not pass; geometry locked")
    folds = (args.fold,) if args.fold is not None else FOLDS; seeds = (args.seed,) if args.seed is not None else SEEDS; results = []
    for fold in folds:
        for seed in seeds: results.append(run_one(fold, seed, device))
    if len(results) == len(FOLDS) * len(SEEDS): print(json.dumps(base.clean(finalize(results)), indent=2), flush=True)


if __name__ == "__main__": main()
