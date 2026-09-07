"""Signed Audit V3.1 reproducibility repair.

The scientific definitions are copied from Signed Audit V3.  The only
intentional change is provenance: every capped sample is selected using a
stable SHA256 seed and persisted, and one canonical full-TRAIN spectrum is
saved per fold/seed.  This script never accesses outer-test data and never
trains a method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import p4_persist_ct as base
import p4_persist_ct_v2 as v2


ROOT = base.ROOT
OUT = ROOT / "outputs" / "persist_eeg_p4_signed_v3_1"
OLD = ROOT / "outputs" / "persist_eeg_p4_signed" / "audit_v3"
TASKS = tuple(base.TASKS)
CLASSES = dict(base.CLASSES)
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
INNER = 5
DRAWS = 100
BOOT = 10_000
EPS = 0.005
PER_GROUP = 32


def clean(v: Any) -> Any:
    return base.clean(v)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_uint64(*parts: Any) -> int:
    """Cross-process deterministic uint64 seed; never use Python's process hash."""
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def array_sha(a: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def spectrum_fingerprint(spec: Mapping[str, Any], artifact_path: Path | None = None) -> dict[str, Any]:
    names = ("mean", "whitener", "dewhitener", "directions", "rho")
    arrays = {n: array_sha(np.asarray(spec[n], dtype=np.float32)) for n in names}
    payload = {"arrays": arrays, "blocks": [list(map(int, b)) for b in spec["blocks"]],
               "block_dimensions": [len(b) for b in spec["blocks"]]}
    payload["combined_arrays_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if artifact_path is not None and artifact_path.exists():
        payload["npz_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return payload


def task_labels(meta: pd.DataFrame, task: str, idx: np.ndarray) -> np.ndarray:
    mapping = v2.labels(meta, task, np.arange(len(meta)))
    # v2.labels is based on the frozen label map; indexing it avoids any
    # dependence on local dataframe ordering after sampling.
    return mapping[idx]


def stable_sample_indices(meta: pd.DataFrame, task: str, subjects: Sequence[str], fold: int, seed: int,
                          inner_split: int, purpose: str, per_group: int = PER_GROUP) -> tuple[np.ndarray, pd.DataFrame]:
    """Stable subject/session/event sample with persisted row identifiers."""
    frame = meta[(meta.paradigm.astype(str) == str(task)) &
                 meta.subject_id.astype(str).isin(set(map(str, subjects)))]
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    # Arrow-backed string columns can intermittently fail on Series.iloc in
    # long runs; convert the persisted identifier column once to a NumPy view.
    global_values = meta["global_index"].to_numpy()
    for (subject, session, event), group in frame.groupby(["subject_id", "session_id", "event_label"], sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        count_before = len(idx)
        if per_group and len(idx) > per_group:
            rng = np.random.default_rng(stable_uint64(fold, seed, task, subject, session, event, inner_split, purpose))
            idx = np.sort(rng.choice(idx, size=per_group, replace=False))
        selected.extend(map(int, idx))
        for row in idx:
            records.append({"frame_index": int(row), "global_index": int(global_values[int(row)]),
                            "task": str(task), "subject_id": str(subject), "session_id": str(session),
                            "event_label": str(event), "fold": fold, "seed": seed,
                            "inner_split": inner_split, "purpose": purpose,
                            "group_count_before": count_before, "selected_count": len(idx)})
    selected_arr = np.asarray(sorted(selected), dtype=np.int64)
    return selected_arr, pd.DataFrame(records).sort_values("frame_index").reset_index(drop=True)


def save_sampling(run_dir: Path, task: str, inner_split: str | int, purpose: str,
                  idx: np.ndarray, metadata: pd.DataFrame) -> None:
    d = run_dir / "sampling" / str(task) / str(inner_split)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{purpose}_indices.npy", np.asarray(idx, dtype=np.int64))
    metadata.to_csv(d / f"{purpose}_metadata.csv", index=False)


def save_spectrum(run_dir: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    d = run_dir / "spectrum"; d.mkdir(parents=True, exist_ok=True)
    npz = d / "PERSISTENCE_SPECTRUM.npz"
    np.savez_compressed(npz,
                        mean=np.asarray(spec["mean"], dtype=np.float32),
                        whitener=np.asarray(spec["whitener"], dtype=np.float32),
                        dewhitener=np.asarray(spec["dewhitener"], dtype=np.float32),
                        directions=np.asarray(spec["directions"], dtype=np.float32),
                        rho=np.asarray(spec["rho"], dtype=np.float32),
                        blocks_json=np.asarray(json.dumps([list(map(int, b)) for b in spec["blocks"]], sort_keys=True)),
                        support_json=np.asarray(json.dumps(spec["audit"].get("persistence_support", []), sort_keys=True)),
                        audit_json=np.asarray(json.dumps(spec["audit"], sort_keys=True)))
    fp = spectrum_fingerprint(spec, npz)
    write_json(d / "PERSISTENCE_SPECTRUM_FINGERPRINT.json", fp)
    write_json(d / "PERSISTENCE_SPECTRUM_AUDIT.json", spec["audit"])
    return fp


def load_canonical_spectrum(run_dir: Path) -> dict[str, Any]:
    z = np.load(run_dir / "spectrum" / "PERSISTENCE_SPECTRUM.npz", allow_pickle=False)
    blocks = json.loads(str(z["blocks_json"].item()))
    audit = json.loads(str(z["audit_json"].item()))
    return {"mean": z["mean"], "whitener": z["whitener"], "dewhitener": z["dewhitener"],
            "directions": z["directions"], "rho": z["rho"], "blocks": blocks, "audit": audit}


def risk(Xf: np.ndarray, yf: np.ndarray, Xe: np.ndarray, ye: np.ndarray, classes: int) -> tuple[float, float]:
    pack = base.ridge_probe(np.asarray(Xf), yf, classes)
    pred, prob = base.probe_predict(np.asarray(Xe), pack, classes)
    yy = np.asarray(ye, dtype=np.int64); pp = np.asarray(prob, dtype=np.float64)
    ce = float(-np.mean(np.log(np.clip(pp[np.arange(len(yy)), yy], 1e-12, 1.0))))
    ba = float(np.mean([np.mean(pred[yy == k] == k) for k in range(classes) if np.any(yy == k)]))
    return ce, ba


def boot(values: Sequence[float], seed: int) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return {"mean": None, "ci95": [None, None], "sign_probability": None, "draws": BOOT, "n_unique_subjects": 0}
    rng = np.random.default_rng(seed); draws = rng.choice(vals, size=(BOOT, len(vals)), replace=True).mean(axis=1)
    return {"mean": float(vals.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "sign_probability": float(np.mean(draws > 0)), "draws": BOOT, "n_unique_subjects": int(len(vals))}


def load_saved_indices(run_dir: Path, task: str, split: str | int, purpose: str) -> np.ndarray:
    return np.load(run_dir / "sampling" / str(task) / str(split) / f"{purpose}_indices.npy").astype(np.int64)


def run_one(fold: int, seed: int, device: torch.device) -> dict[str, Any]:
    manifest = base.load_manifest(); split = next(x for x in base.load_splits() if int(x["fold"]) == fold)
    ckpt, mean, std = base.historical(fold, seed); model = base.load_model(ckpt, manifest, device)
    print(f"[V3.1] fold={fold} seed={seed} extracting full TRAIN", flush=True)
    trm, trh, tr_y = base.extract(model, manifest, split["train_subjects"], mean, std, device, 190000 + fold * 101 + seed, cap=0)
    vam, vah, va_y = base.extract(model, manifest, split["validation_subjects"], mean, std, device, 200000 + fold * 101 + seed, cap=0)
    spec = v2.build_spectrum_v2(trm, trh, 30000 + fold * 101 + seed)
    run_dir = OUT / "runs" / f"fold-{fold}" / f"seed-{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    fp = save_spectrum(run_dir, spec)
    write_json(run_dir / "RUN_PROVENANCE.json", {"fold": fold, "seed": seed, "historical_checkpoint": str(ckpt),
                                                  "train_n": len(trh), "validation_n": len(vah),
                                                  "basis_seed": 30000 + fold * 101 + seed, "outer_test_used": False})
    subjects = v2.subject_sort(split["train_subjects"])
    rows: list[dict[str, Any]] = []; assignments: dict[str, Any] = {}; sampling_index_summary: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = []
        for inner in range(INNER):
            fit_subjects, eval_subjects = v2.split_subjects(subjects, inner, seed)
            fi, fm = stable_sample_indices(trm, task, fit_subjects, fold, seed, inner, "fit")
            ei, em = stable_sample_indices(trm, task, eval_subjects, fold, seed, inner, "eval")
            save_sampling(run_dir, task, inner, "fit", fi, fm); save_sampling(run_dir, task, inner, "eval", ei, em)
            sampling_index_summary.append({"task": task, "inner_split": inner, "fit_n": len(fi), "eval_n": len(ei),
                                           "fit_subjects": fit_subjects, "eval_subjects": eval_subjects})
        for bi, block in enumerate(spec["blocks"]):
            abs_by: dict[str, list[float]] = {}; spec_by: dict[str, list[float]] = {}; ba_by: dict[str, list[float]] = {}; rand_ba_by: dict[str, list[float]] = {}
            supported = bool(spec["audit"]["persistence_support"][bi]["persistence_supported"])
            split_obs = 0
            for inner in range(INNER):
                fi = load_saved_indices(run_dir, task, inner, "fit"); ei = load_saved_indices(run_dir, task, inner, "eval")
                yf, ye = task_labels(trm, task, fi), task_labels(trm, task, ei)
                base_risk, base_ba = risk(trh[fi], yf, trh[ei], ye, CLASSES[task])
                erased_fit = base.erase(trh[fi], spec, block); erased_eval = base.erase(trh[ei], spec, block)
                candidates = np.setdiff1d(np.arange(len(spec["rho"])), np.asarray(block, dtype=np.int64))
                rng = np.random.default_rng(41000 + fold * 1000 + seed * 100 + inner * 10 + bi)
                choices = [rng.choice(candidates if len(candidates) >= len(block) else np.arange(len(spec["rho"])), size=len(block), replace=False) for _ in range(DRAWS)]
                # The V3 scientific protocol fits one random-control probe per
                # draw and reuses it for every evaluation subject.  Cache both
                # the fitted probes and erased evaluation matrix; doing this
                # per subject is numerically equivalent but needlessly slow.
                random_packs = [base.ridge_probe(base.erase(trh[fi], spec, ch), yf, CLASSES[task]) for ch in choices]
                random_eval = [base.erase(trh[ei], spec, ch) for ch in choices]
                em = trm.iloc[ei].reset_index(drop=True)
                for subj, group in em.groupby(em.subject_id.astype(str), sort=True):
                    loc = group.index.to_numpy(dtype=np.int64)
                    if len(loc) < 2:
                        continue
                    rr, bb = risk(trh[fi], yf, trh[ei][loc], ye[loc], CLASSES[task])
                    re, be = risk(erased_fit, yf, erased_eval[loc], ye[loc], CLASSES[task])
                    rs, rba = [], []
                    for pack, erased_random_eval in zip(random_packs, random_eval):
                        _, prob = base.probe_predict(erased_random_eval[loc], pack, CLASSES[task])
                        pp = np.asarray(prob); yy = ye[loc]; rs.append(float(-np.mean(np.log(np.clip(pp[np.arange(len(yy)), yy], 1e-12, 1.0))) - rr))
                        rba.append(float(np.mean([np.mean(pp.argmax(1)[yy == k] == k) for k in range(CLASSES[task]) if np.any(yy == k)]) - bb))
                    u_abs = re - rr; u_spec = u_abs - float(np.mean(rs)); s = str(subj)
                    abs_by.setdefault(s, []).append(float(u_abs)); spec_by.setdefault(s, []).append(float(u_spec)); ba_by.setdefault(s, []).append(float(be - bb)); rand_ba_by.setdefault(s, []).append(float(np.mean(rba))); split_obs += 1
            abs_u = {s: float(np.mean(v)) for s, v in abs_by.items()}; spec_u = {s: float(np.mean(v)) for s, v in spec_by.items()}; ba_u = {s: float(np.mean(v)) for s, v in ba_by.items()}; rb_u = {s: float(np.mean(v)) for s, v in rand_ba_by.items()}
            ab = boot(list(abs_u.values()), 70000 + fold * 101 + seed * 11 + bi); sb = boot(list(spec_u.values()), 71000 + fold * 101 + seed * 11 + bi)
            task_rows.append({"fold": fold, "seed": seed, "task": task, "block": bi, "dimensions": len(block), "persistence_supported": supported,
                              "n_unique_subjects": len(abs_u), "n_split_observations": split_obs,
                              "bootstrap_hierarchy": "aggregate_unique_subject_across_inner_splits_then_subject_bootstrap",
                              "u_abs_mean": ab["mean"], "u_abs_CI95": ab["ci95"], "u_abs_sign_probability": ab["sign_probability"],
                              "u_spec_mean": sb["mean"], "u_spec_CI95": sb["ci95"], "u_spec_sign_probability": sb["sign_probability"],
                              "raw_BA_change": float(np.mean(list(ba_u.values()))) if ba_u else None,
                              "same_rank_random_BA_change": float(np.mean(list(rb_u.values()))) if rb_u else None,
                              "u_abs_bootstrap": ab, "u_spec_bootstrap": sb, "random_interventions": DRAWS})
        prot = [r["block"] for r in task_rows if r["persistence_supported"] and r["u_abs_CI95"][0] is not None and r["u_abs_CI95"][0] > 0 and r["u_spec_CI95"][0] > 0]
        harm = [r["block"] for r in task_rows if r["persistence_supported"] and r["u_abs_CI95"][1] is not None and r["u_abs_CI95"][1] < 0 and r["u_spec_CI95"][1] < 0]
        neutral = [r["block"] for r in task_rows if r["persistence_supported"] and r["u_abs_CI95"][0] is not None and r["u_abs_CI95"][0] >= -EPS and r["u_abs_CI95"][1] <= EPS]
        assignments[task] = {"protected": prot, "harmful": harm, "neutral": neutral,
                             "uncertain": [r["block"] for r in task_rows if r["block"] not in prot + harm + neutral]}
        rows.extend(task_rows)
    # Deterministic validation transfer sampling is persisted separately.
    validation_rows: list[dict[str, Any]] = []
    for task in TASKS:
        ti, tm = stable_sample_indices(trm, task, split["train_subjects"], fold, seed, -1, "validation_train")
        vi, vm = stable_sample_indices(vam, task, split["validation_subjects"], fold, seed, -1, "validation_eval")
        save_sampling(run_dir, task, "validation", "train", ti, tm); save_sampling(run_dir, task, "validation", "eval", vi, vm)
        ytr, yv = task_labels(trm, task, ti), task_labels(vam, task, vi); raw_train, raw_val = trh[ti], vah[vi]
        _, raw_ba = risk(raw_train, ytr, raw_val, yv, CLASSES[task])
        blocks_to_test = [("protected", b) for b in assignments[task]["protected"]] + [("harmful", b) for b in assignments[task]["harmful"]] + [("neutral", b) for b in assignments[task]["neutral"]]
        for kind, bi in blocks_to_test:
            block = spec["blocks"][bi]; _, ba = risk(base.erase(raw_train, spec, block), ytr, base.erase(raw_val, spec, block), yv, CLASSES[task])
            validation_rows.append({"fold": fold, "seed": seed, "task": task, "kind": kind, "block": bi, "validation_gain_BA": float(ba - raw_ba), "raw_BA": float(raw_ba)})
        for kind, bl in [("protected_union", assignments[task]["protected"]), ("harmful_union", assignments[task]["harmful"])]:
            ids = sorted(set(sum((spec["blocks"][b] for b in bl), [])))
            if ids:
                _, ba = risk(base.erase(raw_train, spec, ids), ytr, base.erase(raw_val, spec, ids), yv, CLASSES[task])
                validation_rows.append({"fold": fold, "seed": seed, "task": task, "kind": kind, "block": "union", "validation_gain_BA": float(ba - raw_ba), "raw_BA": float(raw_ba)})
        # Exactly the old same-block random control protocol.
        rank = len(spec["blocks"][0]); rng = np.random.default_rng(81000 + fold * 100 + seed); vals = []
        for _ in range(DRAWS):
            ch = rng.choice(np.arange(len(spec["rho"])), size=rank, replace=False); _, ba = risk(base.erase(raw_train, spec, ch), ytr, base.erase(raw_val, spec, ch), yv, CLASSES[task]); vals.append(ba - raw_ba)
        validation_rows.append({"fold": fold, "seed": seed, "task": task, "kind": "same_block", "block": "random", "validation_gain_BA": float(np.mean(vals)), "raw_BA": float(raw_ba)})
    result = {"fold": fold, "seed": seed, "rows": rows, "assignments": assignments, "validation": validation_rows,
              "sampling_index_summary": sampling_index_summary, "spectrum_fingerprint": fp,
              "outer_test_used": False, "sampling": {"per_group": PER_GROUP, "stable_seed": "sha256(fold|seed|task|subject|session|event|inner_split|purpose)", "persisted": True}}
    pd.DataFrame(rows).to_csv(run_dir / "SIGNED_UTILITY_V3_1.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(run_dir / "VALIDATION_SIGN_TRANSFER_V3_1.csv", index=False)
    write_json(run_dir / "SIGNED_ASSIGNMENTS_V3_1.json", assignments)
    write_json(run_dir / "SIGNED_AUDIT_RUN_V3_1.json", result)
    return result


def reference_comparison(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        fold, seed = result["fold"], result["seed"]
        old_assign = json.loads((OLD / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_ASSIGNMENTS.json").read_text(encoding="utf-8"))
        old_val = pd.read_csv(OLD / f"fold-{fold}" / f"seed-{seed}" / "VALIDATION_SIGN_TRANSFER.csv")
        new_val = pd.DataFrame(result["validation"])
        for task in TASKS:
            oldp = set(old_assign[task].get("protected", [])); newp = set(result["assignments"][task].get("protected", [])); union = oldp | newp
            j = len(oldp & newp) / max(len(union), 1)
            old_row = old_val[(old_val.task == task) & (old_val.kind == "protected_union")]
            new_row = new_val[(new_val.task == task) & (new_val.kind == "protected_union")]
            old_gain = float(old_row.validation_gain_BA.iloc[0]) if len(old_row) else None; new_gain = float(new_row.validation_gain_BA.iloc[0]) if len(new_row) else None
            old_harm = set(old_assign[task].get("harmful", [])); new_harm = set(result["assignments"][task].get("harmful", []))
            rows.append({"fold": fold, "seed": seed, "task": task, "old_protected": sorted(oldp), "new_protected": sorted(newp),
                         "protected_jaccard": j, "old_rank": len(oldp), "new_rank": len(newp), "old_validation_gain_BA": old_gain,
                         "new_validation_gain_BA": new_gain, "validation_sign_agreement": bool(old_gain is not None and new_gain is not None and np.sign(old_gain) == np.sign(new_gain)),
                         "old_harmful": sorted(old_harm), "new_harmful": sorted(new_harm), "harmful_assignment_agreement": old_harm == new_harm})
    return pd.DataFrame(rows)


def hier_boot(values_by_run: Mapping[str, Sequence[float]], seed: int) -> dict[str, Any]:
    keys = sorted(values_by_run); rng = np.random.default_rng(seed); draws = []
    for _ in range(BOOT):
        sampled = rng.choice(keys, size=len(keys), replace=True); vals = []
        for key in sampled:
            arr = np.asarray(values_by_run[key], dtype=np.float64)
            if len(arr): vals.append(float(rng.choice(arr, size=len(arr), replace=True).mean()))
        draws.append(float(np.mean(vals)) if vals else np.nan)
    d = np.asarray(draws); d = d[np.isfinite(d)]; raw = np.asarray([v for key in keys for v in values_by_run[key]], dtype=np.float64)
    return {"mean": float(raw.mean()) if len(raw) else None, "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))] if len(d) else [None, None],
            "sign_probability": float(np.mean(d > 0)) if len(d) else None, "draws": BOOT, "n_runs": len(keys)}


def finalize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    utility = []
    prereq_values: dict[str, list[float]] = {}
    for result in results:
        fold, seed = result["fold"], result["seed"]; key = f"fold-{fold}_seed-{seed}"
        mi = result["assignments"]["mi"]; vr = pd.DataFrame(result["validation"])
        p = vr[(vr.task == "mi") & (vr.kind == "protected_union")]; r = vr[(vr.task == "mi") & (vr.kind == "same_block")]
        pg = float(p.validation_gain_BA.iloc[0]) if len(p) else np.nan; rg = float(r.validation_gain_BA.iloc[0]) if len(r) else np.nan
        difference = rg - pg if np.isfinite(pg) and np.isfinite(rg) else np.nan
        prereq_values[key] = [difference] if np.isfinite(difference) else []
        rows.append({"fold": fold, "seed": seed, "mi_protected_blocks": mi["protected"], "mi_protected_exists": bool(mi["protected"]),
                     "mi_protected_harm_BA": -pg if np.isfinite(pg) else None, "mi_same_rank_random_harm_BA": -rg if np.isfinite(rg) else None,
                     "mi_protected_minus_random_harm_BA": difference if np.isfinite(difference) else None, "outer_test_used": False})
        utility.extend(result["rows"])
    comparison = reference_comparison(results)
    OUT.mkdir(parents=True, exist_ok=True); comparison.to_csv(OUT / "HISTORICAL_V3_VS_V3_1.csv", index=False); pd.DataFrame(rows).to_csv(OUT / "MI_REPRODUCIBLE_PREREQUISITE.csv", index=False); pd.DataFrame(utility).to_csv(OUT / "SIGNED_UTILITY_V3_1_ALL.csv", index=False)
    exists_runs = int(sum(bool(x["mi_protected_exists"]) for x in rows)); diff_vals = [float(x) for v in prereq_values.values() for x in v]
    hb = hier_boot(prereq_values, 991337)
    prereq = {"protected_assignment_runs": exists_runs, "protected_assignment_ge5_of6": exists_runs >= 5,
              "harm_exceeds_random_positive_runs": int(sum(x > 0 for x in diff_vals)), "harm_exceeds_random_ge5_of6": int(sum(x > 0 for x in diff_vals)) >= 5,
              "harm_exceeds_random_hierarchical_ci_lower_gt0": hb["ci95"][0] is not None and hb["ci95"][0] > 0,
              "harm_difference_bootstrap": hb, "pass": exists_runs >= 5 and int(sum(x > 0 for x in diff_vals)) >= 5 and (hb["ci95"][0] or -np.inf) > 0}
    status = "PERSISTENCE_UTILITY_ASSIGNMENT_REPRODUCIBLE" if prereq["pass"] else "PERSISTENCE_UTILITY_ASSIGNMENT_NOT_REPRODUCIBLE"
    payload = {"status": status, "version": "Signed Audit V3.1", "prerequisite": prereq, "runs": rows,
               "outer_test_used": False, "scientific_definitions_changed": False, "shared_geometry_started": False,
               "historical_comparison": {"rows": int(len(comparison)), "path": "HISTORICAL_V3_VS_V3_1.csv"}}
    write_json(OUT / "SIGNED_V3_1_FINAL_REPORT.json", payload)
    (OUT / "SIGNED_V3_1_FINAL_REPORT.md").write_text(
        f"# Signed Audit V3.1\n\nStatus: `{status}`\n\n"
        f"MI Protected assignment runs: `{exists_runs}/6`.\n\n"
        f"Harm-vs-random hierarchical CI: `{hb['ci95']}`.\n\n"
        "Scientific definitions unchanged; outer-test used: `false`.\n", encoding="utf-8")
    write_json(OUT / "protocol" / "SIGNED_V3_1_PROTOCOL.json", {"inner_splits": INNER, "random_draws": DRAWS, "bootstrap_draws": BOOT,
                                                                    "epsilon_neutral": EPS, "per_group": PER_GROUP, "full_train_spectrum": True,
                                                                    "stable_seed": "sha256(fold|seed|task|subject|session|event|inner_split|purpose)",
                                                                    "outer_test_used": False})
    write_json(OUT / "protocol" / "SIGNED_V3_1_ADAPTATION_LOG.json", {
        "issue": "Signed V3 used process-dependent Python hash and did not persist sampled row identifiers",
        "change": "stable SHA256 sampling, persisted fit/eval/validation indices, canonical spectrum npz and fingerprints",
        "scientific_impact": "definitions, thresholds, probe, blocks, bootstrap, and random draws unchanged",
        "data_used": ["TRAIN", "DEVELOPMENT_VALIDATION"], "outer_test_used": False,
    })
    return payload


def recompute_u_spec_mean(run_dir: Path, trm: pd.DataFrame, trh: np.ndarray, spec: Mapping[str, Any],
                          task: str, block_id: int, fold: int, seed: int) -> float:
    """Recompute one saved V3.1 block utility from persisted indices."""
    block = spec["blocks"][int(block_id)]
    subject_values: dict[str, list[float]] = {}
    for inner in range(INNER):
        fi = load_saved_indices(run_dir, task, inner, "fit"); ei = load_saved_indices(run_dir, task, inner, "eval")
        yf, ye = task_labels(trm, task, fi), task_labels(trm, task, ei)
        rr, _ = risk(trh[fi], yf, trh[ei], ye, CLASSES[task])
        erased_fit = base.erase(trh[fi], spec, block); erased_eval = base.erase(trh[ei], spec, block)
        candidates = np.setdiff1d(np.arange(len(spec["rho"])), np.asarray(block, dtype=np.int64))
        rng = np.random.default_rng(41000 + fold * 1000 + seed * 100 + inner * 10 + int(block_id))
        choices = [rng.choice(candidates if len(candidates) >= len(block) else np.arange(len(spec["rho"])), size=len(block), replace=False) for _ in range(DRAWS)]
        random_packs = [base.ridge_probe(base.erase(trh[fi], spec, ch), yf, CLASSES[task]) for ch in choices]
        random_eval = [base.erase(trh[ei], spec, ch) for ch in choices]
        em = trm.iloc[ei].reset_index(drop=True)
        for subj, group in em.groupby(em.subject_id.astype(str), sort=True):
            loc = group.index.to_numpy(dtype=np.int64)
            if len(loc) < 2:
                continue
            rr_sub, _ = risk(trh[fi], yf, trh[ei][loc], ye[loc], CLASSES[task])
            re_sub, _ = risk(erased_fit, yf, erased_eval[loc], ye[loc], CLASSES[task])
            rand_ce = []
            for pack, erased_random_eval in zip(random_packs, random_eval):
                _, prob = base.probe_predict(erased_random_eval[loc], pack, CLASSES[task])
                yy = ye[loc]
                rand_ce.append(float(-np.mean(np.log(np.clip(prob[np.arange(len(yy)), yy], 1e-12, 1.0))) - rr_sub))
            subject_values.setdefault(str(subj), []).append(float((re_sub - rr_sub) - np.mean(rand_ce)))
    return float(np.mean([np.mean(v) for v in subject_values.values()]))


def self_test(fold: int, seed: int) -> dict[str, Any]:
    """Fresh-process replay from persisted indices and canonical spectrum."""
    manifest = base.load_manifest(); split = next(x for x in base.load_splits() if int(x["fold"]) == fold)
    ckpt, mean, std = base.historical(fold, seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.load_model(ckpt, manifest, device)
    trm, trh, _ = base.extract(model, manifest, split["train_subjects"], mean, std, device, 290000 + fold * 101 + seed, cap=0)
    vam, vah, _ = base.extract(model, manifest, split["validation_subjects"], mean, std, device, 300000 + fold * 101 + seed, cap=0)
    run_dir = OUT / "runs" / f"fold-{fold}" / f"seed-{seed}"; spec = load_canonical_spectrum(run_dir)
    assignments = json.loads((run_dir / "SIGNED_ASSIGNMENTS_V3_1.json").read_text(encoding="utf-8"))
    ref = json.loads((run_dir / "SIGNED_AUDIT_RUN_V3_1.json").read_text(encoding="utf-8")); checks: list[dict[str, Any]] = []
    for task in TASKS:
        ti = load_saved_indices(run_dir, task, "validation", "train"); vi = load_saved_indices(run_dir, task, "validation", "eval")
        ytr, yv = task_labels(trm, task, ti), task_labels(vam, task, vi); raw_train, raw_val = trh[ti], vah[vi]
        _, raw_ba = risk(raw_train, ytr, raw_val, yv, CLASSES[task]); ids = sorted(set(sum((spec["blocks"][b] for b in assignments[task]["protected"]), [])))
        _, erased_ba = risk(base.erase(raw_train, spec, ids), ytr, base.erase(raw_val, spec, ids), yv, CLASSES[task]); observed = erased_ba - raw_ba
        saved = float(pd.DataFrame(ref["validation"])[lambda x: (x.task == task) & (x.kind == "protected_union")].validation_gain_BA.iloc[0])
        checks.append({"kind": "protected_union_validation", "task": task, "observed": observed, "saved": saved, "abs_diff": abs(observed - saved), "pass": bool(np.isclose(observed, saved, rtol=0, atol=1e-12))})
    utility = pd.DataFrame(ref["rows"])
    for _, row in utility.head(5).iterrows():
        observed = recompute_u_spec_mean(run_dir, trm, trh, spec, str(row.task), int(row.block), fold, seed)
        saved = float(row.u_spec_mean)
        checks.append({"kind": "block_utility_replay", "task": row.task, "block": int(row.block),
                       "recomputed_u_spec_mean": observed, "saved_u_spec_mean": saved,
                       "abs_diff": abs(observed - saved), "pass": bool(np.isclose(observed, saved, rtol=0, atol=1e-12))})
    payload = {"status": "PASS" if all(x["pass"] for x in checks) else "SIGNED_V3_1_REPRODUCIBILITY_FAIL", "fold": fold, "seed": seed,
               "checks": checks, "outer_test_used": False}
    write_json(run_dir / "REPRODUCIBILITY_SELF_TEST.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--seed", type=int); ap.add_argument("--self-test", action="store_true"); ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Signed V3.1 requires server GPU")
    if args.self_test:
        result = self_test(args.fold if args.fold is not None else 0, args.seed if args.seed is not None else 0); print(json.dumps(clean(result), indent=2), flush=True); return
    folds = (args.fold,) if args.fold is not None else FOLDS; seeds = (args.seed,) if args.seed is not None else SEEDS; results = []
    for fold in folds:
        for seed in seeds:
            saved = OUT / "runs" / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_AUDIT_RUN_V3_1.json"
            if args.skip_existing and saved.exists():
                print(f"[V3.1] fold={fold} seed={seed} loading saved canonical run", flush=True)
                results.append(json.loads(saved.read_text(encoding="utf-8")))
            else:
                print(f"[V3.1] fold={fold} seed={seed}", flush=True); results.append(run_one(fold, seed, device))
    if len(results) == len(FOLDS) * len(SEEDS):
        print(json.dumps(clean(finalize(results)), indent=2), flush=True)


if __name__ == "__main__":
    main()
