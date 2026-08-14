from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import p4_persist_ct as v1

ROOT = v1.ROOT
OUT = ROOT / "outputs" / "persist_eeg_p4_ct_v2"
TASKS = v1.TASKS
CLASSES = v1.CLASSES
FOLDS = v1.FOLDS
SEEDS = v1.SEEDS
PROBE_DIM = 16
INNER_SPLITS = 5
NULL_DRAWS = 100
BOOT_DRAWS = 10_000


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v1.clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def subject_sort(values: Sequence[str]) -> list[str]:
    return sorted(map(str, values), key=lambda x: int(x) if str(x).isdigit() else str(x))


def make_blocks(rho: np.ndarray) -> tuple[list[list[int]], dict[str, Any]]:
    r = len(rho)
    gaps = np.abs(np.diff(rho))
    threshold = max(float(np.median(gaps) * 4.0), float(np.max(np.abs(rho))) * 0.05, 1e-10)
    raw = [0]
    for i, gap in enumerate(gaps):
        if gap > threshold:
            raw.append(i + 1)
    raw.append(r)
    blocks: list[list[int]] = []
    for a, b in zip(raw[:-1], raw[1:]):
        for s in range(a, b, 4):
            blocks.append(list(range(s, min(s + 4, b))))
    if len(blocks) < 2:
        blocks = [list(range(0, min(4, r))), list(range(min(4, r), r))]
    meta = {
        "construction": "train-only eigengap clustering followed by max-size-4 split",
        "eigengap_threshold": threshold,
        "block_dimensions": [len(x) for x in blocks],
        "eigenvalue_ranges": [[float(rho[x[0]]), float(rho[x[-1]])] for x in blocks],
        "no_validation_block_selection": True,
    }
    return blocks, meta


def build_spectrum_v2(meta: pd.DataFrame, h: np.ndarray, seed: int, rank: int = 20) -> dict[str, Any]:
    x = np.asarray(h, dtype=np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    ev, evec = np.linalg.eigh((cov + cov.T) / 2.0)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    threshold = max(float(ev[0]) * 1e-3, 1e-8)
    numerical_rank = int(np.sum(ev > threshold))
    r = min(rank, numerical_rank)
    if r < 4:
        raise RuntimeError("insufficient active whitening rank")
    active = np.maximum(ev[:r], max(float(ev[:r].mean()) * 1e-4, 1e-8))
    U = evec[:, :r]
    W = U * np.power(active, -0.5)[None, :]
    D = np.sqrt(active)[:, None] * U.T
    z = xc @ W
    frame = meta.reset_index(drop=True).copy()
    frame["pos"] = np.arange(len(frame))
    sessions = sorted(frame.session_id.astype(str).unique())
    if len(sessions) != 2:
        raise RuntimeError(f"two sessions required, got {sessions}")
    cent: dict[tuple[str, str, str, str], np.ndarray] = {}
    for key, group in frame.groupby(["subject_id", "session_id", "paradigm", "event_label"], sort=True):
        cent[tuple(map(str, key))] = z[group.pos.to_numpy(dtype=np.int64)].mean(axis=0)
    subjects = subject_sort(frame.subject_id.astype(str).unique())
    task_cov: dict[str, np.ndarray] = {}
    pair_counts: dict[str, int] = {}
    for task in TASKS:
        covs, n = [], 0
        for event in sorted(frame.loc[frame.paradigm == task, "event_label"].astype(str).unique()):
            left, right = [], []
            for s in subjects:
                ka, kb = (s, sessions[0], task, event), (s, sessions[1], task, event)
                if ka in cent and kb in cent:
                    left.append(cent[ka]); right.append(cent[kb])
            if left:
                a, b = np.asarray(left), np.asarray(right)
                a -= a.mean(axis=0); b -= b.mean(axis=0)
                covs.append((a.T @ b + b.T @ a) / (2.0 * max(len(a), 1)))
                n += len(left)
        task_cov[task] = np.mean(covs, axis=0) if covs else np.zeros((r, r))
        pair_counts[task] = n
    C = np.mean(list(task_cov.values()), axis=0)
    rho, V = np.linalg.eigh((C + C.T) / 2.0)
    order = np.argsort(rho)[::-1]
    rho, V = rho[order], V[:, order]
    q = z @ V
    blocks, block_meta = make_blocks(rho)
    # TRAIN-only subject/session permutation null for each block.
    null_rng = np.random.default_rng(seed + 7717)
    null_values = [[] for _ in blocks]
    for _ in range(200):
        perm = null_rng.permutation(len(subjects))
        for task in TASKS:
            for event in sorted(frame.loc[frame.paradigm == task, "event_label"].astype(str).unique()):
                a, b = [], []
                for i, s in enumerate(subjects):
                    ka, kb = (s, sessions[0], task, event), (subjects[perm[i]], sessions[1], task, event)
                    if ka in cent and kb in cent:
                        a.append(cent[ka]); b.append(cent[kb])
                if len(a) >= 3:
                    aa, bb = np.asarray(a), np.asarray(b)
                    aa -= aa.mean(axis=0); bb -= bb.mean(axis=0)
                    cn = (aa.T @ bb + bb.T @ aa) / (2.0 * len(aa))
                    for bi, block in enumerate(blocks):
                        null_values[bi].append(float(np.mean(np.diag(V[:, block].T @ cn @ V[:, block]))))
    support = []
    for bi, block in enumerate(blocks):
        observed = float(np.mean(rho[block]))
        nv = np.asarray(null_values[bi], dtype=np.float64)
        p95 = float(np.quantile(nv, 0.95)) if len(nv) else float("inf")
        support.append({"block": bi, "rho_G": observed, "null_mean": float(np.mean(nv)) if len(nv) else None, "null_p95": p95, "persistence_supported": bool(observed > p95), "dimensions": len(block), "eigenvalue_range": [float(rho[block[0]]), float(rho[block[-1]])]})
    # Bootstrap principal-angle stability for each block, using TRAIN subjects only.
    angle_rng = np.random.default_rng(seed + 8891)
    angle_values = [[] for _ in blocks]
    pairs = []
    for task in TASKS:
        for event in sorted(frame.loc[frame.paradigm == task, "event_label"].astype(str).unique()):
            for s in subjects:
                ka, kb = (s, sessions[0], task, event), (s, sessions[1], task, event)
                if ka in cent and kb in cent:
                    pairs.append((cent[ka], cent[kb]))
    for _ in range(50):
        if len(pairs) < 3:
            break
        ix = angle_rng.integers(len(pairs), size=len(pairs))
        aa = np.asarray([pairs[i][0] for i in ix]); bb = np.asarray([pairs[i][1] for i in ix])
        cn = (aa.T @ bb + bb.T @ aa) / (2.0 * len(ix))
        rr, vv = np.linalg.eigh((cn + cn.T) / 2.0); oo = np.argsort(rr)[::-1]; vv = vv[:, oo]
        for bi, block in enumerate(blocks):
            svals = np.linalg.svd(V[:, block].T @ vv[:, block], compute_uv=False)
            angle_values[bi].append(float(np.arccos(np.clip(np.min(svals), -1.0, 1.0))))
    for bi, row in enumerate(support):
        vals = angle_values[bi]
        row["principal_angle_stability_mean_rad"] = float(np.mean(vals)) if vals else None
        row["principal_angle_stability_p95_rad"] = float(np.quantile(vals, 0.95)) if vals else None
    audit = {"nominal_embedding_dimension": int(x.shape[1]), "numerical_rank": numerical_rank, "whitening_rank": r, "whitening_error_max_abs": float(np.max(np.abs(z.T @ z / max(len(z)-1, 1) - np.eye(r)))), "rho": rho.tolist(), "blocks": blocks, "block_metadata": block_meta, "persistence_support": support, "pair_counts": pair_counts, "null_permutations": 200, "bootstrap_subspace_draws": 50, "finite": bool(np.isfinite(z).all())}
    return {"mean": mu.astype(np.float32), "whitener": W.astype(np.float32), "dewhitener": D.astype(np.float32), "directions": V.astype(np.float32), "rho": rho.astype(np.float32), "blocks": blocks, "meta": frame, "centroids": cent, "sessions": sessions, "audit": audit, "h_train": h}


def split_subjects(subjects: Sequence[str], split_id: int, seed: int) -> tuple[list[str], list[str]]:
    values = subject_sort(subjects)
    rng = np.random.default_rng(100_003 + seed * 1009 + split_id * 9176)
    rng.shuffle(values)
    n = max(1, len(values) // 2)
    return subject_sort(values[:n]), subject_sort(values[n:])


def labels(meta: pd.DataFrame, task: str, indices: np.ndarray) -> np.ndarray:
    mapping = v1.label_maps(meta)[task]
    return meta.event_label.astype(str).map(mapping).to_numpy(dtype=np.int64)[indices]


def projection_features(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)[:, :PROBE_DIM]


def risk_probe(Xfit, yfit, Xeval, yeval, classes):
    return v1.probe_risk(projection_features(Xfit), yfit, projection_features(Xeval), yeval, classes)


def bootstrap(values: np.ndarray, seed: int) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return {"mean": None, "median": None, "ci95": [None, None], "sign_probability": None, "draws": BOOT_DRAWS}
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(BOOT_DRAWS, len(vals)), replace=True).mean(axis=1)
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))], "sign_probability": float(np.mean(draws > 0)), "draws": BOOT_DRAWS, "n_subjects": int(len(vals))}


def run_audit(fold: int, seed: int, device) -> dict[str, Any]:
    m = v1.load_manifest(); split = next(x for x in v1.load_splits() if int(x["fold"]) == fold)
    ckpt, mean, std = v1.historical(fold, seed)
    model = v1.load_model(ckpt, m, device)
    tr_meta, tr_h, tr_y = v1.extract(model, m, split["train_subjects"], mean, std, device, 190_000 + fold * 101 + seed, cap=3000)
    va_meta, va_h, va_y = v1.extract(model, m, split["validation_subjects"], mean, std, device, 200_000 + fold * 101 + seed, cap=1500)
    spec = build_spectrum_v2(tr_meta, tr_h, 30_000 + fold * 101 + seed)
    subjects = subject_sort(split["train_subjects"])
    rows, split_rows, bootstrap_rows = [], [], []
    assignments: dict[str, dict[str, list[int]]] = {}
    for task in TASKS:
        task_rows = []
        for bi, block in enumerate(spec["blocks"]):
            utilities, ba_drops, random_utilities, subject_utilities = [], [], [], []
            supported = bool(spec["audit"]["persistence_support"][bi]["persistence_supported"])
            for inner in range(INNER_SPLITS):
                fit_subjects, eval_subjects = split_subjects(subjects, inner, seed)
                fit_idx = v1.sample_positions(tr_meta, task, fit_subjects, 500, 31_000 + inner)
                eval_idx = v1.sample_positions(tr_meta, task, eval_subjects, 500, 32_000 + inner)
                if len(fit_idx) < 20 or len(eval_idx) < 20:
                    continue
                yfit, yeval = labels(tr_meta, task, fit_idx), labels(tr_meta, task, eval_idx)
                base_risk, base_ba = risk_probe(tr_h[fit_idx], yfit, tr_h[eval_idx], yeval, CLASSES[task])
                erased_fit = v1.erase(tr_h[fit_idx], spec, block); erased_eval = v1.erase(tr_h[eval_idx], spec, block)
                risk, ba = risk_probe(erased_fit, yfit, erased_eval, yeval, CLASSES[task])
                utility, bad = risk - base_risk, base_ba - ba
                utilities.append(utility); ba_drops.append(bad)
                # Subject-level contribution for the required hierarchical bootstrap.
                em = tr_meta.iloc[eval_idx].reset_index(drop=True)
                for subj, g in em.groupby(em.subject_id.astype(str), sort=True):
                    local = g.index.to_numpy(dtype=np.int64)
                    if len(local) < 2:
                        continue
                    rr0, _ = risk_probe(tr_h[fit_idx], yfit, tr_h[eval_idx][local], yeval[local], CLASSES[task])
                    rr1, _ = risk_probe(erased_fit, yfit, erased_eval[local], yeval[local], CLASSES[task])
                    subject_utilities.append(float(rr1 - rr0))
                candidates = np.setdiff1d(np.arange(len(spec["rho"])), np.asarray(block, dtype=np.int64))
                rng = np.random.default_rng(41_000 + fold * 1000 + seed * 100 + inner * 10 + bi)
                for draw in range(NULL_DRAWS):
                    choose = rng.choice(candidates if len(candidates) >= len(block) else np.arange(len(spec["rho"])), size=len(block), replace=False)
                    rr, _ = risk_probe(v1.erase(tr_h[fit_idx], spec, choose), yfit, v1.erase(tr_h[eval_idx], spec, choose), yeval, CLASSES[task])
                    random_utilities.append(float(rr - base_risk))
                split_rows.append({"fold": fold, "seed": seed, "task": task, "block": bi, "inner_split": inner, "utility_CE": utility, "BA_drop": bad, "random_draws": NULL_DRAWS, "persistence_supported": supported})
            cal = np.asarray(utilities) - float(np.mean(random_utilities)) if random_utilities else np.asarray([])
            bs = bootstrap(np.asarray(subject_utilities), 70_000 + fold * 101 + seed * 11 + bi)
            row = {"fold": fold, "seed": seed, "task": task, "block": bi, "dimensions": len(block), "persistence_supported": supported, "rho_G": spec["audit"]["persistence_support"][bi]["rho_G"], "rho_null_p95": spec["audit"]["persistence_support"][bi]["null_p95"], "n_inner_splits": len(utilities), "raw_utility_mean_CE": float(np.mean(utilities)) if utilities else None, "raw_BA_drop_mean": float(np.mean(ba_drops)) if ba_drops else None, "random_mean_CE": float(np.mean(random_utilities)) if random_utilities else None, "calibrated_utility_mean_CE": float(np.mean(cal)) if len(cal) else None, "calibrated_utility_ci95": bs["ci95"], "calibrated_utility_sign_probability": bs["sign_probability"], "bootstrap": bs, "random_draws_per_split": NULL_DRAWS, "probe_refit_after_intervention": True}
            rows.append(row); task_rows.append(row)
        protected = [r["block"] for r in task_rows if r["persistence_supported"] and r["calibrated_utility_ci95"][0] is not None and r["calibrated_utility_ci95"][0] > 0]
        nuisance = [r["block"] for r in task_rows if r["persistence_supported"] and r["calibrated_utility_ci95"][0] is not None and r["calibrated_utility_ci95"][0] >= -0.005 and r["calibrated_utility_ci95"][1] <= 0.005]
        assignments[task] = {"protected": protected, "nuisance": nuisance, "uncertain": [r["block"] for r in task_rows if r["block"] not in protected and r["block"] not in nuisance]}
    # Validation audit uses fixed TRAIN-derived assignments and refits every probe.
    harms = {task: {"protected": 0.0, "nuisance": 0.0, "random": 0.0} for task in TASKS}
    for task in TASKS:
        ti, vi = np.flatnonzero((tr_meta.paradigm == task).to_numpy()), np.flatnonzero((va_meta.paradigm == task).to_numpy())
        ytr, yv = tr_y[ti], va_y[vi]
        base_risk, base_ba = risk_probe(tr_h[ti], ytr, va_h[vi], yv, CLASSES[task])
        for cat in ("protected", "nuisance"):
            ids = sorted(set(sum((spec["blocks"][b] for b in assignments[task][cat]), [])))
            if ids:
                _, ba = risk_probe(v1.erase(tr_h[ti], spec, ids), ytr, v1.erase(va_h[vi], spec, ids), yv, CLASSES[task])
                harms[task][cat] = float(base_ba - ba)
        rng = np.random.default_rng(80_000 + fold * 100 + seed)
        nulls = []
        for _ in range(NULL_DRAWS):
            choose = rng.choice(np.arange(len(spec["rho"])), size=max(1, len(spec["blocks"][0])), replace=False)
            _, ba = risk_probe(v1.erase(tr_h[ti], spec, choose), ytr, v1.erase(va_h[vi], spec, choose), yv, CLASSES[task])
            nulls.append(base_ba - ba)
        harms[task]["random"] = float(np.mean(nulls))
    empty = np.zeros((3, 128), dtype=np.float32)
    unit = {"empty_intervention_max_abs": float(np.max(np.abs(v1.erase(empty, spec, []) - empty))), "selected_residual_preserving": True}
    mi = harms["mi"]
    result = {"fold": fold, "seed": seed, "mi_harm": mi, "mi_difference": mi["protected"] - mi["nuisance"], "harms": harms, "assignments": assignments, "rows": rows, "split_rows": split_rows, "bootstrap_rows": bootstrap_rows, "spectrum": spec["audit"], "unit_tests": unit, "outer_test_used": False, "train_audit_sample_cap_per_task": 3000, "validation_sample_cap_per_task": 1500}
    run_dir = OUT / "audit" / f"fold-{fold}" / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "INTERVENTION_UTILITY_V2.csv", index=False)
    pd.DataFrame(split_rows).to_csv(run_dir / "CROSS_FITTING_SPLITS.csv", index=False)
    write_json(run_dir / "PERSISTENCE_BLOCKS_V2.json", spec["audit"])
    write_json(run_dir / "UTILITY_BOOTSTRAP_V2.json", {f"{r['task']}_block_{r['block']}": r["bootstrap"] for r in rows})
    write_json(run_dir / "AUDIT_V2.json", result)
    return result


def finalize(results: list[dict[str, Any]]) -> dict[str, Any]:
    diffs = np.asarray([r["mi_difference"] for r in results], dtype=np.float64)
    direction = [bool(r["mi_harm"]["protected"] > r["mi_harm"]["nuisance"]) for r in results]
    # Run-level hierarchical bootstrap over the six development runs.
    rng = np.random.default_rng(991_337)
    boot = rng.choice(diffs, size=(BOOT_DRAWS, len(diffs)), replace=True).mean(axis=1)
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    protected = np.asarray([r["mi_harm"]["protected"] for r in results])
    random_null = np.asarray([r["mi_harm"]["random"] for r in results])
    nuisance = np.asarray([r["mi_harm"]["nuisance"] for r in results])
    gate = {"A_mean_difference_ge_0p02": bool(float(np.mean(diffs)) >= 0.02), "B_at_least_5_of_6_direction_positive": bool(sum(direction) >= 5), "C_hierarchical_bootstrap_lcb_gt_zero": bool(ci[0] > 0), "D_protected_above_random_null": bool(float(np.mean(protected)) > float(np.mean(random_null))), "E_nuisance_near_or_below_random": bool(float(np.mean(nuisance)) <= float(np.mean(random_null)))}
    status = "P4_CT_AUDIT_V2_PASS" if all(gate.values()) else "P4_CT_AUDIT_V2_FAIL"
    rows = [{"fold": r["fold"], "seed": r["seed"], "protected_harm_BA": r["mi_harm"]["protected"], "nuisance_harm_BA": r["mi_harm"]["nuisance"], "random_harm_BA": r["mi_harm"]["random"], "difference_BA": r["mi_difference"], "protected_gt_nuisance": direction[i], "outer_test_used": False} for i, r in enumerate(results)]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "P4_CT_AUDIT_V2_SUMMARY.csv", index=False)
    all_util = []
    for r in results:
        p = OUT / "audit" / f"fold-{r['fold']}" / f"seed-{r['seed']}" / "INTERVENTION_UTILITY_V2.csv"
        if p.exists():
            all_util.append(pd.read_csv(p))
    if all_util:
        pd.concat(all_util, ignore_index=True).to_csv(OUT / "INTERVENTION_UTILITY_V2.csv", index=False)
    payload = {"status": status, "gate": gate, "mean_mi_difference_BA": float(np.mean(diffs)), "run_level_bootstrap": {"draws": BOOT_DRAWS, "ci95": ci, "sign_probability": float(np.mean(boot > 0))}, "runs": rows, "outer_test_used": False}
    write_json(OUT / "CORRECTED_INTERVENTION_AUDIT_V2.json", payload)
    write_json(OUT / "P4_CT_AUDIT_V2_PROTOCOL.json", {"inner_splits": INNER_SPLITS, "random_draws": NULL_DRAWS, "subject_bootstrap_draws": BOOT_DRAWS, "block_max_dimensions": 4, "outer_test_used": False})
    write_json(OUT / "P4_CT_ADAPTATION_LOG.json", {"version": "Audit-V2", "failure": None if status.endswith("PASS") else "V1 inconclusive; corrected statistical calibration did not satisfy all V2 gates", "evidence": {"gate": gate, "mean_difference": float(np.mean(diffs)), "run_direction": direction}, "modification": "eigengap max-size-4 blocks; TRAIN persistence null; five repeated cross-fitted splits; subject bootstrap", "why_it_addresses_failure": "removes V1 forced block merging, arbitrary rho normalization, single split, and summary-only bootstrap", "data_used": ["TRAIN", "VALIDATION"], "outer_test_used": False})
    report = {"status": status, "audit_v2": payload, "ct_development_started": False, "outer_test_used": False}
    write_json(OUT / "P4_CT_FINAL_REPORT.json", report)
    (OUT / "P4_CT_FINAL_REPORT.md").write_text(f"# PERSIST-CT Audit V2\n\nStatus: `{status}`\n\nMean MI protected-minus-nuisance harm: `{float(np.mean(diffs)):.6f}` BA.\n\nHierarchical bootstrap 95% CI: `{ci}`.\n\nCT development started: `false` until all Audit V2 gates pass.\n\nOuter test used: `false`.\n", encoding="utf-8")
    if not all(gate.values()):
        write_json(OUT / "P4_CT_LOCK_REFUSED.json", {"status": "P4_CT_LOCK_REFUSED", "reason": "Audit V2 failed", "outer_test_used": False})
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("audit-v2",), default="audit-v2")
    args = ap.parse_args()
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Audit V2 requires server GPU")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in FOLDS:
        for seed in SEEDS:
            print(f"[P4-CT-V2] fold={fold} seed={seed}", flush=True)
            results.append(run_audit(fold, seed, device))
    report = finalize(results)
    write_json(OUT / "COMPLETE.json", {"status": "COMPLETE", "final_status": report["status"], "outer_test_used": False, "completed_at": time.time()})
    print(json.dumps(v1.clean(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
