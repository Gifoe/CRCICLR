"""TRAIN pseudo-target tuning, frozen OUTER evaluation and compact fold audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

CORE_PATH = Path(os.environ.get("PCDA_CORE_CODE", str(Path(__file__).with_name("run.py"))))
import importlib.util
spec = importlib.util.spec_from_file_location("pcda_core", CORE_PATH)
assert spec and spec.loader
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
ALPHAS = tuple(map(float, core.ALPHAS))
PAIRS = tuple((p, c) for p in ALPHAS for c in ALPHAS)
REAL = ("PROTECTED", "PERSIST_ALL", "PCA_R", "SUPERVISED_DECISION_R")
EPS = 1e-12


def metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = np.argmax(logits, axis=1)
    t0 = y == 0; t1 = y == 1
    if not t0.any() or not t1.any(): raise RuntimeError("evaluation subject lacks a class")
    ba = float(((pred[t0] == 0).mean() + (pred[t1] == 1).mean()) / 2)
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    u = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(u); prob /= prob.sum(axis=1, keepdims=True)
    nll = float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())
    return {"BA": ba, "macro_F1": f1, "NLL": nll}


def projector(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0: return np.zeros_like(a)
    return (a @ q) @ q.T


def orthonormal(columns: np.ndarray, rank: int) -> np.ndarray:
    if rank == 0: return np.empty((columns.shape[0], 0), np.float64)
    u, s, _ = np.linalg.svd(np.asarray(columns, np.float64), full_matrices=False)
    if len(s) < rank or s[rank - 1] < s[0] * 1e-8: raise RuntimeError("control basis rank deficient")
    return u[:, :rank]


def source_stats(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = h.mean(axis=0, dtype=np.float64)
    sd = np.maximum(h.std(axis=0, dtype=np.float64), 1e-6)
    return mu, sd


def delta_affine(eval_h: np.ndarray, context_h: np.ndarray, mu_src: np.ndarray,
                 std_src: np.ndarray, lam: float) -> np.ndarray:
    mu_tgt = context_h.mean(axis=0, dtype=np.float64)
    std_tgt = context_h.std(axis=0, dtype=np.float64)
    mu_hat = (1 - lam) * mu_src + lam * mu_tgt
    std_hat = (1 - lam) * std_src + lam * std_tgt
    adapted = mu_src + std_src / (std_hat + core.EPS) * (eval_h - mu_hat)
    delta = adapted - eval_h
    if not np.isfinite(delta).all(): raise RuntimeError("non-finite adaptation")
    return delta


def context_indices(y: np.ndarray, task: str, budget: int, subject: str, fold: int) -> tuple[np.ndarray, dict]:
    if task != "OpenBMI_ERP":
        if len(y) < budget: raise RuntimeError(f"MI context shorter than {budget}: {subject}/f{fold}")
        return np.arange(budget), {"unit": "trial", "units": budget, "epochs": budget}
    # Cache rows preserve stimulus order. One repetition is twelve contiguous
    # flashes with exactly two target codes. Never inspect labels to choose
    # individual target or nontarget flashes: only complete sequences qualify.
    complete = len(y) // 12
    if complete < budget: raise RuntimeError(f"ERP context has fewer than {budget} complete sequences")
    counts = np.asarray([int(y[i * 12:(i + 1) * 12].sum()) for i in range(complete)])
    invalid = np.flatnonzero(counts != 2)
    contiguous = int(invalid[0]) if len(invalid) else complete
    if contiguous < budget:
        raise RuntimeError(f"ERP first {budget} sequences lack validated 12-flash grouping: {subject}/f{fold}")
    return np.arange(12 * budget), {"unit": "validated_12_stimulus_sequence", "units": budget,
                                      "epochs": 12 * budget, "incomplete_tail": int(len(y) % 12),
                                      "first_invalid_sequence": None if not len(invalid) else int(invalid[0])}


def base_episode(data: dict, subject: str, budget: int, lam: float, task: str, fold: int) -> dict:
    h, y, sub, sess = (data[k] for k in ("h", "y", "subject", "session"))
    source_mask = (sess == 1) & (sub != subject)
    context_mask = (sess == 1) & (sub == subject)
    eval_mask = (sess == 2) & (sub == subject)
    src_h = np.asarray(h[source_mask], np.float64)
    context_h = np.asarray(h[context_mask], np.float64)
    context_y = np.asarray(y[context_mask], np.int64)
    eval_h = np.asarray(h[eval_mask], np.float64)
    eval_y = np.asarray(y[eval_mask], np.int64)
    ix, ctx = context_indices(context_y, task, budget, subject, fold)
    mu, sd = source_stats(src_h)
    delta = delta_affine(eval_h, context_h[ix], mu, sd, lam)
    w = np.asarray(data["head_weight"], np.float64)
    bias = np.asarray(data["head_bias"], np.float64)
    return {"subject": subject, "h": eval_h, "y": eval_y, "base": eval_h @ w.T + bias,
            "delta": delta, "context": ctx, "source_mu": mu, "source_std": sd,
            "source_subject_count": int(len(set(sub[source_mask]))), "eval_rows": len(eval_y)}


def scores(ep: dict, q: np.ndarray, pair: tuple[float, float]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    dp = projector(ep["delta"], q)
    dc = ep["delta"] - dp
    p, c = pair
    update = p * dp + c * dc
    w = np.asarray(ep["w"], np.float64)
    logits = ep["base"] + update @ w.T
    m = metrics(ep["y"], logits)
    m.update({"P_drift": float(np.linalg.norm(p * dp, axis=1).mean()),
              "C_drift": float(np.linalg.norm(c * dc, axis=1).mean()),
              "total_drift": float(np.linalg.norm(update, axis=1).mean()),
              "update_norm_sq": float(np.mean(np.sum(update ** 2, axis=1))),
              "generic_norm_sq": float(np.mean(np.sum(ep["delta"] ** 2, axis=1))),
              "classifier_margin_drift": float(np.mean(np.abs((logits[:, 1] - logits[:, 0]) -
                                                            (ep["base"][:, 1] - ep["base"][:, 0]))))})
    return m, logits, dp, dc


def avg(rows: list[dict], keys: tuple[str, ...] | None = None) -> dict:
    if not rows: raise RuntimeError("cannot average empty subject rows")
    keys = keys or tuple(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def choose(rows: list[dict], *, uniform_only: bool = False) -> dict:
    allowed = [r for r in rows if not uniform_only or r["alpha_P"] == r["alpha_C"]]
    if not allowed: raise RuntimeError("no alpha candidates")
    return min(allowed, key=lambda r: (-r["BA"], r["NLL"], r["update_norm_sq"], r["alpha_P"], r["alpha_C"]))


def supervised_basis(h: np.ndarray, y: np.ndarray, sub: np.ndarray, sess: np.ndarray,
                     rank: int, task: str) -> np.ndarray:
    if rank == 0: return np.empty((h.shape[1], 0), np.float64)
    scaler = StandardScaler().fit(h)
    std = np.maximum(scaler.scale_.astype(np.float64), 1e-6)
    normals = []
    for subject in sorted(set(sub), key=int):
        for session in (1, 2):
            mask = (sub == subject) & (sess == session)
            if set(y[mask]) != {0, 1}: raise RuntimeError("local supervised control class missing")
            clf = core.SDG.classifier(task).fit(scaler.transform(h[mask]), y[mask])
            normal = clf.coef_[0].astype(np.float64) / std
            normal /= max(np.linalg.norm(normal), EPS)
            normals.append(normal)
    _, s, vt = np.linalg.svd(np.asarray(normals), full_matrices=False)
    if len(s) < rank or s[rank - 1] < s[0] * 1e-8: raise RuntimeError("supervised control rank deficient")
    return vt[:rank].T


def random_basis(dim: int, rank: int, model: str, task: str, fold: int, half: int, draw: int) -> np.ndarray:
    if rank == 0: return np.empty((dim, 0), np.float64)
    a = np.random.default_rng(core.seed("partition", model, task, fold, half, draw)).normal(size=(dim, rank))
    return np.linalg.qr(a, mode="reduced")[0]


def partition_bases(data: dict, audit: dict, model: str, task: str, fold: int):
    h, y, sub, sess = (data[k] for k in ("h", "y", "subject", "session"))
    halves = audit["crossfit_half_subjects"]
    bases = {}
    final_rank = int(audit["full"]["rank_protected"])
    for half in (0, 1):
        fit_subjects = halves[1 - half]
        mask = np.isin(sub, fit_subjects)
        p = np.asarray(data[f"p_half{half}"], np.float64)
        r = p.shape[1] if p.shape[1] else final_rank
        hb = np.asarray(h[mask], np.float64)
        bases[half] = {"PROTECTED": p, "PERSIST_ALL": np.asarray(data[f"persist_half{half}"], np.float64),
                       "PCA_R": PCA(n_components=r, svd_solver="full").fit(hb).components_.T if r else np.empty((h.shape[1], 0)),
                       "SUPERVISED_DECISION_R": supervised_basis(hb, y[mask], sub[mask], sess[mask], r, task)}
        bases[half]["RANDOM_R"] = [random_basis(h.shape[1], r, model, task, fold, half, draw) for draw in range(100)]
    return bases


def select(model: str, task: str, fold: int) -> None:
    name = core.cell_id(model, task, fold)
    cache = core.RUNTIME / "train" / f"{name}.npz"
    audit_path = core.RUNTIME / "train" / f"{name}.json"
    target = core.RUNTIME / "selection" / f"{name}.json"
    if target.exists(): raise FileExistsError(target)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if core.sha(cache) != audit["train_cache_sha256"]: raise RuntimeError("TRAIN cache SHA drift")
    with np.load(cache, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    subjects = sorted(set(data["subject"]), key=int)
    half_lookup = {str(s): k for k, group in enumerate(audit["crossfit_half_subjects"]) for s in group}
    if len(half_lookup) != len(subjects): raise RuntimeError("crossfit subject coverage mismatch")
    budgets = tuple(core.PROTOCOL["erp_context_budgets_sequences" if task == "OpenBMI_ERP" else "mi_context_budgets_trials"])
    # Select lambda and context on the identical homogeneous family, entirely
    # inside TRAIN. This shared choice cannot advantage Protected partitions.
    uniform_grid = []
    for budget in budgets:
        for lam in core.LAMBDAS:
            per_alpha = {a: [] for a in ALPHAS}
            for subject in subjects:
                ep = base_episode(data, subject, budget, lam, task, fold)
                ep["w"] = data["head_weight"]
                for a in ALPHAS:
                    per_alpha[a].append(scores(ep, np.empty((data["h"].shape[1], 0)), (a, a))[0])
            for a in ALPHAS:
                row = avg(per_alpha[a], ("BA", "macro_F1", "NLL", "total_drift", "update_norm_sq"))
                uniform_grid.append({"budget": budget, "lambda": lam, "alpha": a, **row})
    selected_base = min(uniform_grid, key=lambda r: (-r["BA"], r["NLL"], r["budget"], r["lambda"], r["alpha"]))
    budget, lam = int(selected_base["budget"]), float(selected_base["lambda"])
    print("TRAIN_BASE_SELECTED", name, budget, lam, flush=True)
    episodes = {}
    for subject in subjects:
        ep = base_episode(data, subject, budget, lam, task, fold)
        ep["w"] = data["head_weight"]
        ep["half"] = half_lookup[str(subject)]
        episodes[str(subject)] = ep
    bases = partition_bases(data, audit, model, task, fold)
    surfaces = []
    selections = {}
    for family in REAL:
        for p, c in PAIRS:
            by_subject = [scores(episodes[str(s)], bases[half_lookup[str(s)]][family], (p, c))[0] for s in subjects]
            surfaces.append({"partition": family, "alpha_P": p, "alpha_C": c,
                             **avg(by_subject)})
        family_rows = [r for r in surfaces if r["partition"] == family]
        selections[family] = {"best": choose(family_rows), "uniform": choose(family_rows, uniform_only=True)}
        print("TRAIN_PARTITION_SELECTED", name, family, selections[family]["best"]["alpha_P"],
              selections[family]["best"]["alpha_C"], flush=True)
    random_rows = []
    for draw in range(100):
        rows = []
        for p, c in PAIRS:
            by_subject = [scores(episodes[str(s)], bases[half_lookup[str(s)]]["RANDOM_R"][draw], (p, c))[0]
                          for s in subjects]
            rows.append({"partition": "RANDOM_R", "draw": draw, "alpha_P": p, "alpha_C": c, **avg(by_subject)})
        random_rows.append({"draw": draw, "best": choose(rows), "uniform": choose(rows, uniform_only=True)})
        if draw % 10 == 9: print("TRAIN_RANDOM_PARTITIONS", name, draw + 1, flush=True)
    chosen = selections["PROTECTED"]["best"]
    ratio = float(np.sqrt(max(chosen["update_norm_sq"], 0) / max(chosen["generic_norm_sq"], EPS)))
    ratio = float(np.clip(ratio, 0, 1))
    result = {"schema": "PCDA_TRAIN_SELECTION_V1", "model": model, "task": task, "fold": fold,
              "status": "PROTECTED_NOT_ESTIMABLE" if audit["full"]["rank_protected"] == 0 or
                        any(r["rank_protected"] == 0 for r in audit["crossfit"]) else "COMPLETE",
              "train_cache_sha256": core.sha(cache), "train_audit_sha256": core.sha(audit_path),
              "protocol_sha256": core.sha(core.EXP / "protocol/PROTOCOL_LOCK.json"),
              "source_only_selected_base": selected_base, "all_budget_lambda_uniform_sensitivity": uniform_grid,
              "response_surface": surfaces, "partition_selections": selections,
              "random_partition_selections": random_rows, "normmatched_uniform_alpha": ratio,
              "train_subject_count": len(subjects), "train_subject_ids": subjects,
              "final_heldout_eeg_reads": 0, "outer_development_accessed": False}
    core.write_json_new(target, result)
    print("SELECT_COMPLETE", name, core.sha(target), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "outer"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "select": select(args.model, args.task, args.fold)
    else: raise NotImplementedError("OUTER stage not yet implemented")


if __name__ == "__main__": main()
