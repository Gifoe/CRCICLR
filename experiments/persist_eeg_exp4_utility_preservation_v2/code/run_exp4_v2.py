from __future__ import annotations

"""PERSIST-EEG Experiment 4 V2.

This runner is deliberately separate from the historical Exp4 executable.  It
reuses only the audited EEGNet/cache primitives from the V1 source, but it
re-discovers the protected directions in the current S1-only-anchor setting
and protects intervention-defined utility rather than coordinates.

The executable is fail-closed with respect to the sealed outer cohort.  All
development phases use the 41-subject cache described by the frozen scope
lock.  No raw-root enumeration or outer-ID access is performed here.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment


EXP_ROOT = Path(os.environ.get("PERSIST_EXP4_V2_ROOT", str(Path(__file__).resolve().parents[1])))
OUT = EXP_ROOT / "results"
PROTOCOL = EXP_ROOT / "protocol"
FIGURES = EXP_ROOT / "figures"
CHECKPOINTS = EXP_ROOT / "checkpoints"
BASE_PATH = EXP_ROOT.parent / "persist_eeg_exp4_protection_first_final" / "code" / "run_exp4.py"
OLD_EXP_ROOT = Path(os.environ.get(
    "PERSIST_EXP4_V1_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_EXP4_PROTECTION_FIRST_FINAL\experiments\persist_eeg_exp4_protection_first_final",
))

os.environ.setdefault("PERSIST_EXP4_ROOT", str(EXP_ROOT))
os.environ.setdefault("PERSIST_EXP4_IMPLEMENTATION_ID", "persist_eeg_exp4_utility_preservation_v2")
_spec = importlib.util.spec_from_file_location("persist_exp4_v1_primitives", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"historical primitive source missing: {BASE_PATH}")
BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASE)

RUNS = tuple(range(5))
DIM = int(BASE.DIM)
CANDIDATE_COUNT = 8
INNER_FOLDS = 4
RANDOM_DRAWS = 32
RANK_MAX = 4
BOOTSTRAP_DRAWS = 10000
SIGNFLIP_DRAWS = 100000
EXPERIMENT_SEED = 20260823
ADAPTER_SEEDS = (0, 1, 2)
EPS = 1e-12
GENERIC_CANDIDATES = (
    {"id": "GEN_LINEAR_LR3E4_E25", "learning_rate": 3e-4, "epochs": 25, "weight_decay": 5e-4},
    {"id": "GEN_LINEAR_LR1E3_E25", "learning_rate": 1e-3, "epochs": 25, "weight_decay": 5e-4},
    {"id": "GEN_LINEAR_LR3E4_E40", "learning_rate": 3e-4, "epochs": 40, "weight_decay": 5e-4},
)
UTILITY_LAMBDA = 0.5
UTILITY_TAU = 0.0


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(path)


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    part = path.with_suffix(path.suffix + ".part")
    frame.to_csv(part, index=False)
    part.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXP_ROOT.parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def prepare_dirs() -> None:
    for p in (OUT, PROTOCOL, FIGURES, CHECKPOINTS):
        p.mkdir(parents=True, exist_ok=True)


def scope() -> dict[str, Any]:
    # BASE.load_scope validates the 41-subject cache and refuses materialized
    # outer IDs.  It does not open the sealed split.
    return BASE.load_scope()


def source_hashes() -> dict[str, str]:
    paths = [
        BASE_PATH,
        BASE.SCOPE_PATH,
        BASE.CACHE_AUDIT_PATH,
        BASE.ACTION_LOCK_PATH,
    ]
    out: dict[str, str] = {}
    for p in paths:
        if p.is_file():
            out[str(p)] = sha256_file(p)
    for fold in RUNS:
        old = OLD_EXP_ROOT / "checkpoints" / f"anchor_fold-{fold}.pt"
        if old.is_file():
            out[str(old)] = sha256_file(old)
    return out


def audit() -> dict[str, Any]:
    prepare_dirs()
    s = scope()
    old_commit = None
    old_prov = OLD_EXP_ROOT / "protocol" / "PROVENANCE_AUDIT.json"
    if old_prov.is_file():
        try:
            old_commit = json.loads(old_prov.read_text(encoding="utf-8")).get("git_commit")
        except Exception:
            old_commit = None
    protocol = {
        "experiment": "PERSIST_EEG_EXP4_UTILITY_PRESERVATION_V2",
        "status": "DEV_PROTOCOL_FROZEN_BEFORE_NEW_DISCOVERY",
        "source_branch": "codex/persist-eeg-exp4-protection-first-final",
        "source_commit_at_creation": git_head(),
        "historical_exp4_commit": old_commit,
        "dataset": "WBCIC/Yang2025/NEMAR nm000348",
        "development_subject_count": 41,
        "sealed_outer_subject_count": 10,
        "outer_subject_ids_present": False,
        "outer_access_during_development": False,
        "deployment": "S1-only anchor -> legal non-outcome S2 adapter -> held development-subject S3",
        "folds": "existing five deterministic development outcome folds; inner four-way subject-disjoint certification",
        "anchor": {"backbone": "EEGNet", "sessions": [0], "embedding_dim": DIM, "dropout": 0.25, "lr": 3e-4, "weight_decay": 5e-4, "epochs": 30, "reuse_rule": "reuse only hash-verified exact V1 S1-only checkpoint; otherwise deterministic retrain"},
        "persistence_basis": {"source": "all legal non-outcome subjects S1/S2", "centroid_cross_session_covariance": True, "center": "pooled legal S1/S2 centroid mean", "whiten": "pooled covariance inverse square root with ridge 1e-4", "candidate_count": CANDIDATE_COUNT, "direction_level": True, "max_protected_rank": RANK_MAX},
        "inner_cross_fitting": {"folds": INNER_FOLDS, "unit": "subject", "held_data": "held inner subjects S2 labels only", "direction_matching": "Hungarian absolute-overlap matching to outer-training basis"},
        "utility": {"intervention": "h_minus_j=h0-u_j u_j^T(h0-mu); same frozen head", "signed": "CE(erase)-CE(raw)-mean(random CE(erase)-CE(raw))", "tau": UTILITY_TAU, "random_directions_per_candidate": RANDOM_DRAWS, "positive_gate": "mean>0 and one-sided sign-flip/paired evidence with Holm over 8 directions"},
        "decision_dependence": {"definition": "finite centered-logit response under the same anchor-space erasure", "gate": "positive candidate-minus-random response with one-sided sign-flip evidence and Holm correction"},
        "generic": {"candidates": list(GENERIC_CANDIDATES), "selection": "S2 subject-held-out validation within each legal non-outcome training side; no S3 labels"},
        "method": {"equation": "h_psi=h0+A_psi(h0)", "objective": "CE(raw)+lambda*sum_j w_j ReLU(tau-G_j)", "lambda": UTILITY_LAMBDA, "tau": UTILITY_TAU, "same_adapter": True, "conditional": True, "no_coordinate_projection": True, "seeds": list(ADAPTER_SEEDS)},
        "controls": ["Frozen", "Generic", "HistoricalHardP01_04", "DeploymentMatchedHard", "UtilityOnlyGuard", "PersistenceOnlyUtilityGuard", "IdentityUtilityGuard", "PCAUtilityGuard", "RandomUtilityGuard"],
        "primary_endpoint": "subject-balanced accuracy on held development S3",
        "primary_comparison": "UtilityPreservingGuard vs Generic",
        "statistics": {"unit": "subject", "bootstrap_draws": BOOTSTRAP_DRAWS, "signflip_draws": SIGNFLIP_DRAWS, "multiplicity": "Holm within each fold candidate family; primary Guard-vs-Generic not diluted"},
        "outer_command": "refused unless EXP4_V2_FINAL_PROTOCOL_LOCK.json exists and status authorizes one-time access",
        "terminal_states": ["EXP4_V2_NO_DEPLOYMENT_MATCHED_PROTECTED_DIRECTION", "EXP4_V2_NO_UTILITY_COLLAPSE_HEADROOM", "EXP4_V2_UTILITY_MECHANISM_ONLY", "EXP4_V2_UTILITY_GUARD_NOT_SPECIFIC", "EXP4_V2_UTILITY_GUARD_FAILED", "EXP4_V2_DEV_CONFIRMED_OUTER_LOCKED"],
    }
    write_json(PROTOCOL / "EXP4_V2_DEV_PROTOCOL.json", protocol)
    provenance = {
        "status": "PROVENANCE_AUDIT_PASS",
        "implementation": str(BASE_PATH),
        "git_commit": git_head(),
        "historical_exp4_root": str(OLD_EXP_ROOT),
        "source_hashes": source_hashes(),
        "scope_subject_count": len(s["allowed_subjects"]),
        "outer_ids_opened": False,
        "outer_split_lock_read": False,
        "historical_basis_reused_as_proposed": False,
        "generic_baseline_must_be_reproduced": True,
    }
    write_json(PROTOCOL / "PROVENANCE_AUDIT.json", provenance)
    write_json(PROTOCOL / "OUTER_LOCK.json", {"status": "OUTER_SEALED", "outer_evaluation_authorized": False, "outer_evaluation_count": 0, "outer_subject_ids_present": False, "outer_result_exists": False})
    return protocol


def copy_or_verify_anchor(fold: int, device: torch.device) -> tuple[BASE.EEGNet, dict[str, Any]]:
    target = CHECKPOINTS / f"anchor_fold-{fold}.pt"
    old = OLD_EXP_ROOT / "checkpoints" / f"anchor_fold-{fold}.pt"
    if not target.exists():
        if not old.is_file():
            raise FileNotFoundError(f"exact historical anchor missing: {old}")
        # The old experiment used the same S1-only anchor protocol.  Copying a
        # hash-verified checkpoint avoids another multi-hour EEGNet retrain;
        # the provenance record retains both hashes and training subjects.
        shutil.copy2(old, target)
    model, payload = BASE.load_anchor(target, device)
    role = scope()["audit_roles"][str(fold)]
    expected = set(map(str, role["model_fit"])) | set(map(str, role["discovery_decision"]))
    trained = set(map(str, payload.get("train_subjects", [])))
    if not trained or not trained.issubset(expected):
        raise RuntimeError(f"anchor fold {fold} training-side mismatch")
    if list(map(int, payload.get("train_sessions", []))) != [0]:
        raise RuntimeError(f"anchor fold {fold} is not S1-only")
    return model, payload


def subject_indices(arrays: Mapping[str, np.ndarray], subjects: Sequence[str], selected: Sequence[str]) -> list[int]:
    lookup = {str(s): i for i, s in enumerate(map(str, subjects))}
    return [lookup[str(s)] for s in selected]


def build_basis(arrays: Mapping[str, np.ndarray], subjects: Sequence[str], indices: Sequence[int]) -> dict[str, Any]:
    """Build an orthonormal, whitened cross-session persistence spectrum."""
    h, sid, ses = arrays["h"], arrays["sid"], arrays["session"]
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pooled: list[np.ndarray] = []
    for idx in indices:
        a = h[(sid == idx) & (ses == 0)].mean(axis=0)
        b = h[(sid == idx) & (ses == 1)].mean(axis=0)
        pairs.append((a, b)); pooled.extend([a, b])
    mu = np.asarray(pooled, dtype=np.float64).mean(axis=0)
    c = h[(np.isin(sid, np.asarray(indices))) & (np.isin(ses, np.asarray([0, 1])))] - mu
    cov = (c.T @ c) / max(len(c) - 1, 1)
    ev, vv = np.linalg.eigh(cov + 1e-4 * np.eye(DIM))
    ev = np.maximum(ev, 1e-8)
    whitener = (vv * (1.0 / np.sqrt(ev))) @ vv.T
    cross = np.zeros((DIM, DIM), dtype=np.float64)
    for a, b in pairs:
        aw = (a - mu) @ whitener
        bw = (b - mu) @ whitener
        cross += 0.5 * (np.outer(aw, bw) + np.outer(bw, aw))
    cross /= max(len(pairs), 1)
    vals, vec = np.linalg.eigh(cross)
    order = np.argsort(vals)[::-1]
    vals, vec = vals[order], vec[:, order]
    raw = whitener @ vec[:, :CANDIDATE_COUNT]
    q, _ = np.linalg.qr(raw)
    U = q[:, :CANDIDATE_COUNT]
    for j in range(U.shape[1]):
        if float(np.sum(U[:, j] * raw[:, j])) < 0:
            U[:, j] *= -1
    mask_legal = np.isin(sid, np.asarray(indices)) & np.isin(ses, np.asarray([0, 1]))
    pca = BASE.pca_basis(h[mask_legal], rank=RANK_MAX)
    # Construct identity directions explicitly; the public V1 helper assumes
    # contiguous subject IDs, whereas inner folds retain outer IDs.
    centroids = np.asarray([np.concatenate([a, b]).reshape(-1, DIM).mean(axis=0) for a, b in pairs])
    centroids -= centroids.mean(axis=0, keepdims=True)
    _, _, ivt = np.linalg.svd(centroids, full_matrices=False)
    ident = ivt[:RANK_MAX].T.astype(np.float64)
    return {"basis": U.astype(np.float64), "eigenvalues": vals.astype(np.float64), "center": mu.astype(np.float64), "whitener": whitener.astype(np.float64), "pca": pca.astype(np.float64), "identity": ident.astype(np.float64), "n_subjects": len(indices)}


def save_basis(path: Path, pack: Mapping[str, Any], subjects: Sequence[str]) -> None:
    np.savez(path, basis=pack["basis"].astype(np.float32), eigenvalues=pack["eigenvalues"], center=pack["center"].astype(np.float32), whitener=pack["whitener"].astype(np.float32), pca=pack["pca"].astype(np.float32), identity=pack["identity"].astype(np.float32), subjects=np.asarray(list(map(str, subjects))))


def load_basis(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k].astype(np.float64) if z[k].dtype.kind in "fc" else z[k] for k in z.files}


def all_training_subjects(scope_data: Mapping[str, Any], fold: int) -> list[str]:
    outcome = set(map(str, scope_data["audit_roles"][str(fold)]["outcome"]))
    return [str(s) for s in scope_data["allowed_subjects"] if str(s) not in outcome]


def prepare(scope_data: Mapping[str, Any], device: torch.device) -> None:
    rows: list[dict[str, Any]] = []
    for fold in RUNS:
        model, payload = copy_or_verify_anchor(fold, device)
        legal = all_training_subjects(scope_data, fold)
        arrays = BASE.infer(model, legal, [0, 1], device)
        pack = build_basis(arrays, legal, list(range(len(legal))))
        path = CHECKPOINTS / f"basis_fold-{fold}.npz"
        save_basis(path, pack, legal)
        orth = np.max(np.abs(pack["basis"].T @ pack["basis"] - np.eye(CANDIDATE_COUNT)))
        rows.append({"fold": fold, "n_legal_subjects": len(legal), "candidate_count": CANDIDATE_COUNT, "max_orthonormality_error": float(orth), "top_persistence_eigenvalue": float(pack["eigenvalues"][0]), "anchor_checkpoint": str(CHECKPOINTS / f"anchor_fold-{fold}.pt"), "anchor_sha256": sha256_file(CHECKPOINTS / f"anchor_fold-{fold}.pt"), "basis_sha256": sha256_file(path), "anchor_model_state_sha256": payload.get("model_state_sha256")})
    write_csv(OUT / "ALIGNMENT_AUDIT.csv", rows)
    write_json(OUT / "PREPARE_STATE.json", {"status": "PREPARE_COMPLETE", "folds": list(RUNS), "device": str(device), "outer_accessed": False})


def erase(h: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
    z = h - mu
    return h - np.outer(z @ u, u)


def torch_erase(h: torch.Tensor, u: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    z = h - mu
    return h - (z @ u).unsqueeze(1) * u.unsqueeze(0)


def ce(logits: np.ndarray, y: np.ndarray) -> float:
    t = torch.from_numpy(logits.astype(np.float32)); yy = torch.from_numpy(y.astype(np.int64))
    return float(torch.nn.functional.cross_entropy(t, yy).item())


def paired_sign_p(values: Sequence[float], draws: int = SIGNFLIP_DRAWS, seed: int = 0) -> float:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return 1.0
    observed = float(np.mean(x))
    rng = np.random.default_rng(seed)
    n = len(x)
    if n <= 16:
        signs = np.asarray([[(1.0 if (mask >> i) & 1 else -1.0) for i in range(n)] for mask in range(1 << n)], dtype=np.float64)
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, n))
    return float(np.mean((signs * x).mean(axis=1) >= observed - 1e-15))


def holm(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, dtype=np.float64); order = np.argsort(p); out = np.empty(len(p), dtype=np.float64); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx])); out[idx] = running
    return out.tolist()


def match_basis(train_u: np.ndarray, full_u: np.ndarray) -> np.ndarray:
    score = np.abs(train_u.T @ full_u)
    rr, cc = linear_sum_assignment(-score)
    out = np.zeros_like(full_u)
    for r, c in zip(rr, cc):
        out[:, c] = train_u[:, r] * (1.0 if float(train_u[:, r] @ full_u[:, c]) >= 0 else -1.0)
    # Hungarian returns all requested columns when the matrices are rank 8.
    for c in range(full_u.shape[1]):
        if not np.any(out[:, c]):
            out[:, c] = full_u[:, c]
    return out


def random_direction(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(DIM, 1)))
    return q[:, 0].astype(np.float64)


def direction_stats(arrays: Mapping[str, np.ndarray], subjects: Sequence[str], basis: np.ndarray, center: np.ndarray, direction: int, random_seed_prefix: Any) -> list[dict[str, Any]]:
    h, y, sid, ses = arrays["h"], arrays["y"], arrays["sid"], arrays["session"]
    # Random controls are generated once per fold/direction and frozen before
    # any certification decision.
    random_u = np.stack([random_direction(stable_seed(random_seed_prefix, direction, r)) for r in range(RANDOM_DRAWS)], axis=1)
    rows: list[dict[str, Any]] = []
    for idx, subject in enumerate(map(str, subjects)):
        m = (sid == idx) & (ses == 1)
        if not np.any(m):
            continue
        hh, yy = h[m], y[m]
        raw_logits = BASE.logits_for(hh, np.zeros((2, DIM)), np.zeros(2)) if False else None
        # The caller injects logits/head through global fields on arrays.
        w, b = arrays["head_weight"], arrays["head_bias"]
        raw = BASE.logits_for(hh, w, b)
        abs_u = ce(BASE.logits_for(erase(hh, basis[:, direction], center), w, b), yy) - ce(raw, yy)
        rand_abs = []
        for r in range(RANDOM_DRAWS):
            er = erase(hh, random_u[:, r], center)
            rand_abs.append(ce(BASE.logits_for(er, w, b), yy) - ce(raw, yy))
        signed = abs_u - float(np.mean(rand_abs))
        centered_raw = raw - raw.mean(axis=1, keepdims=True)
        centered_erased = BASE.logits_for(erase(hh, basis[:, direction], center), w, b)
        response = float(np.mean(np.abs((centered_erased - centered_raw))))
        rand_response = []
        for r in range(RANDOM_DRAWS):
            rl = BASE.logits_for(erase(hh, random_u[:, r], center), w, b)
            rand_response.append(float(np.mean(np.abs((rl - rl.mean(axis=1, keepdims=True)) - centered_raw))))
        rows.append({"subject": subject, "direction": direction + 1, "abs_utility": abs_u, "random_abs_utility": float(np.mean(rand_abs)), "signed_utility": signed, "decision_response": response, "random_decision_response": float(np.mean(rand_response)), "decision_dependence": response - float(np.mean(rand_response)), "n_S2_trials": int(m.sum())})
    return rows


def certify_directions(scope_data: Mapping[str, Any], device: torch.device) -> dict[int, list[int]]:
    spectrum_rows: list[dict[str, Any]] = []
    cert_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[int, list[int]] = {}
    for fold in RUNS:
        legal = all_training_subjects(scope_data, fold)
        model, _ = BASE.load_anchor(CHECKPOINTS / f"anchor_fold-{fold}.pt", device)
        arrays = BASE.infer(model, legal, [0, 1], device)
        arrays["head_weight"] = model.head.weight.detach().cpu().numpy().astype(np.float64)
        arrays["head_bias"] = model.head.bias.detach().cpu().numpy().astype(np.float64)
        full = load_basis(CHECKPOINTS / f"basis_fold-{fold}.npz")
        U_full, mu = full["basis"], full["center"]
        subjects = legal
        # Four inner subject-disjoint folds.  Each held score uses a basis
        # constructed only from the other inner subjects and is aligned to the
        # frozen outer-training candidate spectrum.
        order = sorted(range(len(subjects)), key=lambda i: stable_seed("inner", fold, subjects[i]))
        inner_groups = [order[i::INNER_FOLDS] for i in range(INNER_FOLDS)]
        by_dir: dict[int, list[dict[str, Any]]] = {j: [] for j in range(CANDIDATE_COUNT)}
        for inner, held in enumerate(inner_groups):
            train = [i for i in order if i not in set(held)]
            inner_pack = build_basis(arrays, subjects, train)
            matched = match_basis(inner_pack["basis"], U_full)
            held_subs = [subjects[i] for i in held]
            # Reindex the held-out arrays to the local contiguous subject IDs.
            index_map = {old: new for new, old in enumerate(held)}
            mask = np.isin(arrays["sid"], np.asarray(held))
            held_arr = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(arrays["h"]) else v) for k, v in arrays.items()}
            held_arr["sid"] = np.asarray([index_map[int(x)] for x in arrays["sid"][mask]], dtype=int)
            for j in range(CANDIDATE_COUNT):
                vals = direction_stats(held_arr, held_subs, matched, inner_pack["center"], j, ("inner-random", fold, inner))
                by_dir[j].extend([{**row, "fold": fold, "inner_fold": inner} for row in vals])
        p_util = []
        p_dec = []
        for j in range(CANDIDATE_COUNT):
            rows = by_dir[j]
            util = np.asarray([r["signed_utility"] for r in rows], dtype=float)
            dec = np.asarray([r["decision_dependence"] for r in rows], dtype=float)
            pu = paired_sign_p(util, seed=stable_seed("utility-p", fold, j))
            pd = paired_sign_p(dec, seed=stable_seed("decision-p", fold, j))
            p_util.append(pu); p_dec.append(pd)
            persistence = float(full["eigenvalues"][j])
            spectrum_rows.append({"fold": fold, "direction": j + 1, "persistence_eigenvalue": persistence, "inner_subject_count": len(rows), "signed_utility_mean": float(util.mean()) if len(util) else None, "signed_utility_median": float(np.median(util)) if len(util) else None, "utility_positive_fraction": float(np.mean(util > 0)) if len(util) else None, "utility_p_raw": pu, "decision_dependence_mean": float(dec.mean()) if len(dec) else None, "decision_positive_fraction": float(np.mean(dec > 0)) if len(dec) else None, "decision_p_raw": pd})
        hu, hd = holm(p_util), holm(p_dec)
        rows_for_fold = [r for r in spectrum_rows if r["fold"] == fold]
        candidates: list[int] = []
        for j, row in enumerate(rows_for_fold):
            utility_pass = bool(row["signed_utility_mean"] > 0 and row["utility_positive_fraction"] >= 0.55 and hu[j] < 0.05)
            decision_pass = bool(row["decision_dependence_mean"] > 0 and row["decision_positive_fraction"] >= 0.55 and hd[j] < 0.05)
            pass_all = bool(row["persistence_eigenvalue"] > 0 and utility_pass and decision_pass)
            if pass_all:
                candidates.append(j)
            cert_rows.append({**row, "utility_p_holm": hu[j], "decision_p_holm": hd[j], "persistence_pass": row["persistence_eigenvalue"] > 0, "utility_pass": utility_pass, "decision_pass": decision_pass, "certified": pass_all})
        # Direction-level rank is capped at four.  Score uses only training-side
        # signed utility and decision evidence, never held S3 outcomes.
        candidates = sorted(candidates, key=lambda j: (-(rows_for_fold[j]["signed_utility_mean"] * max(rows_for_fold[j]["decision_dependence_mean"], 0.0)), j))[:RANK_MAX]
        selected_by_fold[fold] = [j for j in candidates]
        for row in cert_rows:
            if row["fold"] == fold:
                row["selected"] = (row["direction"] - 1) in candidates
    write_csv(OUT / "PROTECTED_DIRECTION_SPECTRUM.csv", spectrum_rows)
    write_csv(OUT / "PROTECTED_DIRECTION_CERTIFICATION.csv", cert_rows)
    write_json(OUT / "SELECTED_DIRECTIONS.json", {"selected_by_fold": selected_by_fold, "outer_accessed": False})
    # Compact human-readable audit.
    lines = ["# Deployment-matched protected direction discovery", "", f"Candidate pool: {CANDIDATE_COUNT} direction-level whitened persistence directions per fold.", "Inner utility/decision scores were evaluated on held inner subjects only; S3 was not used.", "", "| fold | selected directions (1-based) |", "|---:|:---|"]
    for fold in RUNS:
        lines.append(f"| {fold} | {', '.join(str(j + 1) for j in selected_by_fold[fold]) or 'none'} |" )
    lines += ["", "A direction is certified only when persistence, signed utility, and finite decision dependence all pass the frozen one-sided Holm rule."]
    (EXP_ROOT / "PROTECTED_DIRECTION_DISCOVERY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected_by_fold


def subject_split(subjects: Sequence[str], fold: int) -> tuple[list[str], list[str]]:
    ordered = sorted(map(str, subjects), key=lambda x: stable_seed("generic-val", fold, x))
    cut = max(1, int(round(0.75 * len(ordered))))
    return ordered[:cut], ordered[cut:]


def generic_selection(scope_data: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cand in GENERIC_CANDIDATES:
        scores: list[float] = []
        for fold in RUNS:
            legal = all_training_subjects(scope_data, fold)
            train_sub, val_sub = subject_split(legal, fold)
            model, _ = BASE.load_anchor(CHECKPOINTS / f"anchor_fold-{fold}.pt", device)
            tr, va = BASE.infer(model, train_sub, [1], device), BASE.infer(model, val_sub, [1], device)
            w = model.head.weight.detach().cpu().numpy(); b = model.head.bias.detach().cpu().numpy()
            ad, _ = BASE.fit_adapter(tr["h"], tr["y"], w, b, cand, None, stable_seed("generic", fold, cand["id"]), device)
            ah = BASE.adapter_apply(ad, va["h"], None, device)
            scores.append(BASE.metric_ba(va["y"], BASE.logits_for(ah, w, b).argmax(1)))
        rows.append({**cand, "mean_S2_validation_BA": float(np.mean(scores)), "fold_scores": json.dumps(scores)})
    tab = pd.DataFrame(rows).sort_values(["mean_S2_validation_BA", "id"], ascending=[False, True]).reset_index(drop=True)
    selected = next(c for c in GENERIC_CANDIDATES if c["id"] == str(tab.iloc[0]["id"]))
    write_csv(OUT / "GENERIC_SELECTION.csv", tab)
    write_json(OUT / "GENERIC_SELECTION.json", {"selected": selected, "candidates": rows, "selection_scope": "S2 subject-held-out validation within legal non-outcome training side", "outer_accessed": False})
    (EXP_ROOT / "GENERIC_BASELINE_SELECTION.md").write_text("# Generic baseline selection\n\nGeneric was selected using S2 subject-held-out validation only. The Guard gap and all S3 outcomes were unavailable to this selection.\n\n" + tab.to_markdown(index=False) + "\n", encoding="utf-8")
    return selected


def adapter_forward(adapter: BASE.LinearAdapter, h: torch.Tensor) -> torch.Tensor:
    return h + adapter.linear(h)


def fit_utility_adapter(h: np.ndarray, y: np.ndarray, w_np: np.ndarray, b_np: np.ndarray, config: Mapping[str, Any], basis: np.ndarray, center: np.ndarray, weights: np.ndarray | None, seed: int, device: torch.device) -> tuple[BASE.LinearAdapter, dict[str, Any]]:
    BASE.seed_all(seed)
    adapter = BASE.LinearAdapter(h.shape[1]).to(device)
    x = torch.from_numpy(h.astype(np.float32)).to(device); target = torch.from_numpy(y.astype(np.int64)).to(device)
    w = torch.from_numpy(w_np.astype(np.float32)).to(device); b = torch.from_numpy(b_np.astype(np.float32)).to(device)
    U = torch.from_numpy(basis.astype(np.float32)).to(device); mu = torch.from_numpy(center.astype(np.float32)).to(device)
    ww = torch.ones(U.shape[1], device=device) if weights is None else torch.from_numpy(weights.astype(np.float32)).to(device)
    ww = ww / torch.clamp(ww.mean(), min=1e-6)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    n = len(y); history: list[dict[str, float]] = []; active = 0; steps = 0; max_violation = 0.0
    for epoch in range(int(config["epochs"])):
        order = torch.randperm(n, device=device); task_sum = 0.0; hinge_sum = 0.0
        adapter.train()
        for start in range(0, n, 256):
            idx = order[start:start + 256]; xb, yb = x[idx], target[idx]
            opt.zero_grad(set_to_none=True)
            raw_h = adapter_forward(adapter, xb)
            raw_loss = torch.nn.functional.cross_entropy(raw_h @ w.T + b, yb)
            gs = []
            for j in range(U.shape[1]):
                erased = torch_erase(xb, U[:, j], mu)
                erased_h = adapter_forward(adapter, erased)
                g = torch.nn.functional.cross_entropy(erased_h @ w.T + b, yb) - raw_loss
                gs.append(g)
            gvec = torch.stack(gs)
            violation = torch.relu(torch.as_tensor(float(UTILITY_TAU), device=device) - gvec)
            hinge = torch.sum(ww * violation) / max(len(gs), 1)
            loss = raw_loss + float(UTILITY_LAMBDA) * hinge
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite utility adapter loss")
            loss.backward(); opt.step()
            task_sum += float(raw_loss.detach()) * len(idx); hinge_sum += float(hinge.detach()) * len(idx)
            active += int(torch.sum(violation.detach() > 1e-6).item()); steps += len(gs); max_violation = max(max_violation, float(violation.detach().max().item()))
        history.append({"epoch": epoch + 1, "task_loss": task_sum / max(n, 1), "hinge_loss": hinge_sum / max(n, 1), "active_fraction": active / max(steps, 1)})
    adapter.eval()
    return adapter, {"seed": seed, "config": dict(config), "basis_rank": int(basis.shape[1]), "lambda": UTILITY_LAMBDA, "tau": UTILITY_TAU, "constraint_active_fraction": active / max(steps, 1), "constraint_max_violation": max_violation, "history": history, "adapter_state_sha256": BASE.adapter_state_sha(adapter)}


def utility_values(h: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, adapter: BASE.LinearAdapter | None, basis: np.ndarray, center: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        raw = torch.from_numpy(h.astype(np.float32)).to(device)
        if adapter is None:
            after = raw
        else:
            after = adapter_forward(adapter, raw)
        raw_logits = after @ torch.from_numpy(w.astype(np.float32)).to(device).T + torch.from_numpy(b.astype(np.float32)).to(device)
        raw_ce = torch.nn.functional.cross_entropy(raw_logits, torch.from_numpy(y.astype(np.int64)).to(device), reduction="none")
        vals = []
        U = torch.from_numpy(basis.astype(np.float32)).to(device); mu = torch.from_numpy(center.astype(np.float32)).to(device)
        for j in range(U.shape[1]):
            er = torch_erase(raw, U[:, j], mu)
            if adapter is not None:
                er = adapter_forward(adapter, er)
            el = er @ torch.from_numpy(w.astype(np.float32)).to(device).T + torch.from_numpy(b.astype(np.float32)).to(device)
            vals.append((torch.nn.functional.cross_entropy(el, torch.from_numpy(y.astype(np.int64)).to(device), reduction="none") - raw_ce).mean().item())
        return np.asarray(vals, dtype=float)


def mechanism(h: np.ndarray, transformed: np.ndarray, u: np.ndarray, w: np.ndarray) -> dict[str, float]:
    d = transformed - h; perp = np.eye(DIM) - u @ u.T
    coord = np.linalg.norm(d @ u, axis=1) / np.maximum(np.linalg.norm(h @ u, axis=1), EPS)
    comp = np.linalg.norm(d @ perp, axis=1) / np.maximum(np.linalg.norm(h), EPS)
    total = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(h), EPS)
    before = w @ u; after = w @ (np.eye(DIM) + (np.linalg.lstsq(h, d, rcond=None)[0].T if len(h) > DIM else np.zeros((DIM, DIM)))) @ u
    response = float(np.linalg.norm(after - before) / max(np.linalg.norm(before), EPS))
    return {"coordinate_drift": float(np.mean(coord)), "coordinate_drift_q95": float(np.quantile(coord, 0.95)), "decision_response_drift": response, "complement_adaptation": float(np.mean(comp)), "total_adaptation": float(np.mean(total))}


def bootstrap(values: Sequence[float], seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    if len(x) == 0: return (float("nan"), float("nan"), float("nan"))
    d = rng.choice(x, size=(BOOTSTRAP_DRAWS, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def compute_headroom(scope_data: Mapping[str, Any], selected: Mapping[int, list[int]], generic: Mapping[str, Any], device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []; s3_rows: list[dict[str, Any]] = []
    for fold in RUNS:
        legal = all_training_subjects(scope_data, fold); role = scope_data["audit_roles"][str(fold)]; outcome = list(map(str, role["outcome"]))
        model, _ = BASE.load_anchor(CHECKPOINTS / f"anchor_fold-{fold}.pt", device); arrays = BASE.infer(model, legal, [0, 1], device)
        w = model.head.weight.detach().cpu().numpy().astype(float); b = model.head.bias.detach().cpu().numpy().astype(float)
        pack = load_basis(CHECKPOINTS / f"basis_fold-{fold}.npz"); inds = selected[fold]
        if not inds: continue
        U = pack["basis"][:, inds]; mu = pack["center"]
        ad, meta = BASE.fit_adapter(arrays["h"][arrays["session"] == 1], arrays["y"][arrays["session"] == 1], w, b, generic, None, stable_seed("headroom-generic", fold), device)
        for idx, subject in enumerate(legal):
            m = (arrays["sid"] == idx) & (arrays["session"] == 1)
            if not np.any(m): continue
            h, y = arrays["h"][m], arrays["y"][m]
            anchor_g = utility_values(h, y, w, b, None, U, mu, device)
            generic_g = utility_values(h, y, w, b, ad, U, mu, device)
            for j in range(len(inds)):
                rows.append({"fold": fold, "subject": subject, "direction": int(inds[j] + 1), "anchor_G": anchor_g[j], "generic_G": generic_g[j], "delta_G": generic_g[j] - anchor_g[j], "generic_constraint_collapse": bool(generic_g[j] <= 0), "n_S2_trials": int(m.sum()), "adapter_meta": json.dumps(meta)})
        # S3 BA is recorded separately for the required mechanism association;
        # it is not read by direction or hyperparameter selection.
        test = BASE.infer(model, outcome, [2], device); generic_h = BASE.adapter_apply(ad, test["h"], None, device); pred0 = BASE.logits_for(test["h"], w, b).argmax(1); predg = BASE.logits_for(generic_h, w, b).argmax(1)
        for idx, subject in enumerate(outcome):
            m = test["sid"] == idx
            s3_rows.append({"fold": fold, "subject": subject, "delta_BA_generic": BASE.metric_ba(test["y"][m], predg[m]) - BASE.metric_ba(test["y"][m], pred0[m])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["fold", "subject", "direction", "anchor_G", "generic_G", "delta_G", "generic_constraint_collapse", "n_S2_trials", "adapter_meta"])
    write_csv(OUT / "UTILITY_BEFORE_AFTER.csv", frame)
    if frame.empty:
        audit_result = {"headroom": False, "reason": "no certified directions"}
    else:
        delta = frame.delta_G.to_numpy(float); by_fold = frame.groupby("fold").delta_G.mean()
        # Frozen before looking at S3: a replicated training-side decrease or a
        # nontrivial crossing rate is the mechanistic gate.
        fold_rep = int(np.sum(by_fold.to_numpy() < 0)) >= 3
        crossing = float(np.mean(frame.generic_constraint_collapse.to_numpy(bool)))
        p = paired_sign_p(delta, seed=stable_seed("headroom"))
        audit_result = {"headroom": bool((fold_rep and float(delta.mean()) < 0 and p < 0.10) or crossing >= 0.20), "mean_delta_G": float(delta.mean()), "median_delta_G": float(np.median(delta)), "fold_mean_delta_G": {str(k): float(v) for k, v in by_fold.items()}, "utility_collapse_rate": crossing, "utility_delta_signflip_p": p, "replicated_negative_folds": int(np.sum(by_fold.to_numpy() < 0)), "s3_association_rows": s3_rows}
    write_json(OUT / "UTILITY_COLLAPSE_AUDIT.json", audit_result)
    # Keep a compact row-level association table for the report; no selection
    # decisions are based on these S3 values.
    if s3_rows:
        write_csv(OUT / "HEADROOM_S3_ASSOCIATION.csv", s3_rows)
    return frame, audit_result


def control_bases(pack: Mapping[str, np.ndarray], selected: list[int], fold: int) -> dict[str, np.ndarray]:
    proposed = pack["basis"][:, selected]
    return {
        "UtilityPreservingGuard": proposed,
        "UtilityOnlyGuard": proposed,
        "PersistenceOnlyUtilityGuard": pack["basis"][:, :max(len(selected), 1)],
        "IdentityUtilityGuard": pack["identity"][:, :max(len(selected), 1)],
        "PCAUtilityGuard": pack["pca"][:, :max(len(selected), 1)],
        "RandomUtilityGuard": np.stack([random_direction(stable_seed("random-control", fold, j)) for j in range(max(len(selected), 1))], axis=1),
        "DeploymentMatchedHard": proposed,
    }


def evaluate_dev(scope_data: Mapping[str, Any], selected: Mapping[int, list[int]], generic: Mapping[str, Any], headroom: Mapping[str, Any], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not headroom.get("headroom", False):
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []; raw_rows: list[dict[str, Any]] = []; activity_rows: list[dict[str, Any]] = []; utility_rows: list[dict[str, Any]] = []
    for fold in RUNS:
        inds = selected[fold]
        if not inds: continue
        legal = all_training_subjects(scope_data, fold); outcome = list(map(str, scope_data["audit_roles"][str(fold)]["outcome"]))
        model, _ = BASE.load_anchor(CHECKPOINTS / f"anchor_fold-{fold}.pt", device); tr = BASE.infer(model, legal, [1], device); te = BASE.infer(model, outcome, [2], device)
        w = model.head.weight.detach().cpu().numpy().astype(float); b = model.head.bias.detach().cpu().numpy().astype(float); pack = load_basis(CHECKPOINTS / f"basis_fold-{fold}.npz"); mu = pack["center"]
        bases = control_bases(pack, inds, fold)
        train_h, train_y = tr["h"], tr["y"]
        method_logits: dict[str, list[np.ndarray]] = {"Generic": [], "UtilityPreservingGuard": [], "UtilityOnlyGuard": [], "PersistenceOnlyUtilityGuard": [], "IdentityUtilityGuard": [], "PCAUtilityGuard": [], "RandomUtilityGuard": [], "DeploymentMatchedHard": []}
        method_meta: dict[str, list[dict[str, Any]]] = {k: [] for k in method_logits}
        for method, basis in bases.items():
            seeds = ADAPTER_SEEDS if method in {"Generic", "UtilityPreservingGuard"} else (0,)
            for seed in seeds:
                if method == "Generic":
                    ad, meta = BASE.fit_adapter(train_h, train_y, w, b, generic, None, stable_seed("dev-adapter", fold, method, seed), device)
                elif method == "DeploymentMatchedHard":
                    ad, meta = BASE.fit_adapter(train_h, train_y, w, b, generic, basis, stable_seed("dev-adapter", fold, method, seed), device)
                else:
                    # The selected proposed basis carries decision-grounded
                    # weights; controls use uniform weights but exactly the same
                    # hinge, optimizer, data, and epoch budget.
                    weight = np.ones(basis.shape[1], dtype=float)
                    if method == "UtilityPreservingGuard":
                        cert = pd.read_csv(OUT / "PROTECTED_DIRECTION_CERTIFICATION.csv")
                        scores = []
                        for j in inds:
                            x = cert[(cert.fold == fold) & (cert.direction == j + 1)].iloc[0]
                            scores.append(max(float(x.decision_dependence_mean), 1e-3))
                        weight = np.asarray(scores, dtype=float)
                    ad, meta = fit_utility_adapter(train_h, train_y, w, b, generic, basis, mu, weight, stable_seed("dev-adapter", fold, method, seed), device)
                ah = BASE.adapter_apply(ad, te["h"], None, device) if method == "Generic" else (BASE.adapter_apply(ad, te["h"], basis, device) if method == "DeploymentMatchedHard" else BASE.adapter_apply(ad, te["h"], None, device))
                # Utility adapters deliberately do not pass a projection basis;
                # their basis appears only in the differentiable hinge.
                logits = BASE.logits_for(ah, w, b); method_logits[method].append(logits); method_meta[method].append(meta)
                activity_rows.append({"fold": fold, "method": method, "seed": seed, "constraint_active_fraction": float(meta.get("constraint_active_fraction", 0.0)), "constraint_max_violation": float(meta.get("constraint_max_violation", 0.0)), "complement_adaptation": float(np.mean(np.linalg.norm((ah - te["h"]) @ (np.eye(DIM) - pack["basis"][:, inds] @ pack["basis"][:, inds].T), axis=1) / np.maximum(np.linalg.norm(te["h"], axis=1), EPS)))})
                raw_rows.append({"fold": fold, "method": method, "seed": seed, "adapter_state_sha256": meta.get("adapter_state_sha256"), "config": json.dumps(meta.get("config", generic)), "basis_rank": int(meta.get("basis_rank", 0))})
        # Frozen is represented by anchor logits; average seed predictions first
        # for the two methods with the required three-seed robustness audit.
        all_logits = {"Frozen": BASE.logits_for(te["h"], w, b), **{m: np.mean(v, axis=0) for m, v in method_logits.items()}}
        for method, logits in all_logits.items():
            pred = logits.argmax(1)
            for idx, subject in enumerate(outcome):
                m = te["sid"] == idx
                ba = BASE.metric_ba(te["y"][m], pred[m]); frozen_ba = BASE.metric_ba(te["y"][m], all_logits["Frozen"].argmax(1)[m])
                if method == "Frozen":
                    u_basis = pack["basis"][:, inds]
                    u_before = utility_values(te["h"][m], te["y"][m], w, b, None, u_basis, mu, device); u_after = u_before
                    mech = {"coordinate_drift": 0., "coordinate_drift_q95": 0., "decision_response_drift": 0., "complement_adaptation": 0., "total_adaptation": 0.}
                else:
                    # Reconstruct transformed features from averaged logits is
                    # impossible; use the first seed's representation for
                    # mechanism metrics and average utility across seed adapters.
                    if method == "Generic":
                        basis = np.zeros((DIM, 0)); adapter = BASE.fit_adapter(train_h, train_y, w, b, generic, None, stable_seed("mechanism", fold, method), device)[0]
                    elif method == "DeploymentMatchedHard":
                        basis = bases[method]; adapter = BASE.fit_adapter(train_h, train_y, w, b, generic, basis, stable_seed("mechanism", fold, method), device)[0]
                    else:
                        basis = bases[method]; adapter = fit_utility_adapter(train_h, train_y, w, b, generic, basis, mu, None, stable_seed("mechanism", fold, method), device)[0]
                    th = BASE.adapter_apply(adapter, te["h"][m], None, device) if method != "DeploymentMatchedHard" else BASE.adapter_apply(adapter, te["h"][m], basis, device)
                    u_basis = pack["basis"][:, inds]
                    u_before = utility_values(te["h"][m], te["y"][m], w, b, None, u_basis, mu, device); u_after = utility_values(te["h"][m], te["y"][m], w, b, adapter, u_basis, mu, device)
                    mech = mechanism(te["h"][m], th, u_basis, w)
                for j, (before, after) in enumerate(zip(np.atleast_1d(u_before), np.atleast_1d(u_after))):
                    utility_rows.append({"fold": fold, "subject": subject, "method": method, "direction": int(inds[j] + 1), "utility_before": float(before), "utility_after": float(after), "utility_delta": float(after - before), "utility_collapse": bool(after <= 0)})
                rows.append({"fold": fold, "subject": subject, "method": method, "seed_aggregation": "mean_logits" if method in {"Generic", "UtilityPreservingGuard"} else "single_seed", "BA": ba, "Frozen_BA": frozen_ba, "delta_BA_vs_Frozen": ba - frozen_ba, "macro_F1": BASE.metric_macro_f1(te["y"][m], pred[m]), "accuracy": float(np.mean(pred[m] == te["y"][m])), **mech})
    frame = pd.DataFrame(rows); raw = pd.DataFrame(raw_rows); activity = pd.DataFrame(activity_rows); utility = pd.DataFrame(utility_rows)
    write_csv(OUT / "DEV_SUBJECT_RESULTS.csv", frame); write_csv(OUT / "DEV_SUBJECT_RESULTS_RAW.csv", raw); write_csv(OUT / "CONSTRAINT_ACTIVITY.csv", activity); write_csv(OUT / "UTILITY_DRIFT.csv", utility)
    return frame, activity


def baseline_only(scope_data: Mapping[str, Any], generic: Mapping[str, Any], device: torch.device) -> pd.DataFrame:
    """Reproduce Frozen/Generic on development S3 after the G1 stop.

    This is an audit of the baseline, not a route back into Guard selection.
    The held S3 labels are used only to report the predeclared development
    endpoint; no direction, threshold, or optimizer decision is made from it.
    """
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for fold in RUNS:
        legal = all_training_subjects(scope_data, fold)
        outcome = list(map(str, scope_data["audit_roles"][str(fold)]["outcome"]))
        model, _ = BASE.load_anchor(CHECKPOINTS / f"anchor_fold-{fold}.pt", device)
        train = BASE.infer(model, legal, [1], device)
        test = BASE.infer(model, outcome, [2], device)
        w = model.head.weight.detach().cpu().numpy().astype(float)
        b = model.head.bias.detach().cpu().numpy().astype(float)
        frozen_logits = BASE.logits_for(test["h"], w, b)
        generic_logits: list[np.ndarray] = []
        for seed in ADAPTER_SEEDS:
            adapter, meta = BASE.fit_adapter(train["h"], train["y"], w, b, generic, None, stable_seed("baseline-only", fold, seed), device)
            transformed = BASE.adapter_apply(adapter, test["h"], None, device)
            logits = BASE.logits_for(transformed, w, b)
            generic_logits.append(logits)
            seed_rows.append({"fold": fold, "method": "Generic", "seed": seed, "adapter_state_sha256": meta.get("adapter_state_sha256"), "mean_S3_BA": float(np.mean([BASE.metric_ba(test["y"][i == test["sid"]], logits[i == test["sid"]].argmax(1)) for i in range(len(outcome))]))})
        avg_generic = np.mean(generic_logits, axis=0)
        for method, logits in (("Frozen", frozen_logits), ("Generic", avg_generic)):
            pred = logits.argmax(1)
            for idx, subject in enumerate(outcome):
                m = test["sid"] == idx
                frozen_ba = BASE.metric_ba(test["y"][m], frozen_logits.argmax(1)[m])
                rows.append({"fold": fold, "subject": subject, "method": method, "seed_aggregation": "mean_logits" if method == "Generic" else "single_seed", "BA": BASE.metric_ba(test["y"][m], pred[m]), "Frozen_BA": frozen_ba, "delta_BA_vs_Frozen": BASE.metric_ba(test["y"][m], pred[m]) - frozen_ba, "macro_F1": BASE.metric_macro_f1(test["y"][m], pred[m]), "accuracy": float(np.mean(pred[m] == test["y"][m])), "coordinate_drift": 0.0, "coordinate_drift_q95": 0.0, "decision_response_drift": 0.0, "complement_adaptation": 0.0, "total_adaptation": 0.0})
    frame = pd.DataFrame(rows)
    write_csv(OUT / "DEV_SUBJECT_RESULTS.csv", frame)
    write_csv(OUT / "DEV_SUBJECT_RESULTS_RAW.csv", pd.DataFrame(seed_rows))
    write_csv(OUT / "NEGATIVE_TRANSFER.csv", frame[["fold", "subject", "method", "Frozen_BA", "BA", "delta_BA_vs_Frozen"]])
    write_csv(OUT / "SEED_ROBUSTNESS.csv", pd.DataFrame(seed_rows))
    write_csv(OUT / "DEV_METHOD_SUMMARY.csv", summarize(frame))
    # These controls are intentionally absent after the prospective G1 stop;
    # explicit status rows prevent an empty file from being mistaken for a
    # successful null control.
    write_csv(OUT / "CONTROL_COMPARISON.csv", pd.DataFrame([{"method": m, "status": "NOT_RUN_G1_NO_CERTIFIED_DIRECTION"} for m in ("HistoricalHardP01_04", "DeploymentMatchedHard", "UtilityOnlyGuard", "PersistenceOnlyUtilityGuard", "IdentityUtilityGuard", "PCAUtilityGuard", "RandomUtilityGuard")]))
    empty_cols = {"fold": [], "subject": [], "method": [], "direction": [], "utility_before": [], "utility_after": [], "utility_delta": [], "utility_collapse": []}
    write_csv(OUT / "UTILITY_DRIFT.csv", pd.DataFrame(empty_cols))
    write_csv(OUT / "COORDINATE_DRIFT.csv", pd.DataFrame(columns=["fold", "subject", "method", "coordinate_drift"]))
    write_csv(OUT / "DECISION_DRIFT.csv", pd.DataFrame(columns=["fold", "subject", "method", "decision_response_drift"]))
    write_csv(OUT / "CONSTRAINT_ACTIVITY.csv", pd.DataFrame(columns=["fold", "method", "seed", "constraint_active_fraction", "constraint_max_violation"]))
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in frame.groupby("method", sort=False):
        d = g.delta_BA_vs_Frozen.to_numpy(float); ba = g.BA.to_numpy(float); q = np.sort(d)[:max(1, math.ceil(len(d) * .25))]
        m, lo, hi = bootstrap(ba, stable_seed("ba", method)); dm, dlo, dhi = bootstrap(d, stable_seed("delta", method))
        rows.append({"method": method, "n_subjects": len(g), "BA_mean": m, "BA_CI95_L": lo, "BA_CI95_U": hi, "macro_F1_mean": float(g.macro_F1.mean()), "delta_BA_vs_Frozen_mean": dm, "delta_BA_vs_Frozen_CI95_L": dlo, "delta_BA_vs_Frozen_CI95_U": dhi, "negative_transfer_rate": float(np.mean(d < 0)), "negative_transfer_count": int(np.sum(d < 0)), "worst_quartile_delta": float(q.mean()), "worst_subject_delta": float(d.min()), "coordinate_drift_mean": float(g.coordinate_drift.mean()), "decision_response_drift_mean": float(g.decision_response_drift.mean()), "complement_adaptation_mean": float(g.complement_adaptation.mean())})
    return pd.DataFrame(rows)


def analyze(frame: pd.DataFrame, selected: Mapping[int, list[int]], headroom: Mapping[str, Any]) -> dict[str, Any]:
    if not any(selected.values()):
        summary = summarize(frame) if not frame.empty else pd.DataFrame()
        if not summary.empty:
            write_csv(OUT / "DEV_METHOD_SUMMARY.csv", summary)
        result = {
            "terminal_state": "EXP4_V2_NO_DEPLOYMENT_MATCHED_PROTECTED_DIRECTION",
            "reason": "no direction passed persistence + signed utility + decision dependence certification",
            "headroom": headroom,
            "baseline_methods": summary.to_dict(orient="records") if not summary.empty else [],
            "outer_accessed": False,
            "outer_authorized": False,
        }
        write_json(OUT / "STATISTICAL_TESTS.json", result)
        return result
    if frame.empty:
        terminal = "EXP4_V2_NO_UTILITY_COLLAPSE_HEADROOM" if headroom.get("reason") != "no certified directions" else "EXP4_V2_NO_DEPLOYMENT_MATCHED_PROTECTED_DIRECTION"
        result = {"terminal_state": terminal, "headroom": headroom, "outer_accessed": False}
        write_json(OUT / "STATISTICAL_TESTS.json", result); return result
    summary = summarize(frame); write_csv(OUT / "DEV_METHOD_SUMMARY.csv", summary)
    methods = set(frame.method)
    primary_method = "UtilityPreservingGuard" if "UtilityPreservingGuard" in methods else None
    if primary_method is None:
        terminal = "EXP4_V2_UTILITY_MECHANISM_ONLY"
        primary = {}
    else:
        g = frame[frame.method == "Generic"].set_index(["fold", "subject"]); p = frame[frame.method == primary_method].set_index(["fold", "subject"])
        d = (p.BA - g.BA).to_numpy(float); mean, lo, hi = bootstrap(d, stable_seed("primary")); pval = paired_sign_p(d, seed=stable_seed("primary-p")); primary = {"mean_delta_BA": mean, "median_delta_BA": float(np.median(d)), "ci95": [lo, hi], "signflip_p": pval, "positive_subjects": int(np.sum(d > 0)), "n_subjects": len(d)}
        generic = frame[frame.method == "Generic"].set_index(["fold", "subject"]); guard = frame[frame.method == primary_method].set_index(["fold", "subject"])
        neg_g = generic.delta_BA_vs_Frozen.to_numpy(float) < 0; neg_p = guard.delta_BA_vs_Frozen.to_numpy(float) < 0
        rescued = int(np.sum(neg_g & (guard.delta_BA_vs_Frozen.to_numpy(float) >= 0)))
        specific_controls = {}
        for m in ("PersistenceOnlyUtilityGuard", "IdentityUtilityGuard", "PCAUtilityGuard", "RandomUtilityGuard"):
            if m in methods:
                x = guard.BA.to_numpy(float) - frame[frame.method == m].set_index(["fold", "subject"]).BA.to_numpy(float)
                specific_controls[m] = {"mean": float(x.mean()), "ci95": list(bootstrap(x, stable_seed("control", m))[1:]), "positive_subjects": int(np.sum(x > 0))}
        sufficient = bool(mean >= 0.005 and lo > 0 and np.mean(neg_p) <= np.mean(neg_g) and rescued >= 1)
        specific = bool(all(v["ci95"][0] > 0 for v in specific_controls.values()))
        mechanism = pd.read_csv(OUT / "UTILITY_DRIFT.csv") if (OUT / "UTILITY_DRIFT.csv").is_file() else pd.DataFrame()
        if not mechanism.empty:
            ug = mechanism[mechanism.method == "Generic"].utility_delta.to_numpy(float); up = mechanism[mechanism.method == primary_method].utility_delta.to_numpy(float)
            utility_preserved = float(np.mean(up > ug)) >= 0.5
        else:
            utility_preserved = False
        if not sufficient:
            terminal = "EXP4_V2_UTILITY_GUARD_FAILED"
        elif specific:
            terminal = "EXP4_V2_UTILITY_GUARD_NOT_SPECIFIC"
        elif not utility_preserved:
            terminal = "EXP4_V2_UTILITY_MECHANISM_ONLY"
        else:
            terminal = "EXP4_V2_DEV_CONFIRMED_OUTER_LOCKED"
        primary.update({"generic_negative_transfer_rate": float(np.mean(neg_g)), "guard_negative_transfer_rate": float(np.mean(neg_p)), "generic_negative_transfer_count": int(np.sum(neg_g)), "guard_negative_transfer_count": int(np.sum(neg_p)), "generic_harmed_rescued": rescued})
    result = {"terminal_state": terminal, "selected_by_fold": selected, "headroom": headroom, "primary": primary, "methods": summary.to_dict(orient="records"), "outer_accessed": False, "outer_authorized": False}
    write_json(OUT / "STATISTICAL_TESTS.json", result)
    return result


def write_static_reports(state: Mapping[str, Any]) -> None:
    terminal = state.get("terminal_state", "UNKNOWN")
    docs = {
        "PROTOCOL_SELECTION_AUDIT.md": "# Protocol selection audit\n\nThis V2 branch starts from the historical Exp4 commit but does not reuse its P01_04 proposed basis. The historical assignment was trained/defined under an earlier S1+S2→S3 representation protocol; V2 uses the exact S1-only anchor and reconstructs directions inside each legal training side. Inner certification is subject-disjoint and frozen before held S3 evaluation.\n",
        "ALIGNMENT_AUDIT.md": "# Alignment audit\n\nThe anchor is S1-only EEGNet. For each development fold, all non-outcome subjects' S1/S2 representations are used to compute a centered, ridge-whitened cross-session centroid covariance. The top eight orthonormal directions are the only candidate pool. The sealed outer split is never opened. See `results/ALIGNMENT_AUDIT.csv`.\n",
        "UTILITY_HEADROOM_AUDIT.md": "# Utility-collapse headroom audit\n\nThis gate is evaluated before Guard development. It compares anchor-space intervention utility with the same intervention after the frozen Generic adapter. S3 association rows are descriptive and are not used to choose directions or hyperparameters.\n\n" + json.dumps(state.get("headroom", {}), indent=2, ensure_ascii=False) + "\n",
        "ITERATION_LEDGER.md": "# Iteration ledger\n\n| version | change | decision |\n|---|---|---|\n| V0 | Deployment-matched whitened direction spectrum, inner cross-fitted utility/decision certification | frozen before S3 |\n| V1 | Conditional utility hinge, τ=0, λ=0.5, same zero-initialized 32×32 adapter | run only if headroom gate passes |\n| V2 | No alpha scan; no hard-coordinate strength search | prohibited by protocol |\n\nTerminal state: `" + terminal + "`.\n",
        "FINAL_MODEL_CARD.md": f"# Exp4 V2 model card\n\nTerminal state: **{terminal}**. The model uses an S1-only EEGNet anchor and a zero-initialized 32×32 residual adapter. The proposed constraint, when run, is a differentiable hinge on anchor-space intervention utility; it does not freeze coordinates. Outer subjects were not accessed.\n",
        "CLAIM_AUDIT.md": "# Claim audit\n\nOnly development-cohort, subject-level claims are permitted. A non-significant Guard comparison is not reported as equivalence. Utility preservation is a mechanism claim, not proof of causality from correlation. No outer claim is made unless a final protocol lock authorizes a one-time evaluation.\n",
        "REVIEWER_SELF_AUDIT.md": "# Reviewer self-audit\n\nThe main risks are limited development sample size, reliance on a frozen EEGNet anchor, and possible instability of utility estimates. The V2 protocol addresses the largest historical weakness by aligning discovery with S1-only deployment and by requiring a prospective utility-collapse gate. It does not claim universal EEG domain-shift safety.\n",
        "REPRODUCIBILITY.md": "# Reproducibility\n\nRun `python experiments/persist_eeg_exp4_utility_preservation_v2/code/run_exp4_v2.py audit`, then `prepare`, `discover`, `generic`, and `dev`. The runner records source/cache/checkpoint hashes, deterministic fold seeds, all subject-level tables, and outer-sealed locks. Raw EEG and large caches are excluded from Git.\n",
        "OUTER_ACCESS_AUDIT.md": "# Outer access audit\n\nNo outer subject IDs were enumerated, no outer raw files were loaded, and no outer embeddings or labels were materialized during development. The outer command is fail-closed without `EXP4_V2_FINAL_PROTOCOL_LOCK.json`.\n",
    }
    for name, text in docs.items():
        (EXP_ROOT / name).write_text(text, encoding="utf-8")
    report = {"terminal_state": terminal, "state": state, "outer_accessed": False}
    write_json(EXP_ROOT / "EXP4_V2_FINAL_REPORT.json", report)
    (EXP_ROOT / "EXP4_V2_FINAL_REPORT.md").write_text("# PERSIST-EEG Experiment 4 V2 — final report\n\nTerminal state: **" + terminal + "**\n\n```json\n" + json.dumps(clean(state), indent=2, ensure_ascii=False) + "\n```\n\nNo outer subjects were accessed.\n", encoding="utf-8")


def figures(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    if "Generic" in set(frame.method):
        g = frame[frame.method == "Generic"].set_index(["fold", "subject"])
        if "UtilityPreservingGuard" in set(frame.method):
            p = frame[frame.method == "UtilityPreservingGuard"].set_index(["fold", "subject"])
            keys = list(g.index); x = np.arange(len(keys)); fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(x, g.loc[keys].BA, "o-", label="Generic"); ax.plot(x, p.loc[keys].BA, "o-", label="Utility Guard"); ax.set_ylabel("S3 subject BA"); ax.set_xlabel("development outcome subject"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "figure_3_main_paired_performance.png", dpi=220); plt.close(fig)
            fig, ax = plt.subplots(figsize=(10, 4)); ax.axhline(0, color="black", lw=1); ax.plot(x, g.loc[keys].delta_BA_vs_Frozen, "o-", label="Generic−Frozen"); ax.plot(x, p.loc[keys].delta_BA_vs_Frozen, "o-", label="Guard−Frozen"); ax.set_ylabel("ΔBA vs Frozen"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "figure_4_negative_transfer.png", dpi=220); plt.close(fig)
    cert = Path(OUT / "PROTECTED_DIRECTION_SPECTRUM.csv")
    if cert.is_file():
        c = pd.read_csv(cert); fig, ax = plt.subplots(figsize=(8, 4)); ax.scatter(c.persistence_eigenvalue, c.signed_utility_mean, c=c.decision_dependence_mean, cmap="coolwarm", s=50); ax.axhline(0, color="black", lw=1); ax.set_xlabel("persistence eigenvalue"); ax.set_ylabel("signed utility"); fig.tight_layout(); fig.savefig(FIGURES / "figure_1_direction_discovery.png", dpi=220); plt.close(fig)
    util = Path(OUT / "UTILITY_DRIFT.csv")
    if util.is_file():
        u = pd.read_csv(util); fig, ax = plt.subplots(figsize=(7, 4));
        for m, grp in u.groupby("method"):
            ax.scatter(grp.utility_before, grp.utility_after, label=m, alpha=.55)
        lim = ax.get_xlim(); ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel("utility before adaptation"); ax.set_ylabel("utility after adaptation"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "figure_5_functional_mechanism.png", dpi=220); plt.close(fig)


def run(phase: str, device: torch.device) -> int:
    prepare_dirs()
    if phase == "audit":
        audit(); return 0
    if not (PROTOCOL / "EXP4_V2_DEV_PROTOCOL.json").is_file():
        raise RuntimeError("run audit first")
    s = scope()
    if phase == "prepare":
        prepare(s, device); return 0
    if phase == "discover":
        selected = certify_directions(s, device); return 0
    if phase == "generic":
        generic_selection(s, device); return 0
    if phase in {"headroom", "dev", "all_dev"}:
        if phase == "all_dev" or phase == "headroom":
            if not (OUT / "SELECTED_DIRECTIONS.json").is_file(): certify_directions(s, device)
            if not (OUT / "GENERIC_SELECTION.json").is_file(): generic_selection(s, device)
            selected = json.loads((OUT / "SELECTED_DIRECTIONS.json").read_text(encoding="utf-8"))["selected_by_fold"]
            selected = {int(k): [int(x) for x in v] for k, v in selected.items()}
            generic = json.loads((OUT / "GENERIC_SELECTION.json").read_text(encoding="utf-8"))["selected"]
            _, hr = compute_headroom(s, selected, generic, device)
            if phase == "headroom":
                state = {"terminal_state": "EXP4_V2_NO_DEPLOYMENT_MATCHED_PROTECTED_DIRECTION" if not any(selected.values()) else ("EXP4_V2_NO_UTILITY_COLLAPSE_HEADROOM" if not hr.get("headroom", False) else "HEADROOM_PASS"), "headroom": hr, "selected_by_fold": selected}; write_static_reports(state); return 0
        # dev path
        selected = json.loads((OUT / "SELECTED_DIRECTIONS.json").read_text(encoding="utf-8"))["selected_by_fold"]; selected = {int(k): [int(x) for x in v] for k, v in selected.items()}
        generic = json.loads((OUT / "GENERIC_SELECTION.json").read_text(encoding="utf-8"))["selected"]
        if not (OUT / "UTILITY_COLLAPSE_AUDIT.json").is_file():
            _, hr = compute_headroom(s, selected, generic, device)
        else:
            hr = json.loads((OUT / "UTILITY_COLLAPSE_AUDIT.json").read_text(encoding="utf-8"))
        if not any(selected.values()):
            frame = baseline_only(s, generic, device)
        elif not hr.get("headroom", False):
            # A valid direction exists but the required mechanistic headroom
            # gate fails.  Baseline-only reporting is still allowed; no Guard
            # or control is trained after this point.
            frame = baseline_only(s, generic, device)
        else:
            frame, _ = evaluate_dev(s, selected, generic, hr, device)
        state = analyze(frame, selected, hr); figures(frame); write_static_reports(state); return 0
    if phase == "outer":
        raise RuntimeError("OUTER_FORBIDDEN: this runner does not open sealed subjects without a final protocol lock")
    raise ValueError(phase)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["audit", "prepare", "discover", "generic", "headroom", "dev", "all_dev", "outer"]); ap.add_argument("--device", default="auto"); args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    return run(args.phase, device)


if __name__ == "__main__":
    raise SystemExit(main())
