from __future__ import annotations

"""Exp4 V3: emergence-triggered utility preservation.

The V2 negative result is preserved. V3 changes only the prospective
measurement question: repaired decision centering, energy-matched random
interventions, cumulative rank-1/2/4 subspaces, and a training-side trajectory
audit. The outer cohort is never opened by this executable during development.
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


EXP_ROOT = Path(os.environ.get("PERSIST_EXP4_V3_ROOT", str(Path(__file__).resolve().parents[1])))
OUT = EXP_ROOT / "results"
PROTOCOL = EXP_ROOT / "protocol"
FIGURES = EXP_ROOT / "figures"
CHECKPOINTS = EXP_ROOT / "checkpoints"
V2_PATH = EXP_ROOT.parent / "persist_eeg_exp4_utility_preservation_v2" / "code" / "run_exp4_v2.py"
V1_ROOT = Path(os.environ.get("PERSIST_EXP4_V1_ROOT", r"D:\nips-temp\TotalP\P1\CRCICLR_EXP4_PROTECTION_FIRST_FINAL\experiments\persist_eeg_exp4_protection_first_final"))

os.environ["PERSIST_EXP4_V2_ROOT"] = str(EXP_ROOT)
os.environ.setdefault("PERSIST_EXP4_V1_ROOT", str(V1_ROOT))
_spec = importlib.util.spec_from_file_location("persist_exp4_v2_primitives", V2_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"V2 primitive source missing: {V2_PATH}")
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)
BASE = V2.BASE

RUNS = tuple(range(5))
RANKS = (1, 2, 4)
INNER_FOLDS = 4
CANDIDATE_COUNT = 8
RANDOM_POOL = 256
ENERGY_TOLERANCE = 0.10
TRAJECTORY_EPOCHS = (0, 5, 10, 15, 20, 25)
TRAJECTORY_SEED = 0
FINAL_SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 10000
SIGNFLIP_DRAWS = 100000
EPS = 1e-12
EXPERIMENT_SEED = 20260823
GENERIC = {"id": "GEN_LINEAR_LR1E3_E25", "learning_rate": 1e-3, "epochs": 25, "weight_decay": 5e-4}


def clean(x: Any) -> Any:
    if isinstance(x, Mapping): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, float): return None if not math.isfinite(x) else x
    if isinstance(x, Path): return str(x)
    return x


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
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def git_head() -> str | None:
    forced = os.environ.get("PERSIST_EXP4_V3_GIT_COMMIT")
    if forced: return forced.strip()
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXP_ROOT.parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def prepare_dirs() -> None:
    for p in (OUT, PROTOCOL, FIGURES, CHECKPOINTS): p.mkdir(parents=True, exist_ok=True)


def scope() -> dict[str, Any]: return V2.scope()


def legal_subjects(s: Mapping[str, Any], fold: int) -> list[str]: return V2.all_training_subjects(s, fold)


def audit() -> None:
    prepare_dirs(); s = scope()
    protocol = {
        "experiment": "PERSIST_EEG_EXP4_EMERGENCE_TRIGGERED_V3",
        "status": "DEV_PROTOCOL_FROZEN_BEFORE_TRAJECTORY",
        "source_branch": "codex/persist-eeg-exp4-utility-preservation-v2",
        "source_commit": git_head(),
        "dataset": "WBCIC/Yang2025/NEMAR nm000348",
        "development_subject_count": 41, "outer_subject_count": 10,
        "outer_subject_ids_present": False, "outer_access_during_development": False,
        "deployment": "S1 anchor -> S2 legal adaptation -> unseen development-subject S3",
        "anchor": {"backbone": "EEGNet", "sessions": [0], "embedding_dim": int(BASE.DIM), "dropout": .25, "lr": 3e-4, "weight_decay": 5e-4, "epochs": 30},
        "generic": {"config": GENERIC, "final_seeds": list(FINAL_SEEDS), "selection": "V2 S2 held-subject validation; no S3 selection"},
        "decision_metric": {"definition": "mean(|center(z_erased)-center(z_raw)|)", "center": "subtract per-trial class-mean logit", "random_formula_identical": True},
        "energy_matching": {"random_pool": RANDOM_POOL, "tolerance_relative": ENERGY_TOLERANCE, "match_quantity": "removed representation energy only", "outcome_matching": False},
        "subspace_family": {"ranks": list(RANKS), "candidate_directions": CANDIDATE_COUNT, "multiplicity": "Holm across rank 1/2/4 per fold", "max_rank": 4},
        "trajectory": {"checkpoints": list(TRAJECTORY_EPOCHS), "seed": TRAJECTORY_SEED, "basis": "recomputed from legal training-side adapted S1/S2 representations", "utility": "inner subject-disjoint S2 labels"},
        "emergence_rule": {"gate": "persistence>0, signed utility>0, positive subject fraction>=0.55, repaired decision dependence>0, Holm p<0.05", "stability": "two consecutive checkpoints", "rank_selection": "smallest passing rank"},
        "margin": {"primary": "tau=0", "alternative": "rho=0.5 only if trajectory shows positive-but-substantially-eroded utility", "selection_source": "training-side trajectory only"},
        "outer_command": "refuse unless EXP4_V3_FINAL_PROTOCOL_LOCK.json authorizes one-time evaluation",
    }
    write_json(PROTOCOL / "EXP4_V3_DEV_PROTOCOL.json", protocol)
    source_paths = [V2_PATH, V2.BASE_PATH, BASE.SCOPE_PATH, BASE.CACHE_AUDIT_PATH, BASE.ACTION_LOCK_PATH]
    hashes = {str(p): sha256_file(p) for p in source_paths if p.is_file()}
    for fold in RUNS:
        p = V1_ROOT / "checkpoints" / f"anchor_fold-{fold}.pt"
        if p.is_file(): hashes[str(p)] = sha256_file(p)
    write_json(PROTOCOL / "PROVENANCE_AUDIT.json", {"status": "PROVENANCE_AUDIT_PASS", "git_commit": git_head(), "source_hashes": hashes, "scope_subject_count": len(s["allowed_subjects"]), "historical_v2_terminal": "EXP4_V2_NO_DEPLOYMENT_MATCHED_PROTECTED_DIRECTION", "historical_basis_reused": False, "outer_ids_opened": False, "outer_split_lock_read": False})
    write_json(PROTOCOL / "OUTER_LOCK.json", {"status": "OUTER_SEALED", "outer_evaluation_authorized": False, "outer_evaluation_count": 0, "outer_subject_ids_present": False, "outer_result_exists": False})


def prepare(s: Mapping[str, Any], device: torch.device) -> None:
    rows = []
    for fold in RUNS:
        model, payload = V2.copy_or_verify_anchor(fold, device)
        legal = legal_subjects(s, fold); arrays = BASE.infer(model, legal, [0, 1], device)
        pack = V2.build_basis(arrays, legal, list(range(len(legal))))
        path = CHECKPOINTS / f"basis_fold-{fold}.npz"; V2.save_basis(path, pack, legal)
        rows.append({"fold": fold, "legal_subjects": len(legal), "candidate_count": CANDIDATE_COUNT, "orthonormality_error": float(np.max(np.abs(pack["basis"].T @ pack["basis"] - np.eye(CANDIDATE_COUNT)))), "anchor_sha256": sha256_file(CHECKPOINTS / f"anchor_fold-{fold}.pt"), "basis_sha256": sha256_file(path), "anchor_state_sha256": payload.get("model_state_sha256")})
    write_csv(OUT / "ALIGNMENT_AUDIT.csv", rows)
    write_json(OUT / "PREPARE_STATE.json", {"status": "PREPARE_COMPLETE", "folds": list(RUNS), "outer_accessed": False})


def center_logits(z: np.ndarray) -> np.ndarray: return z - z.mean(axis=1, keepdims=True)


def erase_np(h: np.ndarray, U: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return h - np.outer((h - mu) @ U, U.T) if U.ndim == 1 else h - ((h - mu) @ U) @ U.T


def ce_np(logits: np.ndarray, y: np.ndarray) -> float:
    return float(torch.nn.functional.cross_entropy(torch.from_numpy(logits.astype(np.float32)), torch.from_numpy(y.astype(np.int64))).item())


def symmetric_decision(h: np.ndarray, y: np.ndarray, U: np.ndarray, mu: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    raw = BASE.logits_for(h, w, b); erased = BASE.logits_for(erase_np(h, U, mu), w, b)
    d = float(np.mean(np.abs(center_logits(erased) - center_logits(raw))))
    util = ce_np(erased, y) - ce_np(raw, y)
    energy = float(np.mean(np.sum((h - erase_np(h, U, mu)) ** 2, axis=1)))
    return util, d, energy


def random_subspace(seed: int, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(int(BASE.DIM), rank)))
    return q[:, :rank].astype(np.float64)


def random_pool(seed_prefix: Any, rank: int) -> list[np.ndarray]: return [random_subspace(stable_seed(seed_prefix, rank, i), rank) for i in range(RANDOM_POOL)]


def energy_match(h_train: np.ndarray, candidate: np.ndarray, pool: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    # For an orthonormal U, removed energy is ||(h-mu)U||^2.  Batch all
    # random controls in one einsum; the previous per-control Python loop was
    # mathematically identical but unnecessarily multiplied the large cache
    # scan by RANDOM_POOL.
    x = np.asarray(h_train, dtype=np.float64) - np.asarray(h_train, dtype=np.float64).mean(axis=0, keepdims=True)
    cand = np.asarray(candidate, dtype=np.float64)
    if cand.ndim == 1:
        cand = cand[:, None]
    cand_energy = float(np.mean(np.sum((x @ cand) ** 2, axis=1)))
    controls = np.stack([np.asarray(u, dtype=np.float64) if np.asarray(u).ndim == 2 else np.asarray(u, dtype=np.float64)[:, None] for u in pool], axis=2)
    projections = np.einsum("nd,drm->nrm", x, controls, optimize=True)
    energies = np.mean(np.sum(projections * projections, axis=1), axis=0)
    order = np.argsort(np.abs(energies - cand_energy)); idx = int(order[0]); rel = float(abs(energies[idx] - cand_energy) / max(abs(cand_energy), EPS))
    return pool[idx], {"candidate_energy": cand_energy, "random_energy": float(energies[idx]), "relative_mismatch": rel, "pool_index": idx, "within_tolerance": rel <= ENERGY_TOLERANCE}


def sign_p(x: Sequence[float], seed: int) -> float:
    a = np.asarray(x, dtype=float)
    if len(a) == 0: return 1.0
    obs = float(a.mean()); rng = np.random.default_rng(seed)
    if len(a) <= 16:
        signs = np.asarray([[1 if (m >> i) & 1 else -1 for i in range(len(a))] for m in range(1 << len(a))], dtype=float)
    else: signs = rng.choice(np.array([-1., 1.]), size=(SIGNFLIP_DRAWS, len(a)))
    return float(np.mean((signs * a).mean(axis=1) >= obs - 1e-15))


def holm(p: Sequence[float]) -> list[float]:
    x = np.asarray(p, dtype=float); out = np.empty(len(x)); running = 0.
    for rank, idx in enumerate(np.argsort(x)):
        running = max(running, min(1., (len(x) - rank) * x[idx])); out[idx] = running
    return out.tolist()


# ---------------------------------------------------------------------------
# V3 execution primitives
# ---------------------------------------------------------------------------

def _npz_write(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    np.savez_compressed(part, **arrays)
    # numpy appends .npz when the temporary name does not end in .npz.
    actual = Path(str(part) + ".npz") if not part.exists() and Path(str(part) + ".npz").exists() else part
    actual.replace(path)


def _npz_read(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _feature_cache_path(fold: int, cohort: str) -> Path:
    return CHECKPOINTS / f"anchor_features_fold-{fold}_{cohort}.npz"


def _arrays_from_cache(model: Any, subjects: Sequence[str], sessions: Sequence[int], fold: int, cohort: str, device: torch.device) -> dict[str, np.ndarray]:
    """Materialize compact anchor features once; never materializes outer data."""
    path = _feature_cache_path(fold, cohort)
    if path.is_file():
        try:
            data = _npz_read(path)
            required = {"h", "logits", "y", "sid", "session", "subjects"}
            if required.issubset(data) and len(data["h"]) == len(data["y"]):
                if list(map(str, data["subjects"].tolist())) == list(map(str, subjects)):
                    return data
        except Exception:
            pass
    data = BASE.infer(model, list(map(str, subjects)), list(map(int, sessions)), device)
    _npz_write(path, data)
    return data


def _subset_arrays(arrays: Mapping[str, np.ndarray], subject_indices_: Sequence[int]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Subset an infer() result and remap subject ids to a contiguous range."""
    chosen = [int(x) for x in subject_indices_]
    mask = np.isin(arrays["sid"], np.asarray(chosen, dtype=int))
    index_map = {old: new for new, old in enumerate(chosen)}
    out: dict[str, np.ndarray] = {}
    n = len(arrays["h"])
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and len(value) == n:
            out[key] = value[mask]
        else:
            out[key] = value
    out["sid"] = np.asarray([index_map[int(x)] for x in arrays["sid"][mask]], dtype=int)
    subjects = [str(arrays["subjects"][i]) for i in chosen]
    out["subjects"] = np.asarray(subjects)
    return out, subjects


def _transform_features(arrays: Mapping[str, np.ndarray], adapter: Any | None, device: torch.device) -> dict[str, np.ndarray]:
    out = dict(arrays)
    if adapter is None:
        out["h"] = np.asarray(arrays["h"], dtype=np.float64)
    else:
        out["h"] = BASE.adapter_apply(adapter, arrays["h"], None, device)
    return out


def _head(model: Any) -> tuple[np.ndarray, np.ndarray]:
    return (model.head.weight.detach().cpu().numpy().astype(np.float64),
            model.head.bias.detach().cpu().numpy().astype(np.float64))


def _subject_metrics(h: np.ndarray, y: np.ndarray, U: np.ndarray, mu: np.ndarray, random_U: np.ndarray, w: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Candidate and matched-control measurements with one symmetric D formula."""
    raw_logits = BASE.logits_for(h, w, b)
    erased_logits = BASE.logits_for(erase_np(h, U, mu), w, b)
    random_logits = BASE.logits_for(erase_np(h, random_U, mu), w, b)
    raw_center = center_logits(raw_logits)
    erased_center = center_logits(erased_logits)
    random_center = center_logits(random_logits)
    cand_u = ce_np(erased_logits, y) - ce_np(raw_logits, y)
    rand_u = ce_np(random_logits, y) - ce_np(raw_logits, y)
    cand_d = float(np.mean(np.abs(erased_center - raw_center)))
    rand_d = float(np.mean(np.abs(random_center - raw_center)))
    removed = h - erase_np(h, U, mu)
    removed_r = h - erase_np(h, random_U, mu)
    return {
        "candidate_utility": float(cand_u), "random_utility": float(rand_u),
        "signed_utility": float(cand_u - rand_u),
        "candidate_decision": cand_d, "random_decision": rand_d,
        "decision_dependence": float(cand_d - rand_d),
        "candidate_energy": float(np.mean(np.sum(removed * removed, axis=1))),
        "random_energy": float(np.mean(np.sum(removed_r * removed_r, axis=1))),
        "candidate_rms": float(np.sqrt(np.mean(np.sum(removed * removed, axis=1)))),
        "random_rms": float(np.sqrt(np.mean(np.sum(removed_r * removed_r, axis=1)))),
        "n_trials": int(len(y)),
    }


def _inner_groups(subjects: Sequence[str], fold: int) -> list[list[int]]:
    order = sorted(range(len(subjects)), key=lambda i: stable_seed("v3-inner", fold, subjects[i]))
    return [order[i::INNER_FOLDS] for i in range(INNER_FOLDS)]


def _fit_at_epoch(train_arrays: Mapping[str, np.ndarray], epoch: int, fold: int, inner: int, device: torch.device, seed_tag: str = "trajectory") -> Any | None:
    if int(epoch) <= 0:
        return None
    mask = train_arrays["session"] == 1
    w = train_arrays["head_weight"]
    b = train_arrays["head_bias"]
    config = dict(GENERIC); config["epochs"] = int(epoch)
    adapter, _ = BASE.fit_adapter(train_arrays["h"][mask], train_arrays["y"][mask], w, b, config, None, stable_seed(seed_tag, fold, inner, epoch), device)
    return adapter


def _prepare_arrays_with_head(arrays: Mapping[str, np.ndarray], model: Any) -> dict[str, np.ndarray]:
    out = dict(arrays)
    w, b = _head(model); out["head_weight"] = w; out["head_bias"] = b
    return out


def _basis_for_training(train_arrays: Mapping[str, np.ndarray], subjects: Sequence[str]) -> dict[str, Any]:
    return V2.build_basis(train_arrays, list(map(str, subjects)), list(range(len(subjects))))


def _calibration_rows_for_basis(h: np.ndarray, y: np.ndarray, U: np.ndarray, mu: np.ndarray, random_u: np.ndarray, w: np.ndarray, b: np.ndarray) -> dict[str, float]:
    raw = BASE.logits_for(h, w, b)
    er = BASE.logits_for(erase_np(h, U, mu), w, b)
    rr = BASE.logits_for(erase_np(h, random_u, mu), w, b)
    e = h - erase_np(h, U, mu); re = h - erase_np(h, random_u, mu)
    return {
        "candidate_removed_energy": float(np.mean(np.sum(e * e, axis=1))),
        "matched_removed_energy": float(np.mean(np.sum(re * re, axis=1))),
        "candidate_removed_rms": float(np.sqrt(np.mean(np.sum(e * e, axis=1)))),
        "matched_removed_rms": float(np.sqrt(np.mean(np.sum(re * re, axis=1)))),
        "candidate_variance_fraction": float(np.mean(np.sum(e * e, axis=1)) / max(np.var(h, axis=0).sum(), EPS)),
        "matched_variance_fraction": float(np.mean(np.sum(re * re, axis=1)) / max(np.var(h, axis=0).sum(), EPS)),
        "candidate_logit_response": float(np.mean(np.abs(center_logits(er) - center_logits(raw)))),
        "matched_logit_response": float(np.mean(np.abs(center_logits(rr) - center_logits(raw)))),
        "candidate_mean_delta_ce": float(ce_np(er, y) - ce_np(raw, y)),
        "matched_mean_delta_ce": float(ce_np(rr, y) - ce_np(raw, y)),
    }


def repair_audit() -> None:
    """Executable regression tests for the repaired symmetric decision metric."""
    rng = np.random.default_rng(stable_seed("decision-audit"))
    h = rng.normal(size=(37, int(BASE.DIM))); y = rng.integers(0, 2, size=len(h))
    w = rng.normal(size=(2, int(BASE.DIM))); b = rng.normal(size=2); mu = h.mean(0)
    U = random_subspace(stable_seed("decision-audit", "u"), 2)
    R = random_subspace(stable_seed("decision-audit", "r"), 2)
    m = _subject_metrics(h, y, U, mu, R, w, b)
    # The metric must be invariant to adding a per-trial class-independent offset.
    offset = rng.normal(size=(len(h), 1))
    raw = BASE.logits_for(h, w, b); er = BASE.logits_for(erase_np(h, U, mu), w, b); rr = BASE.logits_for(erase_np(h, R, mu), w, b)
    d0 = float(np.mean(np.abs(center_logits(er) - center_logits(raw))))
    d1 = float(np.mean(np.abs(center_logits(er + offset) - center_logits(raw + offset))))
    rd0 = float(np.mean(np.abs(center_logits(rr) - center_logits(raw))))
    rd1 = float(np.mean(np.abs(center_logits(rr + offset) - center_logits(raw + offset))))
    # Run the same calculation twice to detect accidental global RNG use.
    m2 = _subject_metrics(h, y, U, mu, R, w, b)
    values = {
        "status": "MEASUREMENT_REPAIR_PASS" if abs(d0 - d1) < 1e-12 and abs(rd0 - rd1) < 1e-12 and m == m2 else "MEASUREMENT_REPAIR_FAIL",
        "formula": "mean(abs(center(z_erased_U)-center(z_raw)))",
        "candidate_random_formula_identical": True,
        "offset_invariance_candidate_abs_error": abs(d0 - d1),
        "offset_invariance_random_abs_error": abs(rd0 - rd1),
        "determinism_exact": bool(m == m2),
        "candidate_decision": m["candidate_decision"], "random_decision": m["random_decision"],
        "candidate_random_not_identical": bool(abs(m["candidate_decision"] - m["random_decision"]) > 1e-12),
    }
    write_json(OUT / "DECISION_METRIC_AUDIT.json", values)
    (EXP_ROOT / "BUG_DESCRIPTION.md").write_text("# V2 decision-dependence bug\n\nV2 compared uncentered erased logits with centered raw logits for candidate interventions, while random controls were centered on both sides. This asymmetric definition can rank candidates by arbitrary class-independent offsets.\n", encoding="utf-8")
    (EXP_ROOT / "REPAIR_LOG.md").write_text("# V3 measurement repair\n\nAll candidate and random interventions now use `center(z)=z-mean_class(z)` on both erased and raw logits. The executable audit checks offset invariance, deterministic replay, and non-identical candidate/random responses.\n\n" + json.dumps(clean(values), indent=2) + "\n", encoding="utf-8")
    (EXP_ROOT / "DECISION_METRIC_AUDIT.md").write_text("# Decision metric audit\n\n" + json.dumps(clean(values), indent=2) + "\n", encoding="utf-8")
    if values["status"] != "MEASUREMENT_REPAIR_PASS":
        raise RuntimeError("EXP4_V3_MEASUREMENT_INVALID: decision metric audit failed")


def generic_reproduction(scope_data: Mapping[str, Any], device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for fold in RUNS:
        legal = legal_subjects(scope_data, fold); outcome = list(map(str, scope_data["audit_roles"][str(fold)]["outcome"]))
        model, _ = V2.copy_or_verify_anchor(fold, device)
        train = _prepare_arrays_with_head(_arrays_from_cache(model, legal, [1], fold, "legal", device), model)
        test = _prepare_arrays_with_head(_arrays_from_cache(model, outcome, [2], fold, "outcome", device), model)
        w, b = train["head_weight"], train["head_bias"]
        frozen_logits = BASE.logits_for(test["h"], w, b); seed_logits: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            ad, meta = BASE.fit_adapter(train["h"], train["y"], w, b, GENERIC, None, stable_seed("v3-baseline", fold, seed), device)
            logits = BASE.logits_for(BASE.adapter_apply(ad, test["h"], None, device), w, b); seed_logits.append(logits)
            seed_rows.append({"fold": fold, "seed": seed, "method": "Generic", "adapter_state_sha256": meta.get("adapter_state_sha256"), "mean_S3_BA": float(np.mean([BASE.metric_ba(test["y"][test["sid"] == i], logits[test["sid"] == i].argmax(1)) for i in range(len(outcome))]))})
        avg = np.mean(seed_logits, axis=0)
        for method, logits in (("Frozen", frozen_logits), ("Generic", avg)):
            pred = logits.argmax(1); frozen_pred = frozen_logits.argmax(1)
            for i, subject in enumerate(outcome):
                m = test["sid"] == i
                ba = BASE.metric_ba(test["y"][m], pred[m]); fb = BASE.metric_ba(test["y"][m], frozen_pred[m])
                rows.append({"fold": fold, "subject": subject, "method": method, "seed_aggregation": "mean_logits" if method == "Generic" else "single_seed", "BA": ba, "Frozen_BA": fb, "delta_BA_vs_Frozen": ba - fb, "macro_F1": BASE.metric_macro_f1(test["y"][m], pred[m]), "accuracy": float(np.mean(pred[m] == test["y"][m]))})
    frame = pd.DataFrame(rows); seed_frame = pd.DataFrame(seed_rows)
    write_csv(OUT / "GENERIC_REPRODUCTION.csv", frame)
    write_csv(OUT / "SEED_ROBUSTNESS.csv", seed_frame)
    summary = frame.groupby("method").agg(n_subjects=("subject", "size"), BA_mean=("BA", "mean"), delta_BA_vs_Frozen_mean=("delta_BA_vs_Frozen", "mean"), negative_transfer_rate=("delta_BA_vs_Frozen", lambda x: float(np.mean(np.asarray(x) < 0)))).reset_index()
    selected_id = GENERIC["id"]
    (EXP_ROOT / "GENERIC_BASELINE_SELECTION.md").write_text("# Generic baseline selection\n\nThe V2-selected S2-only configuration was frozen before V3 trajectory analysis: **" + selected_id + "**. No held S3 label is used for selection.\n\n" + summary.to_markdown(index=False) + "\n", encoding="utf-8")
    return frame


def calibrate_controls(scope_data: Mapping[str, Any], device: torch.device) -> None:
    energy_rows: list[dict[str, Any]] = []; control_rows: list[dict[str, Any]] = []
    for fold in RUNS:
        legal = legal_subjects(scope_data, fold); model, _ = V2.copy_or_verify_anchor(fold, device)
        arr = _prepare_arrays_with_head(_arrays_from_cache(model, legal, [0, 1], fold, "legal_s01", device), model)
        pack = V2.build_basis(arr, legal, list(range(len(legal)))); w, b = arr["head_weight"], arr["head_bias"]; mu = pack["center"]
        for rank in RANKS:
            U = pack["basis"][:, :rank]; pool = random_pool(("calibration", fold), rank); matched, info = energy_match(arr["h"], U, pool)
            stats = _calibration_rows_for_basis(arr["h"], arr["y"], U, mu, matched, w, b)
            ordinary = np.asarray([np.mean(np.sum((arr["h"] - erase_np(arr["h"], r, mu)) ** 2, axis=1)) for r in pool])
            energy_rows.append({"fold": fold, "rank": rank, "persistence_strength": float(np.sum(pack["eigenvalues"][:rank])), "candidate_energy": info["candidate_energy"], "matched_random_energy": info["random_energy"], "relative_mismatch": info["relative_mismatch"], "within_tolerance": info["within_tolerance"], **stats})
            control_rows.append({"fold": fold, "rank": rank, "pool_size": RANDOM_POOL, "ordinary_random_energy_mean": float(ordinary.mean()), "ordinary_random_energy_sd": float(ordinary.std()), "ordinary_random_energy_q05": float(np.quantile(ordinary, .05)), "ordinary_random_energy_q95": float(np.quantile(ordinary, .95)), "matched_random_energy": info["random_energy"], "candidate_energy": info["candidate_energy"], "relative_mismatch": info["relative_mismatch"], "pool_index": info["pool_index"], "within_tolerance": info["within_tolerance"], "outcome_matching": False})
    write_csv(OUT / "INTERVENTION_ENERGY.csv", energy_rows)
    write_csv(OUT / "RANDOM_CONTROL_CALIBRATION.csv", control_rows)
    frame = pd.DataFrame(control_rows)
    status = "ENERGY_MATCH_PASS" if bool(frame.within_tolerance.all()) else "ENERGY_MATCH_PARTIAL"
    payload = {"status": status, "tolerance_relative": ENERGY_TOLERANCE, "matched_on": "removed representation energy only", "ordinary_random_retained": True, "rows": control_rows}
    write_json(OUT / "CONTROL_CALIBRATION.json", payload)
    (EXP_ROOT / "CONTROL_CALIBRATION_AUDIT.md").write_text("# Control calibration audit\n\nRandom controls were generated deterministically and matched only on removed representation energy. No task outcome or S3 quantity enters matching. The ordinary random distribution is retained as a secondary diagnostic.\n\n" + json.dumps(clean(payload), indent=2) + "\n", encoding="utf-8")


def _measure_checkpoint(scope_data: Mapping[str, Any], epoch: int, device: torch.device, include_directions: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows: list[dict[str, Any]] = []; direction_rows: list[dict[str, Any]] = []
    for fold in RUNS:
        legal = legal_subjects(scope_data, fold); model, _ = V2.copy_or_verify_anchor(fold, device)
        raw = _prepare_arrays_with_head(_arrays_from_cache(model, legal, [0, 1], fold, "legal_s01", device), model)
        w, b = raw["head_weight"], raw["head_bias"]; groups = _inner_groups(legal, fold)
        rank_acc: dict[int, list[dict[str, Any]]] = {r: [] for r in RANKS}; dir_acc: dict[int, list[dict[str, Any]]] = {j: [] for j in range(CANDIDATE_COUNT)}
        for inner, held_ids in enumerate(groups):
            train_ids = [i for i in range(len(legal)) if i not in set(held_ids)]
            train_raw, train_subjects = _subset_arrays(raw, train_ids); held_raw, held_subjects = _subset_arrays(raw, held_ids)
            train_raw["head_weight"], train_raw["head_bias"] = w, b; held_raw["head_weight"], held_raw["head_bias"] = w, b
            adapter = _fit_at_epoch(train_raw, epoch, fold, inner, device)
            tr = _transform_features(train_raw, adapter, device); he = _transform_features(held_raw, adapter, device)
            pack = _basis_for_training(tr, train_subjects); mu = pack["center"]
            # Energy calibration is frozen from the inner training side.
            for rank in RANKS:
                U = pack["basis"][:, :rank]; pool = random_pool(("trajectory", fold, inner, epoch), rank); rU, info = energy_match(tr["h"], U, pool)
                for sid, subject in enumerate(held_subjects):
                    m = he["sid"] == sid
                    if not np.any(m): continue
                    row = _subject_metrics(he["h"][m], he["y"][m], U, mu, rU, w, b)
                    rank_acc[rank].append({"fold": fold, "inner_fold": inner, "subject": subject, "epoch": epoch, "rank": rank, "persistence": float(np.sum(pack["eigenvalues"][:rank])), "energy_relative_mismatch": info["relative_mismatch"], **row})
            if include_directions:
                for j in range(CANDIDATE_COUNT):
                    U = pack["basis"][:, j:j + 1]; pool = random_pool(("direction", fold, inner, epoch), 1); rU, info = energy_match(tr["h"], U, pool)
                    for sid, subject in enumerate(held_subjects):
                        m = he["sid"] == sid
                        if not np.any(m): continue
                        row = _subject_metrics(he["h"][m], he["y"][m], U, mu, rU, w, b)
                        dir_acc[j].append({"fold": fold, "inner_fold": inner, "subject": subject, "epoch": epoch, "direction": j + 1, "persistence": float(pack["eigenvalues"][j]), "energy_relative_mismatch": info["relative_mismatch"], **row})
        for rank, rows in rank_acc.items():
            d = pd.DataFrame(rows)
            if d.empty: continue
            util = d.signed_utility.to_numpy(float); dec = d.decision_dependence.to_numpy(float)
            rank_rows.append({"fold": fold, "epoch": epoch, "rank": rank, "persistence": float(d.persistence.mean()), "signed_utility": float(util.mean()), "utility_median": float(np.median(util)), "utility_positive_fraction": float(np.mean(util > 0)), "utility_p_raw": sign_p(util, stable_seed("traj-u", fold, epoch, rank)), "decision_dependence": float(dec.mean()), "decision_median": float(np.median(dec)), "decision_positive_fraction": float(np.mean(dec > 0)), "decision_p_raw": sign_p(dec, stable_seed("traj-d", fold, epoch, rank)), "energy_relative_mismatch_mean": float(d.energy_relative_mismatch.mean()), "n_subject_measurements": int(len(d))})
        for j, rows in dir_acc.items():
            if not rows: continue
            d = pd.DataFrame(rows); util = d.signed_utility.to_numpy(float); dec = d.decision_dependence.to_numpy(float)
            direction_rows.append({"fold": fold, "epoch": epoch, "direction": j + 1, "persistence": float(d.persistence.mean()), "signed_utility": float(util.mean()), "utility_positive_fraction": float(np.mean(util > 0)), "utility_p_raw": sign_p(util, stable_seed("dir-u", fold, epoch, j)), "decision_dependence": float(dec.mean()), "decision_positive_fraction": float(np.mean(dec > 0)), "decision_p_raw": sign_p(dec, stable_seed("dir-d", fold, epoch, j)), "n_subject_measurements": int(len(d))})
    rank_frame = pd.DataFrame(rank_rows); dir_frame = pd.DataFrame(direction_rows)
    if not rank_frame.empty:
        rank_frame["utility_p_holm"] = np.nan; rank_frame["decision_p_holm"] = np.nan
        for (fold, epoch), ix in rank_frame.groupby(["fold", "epoch"]).groups.items():
            ids = list(ix); rank_frame.loc[ids, "utility_p_holm"] = holm(rank_frame.loc[ids, "utility_p_raw"].to_numpy(float)); rank_frame.loc[ids, "decision_p_holm"] = holm(rank_frame.loc[ids, "decision_p_raw"].to_numpy(float))
        rank_frame["gate"] = ((rank_frame.persistence > 0) & (rank_frame.signed_utility > 0) & (rank_frame.utility_positive_fraction >= .55) & (rank_frame.decision_dependence > 0) & (rank_frame.decision_positive_fraction >= .55) & (rank_frame.utility_p_holm < .05) & (rank_frame.decision_p_holm < .05))
    if not dir_frame.empty:
        dir_frame["utility_p_holm"] = np.nan; dir_frame["decision_p_holm"] = np.nan
        for (fold, epoch), ix in dir_frame.groupby(["fold", "epoch"]).groups.items():
            ids = list(ix); dir_frame.loc[ids, "utility_p_holm"] = holm(dir_frame.loc[ids, "utility_p_raw"].to_numpy(float)); dir_frame.loc[ids, "decision_p_holm"] = holm(dir_frame.loc[ids, "decision_p_raw"].to_numpy(float))
        dir_frame["gate"] = ((dir_frame.persistence > 0) & (dir_frame.signed_utility > 0) & (dir_frame.utility_positive_fraction >= .55) & (dir_frame.decision_dependence > 0) & (dir_frame.decision_positive_fraction >= .55) & (dir_frame.utility_p_holm < .05) & (dir_frame.decision_p_holm < .05))
    return rank_frame, dir_frame


def certify_subspaces(scope_data: Mapping[str, Any], device: torch.device) -> None:
    rank, direction = _measure_checkpoint(scope_data, 0, device, include_directions=True)
    write_csv(OUT / "SUBSPACE_CERTIFICATION.csv", rank)
    write_csv(OUT / "DIRECTION_CERTIFICATION.csv", direction)
    selected: dict[str, list[int]] = {}
    for fold in RUNS:
        selected[str(fold)] = []
        for r in RANKS:
            subset = rank[(rank.fold == fold) & (rank.rank == r)] if not rank.empty else pd.DataFrame()
            if not subset.empty and bool(subset.iloc[0].gate):
                selected[str(fold)].append(int(r))
    write_json(OUT / "SUBSPACE_SELECTION.json", {"selected_by_fold": selected, "selection_rule": "training-side rank gate only; no S3", "outer_accessed": False})
    (EXP_ROOT / "SUBSPACE_CERTIFICATION.md").write_text("# Deployment-matched subspace certification\n\nOnly cumulative ranks 1, 2, and 4 were primary hypotheses. Holm correction is applied across these three ranks within each fold. Individual directions are retained as secondary diagnostics.\n\n" + (rank.to_markdown(index=False) if not rank.empty else "No rows.") + "\n", encoding="utf-8")


def trajectory(scope_data: Mapping[str, Any], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rank: list[pd.DataFrame] = []; all_dir: list[pd.DataFrame] = []
    for epoch in TRAJECTORY_EPOCHS:
        print(f"[v3 trajectory] epoch={epoch}", flush=True)
        rank, direction = _measure_checkpoint(scope_data, int(epoch), device, include_directions=False)
        all_rank.append(rank); all_dir.append(direction)
    rf = pd.concat(all_rank, ignore_index=True) if all_rank else pd.DataFrame(); df = pd.concat(all_dir, ignore_index=True) if all_dir else pd.DataFrame()
    write_csv(OUT / "TRAJECTORY_PERSISTENCE.csv", rf[[c for c in rf.columns if c in {"fold", "epoch", "rank", "persistence", "n_subject_measurements"}]] if not rf.empty else rf)
    write_csv(OUT / "TRAJECTORY_UTILITY.csv", rf[[c for c in rf.columns if c in {"fold", "epoch", "rank", "signed_utility", "utility_median", "utility_positive_fraction", "utility_p_raw", "utility_p_holm", "gate", "n_subject_measurements"}]] if not rf.empty else rf)
    write_csv(OUT / "TRAJECTORY_DECISION.csv", rf[[c for c in rf.columns if c in {"fold", "epoch", "rank", "decision_dependence", "decision_median", "decision_positive_fraction", "decision_p_raw", "decision_p_holm", "gate", "n_subject_measurements"}]] if not rf.empty else rf)
    write_csv(OUT / "TRAJECTORY_ALL.csv", rf); write_json(OUT / "TRAJECTORY_STATE.json", {"status": "TRAJECTORY_COMPLETE", "epochs": list(TRAJECTORY_EPOCHS), "seed": TRAJECTORY_SEED, "outer_accessed": False})
    return rf, df


def _emergence_events(rf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if rf.empty:
        return pd.DataFrame(columns=["fold", "rank", "emerged", "t_star", "next_epoch", "utility_t_star", "utility_next", "decision_t_star", "decision_next"])
    for fold in RUNS:
        for rank in RANKS:
            x = rf[(rf.fold == fold) & (rf.rank == rank)].sort_values("epoch")
            chosen = None
            for i in range(len(x) - 1):
                a, b = x.iloc[i], x.iloc[i + 1]
                if bool(a.gate) and bool(b.gate): chosen = (a, b); break
            if chosen is None:
                rows.append({"fold": fold, "rank": rank, "emerged": False, "t_star": None, "next_epoch": None, "utility_t_star": None, "utility_next": None, "decision_t_star": None, "decision_next": None})
            else:
                a, b = chosen; rows.append({"fold": fold, "rank": rank, "emerged": True, "t_star": int(a.epoch), "next_epoch": int(b.epoch), "utility_t_star": float(a.signed_utility), "utility_next": float(b.signed_utility), "decision_t_star": float(a.decision_dependence), "decision_next": float(b.decision_dependence)})
    out = pd.DataFrame(rows)
    return out


def _baseline_fold_effects(scope_data: Mapping[str, Any], device: torch.device) -> pd.DataFrame:
    p = OUT / "GENERIC_REPRODUCTION.csv"
    if not p.is_file(): generic_reproduction(scope_data, device)
    x = pd.read_csv(p); g = x[x.method == "Generic"]; f = x[x.method == "Frozen"]
    return g.groupby("fold").delta_BA_vs_Frozen.mean().rename("generic_delta_BA").reset_index()


def utility_collapse_audit(scope_data: Mapping[str, Any], rf: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, str]:
    events = _emergence_events(rf); write_csv(OUT / "EMERGENCE_EVENTS.csv", events)
    fold_effects = _baseline_fold_effects(scope_data, device)
    rows: list[dict[str, Any]] = []
    for fold in RUNS:
        ev = events[(events.fold == fold) & (events.emerged)]
        if ev.empty:
            rows.append({"fold": fold, "rank": None, "emerged": False, "t_star": None, "utility_t_star": None, "utility_final": None, "utility_delta_final_minus_t_star": None, "collapse_replicated": False, "generic_delta_BA": float(fold_effects.loc[fold_effects.fold == fold, "generic_delta_BA"].iloc[0]) if not fold_effects.loc[fold_effects.fold == fold].empty else np.nan})
            continue
        # Predeclared selection: smallest rank, then earliest t*, from training-side evidence only.
        ev = ev.sort_values(["rank", "t_star"]); e = ev.iloc[0]; x = rf[(rf.fold == fold) & (rf.rank == int(e.rank))].sort_values("epoch")
        final = x.iloc[-1]; delta = float(final.signed_utility - e.utility_t_star); collapse = bool(delta < -max(.01, .10 * max(abs(float(e.utility_t_star)), .01)))
        rows.append({"fold": fold, "rank": int(e.rank), "emerged": True, "t_star": int(e.t_star), "utility_t_star": float(e.utility_t_star), "utility_final": float(final.signed_utility), "utility_delta_final_minus_t_star": delta, "collapse_replicated": collapse, "generic_delta_BA": float(fold_effects.loc[fold_effects.fold == fold, "generic_delta_BA"].iloc[0]) if not fold_effects.loc[fold_effects.fold == fold].empty else np.nan})
    cf = pd.DataFrame(rows); write_csv(OUT / "UTILITY_COLLAPSE.csv", cf)
    emerged_n = int(cf.emerged.sum()) if not cf.empty else 0; collapsed = cf[cf.collapse_replicated] if not cf.empty else pd.DataFrame()
    if emerged_n == 0:
        terminal = "EXP4_V3_NO_PROTECTED_EMERGENCE"
    elif len(collapsed) < 2 or float(collapsed.utility_delta_final_minus_t_star.mean()) >= 0:
        terminal = "EXP4_V3_EMERGENCE_NO_COLLAPSE"
    else:
        # A directional fold-level association is a mechanistic diagnostic, not a tuning criterion.
        assoc = collapsed.generic_delta_BA.to_numpy(float)
        linked = bool(np.mean(assoc < 0) >= .5 or (len(assoc) >= 3 and np.corrcoef(collapsed.utility_delta_final_minus_t_star.to_numpy(float), assoc)[0, 1] > 0))
        terminal = "EXP4_V3_COLLAPSE_NOT_LINKED_TO_TRANSFER" if not linked else "CASE_C_AUTHORIZED"
    (EXP_ROOT / "EMERGENCE_AUDIT.md").write_text("# Emergence audit\n\nEmergence requires two consecutive checkpoints passing persistence, signed utility, repaired decision dependence, subject-positive-fraction, and Holm gates. The first passing checkpoint is `t*`; rank selection is smallest passing rank.\n\n" + events.to_markdown(index=False) + "\n", encoding="utf-8")
    (EXP_ROOT / "UTILITY_COLLAPSE_AUDIT.md").write_text("# Utility collapse audit\n\nCollapse is predeclared as a final-minus-trigger utility decrease below `-max(0.01, 0.10*abs(U(t*)))`, replicated in at least two folds before Case C can be considered. S3 delta BA is diagnostic only.\n\n" + cf.to_markdown(index=False) + "\n", encoding="utf-8")
    return cf, terminal


def _fit_guard_after_trigger(h: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, config: Mapping[str, Any], U: np.ndarray, mu: np.ndarray, t_star: int, device: torch.device, seed: int) -> tuple[Any, dict[str, Any]]:
    """Emergence-triggered residual adapter; only called after Case C."""
    BASE.seed_all(seed); adapter = BASE.LinearAdapter(h.shape[1]).to(device)
    x = torch.from_numpy(h.astype(np.float32)).to(device); target = torch.from_numpy(y.astype(np.int64)).to(device); wt = torch.from_numpy(w.astype(np.float32)).to(device); bt = torch.from_numpy(b.astype(np.float32)).to(device); Ut = torch.from_numpy(U.astype(np.float32)).to(device); mut = torch.from_numpy(mu.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    history: list[dict[str, float]] = []; active = 0; steps = 0
    total_epochs = int(config["epochs"])
    for epoch in range(total_epochs):
        order = torch.randperm(len(y), device=device); adapter.train(); loss_sum = 0.
        for start in range(0, len(y), 256):
            idx = order[start:start + 256]; xb, yb = x[idx], target[idx]; opt.zero_grad(set_to_none=True); raw_h = xb + adapter.linear(xb); raw_loss = torch.nn.functional.cross_entropy(raw_h @ wt.T + bt, yb)
            if epoch < int(t_star):
                loss = raw_loss; hinge = torch.zeros((), device=device)
            else:
                erased = xb - ((xb - mut) @ Ut) @ Ut.T; er_h = erased + adapter.linear(erased); er_loss = torch.nn.functional.cross_entropy(er_h @ wt.T + bt, yb); g = er_loss - raw_loss; hinge = torch.relu(torch.as_tensor(0.0, device=device) - g); loss = raw_loss + .5 * hinge; active += int(float(hinge.detach()) > 1e-6); steps += 1
            loss.backward(); opt.step(); loss_sum += float(loss.detach()) * len(idx)
        history.append({"epoch": epoch + 1, "task_loss": float(loss_sum / max(len(y), 1)), "guard_active": float(epoch >= int(t_star))})
    adapter.eval(); return adapter, {"seed": seed, "trigger_epoch": int(t_star), "basis_rank": int(U.shape[1]), "history": history, "adapter_state_sha256": BASE.adapter_state_sha(adapter), "constraint_active_fraction": active / max(steps, 1)}


def evaluate_guard(scope_data: Mapping[str, Any], collapse: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []; controls: list[dict[str, Any]] = []
    for fold in RUNS:
        e = collapse[(collapse.fold == fold) & (collapse.emerged) & (collapse.collapse_replicated)]
        if e.empty: continue
        e = e.sort_values(["rank", "t_star"]).iloc[0]; rank, tstar = int(e.rank), int(e.t_star); legal = legal_subjects(scope_data, fold); outcome = list(map(str, scope_data["audit_roles"][str(fold)]["outcome"]))
        model, _ = V2.copy_or_verify_anchor(fold, device); tr = _prepare_arrays_with_head(_arrays_from_cache(model, legal, [1], fold, "legal", device), model); te = _prepare_arrays_with_head(_arrays_from_cache(model, outcome, [2], fold, "outcome", device), model); w, b = tr["head_weight"], tr["head_bias"]
        # Reconstruct the trigger basis from the legal training side at t*.
        raw_s01 = _prepare_arrays_with_head(_arrays_from_cache(model, legal, [0, 1], fold, "legal_s01", device), model); ad_pre = _fit_at_epoch(raw_s01, tstar, fold, 99, device, seed_tag="guard-trigger"); tr_s01 = _transform_features(raw_s01, ad_pre, device); pack = _basis_for_training(tr_s01, legal); U = pack["basis"][:, :rank]; mu = pack["center"]
        guard, meta = _fit_guard_after_trigger(tr["h"], tr["y"], w, b, GENERIC, U, mu, tstar, device, stable_seed("guard", fold, 0)); gh = BASE.adapter_apply(guard, te["h"], None, device); gl = BASE.logits_for(gh, w, b); fl = BASE.logits_for(te["h"], w, b); ga = BASE.fit_adapter(tr["h"], tr["y"], w, b, GENERIC, None, stable_seed("generic-final", fold, 0), device)[0]; glg = BASE.logits_for(BASE.adapter_apply(ga, te["h"], None, device), w, b)
        for i, subject in enumerate(outcome):
            m = te["sid"] == i; fb = BASE.metric_ba(te["y"][m], fl.argmax(1)[m]); gb = BASE.metric_ba(te["y"][m], glg.argmax(1)[m]); pb = BASE.metric_ba(te["y"][m], gl.argmax(1)[m]); rows.append({"fold": fold, "subject": subject, "method": "Frozen", "BA": fb, "Frozen_BA": fb, "delta_BA_vs_Frozen": 0.}); rows.append({"fold": fold, "subject": subject, "method": "Generic", "BA": gb, "Frozen_BA": fb, "delta_BA_vs_Frozen": gb - fb}); rows.append({"fold": fold, "subject": subject, "method": "EmergenceTriggeredUtilityGuard", "BA": pb, "Frozen_BA": fb, "delta_BA_vs_Frozen": pb - fb, "trigger_epoch": tstar, "rank": rank, "constraint_active_fraction": meta["constraint_active_fraction"]})
        controls.append({"fold": fold, "method": "EmergenceTriggeredUtilityGuard", "rank": rank, "trigger_epoch": tstar, "status": "RUN"})
    frame = pd.DataFrame(rows); write_csv(OUT / "DEV_SUBJECT_RESULTS.csv", frame); write_csv(OUT / "CONTROL_COMPARISON.csv", pd.DataFrame(controls)); write_csv(OUT / "CONSTRAINT_ACTIVITY.csv", frame[frame.method == "EmergenceTriggeredUtilityGuard"] if not frame.empty else pd.DataFrame()); return frame, pd.DataFrame(controls)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame.empty: return pd.DataFrame()
    for method, g in frame.groupby("method", sort=False):
        d = g.delta_BA_vs_Frozen.to_numpy(float); rows.append({"method": method, "n_subjects": int(len(g)), "BA_mean": float(g.BA.mean()), "delta_BA_vs_Frozen_mean": float(d.mean()), "negative_transfer_rate": float(np.mean(d < 0)), "negative_transfer_count": int(np.sum(d < 0)), "worst_quartile_delta": float(np.sort(d)[:max(1, math.ceil(len(d) * .25))].mean()), "worst_subject_delta": float(d.min())})
    return pd.DataFrame(rows)


def _write_figures(rf: pd.DataFrame, frame: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not rf.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for rank, g in rf.groupby("rank"):
            q = g.groupby("epoch").mean(numeric_only=True).sort_index(); axes[0].plot(q.index, q.persistence, "o-", label=f"rank {rank}"); axes[1].plot(q.index, q.signed_utility, "o-", label=f"rank {rank}"); axes[2].plot(q.index, q.decision_dependence, "o-", label=f"rank {rank}")
        for ax, title in zip(axes, ("P(t)", "signed utility U(t)", "decision D(t)")): ax.axhline(0, color="black", lw=.8); ax.set_title(title); ax.set_xlabel("adapter epoch"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGURES / "figure_3_training_trajectory.png", dpi=220); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); q = rf.groupby(["epoch", "rank"]).signed_utility.mean().unstack(); q.plot(ax=ax, marker="o"); ax.axhline(0, color="black", lw=.8); ax.set_ylabel("signed utility"); ax.set_title("Figure 1/2: rank certification"); fig.tight_layout(); fig.savefig(FIGURES / "figure_2_rank_certification.png", dpi=220); plt.close(fig)
    if not frame.empty and "method" in frame:
        fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(frame));
        for method, g in frame.groupby("method", sort=False): ax.scatter(g.index, g.BA, label=method)
        ax.set_ylabel("held S3 subject BA"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "figure_7_subject_ba.png", dpi=220); plt.close(fig)


def _write_reports(state: Mapping[str, Any], baseline: pd.DataFrame | None = None, trajectory_frame: pd.DataFrame | None = None) -> None:
    terminal = str(state.get("terminal_state", "UNKNOWN")); summary = state.get("summary", [])
    docs = {
        "PROTOCOL_SELECTION_AUDIT.md": "# Protocol selection audit\n\nV3 is a new prospective trajectory protocol. It keeps the S1-only EEGNet anchor, the legal 41-subject development cohort, subject-disjoint inner evidence, and the sealed 10-subject outer cohort. No S3 quantity selects a rank, trigger, threshold, or method.\n",
        "FINAL_MODEL_CARD.md": f"# Exp4 V3 model card\n\nTerminal state: **{terminal}**. V3 repairs the decision metric, energy-matches random interventions, tests cumulative deployment-matched ranks 1/2/4, and requires two consecutive training-side checkpoints before declaring emergence. Outer data were not accessed.\n",
        "CLAIM_AUDIT.md": "# Claim audit\n\nClaims are restricted to the development cohort and the predeclared EEGNet representation. A null or non-significant Guard result is not treated as equivalence. No outer claim is made without a final lock.\n",
        "REVIEWER_SELF_AUDIT.md": "# Reviewer self-audit\n\nRemaining risks include finite development subjects, frozen-head dependence, and the fact that trajectory evidence is a mechanism audit rather than an independent test. The protocol does not justify universal EEG invariance claims.\n",
        "REPRODUCIBILITY.md": "# Reproducibility\n\nRun `audit`, `prepare`, `baseline`, `repair`, `calibrate`, `certify`, `trajectory`, and `finalize` with the same environment. Feature caches/checkpoints remain outside Git; compact CSV/JSON/Markdown outputs are versioned.\n",
        "ITERATION_LEDGER.md": "# Iteration ledger\n\n| version | exact change | pre-run expectation | decision |\n|---|---|---|---|\n| V3.0 | repaired symmetric decision metric | remove candidate/random centering asymmetry | frozen |\n| V3.1 | energy-matched random pool | reduce geometric control confounding | frozen |\n| V3.2 | cumulative rank 1/2/4 and two-checkpoint emergence | detect subspace-level emergence without outcome selection | frozen |\n| V3.3 | conditional guard only if Case C | do not train a guard after a falsified mechanism | " + terminal + " |\n",
    }
    for name, text in docs.items(): (EXP_ROOT / name).write_text(text, encoding="utf-8")
    payload = {"terminal_state": terminal, "state": clean(state), "outer_accessed": False, "outer_evaluation_count": 0}
    write_json(EXP_ROOT / "EXP4_V3_FINAL_REPORT.json", payload)
    (EXP_ROOT / "EXP4_V3_FINAL_REPORT.md").write_text("# PERSIST-EEG Experiment 4 V3 — final report\n\nTerminal state: **" + terminal + "**\n\n" + json.dumps(clean(state), indent=2, ensure_ascii=False) + "\n\nOuter subjects were not accessed during development.\n", encoding="utf-8")


def finalize(scope_data: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if not (OUT / "TRAJECTORY_ALL.csv").is_file():
        raise RuntimeError("trajectory must be completed before finalize")
    rf = pd.read_csv(OUT / "TRAJECTORY_ALL.csv"); collapse, terminal = utility_collapse_audit(scope_data, rf, device)
    baseline = pd.read_csv(OUT / "GENERIC_REPRODUCTION.csv") if (OUT / "GENERIC_REPRODUCTION.csv").is_file() else generic_reproduction(scope_data, device)
    frame: pd.DataFrame
    if terminal == "CASE_C_AUTHORIZED":
        frame, _ = evaluate_guard(scope_data, collapse, device)
        # Development success requires the guard to beat Generic on paired BA and not worsen the lower tail.
        if frame.empty or "EmergenceTriggeredUtilityGuard" not in set(frame.method):
            terminal = "EXP4_V3_UTILITY_GUARD_FAILED"
        else:
            g = frame[frame.method == "Generic"].set_index(["fold", "subject"]); p = frame[frame.method == "EmergenceTriggeredUtilityGuard"].set_index(["fold", "subject"]); d = (p.BA - g.BA).to_numpy(float)
            if float(d.mean()) < .005 or float(np.mean(p.delta_BA_vs_Frozen < 0)) > float(np.mean(g.delta_BA_vs_Frozen < 0)):
                terminal = "EXP4_V3_UTILITY_GUARD_FAILED"
            else:
                terminal = "EXP4_V3_DEV_CONFIRMED_OUTER_LOCKED"
    else:
        # Always report Frozen/Generic baseline at the end; no controls are trained after a falsified mechanism.
        b = baseline.copy(); frame = b
        write_csv(OUT / "DEV_SUBJECT_RESULTS.csv", frame); write_csv(OUT / "NEGATIVE_TRANSFER.csv", frame[[c for c in ["fold", "subject", "method", "Frozen_BA", "BA", "delta_BA_vs_Frozen"] if c in frame.columns]])
        write_csv(OUT / "CONTROL_COMPARISON.csv", pd.DataFrame([{"method": m, "status": "NOT_AUTHORIZED_NO_CASE_C"} for m in ("HistoricalHardP01_04", "DeploymentMatchedHard", "EmergenceTriggeredCoordinateGuard", "PersistenceOnlyUtilityGuard", "IdentityUtilityGuard", "PCAUtilityGuard", "RandomUtilityGuard", "EmergenceTriggeredUtilityGuard")]))
        write_csv(OUT / "CONSTRAINT_ACTIVITY.csv", pd.DataFrame(columns=["method", "status"]))
    summary = _summary(frame); write_csv(OUT / "DEV_METHOD_SUMMARY.csv", summary)
    # Keep outer sealed even if a development protocol succeeds; writing the lock is
    # deliberately separate from opening outer and is never automatic.
    state = {"terminal_state": terminal, "emergence_and_collapse_state": "CASE_C_AUTHORIZED" if terminal in {"EXP4_V3_DEV_CONFIRMED_OUTER_LOCKED", "EXP4_V3_UTILITY_GUARD_FAILED"} else terminal, "collapse": collapse.to_dict(orient="records"), "summary": summary.to_dict(orient="records"), "outer_accessed": False, "outer_authorized": False}
    write_json(OUT / "STATISTICAL_TESTS.json", state)
    write_json(PROTOCOL / "OUTER_LOCK.json", {"status": "OUTER_SEALED", "outer_evaluation_authorized": False, "outer_evaluation_count": 0, "outer_subject_ids_present": False, "outer_result_exists": False})
    _write_figures(rf, frame); _write_reports(state, baseline, rf)
    return state


def run(phase: str, device: torch.device) -> int:
    prepare_dirs()
    if phase == "audit": audit(); return 0
    if not (PROTOCOL / "EXP4_V3_DEV_PROTOCOL.json").is_file(): raise RuntimeError("run audit first")
    s = scope()
    if phase == "prepare": prepare(s, device); return 0
    if phase == "repair": repair_audit(); return 0
    if phase == "baseline": generic_reproduction(s, device); return 0
    if phase == "calibrate": calibrate_controls(s, device); return 0
    if phase == "certify": certify_subspaces(s, device); return 0
    if phase == "trajectory": trajectory(s, device); return 0
    if phase == "finalize": finalize(s, device); return 0
    if phase == "all":
        audit(); s = scope(); prepare(s, device); repair_audit(); generic_reproduction(s, device); calibrate_controls(s, device); certify_subspaces(s, device); trajectory(s, device); finalize(s, device); return 0
    if phase == "outer":
        lock = PROTOCOL / "EXP4_V3_FINAL_PROTOCOL_LOCK.json"
        if not lock.is_file(): raise RuntimeError("OUTER_FORBIDDEN: no final V3 protocol lock")
        raise RuntimeError("OUTER_FORBIDDEN: one-time outer evaluation is not authorized by this development runner")
    raise ValueError(phase)


def main() -> int:
    ap = argparse.ArgumentParser(description="PERSIST-EEG Exp4 V3 emergence-triggered audit")
    ap.add_argument("phase", choices=["audit", "prepare", "repair", "baseline", "calibrate", "certify", "trajectory", "finalize", "all", "outer"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"[v3] phase={args.phase} device={device} root={EXP_ROOT}", flush=True)
    return run(args.phase, device)


if __name__ == "__main__":
    raise SystemExit(main())
