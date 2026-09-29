"""One-shot OUTER_DEVELOPMENT evaluation of a frozen TRAIN pooling seal.

This code never ranks tokens or chooses hyperparameters on OUTER. It refuses
execution until the TRAIN selection seal and separately extracted OUTER token
provenance pass all role, identity, and SHA checks. Formal heldout is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aggregate import C_GRID, Q_GRID, TAU_GRID, fit_decoder, pool, predict_decoder, rank_subset, stable_seed, subject_equal, subject_metrics


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_extraction(base: Path, role: str, fold: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory = base / role
    provenance_path = directory / "EXTRACTION_PROVENANCE.json"
    p = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (p.get("fold"), p.get("role"), p.get("formal_final_heldout_eeg_reads")) != (fold, role, 0):
        raise RuntimeError(f"{role} role or final-heldout audit mismatch")
    if set(p["files"]) != {"tokens.npy", "labels.npy", "subjects.npy", "sessions.npy"}:
        raise RuntimeError(f"{role} extraction inventory mismatch")
    for name, expected in p["files"].items():
        if sha(directory / name) != expected:
            raise RuntimeError(f"{role} extraction file SHA mismatch: {name}")
    z = np.load(directory / "tokens.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / "labels.npy", allow_pickle=False)
    subjects = np.load(directory / "subjects.npy", allow_pickle=False)
    sessions = np.load(directory / "sessions.npy", allow_pickle=False)
    if z.shape != (len(y), 248, 200) or len(y) != len(subjects) or len(y) != len(sessions):
        raise RuntimeError(f"{role} token/label/subject inventory mismatch")
    expected_sessions = {1, 2} if role == "train" else {2}
    if set(np.unique(sessions).tolist()) != expected_sessions:
        raise RuntimeError(f"{role} session inventory mismatch")
    for start in range(0, len(z), 128):
        if not np.isfinite(z[start:start+128]).all():
            raise RuntimeError(f"{role} contains nonfinite frozen tokens")
    return p, z, y, subjects, sessions


def decoder(x: np.ndarray, y: np.ndarray, c: float):
    if c not in C_GRID:
        raise RuntimeError("decoder C not in frozen TRAIN grid")
    if x.shape[1] <= 400:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return fit_decoder(x, y, c)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(x)
    model = LogisticRegression(C=c, penalty="l2", solver="liblinear", dual=True,
                               max_iter=5000, tol=1e-4, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(transformed, y)
    return scaler, model


def method_features(z: np.ndarray, method: str, parameter: str,
                    r: np.ndarray, u: np.ndarray, energy: np.ndarray) -> np.ndarray:
    rz = (r - r.mean()) / r.std()
    uz = (u - u.mean()) / u.std()
    joint = rz + uz
    if method in ("MEAN_ALL", "FLATTEN_LINEAR"):
        if parameter != "none":
            raise RuntimeError("baseline received a selected parameter")
        return pool(z, method)
    value = float(parameter)
    if method in ("ENERGY_TOPK", "ENERGY_TWO_STREAM"):
        score = energy
    elif method in ("RELIABLE_TOPK", "RELIABILITY_WEIGHTED"):
        score = r
    elif method == "UTILITY_TOPK":
        score = u
    elif method in ("RELIABLE_UTILITY_TOPK", "RU_WEIGHTED", "RU_TWO_STREAM"):
        score = joint
    else:
        raise KeyError(method)
    if method in ("RELIABILITY_WEIGHTED", "RU_WEIGHTED"):
        if value not in TAU_GRID:
            raise RuntimeError("weight temperature outside frozen TRAIN grid")
        return pool(z, method, score=score, tau=value)
    if value not in Q_GRID:
        raise RuntimeError("subset fraction outside frozen TRAIN grid")
    return pool(z, method, subset=rank_subset(score, value))


def run(fold: int) -> None:
    runtime = Path(os.environ["TOKEN_AUDIT_RUNTIME"]).resolve()
    base = runtime / f"fold{fold}"
    seal_path = base / "TRAIN_SELECTION_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("fold"), seal.get("status"), seal.get("outer_development_eeg_reads"),
        seal.get("formal_final_heldout_eeg_reads")) != (fold, "FROZEN_TRAIN_ONLY", 0, 0):
        raise RuntimeError("invalid or non-TRAIN pooling selection seal")
    for name, expected in seal["files"].items():
        if sha(base / "train_selection_v1" / name) != expected:
            raise RuntimeError(f"TRAIN selection artifact hash mismatch: {name}")
    train_p, train_z, train_y, train_subjects, train_sessions = checked_extraction(base, "train", fold)
    outer_p, outer_z, outer_y, outer_subjects, outer_sessions = checked_extraction(base, "outer", fold)
    if sha(base / "train" / "EXTRACTION_PROVENANCE.json") != seal["source_sha256"]["extraction"]:
        raise RuntimeError("TRAIN selection source extraction SHA mismatch")
    for field in ("checkpoint_sha256", "split_sha256", "normalizer_sha256", "source_code_sha256"):
        if train_p[field] != outer_p[field]:
            raise RuntimeError(f"TRAIN/OUTER frozen source mismatch: {field}")
    if set(train_subjects.tolist()) & set(outer_subjects.tolist()):
        raise RuntimeError("TRAIN/OUTER biological subject overlap")
    r = np.asarray(seal["full_train_R_cos"], dtype=np.float64)
    u = np.asarray(seal["full_train_U_probe"], dtype=np.float64)
    energy = np.asarray(seal["full_train_energy_source_S1"], dtype=np.float64)
    if any(v.shape != (248,) or not np.isfinite(v).all() or v.std() <= 1e-12 for v in (r, u, energy)):
        raise RuntimeError("invalid frozen full TRAIN token score")
    selected = seal["selected"]
    required = {"MEAN_ALL", "FLATTEN_LINEAR", "ENERGY_TOPK", "RELIABLE_TOPK",
                "UTILITY_TOPK", "RELIABLE_UTILITY_TOPK", "RELIABILITY_WEIGHTED",
                "RU_WEIGHTED", "RU_TWO_STREAM", "ENERGY_TWO_STREAM"}
    if set(selected) != required:
        raise RuntimeError("TRAIN-selected method inventory mismatch")
    fit = train_sessions == 1
    if set(np.unique(train_y[fit]).tolist()) != {0, 1} or set(np.unique(outer_y).tolist()) != {0, 1}:
        raise RuntimeError("binary source/OUTER class inventory mismatch")
    output = base / "outer_evaluation_v2"
    output.mkdir(exist_ok=False)
    main_rows = []
    for method in sorted(required):
        choice = selected[method]
        xfit = method_features(train_z[fit], method, choice["parameter"], r, u, energy)
        xtest = method_features(outer_z, method, choice["parameter"], r, u, energy)
        fitted = decoder(xfit, train_y[fit], float(choice["C"]))
        probability = predict_decoder(fitted, xtest)
        for metric in subject_metrics(outer_y, probability, outer_subjects):
            main_rows.append({"fold": fold, "method": method, "selected_parameter": choice["parameter"],
                              "selected_C": choice["C"], **metric})
        print(f"fold={fold} OUTER method={method} complete", flush=True)
    main = pd.DataFrame(main_rows)
    main.to_csv(output / "OUTER_POOLING_RESULTS.csv", index=False)

    random_specs = (("RANDOM_TOPK", "RELIABLE_UTILITY_TOPK", "RANDOM_TOPK_NULL.csv",
                     "RANDOM_TOPK_SUBJECT.csv"),
                    ("RANDOM_TWO_STREAM", "RU_TWO_STREAM", "RANDOM_TWO_STREAM_NULL.csv",
                     "RANDOM_TWO_STREAM_SUBJECT.csv"))
    for random_method, matched_method, filename, subject_filename in random_specs:
        choice = selected[matched_method]
        q = float(choice["parameter"])
        count = len(rank_subset(r, q))
        c = float(choice["C"])
        rows, subject_rows = [], []
        for repeat in range(500):
            rng = np.random.default_rng(stable_seed("token-audit-random-control", fold,
                                                    random_method, repeat))
            subset = np.sort(rng.choice(248, size=count, replace=False))
            xfit = pool(train_z[fit], random_method, subset=subset)
            xtest = pool(outer_z, random_method, subset=subset)
            fitted = decoder(xfit, train_y[fit], c)
            metrics = subject_metrics(outer_y, predict_decoder(fitted, xtest), outer_subjects)
            pooled = subject_equal(metrics)
            for metric in metrics:
                subject_rows.append({"fold": fold, "repeat": repeat, "method": random_method,
                                     "matched_method": matched_method, **metric})
            rows.append({"fold": fold, "repeat": repeat, "matched_method": matched_method,
                         "selected_q_train": q, "selected_C_train": c, "subset_size": count,
                         "subset_sha256": hashlib.sha256(subset.astype(np.int16).tobytes()).hexdigest(),
                         **pooled})
            if repeat % 50 == 49:
                print(f"fold={fold} OUTER {random_method} repeats={repeat+1}/500", flush=True)
        pd.DataFrame(rows).to_csv(output / filename, index=False)
        pd.DataFrame(subject_rows).to_csv(output / subject_filename, index=False)
    report = {"schema": "TOKEN_POOLING_OUTER_EVALUATION_V2", "status": "COMPLETE_OUTER_EVALUATION_ONLY",
              "fold": fold, "train_selection_seal_sha256": sha(seal_path),
              "train_extraction_sha256": sha(base / "train" / "EXTRACTION_PROVENANCE.json"),
              "outer_extraction_sha256": sha(base / "outer" / "EXTRACTION_PROVENANCE.json"),
              "train_subject_count": len(set(train_subjects.tolist())),
              "outer_subject_count": len(set(outer_subjects.tolist())),
              "random_topk_draws": 500, "random_two_stream_draws": 500,
              "outer_use": "evaluation only; no rank, score, subset, C, q, tau, or method selection",
              "formal_final_heldout_eeg_reads": 0,
              "files": {name: sha(output / name) for name in (
                  "OUTER_POOLING_RESULTS.csv", "RANDOM_TOPK_NULL.csv", "RANDOM_TWO_STREAM_NULL.csv",
                  "RANDOM_TOPK_SUBJECT.csv", "RANDOM_TWO_STREAM_SUBJECT.csv")}}
    (output / "OUTER_EVALUATION_SEAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"fold": fold, "status": report["status"],
                      "seal_sha256": sha(output / "OUTER_EVALUATION_SEAL.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    run(parser.parse_args().fold)
