"""TRAIN-only LOSO token utility probes and all-token erasure.

All fits use other TRAIN subjects' Session 1. Every evaluation uses exactly
one held TRAIN biological subject's Session 2. No OUTER array is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _logistic(x: np.ndarray, y: np.ndarray, *, dual: bool = False):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(C=1., penalty="l2", solver="liblinear", dual=dual,
                             max_iter=1000, tol=1e-5, random_state=0)
    clf.fit(x_scaled, y)
    return scaler, clf


def held_subject_utility(z: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                         sessions: np.ndarray, held: str) -> tuple[np.ndarray, np.ndarray, float]:
    fit = (subjects != held) & (sessions == 1)
    test = (subjects == held) & (sessions == 2)
    if not fit.any() or not test.any() or set(np.unique(y[fit])) != {0, 1} or set(np.unique(y[test])) != {0, 1}:
        raise ValueError("LOSO source/held population or binary class missing")
    token_count, dimension = z.shape[1:]
    probe = np.empty(token_count, dtype=np.float64)
    for k in range(token_count):
        scaler, clf = _logistic(z[fit, k, :], y[fit])
        prediction = clf.predict(scaler.transform(z[test, k, :]))
        probe[k] = balanced_accuracy_score(y[test], prediction)
    scaler, clf = _logistic(z[fit].reshape(int(fit.sum()), -1), y[fit], dual=True)
    future = scaler.transform(z[test].reshape(int(test.sum()), -1))
    logit = clf.decision_function(future)
    truth = y[test]
    full = float(balanced_accuracy_score(truth, logit >= 0))
    coefficient = clf.coef_[0].reshape(token_count, dimension)
    contribution = np.einsum("nkd,kd->nk", future.reshape(int(test.sum()), token_count, dimension),
                             coefficient, optimize=True)
    erased_positive = (logit[:, None] - contribution) >= 0
    p1 = erased_positive[truth == 1].mean(axis=0)
    p0 = (~erased_positive[truth == 0]).mean(axis=0)
    erased_ba = .5 * (p1 + p0)
    return probe, full - erased_ba, full


def run(fold: int):
    runtime = Path(os.environ["TOKEN_AUDIT_RUNTIME"]).resolve()
    source = runtime / f"fold{fold}" / "train"
    provenance_path = source / "EXTRACTION_PROVENANCE.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("TRAIN extraction not complete")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["fold"] != fold or provenance["role"] != "train" or provenance["formal_final_heldout_eeg_reads"] != 0:
        raise RuntimeError("TRAIN extraction provenance mismatch")
    for name, expected in provenance["files"].items():
        if sha(source / name) != expected:
            raise RuntimeError(f"TRAIN extraction hash mismatch: {name}")
    z = np.load(source / "tokens.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(source / "labels.npy", allow_pickle=False)
    subjects = np.load(source / "subjects.npy", allow_pickle=False)
    sessions = np.load(source / "sessions.npy", allow_pickle=False)
    if z.ndim != 3 or z.shape[1:] != (248, 200):
        raise RuntimeError("CBraMod token shape mismatch")
    output = runtime / f"fold{fold}" / "utility_v1"
    output.mkdir(parents=False, exist_ok=False)
    ordered = sorted(set(subjects.tolist()))
    probe_values, erase_values, full_values = [], [], []
    for i, held in enumerate(ordered):
        probe, erase, full = held_subject_utility(z, y, subjects, sessions, held)
        np.savez_compressed(output / f"held_subject_{i:02d}.npz", subject=np.asarray(held),
                            U_probe=probe, U_erase=erase, BA_full=full)
        probe_values.append(probe); erase_values.append(erase); full_values.append(full)
        print(f"fold={fold} held_subject={held} completed={i+1}/{len(ordered)}", flush=True)
    table = pd.DataFrame({"fold": fold, "token_index": np.arange(248),
                          "U_probe": np.mean(probe_values, axis=0),
                          "U_erase": np.mean(erase_values, axis=0),
                          "all_token_full_BA": float(np.mean(full_values)),
                          "train_subject_count": len(ordered)})
    table.to_csv(output / "TOKEN_UTILITY.csv", index=False)
    seal = {"status": "COMPLETE_TRAIN_UTILITY", "fold": fold,
            "source_extraction_sha256": sha(provenance_path),
            "probe": "LOSO single-token C=1 L2 liblinear, other-subject S1 fit, held-subject S2 BA",
            "erase": "LOSO all-token C=1 L2 dual-liblinear, other-subject S1 fit, TRAIN-neutral-centroid token erasure on held-subject S2",
            "train_subject_count": len(ordered), "formal_final_heldout_eeg_reads": 0,
            "files": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "TRAIN_UTILITY_SEAL.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fold": fold, "status": seal["status"],
                      "seal_sha256": sha(output / "TRAIN_UTILITY_SEAL.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    run(parser.parse_args().fold)
