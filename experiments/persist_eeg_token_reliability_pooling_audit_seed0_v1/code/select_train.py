"""Nested source-only CBraMod token pooling selection for one frozen fold.

For every held TRAIN biological subject, both R and U are recomputed without
that subject. U_probe is itself a subject-LOSO transfer score among the other
TRAIN subjects. Classifier candidates are fitted to other-subject Session 1
and evaluated on the held subject's Session 2. OUTER is not loadable here.

This is computationally expensive by design. Partial output is preserved on
failure and the output directory is never reused.
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
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from aggregate import C_GRID, Q_GRID, TAU_GRID, fit_decoder, pool, predict_decoder, rank_subset, subject_metrics
from token_scores import cosine_reliability, directions


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or values.std() <= 1e-12:
        raise ValueError("invalid source token score")
    return (values - values.mean()) / values.std()


def probe_score_excluding(z: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                          sessions: np.ndarray, excluded: str) -> np.ndarray:
    """Exact nested LOSO U_probe among TRAIN subjects other than excluded."""
    remaining = sorted(set(subjects.tolist()) - {excluded})
    if len(remaining) < 4:
        raise ValueError("too few TRAIN subjects for nested utility")
    total = np.zeros(z.shape[1], dtype=np.float64)
    for i, inner_held in enumerate(remaining):
        fit = (subjects != excluded) & (subjects != inner_held) & (sessions == 1)
        test = (subjects == inner_held) & (sessions == 2)
        if set(np.unique(y[fit])) != {0, 1} or set(np.unique(y[test])) != {0, 1}:
            raise RuntimeError("nested TRAIN utility class inventory incomplete")
        train_y, test_y = y[fit], y[test]
        for token in range(z.shape[1]):
            scaler = StandardScaler()
            x = scaler.fit_transform(z[fit, token, :])
            model = LogisticRegression(C=1., penalty="l2", solver="liblinear",
                                       max_iter=1000, tol=1e-5, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x, train_y)
            prediction = model.predict(scaler.transform(z[test, token, :]))
            total[token] += balanced_accuracy_score(test_y, prediction)
        print(f"nested_U excluded={excluded} inner={i+1}/{len(remaining)}", flush=True)
    return total / len(remaining)


def checked_sources(runtime: Path, fold: int) -> tuple[Path, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    base = runtime / f"fold{fold}"
    extraction_path = base / "train" / "EXTRACTION_PROVENANCE.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if (extraction.get("fold"), extraction.get("role"), extraction.get("formal_final_heldout_eeg_reads")) != (
        fold, "train", 0
    ):
        raise RuntimeError("invalid TRAIN extraction role")
    for name, expected in extraction["files"].items():
        if sha(base / "train" / name) != expected:
            raise RuntimeError(f"TRAIN extraction file hash mismatch: {name}")
    rdir, udir = base / "reliability_v1", base / "utility_v1"
    rseal_path, useal_path = rdir / "TRAIN_RELIABILITY_SEAL.json", udir / "TRAIN_UTILITY_SEAL.json"
    rseal, useal = (json.loads(path.read_text(encoding="utf-8")) for path in (rseal_path, useal_path))
    if (rseal.get("status"), useal.get("status")) != (
        "COMPLETE_TRAIN_RELIABILITY_ONLY", "COMPLETE_TRAIN_UTILITY"
    ):
        raise RuntimeError("TRAIN score stages incomplete")
    for seal, directory in ((rseal, rdir), (useal, udir)):
        if seal.get("source_extraction_sha256") != sha(extraction_path) or seal.get("fold") != fold:
            raise RuntimeError("TRAIN score stage source identity mismatch")
        for name, expected in seal["files"].items():
            if sha(directory / name) != expected:
                raise RuntimeError(f"TRAIN score file hash mismatch: {name}")
    z = np.load(base / "train" / "tokens.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(base / "train" / "labels.npy", allow_pickle=False)
    subjects = np.load(base / "train" / "subjects.npy", allow_pickle=False)
    sessions = np.load(base / "train" / "sessions.npy", allow_pickle=False)
    if z.shape != (len(y), 248, 200) or len(y) != len(subjects) or len(y) != len(sessions):
        raise RuntimeError("TRAIN token/label/subject/session inventory mismatch")
    for start in range(0, len(z), 128):
        if not np.isfinite(z[start:start+128]).all():
            raise RuntimeError("nonfinite frozen TRAIN token")
    reliability = pd.read_csv(rdir / "TOKEN_RELIABILITY.csv").sort_values("token_index")
    utility = pd.read_csv(udir / "TOKEN_UTILITY.csv").sort_values("token_index")
    if set(reliability.token_index) != set(range(248)) or set(utility.token_index) != set(range(248)):
        raise RuntimeError("TRAIN token score index mismatch")
    return (base, {"extraction": sha(extraction_path), "reliability": sha(rseal_path),
                   "utility": sha(useal_path)}, z, y, subjects, sessions,
            reliability.R_cos.to_numpy(dtype=np.float64), utility.U_probe.to_numpy(dtype=np.float64))


def candidates(r: np.ndarray, u: np.ndarray, energy: np.ndarray) -> list[tuple[str, float | None, np.ndarray | None, np.ndarray | None]]:
    joint = zscore(r) + zscore(u)
    out = [("MEAN_ALL", None, None, None), ("FLATTEN_LINEAR", None, None, None)]
    for method, score in (("ENERGY_TOPK", energy), ("RELIABLE_TOPK", r),
                          ("UTILITY_TOPK", u), ("RELIABLE_UTILITY_TOPK", joint),
                          ("RU_TWO_STREAM", joint), ("ENERGY_TWO_STREAM", energy)):
        for q in Q_GRID:
            out.append((method, q, rank_subset(score, q), None))
    for method, score in (("RELIABILITY_WEIGHTED", r), ("RU_WEIGHTED", joint)):
        for tau in TAU_GRID:
            out.append((method, tau, None, score))
    return out


def decoder(x: np.ndarray, y: np.ndarray, c: float):
    if x.shape[1] <= 400:
        return fit_decoder(x, y, c)
    # FLATTEN_LINEAR is a deliberately strong, high-dimensional baseline.
    # liblinear's dual formulation avoids materializing a 49,600-square Hessian.
    scaler = StandardScaler()
    transformed = scaler.fit_transform(x)
    model = LogisticRegression(C=c, penalty="l2", solver="liblinear", dual=True,
                               max_iter=5000, tol=1e-4, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(transformed, y)
    return scaler, model


def evaluate_held(z: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray,
                  held: str, r: np.ndarray, u: np.ndarray) -> list[dict]:
    fit = (subjects != held) & (sessions == 1)
    test = (subjects == held) & (sessions == 2)
    if set(np.unique(y[fit])) != {0, 1} or set(np.unique(y[test])) != {0, 1}:
        raise RuntimeError("TRAIN LOSO biological-subject inventory mismatch")
    zfit, ztest = z[fit], z[test]
    energy = zfit.var(axis=(0, 2), dtype=np.float64)
    rows = []
    for method, parameter, subset, score in candidates(r, u, energy):
        if method in ("RELIABILITY_WEIGHTED", "RU_WEIGHTED"):
            xfit = pool(zfit, method, score=score, tau=parameter)
            xtest = pool(ztest, method, score=score, tau=parameter)
        else:
            xfit = pool(zfit, method, subset=subset)
            xtest = pool(ztest, method, subset=subset)
        for c in C_GRID:
            fitted = decoder(xfit, y[fit], c)
            probabilities = predict_decoder(fitted, xtest)
            metric = subject_metrics(y[test], probabilities, subjects[test])[0]
            rows.append({"held_train_subject": held, "method": method,
                         "parameter": "none" if parameter is None else str(parameter),
                         "C": c, "subset_size": 0 if subset is None else len(subset),
                         "BA": metric["BA"], "macro_F1": metric["macro_F1"], "NLL": metric["NLL"]})
    return rows


def select(table: pd.DataFrame, subjects: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (table.groupby(["method", "parameter", "C"], as_index=False)
                 .agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"),
                      NLL=("NLL", "mean"), held_subjects=("held_train_subject", "nunique")))
    if not (aggregate.held_subjects == subjects).all():
        raise RuntimeError("not every candidate was evaluated on every held TRAIN subject")
    # BA is the only selection metric. Fixed stable tie breaks use smaller C,
    # then smaller q/tau; no OUTER outcome can influence this order.
    aggregate["parameter_order"] = pd.to_numeric(aggregate.parameter, errors="coerce").fillna(-1.)
    aggregate = aggregate.sort_values(["method", "BA", "C", "parameter_order"],
                                      ascending=[True, False, True, True], kind="stable")
    chosen = aggregate.drop_duplicates("method", keep="first").drop(columns="parameter_order")
    return aggregate.drop(columns="parameter_order"), chosen


def run(fold: int) -> None:
    runtime = Path(os.environ["TOKEN_AUDIT_RUNTIME"]).resolve()
    base, source_hashes, z, y, subjects, sessions, full_r, full_u = checked_sources(runtime, fold)
    ordered = sorted(set(subjects.tolist()))
    d, _ = directions(z, y, subjects, sessions, ordered)
    if not np.allclose(cosine_reliability(d), full_r, atol=1e-7):
        raise RuntimeError("source reliability differs from audited R_cos")
    output = base / "train_selection_v1"
    output.mkdir(exist_ok=False)
    rows = []
    for index, held in enumerate(ordered):
        r = cosine_reliability(d[np.asarray([s != held for s in ordered])])
        u = probe_score_excluding(z, y, subjects, sessions, held)
        np.savez_compressed(output / f"held_score_{index:02d}.npz", subject=np.asarray(held),
                            R_excluding_held=r, U_probe_excluding_held=u)
        result = evaluate_held(z, y, subjects, sessions, held, r, u)
        pd.DataFrame(result).to_csv(output / f"held_candidates_{index:02d}.csv", index=False)
        rows.extend(result)
        print(f"fold={fold} held={held} complete={index+1}/{len(ordered)}", flush=True)
    table = pd.DataFrame(rows)
    summary, chosen = select(table, len(ordered))
    summary.insert(0, "fold", fold)
    chosen.insert(0, "fold", fold)
    summary.to_csv(output / "TRAIN_POOLING_SELECTION_ALL.csv", index=False)
    chosen.to_csv(output / "TRAIN_POOLING_SELECTION.csv", index=False)
    selected = table.merge(chosen[["method", "parameter", "C"]],
                           on=["method", "parameter", "C"], how="inner")
    selected.insert(0, "fold", fold)
    selected.to_csv(output / "TRAIN_LOSO_RESULTS.csv", index=False)
    selected_parameters = {row.method: {"parameter": row.parameter, "C": float(row.C)}
                           for row in chosen.itertuples(index=False)}
    seal = {"schema": "TOKEN_POOLING_TRAIN_SELECTION_SEAL_V1", "status": "FROZEN_TRAIN_ONLY",
            "fold": fold, "train_subject_count": len(ordered),
            "source_sha256": source_hashes,
            "score_rules": {"R": "mean within-subject S1/S2 class-direction cosine; held subject excluded",
                            "U": "nested LOSO single-token transfer probe C=1 among non-held TRAIN subjects",
                            "joint": "TRAIN-token zscore(R)+zscore(U)",
                            "energy": "other-subject TRAIN Session-1 token variance"},
            "selected": selected_parameters,
            "full_train_R_cos": full_r.tolist(), "full_train_U_probe": full_u.tolist(),
            "full_train_energy_source_S1": z[sessions == 1].var(axis=(0, 2), dtype=np.float64).tolist(),
            "outer_development_eeg_reads": 0, "formal_final_heldout_eeg_reads": 0,
            "files": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    seal_path = base / "TRAIN_SELECTION_SEAL.json"
    if seal_path.exists():
        raise FileExistsError("TRAIN selection seal already exists")
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"fold": fold, "status": seal["status"], "seal_sha256": sha(seal_path)},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    run(parser.parse_args().fold)
