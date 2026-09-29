"""Frozen EEGNet trial-context interaction audit. No neural model is trained.

Run `train F` for all five folds before `outer F`. Existing frozen embedding
caches are byte-hashed on every read. The OUTER guard is deliberately global.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
LOCK_PATH = BASE / "protocol" / "PROTOCOL_LOCK.json"
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
UPSTREAM = Path(os.environ.get("REL_RUNTIME", r"D:\nips-temp\TotalP\P1\context_relative_representation_runtime"))
RUNTIME = Path(os.environ.get("INTERACTION_RUNTIME", r"D:\nips-temp\TotalP\P1\trial_context_interaction_runtime"))
ARMS = tuple(LOCK["arms"])
NON = ARMS[:4]
INTER = ARMS[4:]
CS = tuple(LOCK["linear_C_grid"])
KS = tuple(LOCK["bilinear_ranks"])
BS = tuple(LOCK["context"]["budgets"])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed(*pieces: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, pieces)).encode()).digest()[:4], "little")


def save_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def verify_upstream() -> dict:
    p = UPSTREAM / "final_protocol" / "SOURCE_PROVENANCE.json"
    if digest(p) != LOCK["upstream"]["source_provenance_sha256"]:
        raise RuntimeError("upstream source provenance SHA mismatch")
    return json.loads(p.read_text(encoding="utf-8"))


def cache(fold: int, role: str, session: int) -> tuple[dict, dict]:
    if role not in ("TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"):
        raise ValueError(role)
    suffix = "_v2" if fold == 0 and role == "TRAIN_GEOMETRY" else ""
    base = UPSTREAM / "features" / f"fold{fold}_{role.lower()}{suffix}"
    pp = base / "PROVENANCE.json"
    meta = json.loads(pp.read_text(encoding="utf-8"))
    p = base / f"session{session}_embedding.npz"
    if digest(p) != meta["stage_files"][f"session{session}_EMBEDDING"]["sha256"]:
        raise RuntimeError(f"embedding SHA mismatch: {p}")
    if meta["model_state_before"] != meta["model_state_after"] or meta["final_heldout_eeg_reads"] != 0:
        raise RuntimeError("model state/final-heldout audit failure")
    for key in ("checkpoint_path", "checkpoint_sha256", "split_sha256", "normalizer_sha256", "model_source_sha256"):
        if key not in meta:
            raise RuntimeError(f"missing source provenance {key}")
    if digest(Path(meta["checkpoint_path"])) != meta["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint SHA mismatch")
    upstream = verify_upstream()["folds"][str(fold)][role]
    for key in ("checkpoint_sha256", "split_sha256", "normalizer_sha256", "model_source_sha256", "model_state_before", "model_state_after"):
        if meta[key] != upstream[key]:
            raise RuntimeError(f"source provenance mismatch {key}")
    with np.load(p, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    if not set(a) >= {"h", "y", "subject", "session"} or a["h"].shape[1] != 64:
        raise RuntimeError("unexpected embedding schema/dimension")
    if set(map(str, a["subject"])) != set(meta["role_subjects"]):
        raise RuntimeError("role subject mismatch")
    if len(a["h"]) != len(a["y"]) or set(np.unique(a["y"])) != {0, 1}:
        raise RuntimeError("outcome schema failure")
    return a, meta


def ids(a: dict) -> list[str]:
    return sorted(set(map(str, a["subject"])), key=int)


def parts(a: dict, B: int, selected: list[str] | None = None) -> dict:
    chosen = ids(a) if selected is None else selected
    out = {}
    for subject in chosen:
        ix = np.flatnonzero(a["subject"].astype(str) == str(subject))
        h, y = np.asarray(a["h"][ix], np.float64), np.asarray(a["y"][ix], np.int64)
        if len(h) <= B or set(y[B:]) != {0, 1}:
            raise RuntimeError(f"invalid disjoint outcome {subject} B={B}")
        out[str(subject)] = {"r": h[:B].mean(0), "r1": h[:B], "context_y": y[:B],
                             "h": h[B:], "y": y[B:], "indices": ix[B:]}
    return out


def scaler(a: dict, selected: list[str]) -> StandardScaler:
    mask = np.isin(a["subject"].astype(str), selected)
    return StandardScaler().fit(np.asarray(a["h"][mask], np.float64))


def flatten(p: dict, scale: StandardScaler, references: dict[str, np.ndarray] | None = None):
    hh, rr, yy, oo = [], [], [], []
    for subject in sorted(p, key=int):
        item = p[subject]
        h = scale.transform(item["h"])
        raw_ref = item["r"] if references is None else references[subject]
        r = scale.transform(np.asarray(raw_ref)[None, :])[0]
        hh.append(h); rr.append(np.broadcast_to(r, h.shape).copy())
        yy.append(item["y"]); oo.extend([subject] * len(h))
    return np.concatenate(hh), np.concatenate(rr), np.concatenate(yy), np.asarray(oo)


def features(h: np.ndarray, r: np.ndarray, arm: str, basis=None):
    if arm == "TRIAL_ONLY": return h
    if arm == "RELATIVE_SUBTRACTION": return h - r
    if arm == "ADDITIVE_CONTEXT": return np.concatenate((h, r), axis=1)
    if arm == "ADDITIVE_RELATIVE": return np.concatenate((h, r, h-r), axis=1)
    if arm == "ELEMENTWISE_INTERACTION": return np.concatenate((h, r, h*r), axis=1)
    if arm == "FULL_SIMPLE_INTERACTION": return np.concatenate((h, r, h-r, h*r), axis=1)
    if arm == "LOWRANK_BILINEAR":
        if basis is None: raise RuntimeError("missing bilinear TRAIN basis")
        u, v = basis
        return np.concatenate((h, r, (h @ u) * (r @ v)), axis=1)
    if arm == "CONTEXT_ONLY": return r
    raise KeyError(arm)


def moment_basis(h: np.ndarray, r: np.ndarray, y: np.ndarray, K: int):
    # Factorization uses TRAIN labels only; model coefficient for each factor is fitted below.
    m1 = np.einsum("ni,nj->ij", h[y == 1], r[y == 1]) / np.sum(y == 1)
    m0 = np.einsum("ni,nj->ij", h[y == 0], r[y == 0]) / np.sum(y == 0)
    u, _, vt = np.linalg.svd(m1-m0, full_matrices=False)
    # Canonicalize the arbitrary simultaneous sign to stable coordinates.
    for j in range(K):
        z = np.argmax(np.abs(u[:, j]))
        if u[z, j] < 0: u[:, j] *= -1; vt[j, :] *= -1
    return u[:, :K], vt[:K, :].T


def fit_model(source: dict, people: list[str], B: int, arm: str, C: float, K: int | None = None,
              references: dict[str, np.ndarray] | None = None):
    sp = parts(source, B, people)
    sc = scaler(source, people)
    h, r, y, _ = flatten(sp, sc, references)
    basis = moment_basis(h, r, y, K) if arm == "LOWRANK_BILINEAR" else None
    x = features(h, r, arm, basis)
    clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=0)
    clf.fit(x, y)
    if not np.array_equal(clf.classes_, [0, 1]): raise RuntimeError("class mismatch")
    return sc, basis, clf


def score_model(model, data: dict, people: list[str], B: int, arm: str,
                references: dict[str, np.ndarray] | None = None):
    sc, basis, clf = model
    p = parts(data, B, people)
    h, r, y, owners = flatten(p, sc, references)
    x = features(h, r, arm, basis)
    prob = clf.predict_proba(x)[:, 1]
    pred = (prob >= .5).astype(int)
    rows = []
    for subject in people:
        m = owners == str(subject)
        rows.append({"subject": str(subject), "BA": float(balanced_accuracy_score(y[m], pred[m])),
                     "macro_f1": float(f1_score(y[m], pred[m], average="macro")),
                     "NLL": float(log_loss(y[m], prob[m], labels=[0, 1])),
                     "n": int(np.sum(m))})
    return rows


def grouped(source: dict, future: dict, B: int, arm: str, C: float, K: int | None = None):
    people = np.asarray(ids(source))
    if set(people) != set(ids(future)): raise RuntimeError("TRAIN S1/S2 subject mismatch")
    rows = []
    for fit_ix, val_ix in GroupKFold(5).split(people, groups=people):
        fitted = fit_model(source, people[fit_ix].tolist(), B, arm, C, K)
        rows += score_model(fitted, future, people[val_ix].tolist(), B, arm)
    if len(rows) != len(people) or len(set(x["subject"] for x in rows)) != len(people):
        raise RuntimeError("grouped CV inventory mismatch")
    return rows


def mean(rows: list[dict], key="BA") -> float:
    return float(np.mean([r[key] for r in rows]))


def train(fold: int):
    verify_upstream()
    target = RUNTIME / "train_lock" / f"fold{fold}.json"
    if target.exists(): raise FileExistsError(target)
    source, sm = cache(fold, "TRAIN_GEOMETRY", 1)
    future, fm = cache(fold, "TRAIN_GEOMETRY", 2)
    budget = []
    for B in BS:
        anchor = []
        for arm in ("TRIAL_ONLY", "RELATIVE_SUBTRACTION", "ADDITIVE_CONTEXT", "ELEMENTWISE_INTERACTION"):
            row = grouped(source, future, B, arm, .1)
            anchor.append(mean(row))
        budget.append({"B": B, "anchor_BA": anchor, "mean_BA": float(np.mean(anchor))})
    B = sorted(budget, key=lambda x: (-x["mean_BA"], x["B"]))[0]["B"]
    curves = []
    for arm in ARMS:
        grid = ((K, C) for K in KS for C in CS) if arm == "LOWRANK_BILINEAR" else ((None, C) for C in CS)
        for K, C in grid:
            rows = grouped(source, future, B, arm, C, K)
            curves.append({"arm": arm, "K": K, "C": C, "B": B, "BA": mean(rows),
                           "macro_f1": mean(rows, "macro_f1"), "NLL": mean(rows, "NLL"),
                           "subject_rows": rows})
    def choice(arm):
        return sorted((r for r in curves if r["arm"] == arm),
                      key=lambda r: (-r["BA"], r["K"] or 0, r["C"]))[0]
    selected = {arm: {k: v for k, v in choice(arm).items() if k != "subject_rows"} for arm in ARMS}
    best_non = sorted(NON, key=lambda a: (-selected[a]["BA"], NON.index(a)))[0]
    best_inter = sorted(INTER, key=lambda a: (-selected[a]["BA"], INTER.index(a)))[0]
    save_new(target, {"fold": fold, "B": B, "budget_curve": budget, "cv_curves": curves,
                      "selected": selected, "best_noninteraction": best_non,
                      "best_interaction": best_inter,
                      "source_provenance_sha256": digest(UPSTREAM / "features" / (f"fold{fold}_train_geometry_v2" if fold == 0 else f"fold{fold}_train_geometry") / "PROVENANCE.json"),
                      "protocol_lock_sha256": digest(LOCK_PATH), "role": "TRAIN_GEOMETRY",
                      "final_heldout_eeg_reads": 0})
    print(json.dumps({"fold": fold, "B": B, "best_non": best_non, "best_inter": best_inter,
                      "lock_sha256": digest(target)}), flush=True)


def choice(fold: int) -> dict:
    p = RUNTIME / "train_lock" / f"fold{fold}.json"
    a = json.loads(p.read_text(encoding="utf-8"))
    if a["protocol_lock_sha256"] != digest(LOCK_PATH): raise RuntimeError("TRAIN protocol changed")
    return a


def derangement(people: list[str], *label: object) -> dict[str, str]:
    rng = np.random.default_rng(seed(*label))
    base = np.asarray(people)
    for _ in range(10000):
        shuffled = rng.permutation(base)
        if np.all(shuffled != base): return dict(zip(people, shuffled.tolist()))
    raise RuntimeError("cannot construct deterministic subject derangement")


def pairing(fold: int):
    """TRAIN-only pairing null; never loads OUTER or checkpoint validation."""
    verify_upstream()
    selected = choice(fold)
    output = RUNTIME / "train_pairing" / f"fold{fold}.json"
    if output.exists(): raise FileExistsError(output)
    source, _ = cache(fold, "TRAIN_GEOMETRY", 1)
    future, _ = cache(fold, "TRAIN_GEOMETRY", 2)
    people = np.asarray(ids(source))
    fit_ix, val_ix = next(GroupKFold(5).split(people, groups=people))
    fit_people, val_people = people[fit_ix].tolist(), people[val_ix].tolist()
    B, arm = selected["B"], selected["best_interaction"]
    C, K = selected["selected"][arm]["C"], selected["selected"][arm]["K"]
    context = parts(source, B, fit_people)
    true_model = fit_model(source, fit_people, B, arm, C, K)
    true_score = mean(score_model(true_model, future, val_people, B, arm))
    null = []
    for draw in range(LOCK["train_pairing_permutations"]):
        m = derangement(fit_people, "pairing", fold, draw)
        references = {s: context[m[s]]["r"] for s in fit_people}
        model = fit_model(source, fit_people, B, arm, C, K, references)
        null.append({"draw": draw, "BA": mean(score_model(model, future, val_people, B, arm))})
    save_new(output, {"fold": fold, "role": "TRAIN_GEOMETRY", "arm": arm, "B": B,
                      "fit_subjects": fit_people, "held_subjects": val_people,
                      "true_pairing_BA": true_score, "null": null,
                      "train_lock_sha256": digest(RUNTIME / "train_lock" / f"fold{fold}.json"),
                      "final_heldout_eeg_reads": 0})
    print(json.dumps({"fold": fold, "pairing_null": len(null), "sha256": digest(output)}), flush=True)


def subject_predictions(model, p: dict, B: int, arm: str, refs=None, ablate=False):
    sc, basis, clf = model
    result = {}
    for subject, item in p.items():
        h = sc.transform(item["h"])
        r_raw = item["r"] if refs is None else refs[subject]
        r = np.broadcast_to(sc.transform(np.asarray(r_raw)[None, :])[0], h.shape)
        x = features(h, r, arm, basis)
        if ablate:
            if arm == "ELEMENTWISE_INTERACTION": x[:, 128:192] = 0
            elif arm == "FULL_SIMPLE_INTERACTION": x[:, 192:256] = 0
            elif arm == "LOWRANK_BILINEAR": x[:, 128:] = 0
            else: raise ValueError("cannot ablate noninteraction arm")
        prob = clf.predict_proba(x)[:, 1]
        pred = (prob >= .5).astype(int)
        y = item["y"]
        result[subject] = {"subject": subject, "BA": float(balanced_accuracy_score(y, pred)),
                           "macro_f1": float(f1_score(y, pred, average="macro")),
                           "NLL": float(log_loss(y, prob, labels=[0, 1])),
                           "n": int(len(y)), "prediction": pred.tolist(), "probability": prob.tolist()}
    return result


def random_linear_fit(source: dict, people: list[str], B: int, C: float, out_dim: int,
                      family: str, draw: int, h_only=False):
    sc = scaler(source, people)
    sp = parts(source, B, people)
    h, r, y, _ = flatten(sp, sc)
    base = h if h_only else np.concatenate((h, r), axis=1)
    rng = np.random.default_rng(seed("random_linear", family, draw))
    matrix = rng.standard_normal((base.shape[1], out_dim)) / np.sqrt(base.shape[1])
    z = base @ matrix
    clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=0).fit(z, y)
    return sc, matrix, clf


def random_linear_score(model, p: dict, h_only=False):
    sc, matrix, clf = model
    rows = []
    for subject, item in p.items():
        h = sc.transform(item["h"])
        r = np.broadcast_to(sc.transform(item["r"][None, :])[0], h.shape)
        base = h if h_only else np.concatenate((h, r), axis=1)
        prob = clf.predict_proba(base @ matrix)[:, 1]
        rows.append({"subject": subject, "BA": float(balanced_accuracy_score(item["y"], prob >= .5))})
    return rows


def outer(fold: int):
    for f in range(5):
        p = RUNTIME / "train_lock" / f"fold{f}.json"
        if not p.is_file(): raise RuntimeError("all five TRAIN locks required before OUTER")
        if choice(f)["final_heldout_eeg_reads"] != 0: raise RuntimeError("invalid TRAIN lock")
        if not (RUNTIME / "train_pairing" / f"fold{f}.json").is_file():
            raise RuntimeError("all five TRAIN pairing nulls required before OUTER")
    verify_upstream()
    selected = choice(fold)
    output = RUNTIME / "outer" / f"fold{fold}.json"
    if output.exists(): raise FileExistsError(output)
    source, sm = cache(fold, "TRAIN_GEOMETRY", 1)
    target, tm = cache(fold, "OUTER_DEVELOPMENT", 2)
    historical, hm = cache(fold, "OUTER_DEVELOPMENT", 1)
    if ids(target) != ids(historical): raise RuntimeError("OUTER S1/S2 subject mismatch")
    B = selected["B"]
    p, hp = parts(target, B), parts(historical, B)
    subjects = ids(target)
    # The exact same S2 outcome indices/labels are used for every arm and control.
    inventory = {s: {"indices_sha256": hashlib.sha256(p[s]["indices"].tobytes()).hexdigest(),
                     "labels_sha256": hashlib.sha256(p[s]["y"].tobytes()).hexdigest(),
                     "n": len(p[s]["y"])} for s in subjects}
    models, results = {}, {}
    for arm in ARMS:
        spec = selected["selected"][arm]
        models[arm] = fit_model(source, ids(source), B, arm, spec["C"], spec["K"])
        results[arm] = subject_predictions(models[arm], p, B, arm)
    inter = selected["best_interaction"]
    full = results[inter]
    ablated = subject_predictions(models[inter], p, B, inter, ablate=True)
    historical_refs = {s: hp[s]["r"] for s in subjects}
    historical_result = subject_predictions(models[inter], p, B, inter, historical_refs)
    wrong = []
    for draw in range(LOCK["wrong_context_permutations"]):
        m = derangement(subjects, "wrong_outer", fold, draw)
        refs = {s: p[m[s]]["r"] for s in subjects}
        rows = subject_predictions(models[inter], p, B, inter, refs)
        wrong.append({"draw": draw, "subject_BA": {s: rows[s]["BA"] for s in subjects},
                      "BA": float(np.mean([rows[s]["BA"] for s in subjects]))})
    controls = {}
    for family in INTER:
        spec = selected["selected"][family]
        dim = 128 + (spec["K"] if family == "LOWRANK_BILINEAR" else (64 if family == "ELEMENTWISE_INTERACTION" else 128))
        values = []
        for draw in range(100):
            random_model = random_linear_fit(source, ids(source), B, spec["C"], dim, family, draw)
            row = random_linear_score(random_model, p)
            values.append({"draw": draw, "BA": mean(row), "subject_BA": {r["subject"]: r["BA"] for r in row}})
        controls[family] = {"dimension": dim, "random_linear": values}
    selected_dim = controls[inter]["dimension"]
    h_model = random_linear_fit(source, ids(source), B, selected["selected"][inter]["C"], selected_dim, inter, 0, True)
    h_rows = random_linear_score(h_model, p, True)
    context_model = fit_model(source, ids(source), B, "CONTEXT_ONLY", selected["selected"]["ADDITIVE_CONTEXT"]["C"])
    context_result = subject_predictions(context_model, p, B, "CONTEXT_ONLY")
    sensitivity = []
    for s in subjects:
        now, old, abl = full[s], historical_result[s], ablated[s]
        sensitivity.append({"subject": s, "historical_flip_fraction": float(np.mean(np.asarray(now["prediction"]) != np.asarray(old["prediction"]))),
                            "ablation_flip_fraction": float(np.mean(np.asarray(now["prediction"]) != np.asarray(abl["prediction"]))),
                            "historical_probability_shift": float(np.mean(np.abs(np.asarray(now["probability"])-np.asarray(old["probability"])))),
                            "current_minus_historical_BA": now["BA"]-old["BA"],
                            "current_minus_historical_NLL": now["NLL"]-old["NLL"]})
    balance = []
    balanced_diagnostic = {}
    for s in subjects:
        first = p[s]["context_y"]
        n0, n1 = int(np.sum(first == 0)), int(np.sum(first == 1))
        balance.append({"subject": s, "first_B_class0": n0, "first_B_class1": n1,
                        "first_B_class_fraction1": float(n1/len(first)),
                        "balanced_diagnostic_available": bool(min(n0,n1) > 0)})
        if min(n0,n1) > 0:
            q = min(n0, n1)
            raw = p[s]["r1"]
            balanced_ref = np.concatenate((raw[np.flatnonzero(first == 0)[:q]],
                                           raw[np.flatnonzero(first == 1)[:q]])).mean(0)
            one = subject_predictions(models[inter], {s: p[s]}, B, inter, {s: balanced_ref})[s]
            balanced_diagnostic[s] = {"BA": one["BA"], "delta_from_unlabeled_first_B": one["BA"] - full[s]["BA"],
                                      "label_assisted_diagnostic_only": True}
        else:
            balanced_diagnostic[s] = {"BA": None, "delta_from_unlabeled_first_B": None,
                                      "label_assisted_diagnostic_only": True}
    compat = []
    if models["LOWRANK_BILINEAR"][1] is not None:
        sc, basis, _ = models["LOWRANK_BILINEAR"]
        for s in subjects:
            h = sc.transform(p[s]["h"])
            r = sc.transform(p[s]["r"][None, :])[0]
            q = (h @ basis[0]) * (r @ basis[1])
            for cls in (0, 1):
                mask = p[s]["y"] == cls
                compat.append({"subject": s, "class": cls, "mean_q": np.mean(q[mask], axis=0).tolist(),
                               "sd_q": np.std(q[mask], axis=0).tolist(), "n": int(np.sum(mask))})
    save_new(output, {"fold": fold, "role": "OUTER_DEVELOPMENT", "historically_exposed": True,
                      "train_lock_sha256": digest(RUNTIME / "train_lock" / f"fold{fold}.json"),
                      "pairing_sha256": digest(RUNTIME / "train_pairing" / f"fold{fold}.json"),
                      "B": B, "subjects": subjects, "outcome_inventory": inventory,
                      "best_noninteraction": selected["best_noninteraction"], "best_interaction": inter,
                      "results": results, "interaction_ablated": ablated, "historical": historical_result,
                      "wrong_context": wrong, "matched_controls": controls,
                      "h_only_expanded": h_rows, "context_only": context_result,
                      "sensitivity": sensitivity, "context_balance": balance,
                      "balanced_context_diagnostic": balanced_diagnostic,
                      "lowrank_compatibility": compat,
                      "source_checkpoint_sha256": sm["checkpoint_sha256"],
                      "target_checkpoint_sha256": tm["checkpoint_sha256"],
                      "final_heldout_eeg_reads": 0, "outer_selection": False})
    print(json.dumps({"fold": fold, "selected_interaction": inter,
                      "outer_sha256": digest(output)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("train", "pairing", "outer"))
    ap.add_argument("fold", type=int, choices=range(5))
    a = ap.parse_args()
    {"train": train, "pairing": pairing, "outer": outer}[a.phase](a.fold)


if __name__ == "__main__": main()
