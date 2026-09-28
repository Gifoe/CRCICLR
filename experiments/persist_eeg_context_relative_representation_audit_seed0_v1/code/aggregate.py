"""TRAIN-locked frozen-context decoder comparison and compact audit tables.

Invoke `train --fold F` before extracting any OUTER data. Invoke `outer --fold F`
only when all five TRAIN locks exist. The primary stage is the exact EEGNet
head input. All subject/session contexts are computed by index, never label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("REL_RUNTIME", r"D:\nips-temp\TotalP\P1\context_relative_representation_runtime"))
LOCK = json.loads((HERE / "protocol/PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
STAGE = "EMBEDDING"
CS = tuple(LOCK["decoder"]["C"])
BS = tuple(LOCK["budgets"])
LAMBDAS = tuple(LOCK["lambdas"])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed(*values: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, values)).encode()).digest()[:8], "little") % (2**32 - 1)


def save_json_new(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def read_session(fold: int, role: str, session: int, stage: str = STAGE) -> dict:
    suffix = "_v2" if fold == 0 and role == "TRAIN_GEOMETRY" else ""
    base = ROOT / "features" / f"fold{fold}_{role.lower()}{suffix}"
    p = base / f"session{session}_{stage.lower()}.npz"
    meta_path = base / "PROVENANCE.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    key = f"session{session}_{stage}"
    if meta["stage_files"][key]["sha256"] != sha(p):
        raise RuntimeError(f"stage cache SHA mismatch {p}")
    if meta["model_state_before"] != meta["model_state_after"] or meta["final_heldout_eeg_reads"] != 0:
        raise RuntimeError("frozen model/final-heldout audit failed")
    with np.load(p, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    if not set(out) >= {"h", "y", "subject", "session"} or len(out["h"]) != len(out["y"]):
        raise RuntimeError("cache schema failure")
    if set(map(str, out["subject"])) != set(meta["role_subjects"]):
        raise RuntimeError("subject-role cache mismatch")
    return out


def subject_rows(a: dict, subject: str):
    ix = np.flatnonzero(a["subject"].astype(str) == str(subject))
    # Existing _openbmi_rows preserves acquisition/cache order for each ID.
    return np.asarray(a["h"][ix], np.float64), np.asarray(a["y"][ix], np.int64)


def subjects(a: dict) -> list[str]:
    return sorted(set(map(str, a["subject"])), key=int)


def split_context(a: dict, B: int, who: list[str] | None = None):
    who = subjects(a) if who is None else who
    out = {}
    for s in who:
        h, y = subject_rows(a, s)
        if len(h) <= B or set(y[B:]) != {0, 1}:
            raise RuntimeError(f"insufficient disjoint outcomes: subject={s} B={B}")
        out[s] = {"reference": h[:B].mean(0), "context_variance": h[:B].var(0),
                  "h": h[B:], "y": y[B:],
                  "context_class_fraction": float(y[:B].mean()),
                  "context_rows": B, "outcome_rows": len(h) - B}
    return out


def global_mean(source: dict) -> np.ndarray:
    return np.asarray(source["h"], np.float64).mean(0)


def transform(parts: dict, arm: str, mu: np.ndarray, lam: float = 1.0, other: dict | None = None):
    rows, labels, owners = [], [], []
    for s, part in parts.items():
        h, y = part["h"], part["y"]
        if arm == "ABSOLUTE":
            z = h
        elif arm == "GLOBAL_SOURCE_CENTER":
            z = h - mu
        elif arm in ("REL_MEAN", "REL_SHRINK"):
            ref = (1.0 - lam) * mu + lam * part["reference"]
            z = h - ref
        elif arm == "S1_REFERENCE_FOR_S2":
            assert other is not None
            z = h - other[s]["reference"]
        else:
            raise KeyError(arm)
        rows.append(z); labels.append(y); owners.extend([s] * len(y))
    return np.concatenate(rows), np.concatenate(labels), np.asarray(owners)


def transform_z(parts: dict, mu: np.ndarray, global_var: np.ndarray, lam: float):
    rows, labels, owners = [], [], []
    for s, part in parts.items():
        ref = (1.0 - lam) * mu + lam * part["reference"]
        var = (1.0 - lam) * global_var + lam * part["context_variance"]
        z = (part["h"] - ref) / (np.sqrt(np.maximum(var, 0.0)) + 1e-6)
        rows.append(z); labels.append(part["y"]); owners.extend([s] * len(z))
    return np.concatenate(rows), np.concatenate(labels), np.asarray(owners)


def fit(x: np.ndarray, y: np.ndarray, C: float):
    clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=2000,
                             random_state=0)
    clf.fit(x, y)
    if not np.array_equal(clf.classes_, np.array([0, 1])):
        raise RuntimeError("binary class mismatch")
    return clf


def scores(model, x: np.ndarray, y: np.ndarray, owner: np.ndarray) -> list[dict]:
    p = model.predict_proba(x)
    pred = (p[:, 1] >= 0.5).astype(np.int64)
    result = []
    for s in sorted(set(owner), key=int):
        take = owner == s
        result.append({"subject": str(s), "BA": float(balanced_accuracy_score(y[take], pred[take])),
                       "macro_f1": float(f1_score(y[take], pred[take], average="macro")),
                       "nll": float(log_loss(y[take], p[take], labels=[0, 1]))})
    return result


def subject_mean(rows: list[dict], metric: str = "BA") -> float:
    return float(np.mean([r[metric] for r in rows]))


def cv_curve(source: dict, future: dict, B: int, arm: str, C: float,
             lam: float = 1.0, pca_k: int | None = None) -> list[dict]:
    ids = np.asarray(subjects(source))
    rows = []
    for tr_ix, val_ix in GroupKFold(5).split(ids, groups=ids):
        train_ids, val_ids = ids[tr_ix].tolist(), ids[val_ix].tolist()
        train_mask = np.isin(source["subject"].astype(str), train_ids)
        train_all = np.asarray(source["h"][train_mask], np.float64)
        mu, gv = train_all.mean(0), train_all.var(0)
        train_parts = split_context(source, B, train_ids)
        val_parts = split_context(future, B, val_ids)
        if arm == "REL_Z":
            x, y, _ = transform_z(train_parts, mu, gv, lam)
            v, vy, owner = transform_z(val_parts, mu, gv, lam)
        else:
            x, y, _ = transform(train_parts, arm if pca_k is None else "ABSOLUTE", mu, lam)
            v, vy, owner = transform(val_parts, arm if pca_k is None else "ABSOLUTE", mu, lam)
        if pca_k is not None:
            refs = np.stack([train_parts[s]["reference"] for s in train_ids])
            basis = PCA(n_components=pca_k, svd_solver="full").fit(refs).components_
            x = x - ((x - mu) @ basis.T) @ basis
            v = v - ((v - mu) @ basis.T) @ basis
        clf = fit(x, y, C)
        rows.extend(scores(clf, v, vy, owner))
    if len(rows) != len(ids) or len(set(r["subject"] for r in rows)) != len(ids):
        raise RuntimeError("TRAIN pseudo-target CV missing subject")
    return rows


def train(fold: int):
    source = read_session(fold, "TRAIN_GEOMETRY", 1)
    future = read_session(fold, "TRAIN_GEOMETRY", 2)
    result_path = ROOT / "train_lock" / f"fold{fold}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    curves = []
    for B in BS:
        for arm in ("ABSOLUTE", "REL_MEAN"):
            for C in CS:
                rows = cv_curve(source, future, B, arm, C)
                curves.append({"fold": fold, "B": B, "arm": arm, "C": C,
                               "train_pseudotarget_BA": subject_mean(rows),
                               "train_pseudotarget_macro_f1": subject_mean(rows, "macro_f1"),
                               "train_pseudotarget_nll": subject_mean(rows, "nll")})
    best = lambda B, arm: max((r for r in curves if r["B"] == B and r["arm"] == arm),
                              key=lambda r: (r["train_pseudotarget_BA"], -r["C"]))
    B = max(BS, key=lambda b: (best(b, "REL_MEAN")["train_pseudotarget_BA"], -b))
    C_abs, C_rel = best(B, "ABSOLUTE")["C"], best(B, "REL_MEAN")["C"]
    shrink_curve = []
    for lam in LAMBDAS:
        rows = cv_curve(source, future, B, "REL_SHRINK", C_rel, lam)
        shrink_curve.append({"fold": fold, "B": B, "lambda": lam, "C": C_rel,
                             "train_pseudotarget_BA": subject_mean(rows)})
    lam = max(shrink_curve, key=lambda r: (r["train_pseudotarget_BA"], -r["lambda"]))["lambda"]
    z_curve = []
    for z_lam in LAMBDAS:
        rows = cv_curve(source, future, B, "REL_Z", C_rel, z_lam)
        z_curve.append({"fold": fold, "B": B, "lambda": z_lam, "C": C_rel,
                        "train_pseudotarget_BA": subject_mean(rows)})
    z_lam = max(z_curve, key=lambda r: (r["train_pseudotarget_BA"], -r["lambda"]))["lambda"]
    pca_curve = []
    for k in LOCK["pca_k"]:
        rows = cv_curve(source, future, B, "ABSOLUTE", C_abs, pca_k=k)
        pca_curve.append({"fold": fold, "B": B, "k": k, "C": C_abs,
                          "train_pseudotarget_BA": subject_mean(rows)})
    k = max(pca_curve, key=lambda r: (r["train_pseudotarget_BA"], -r["k"]))["k"]
    lock = {"fold": fold, "role": "TRAIN_GEOMETRY", "B": B, "C_absolute": C_abs,
            "C_relative": C_rel, "lambda_shrink": lam, "lambda_z": z_lam, "pca_k": k,
            "selection_rule": LOCK["train_selection"], "source_cache_sha256": sha(ROOT / "features" / (f"fold{fold}_train_geometry_v2" if fold == 0 else f"fold{fold}_train_geometry") / "PROVENANCE.json"),
            "curves": curves, "shrink_curve": shrink_curve, "z_curve": z_curve, "pca_curve": pca_curve,
            "final_heldout_eeg_reads": 0}
    save_json_new(result_path, lock)
    print(json.dumps({"fold": fold, "B": B, "C_abs": C_abs, "C_rel": C_rel,
                      "lambda": lam, "z_lambda": z_lam, "pca_k": k, "lock_sha256": sha(result_path)}))


def outer(fold: int):
    for f in range(5):
        if not (ROOT / "train_lock" / f"fold{f}.json").is_file():
            raise RuntimeError("all five TRAIN locks required before any OUTER evaluation")
    selection_path = ROOT / "train_lock" / f"fold{fold}.json"
    choice = json.loads(selection_path.read_text(encoding="utf-8"))
    outdir = ROOT / "outer" / f"fold{fold}"
    if outdir.exists():
        raise FileExistsError(outdir)
    source = read_session(fold, "TRAIN_GEOMETRY", 1)
    target = read_session(fold, "OUTER_DEVELOPMENT", 2)
    target_s1 = read_session(fold, "OUTER_DEVELOPMENT", 1)
    B, mu = choice["B"], global_mean(source)
    tr, ev, ev_s1 = split_context(source, B), split_context(target, B), split_context(target_s1, B)
    results, effects = [], []
    fitted = {}
    for arm, C, lam in (("ABSOLUTE", choice["C_absolute"], 1.0),
                        ("REL_MEAN", choice["C_relative"], 1.0),
                        ("REL_SHRINK", choice["C_relative"], choice["lambda_shrink"]),
                        ("GLOBAL_SOURCE_CENTER", choice["C_absolute"], 1.0)):
        x, y, _ = transform(tr, arm, mu, lam)
        v, vy, owner = transform(ev, arm, mu, lam)
        clf = fit(x, y, C)
        fitted[arm] = clf
        for row in scores(clf, v, vy, owner):
            effects.append({"fold": fold, "arm": arm, "B": B, "lambda": lam, **row})
        armrows = [r for r in effects if r["arm"] == arm]
        results.append({"fold": fold, "arm": arm, "B": B, "lambda": lam, "C": C,
                        "subject_equal_BA": subject_mean(armrows),
                        "subject_equal_macro_f1": subject_mean(armrows, "macro_f1"),
                        "subject_equal_nll": subject_mean(armrows, "nll")})
    gv = np.asarray(source["h"], np.float64).var(0)
    zx, zy, _ = transform_z(tr, mu, gv, choice["lambda_z"])
    zv, zvy, zowner = transform_z(ev, mu, gv, choice["lambda_z"])
    zrows = scores(fit(zx, zy, choice["C_relative"]), zv, zvy, zowner)
    effects.extend({"fold": fold, "arm": "REL_Z", "B": B, "lambda": choice["lambda_z"], **r} for r in zrows)
    results.append({"fold": fold, "arm": "REL_Z", "B": B, "lambda": choice["lambda_z"],
                    "subject_equal_BA": subject_mean(zrows),
                    "subject_equal_macro_f1": subject_mean(zrows, "macro_f1"),
                    "subject_equal_nll": subject_mean(zrows, "nll")})
    # Current-session versus historical S1 reference: identical frozen REL decoder.
    v, vy, owner = transform(ev, "S1_REFERENCE_FOR_S2", mu, other=ev_s1)
    hist = scores(fitted["REL_MEAN"], v, vy, owner)
    effects.extend({"fold": fold, "arm": "S1_REFERENCE_FOR_S2", "B": B, **r} for r in hist)
    results.append({"fold": fold, "arm": "S1_REFERENCE_FOR_S2", "B": B,
                    "subject_equal_BA": subject_mean(hist)})
    # Generic train-only feature standardization control.
    x, y, _ = transform(tr, "ABSOLUTE", mu)
    v, vy, owner = transform(ev, "ABSOLUTE", mu)
    scaler = StandardScaler().fit(x)
    fs = scores(fit(scaler.transform(x), y, choice["C_absolute"]), scaler.transform(v), vy, owner)
    effects.extend({"fold": fold, "arm": "FEATUREWISE_STANDARDIZATION", "B": B, **r} for r in fs)
    results.append({"fold": fold, "arm": "FEATUREWISE_STANDARDIZATION", "B": B,
                    "subject_equal_BA": subject_mean(fs)})
    # TRAIN-only PCA of source recording means, fixed k from TRAIN pseudo-targets.
    refs = np.stack([tr[s]["reference"] for s in subjects(source)])
    basis = PCA(n_components=choice["pca_k"], svd_solver="full").fit(refs).components_
    xx = x - ((x - mu) @ basis.T) @ basis
    vv = v - ((v - mu) @ basis.T) @ basis
    pc = scores(fit(xx, y, choice["C_absolute"]), vv, vy, owner)
    effects.extend({"fold": fold, "arm": "PCA_REMOVAL", "B": B, "k": choice["pca_k"], **r} for r in pc)
    results.append({"fold": fold, "arm": "PCA_REMOVAL", "B": B, "k": choice["pca_k"],
                    "subject_equal_BA": subject_mean(pc)})
    # Same decoder, false identity/context controls; every draw retained.
    ids = subjects(target)
    wrong, random = [], []
    wrong_subject = {s: [] for s in ids}
    random_subject = {s: [] for s in ids}
    ref_norm = {s: np.linalg.norm(ev[s]["reference"]) for s in ids}
    clf = fitted["REL_MEAN"]
    for draw in range(LOCK["wrong_subject_permutations"]):
        rng = np.random.default_rng(seed("WRONG", fold, draw))
        perm = rng.permutation(len(ids))
        while np.any(perm == np.arange(len(ids))):
            perm = rng.permutation(len(ids))
        fake = {s: {**ev[s], "reference": ev[ids[perm[i]]]["reference"]} for i, s in enumerate(ids)}
        vx, yy, oo = transform(fake, "REL_MEAN", mu)
        rows = scores(clf, vx, yy, oo)
        for row in rows:
            wrong_subject[row["subject"]].append(row["BA"])
        wrong.append({"fold": fold, "draw": draw, "BA": subject_mean(rows),
                      "mapping": ";".join(f"{s}:{ids[perm[i]]}" for i, s in enumerate(ids))})
    for draw in range(LOCK["random_reference_draws"]):
        rng = np.random.default_rng(seed("RANDOM_REFERENCE", fold, draw))
        fake = {}
        for s in ids:
            direction = rng.normal(size=ev[s]["reference"].shape)
            direction *= ref_norm[s] / np.linalg.norm(direction)
            fake[s] = {**ev[s], "reference": direction}
        vx, yy, oo = transform(fake, "REL_MEAN", mu)
        rows = scores(clf, vx, yy, oo)
        for row in rows:
            random_subject[row["subject"]].append(row["BA"])
        random.append({"fold": fold, "draw": draw, "BA": subject_mean(rows)})
    effects.extend({"fold": fold, "arm": "WRONG_SUBJECT_REFERENCE_MEAN", "B": B,
                    "subject": s, "BA": float(np.mean(wrong_subject[s]))} for s in ids)
    effects.extend({"fold": fold, "arm": "RANDOM_REFERENCE_MEAN", "B": B,
                    "subject": s, "BA": float(np.mean(random_subject[s]))} for s in ids)

    # Every B and lambda is reported; none can be selected from these OUTER rows.
    budget_rows, shrink_rows, z_rows, pca_rows = [], [], [], []
    for budget in BS:
        train_b, outer_b = split_context(source, budget), split_context(target, budget)
        for arm in ("ABSOLUTE", "REL_MEAN"):
            c = max((r for r in choice["curves"] if r["B"] == budget and r["arm"] == arm),
                    key=lambda r: (r["train_pseudotarget_BA"], -r["C"]))["C"]
            bx, by, _ = transform(train_b, arm, mu)
            bv, bvy, bo = transform(outer_b, arm, mu)
            sr = scores(fit(bx, by, c), bv, bvy, bo)
            budget_rows.append({"fold": fold, "B": budget, "arm": arm, "C_selected_train": c,
                                "BA": subject_mean(sr), "selected_primary_B": budget == B,
                                "scope": "OUTER_EVALUATION_ONLY"})
    for lam in LAMBDAS:
        sx, sy, _ = transform(tr, "REL_SHRINK", mu, lam)
        sv, svy, so = transform(ev, "REL_SHRINK", mu, lam)
        ss = scores(fit(sx, sy, choice["C_relative"]), sv, svy, so)
        shrink_rows.append({"fold": fold, "B": B, "lambda": lam, "BA": subject_mean(ss),
                            "selected_train_lambda": lam == choice["lambda_shrink"],
                            "scope": "OUTER_EVALUATION_ONLY"})
        sx, sy, _ = transform_z(tr, mu, gv, lam)
        sv, svy, so = transform_z(ev, mu, gv, lam)
        ss = scores(fit(sx, sy, choice["C_relative"]), sv, svy, so)
        z_rows.append({"fold": fold, "B": B, "lambda": lam, "BA": subject_mean(ss),
                       "selected_train_lambda": lam == choice["lambda_z"],
                       "scope": "OUTER_EVALUATION_ONLY"})
    for k in LOCK["pca_k"]:
        basis_k = PCA(n_components=k, svd_solver="full").fit(refs).components_
        xx_k = x - ((x - mu) @ basis_k.T) @ basis_k
        vv_k = v - ((v - mu) @ basis_k.T) @ basis_k
        ss = scores(fit(xx_k, y, choice["C_absolute"]), vv_k, vy, owner)
        pca_rows.append({"fold": fold, "B": B, "k": k, "BA": subject_mean(ss),
                         "selected_train_k": k == choice["pca_k"], "scope": "OUTER_EVALUATION_ONLY"})

    class_rows = []
    effect_map = {(r["arm"], r["subject"]): r["BA"] for r in effects if r["arm"] in ("ABSOLUTE", "REL_MEAN")}
    for s in ids:
        h, labels = subject_rows(target, s)
        # This diagnostic uses labels only after the primary reference is frozen.
        balanced_ix = np.r_[np.flatnonzero(labels == 0)[:B // 2], np.flatnonzero(labels == 1)[:B - B // 2]]
        keep = np.arange(len(h)) >= B
        keep[balanced_ix] = False
        if set(labels[keep]) != {0, 1}:
            raise RuntimeError("label-assisted matched diagnostic lost a class")
        bz = h[keep] - h[balanced_ix].mean(0)
        nz = h[keep] - ev[s]["reference"]
        bscore = scores(clf, bz, labels[keep], np.full(keep.sum(), s))[0]["BA"]
        nscore = scores(clf, nz, labels[keep], np.full(keep.sum(), s))[0]["BA"]
        class_rows.append({"fold": fold, "subject": s, "B": B,
                           "natural_context_class_fraction": ev[s]["context_class_fraction"],
                           "natural_reference_norm": float(np.linalg.norm(ev[s]["reference"])),
                           "natural_REL_minus_ABS_BA": effect_map[("REL_MEAN", s)] - effect_map[("ABSOLUTE", s)],
                           "natural_reference_matched_outcome_BA": nscore,
                           "label_assisted_balanced_context_BA": bscore,
                           "label_assisted_context_diagnostic_only": True,
                           "matched_diagnostic_outcome_rows": int(keep.sum())})

    # Transductive diagnostic: each trial sees every other trial's unlabeled
    # representation. These rows are excluded from the deployable main result.
    loo_rows = []
    for s in ids:
        h, labels = subject_rows(target, s)
        ref = (h.sum(0, keepdims=True) - h) / (len(h) - 1)
        ss = scores(clf, h - ref, labels, np.full(len(h), s))[0]
        loo_rows.append({"fold": fold, "subject": s, "B": B, "arm": "FULL_SESSION_LOO",
                         "BA": ss["BA"], "macro_f1": ss["macro_f1"], "nll": ss["nll"],
                         "transductive_non_deployable": True})

    # TRAIN-only trial-order sensitivity. Same frozen S1 decoder, TRAIN S2
    # outcomes; random windows are deterministic and never target-selected.
    train_future = read_session(fold, "TRAIN_GEOMETRY", 2)
    order_rows = []
    for s in subjects(source):
        h, labels = subject_rows(train_future, s)
        choices = [("FIRST_B", 0, np.arange(B)),
                   ("EVENLY_SPACED", 0, np.linspace(0, len(h) - 1, B, dtype=np.int64))]
        for draw in range(LOCK["trial_order_random_draws"]):
            rng = np.random.default_rng(seed("TRIAL_ORDER", fold, s, draw))
            choices.append(("RANDOM_B", draw, np.sort(rng.choice(len(h), size=B, replace=False))))
        for family, draw, ix in choices:
            keep = np.ones(len(h), dtype=bool); keep[ix] = False
            pred = scores(clf, h[keep] - h[ix].mean(0), labels[keep], np.full(keep.sum(), s))[0]
            order_rows.append({"fold": fold, "subject": s, "window": family, "draw": draw,
                               "B": B, "BA": pred["BA"], "scope": "TRAIN_DIAGNOSTIC_ONLY",
                               "transductive": family != "FIRST_B"})
    outdir.mkdir(parents=True, exist_ok=False)
    write_csv_new(outdir / "OUTER_TRANSFER_RESULTS.csv", results)
    write_csv_new(outdir / "SUBJECT_LEVEL_EFFECTS.csv", effects)
    write_csv_new(outdir / "WRONG_SUBJECT_REFERENCE_NULL.csv", wrong)
    write_csv_new(outdir / "RANDOM_REFERENCE_NULL.csv", random)
    write_csv_new(outdir / "CONTEXT_BUDGET_CURVE.csv", budget_rows)
    write_csv_new(outdir / "CONTEXT_SHRINKAGE_RESULTS.csv", shrink_rows)
    write_csv_new(outdir / "RELATIVE_Z_RESULTS.csv", z_rows)
    write_csv_new(outdir / "PCA_REMOVAL_CONTROL.csv", pca_rows)
    write_csv_new(outdir / "CONTEXT_CLASS_BALANCE_AUDIT.csv", class_rows)
    write_csv_new(outdir / "FULL_SESSION_LOO_DIAGNOSTIC.csv", loo_rows)
    write_csv_new(outdir / "TRIAL_ORDER_AUDIT.csv", order_rows)
    save_json_new(outdir / "AUDIT.json", {"fold": fold, "train_lock_sha256": sha(selection_path),
        "outer_cache_sha256": sha(ROOT / "features" / f"fold{fold}_outer_development" / "PROVENANCE.json"),
        "final_heldout_eeg_reads": 0, "outer_selection": False,
        "files": {p.name: sha(p) for p in outdir.glob("*.csv")}})
    print(json.dumps({"fold": fold, "result": results, "audit_sha256": sha(outdir / "AUDIT.json")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "outer"))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    a = parser.parse_args()
    (train if a.phase == "train" else outer)(a.fold)


if __name__ == "__main__":
    main()
