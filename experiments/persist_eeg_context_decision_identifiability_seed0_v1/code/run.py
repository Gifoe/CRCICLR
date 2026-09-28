"""Frozen EEGNet subject-context direction qualification. Never reads final heldout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
LOCK = json.loads((EXP / "protocol/PROTOCOL_LOCK.json").read_text())
RUNTIME = Path(os.environ.get("CID_RUNTIME", str(EXP / "runtime")))
UPSTREAM = Path(os.environ.get("CID_UPSTREAM_TRAIN", r"D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime\train"))
FAMILIES = tuple(LOCK["descriptor_families"])
RANKS = tuple(LOCK["correction_ranks"])
ALPHAS = tuple(LOCK["ridge_alphas"])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save_json_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def save_npz_new(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    with path.open("xb") as f: np.savez_compressed(f, **arrays)


def load_train(fold):
    name = f"eegnet_openbmi_mi_fold{fold}_seed0"
    npz, meta = UPSTREAM / f"{name}.npz", UPSTREAM / f"{name}.json"
    audit = json.loads(meta.read_text(encoding="utf-8"))
    if (audit["model"], audit["task"], audit["fold"], audit["seed"]) != ("EEGNet", "OpenBMI_MI", fold, 0):
        raise RuntimeError("upstream identity mismatch")
    if sha(npz) != audit["train_cache_sha256"]: raise RuntimeError("upstream TRAIN SHA mismatch")
    p = audit["proof"]
    if p["checkpoint_sha256"] != sha(Path(p["checkpoint_path"])): raise RuntimeError("checkpoint drift")
    if p["model_state_sha256_before"] != p["model_state_sha256_after"]: raise RuntimeError("neural state changed")
    if audit["final_heldout_eeg_reads"] or p["formal_final_heldout_eeg_reads"]: raise RuntimeError("heldout read")
    with np.load(npz, allow_pickle=False) as f: a = {k: f[k] for k in f.files}
    if a["h"].shape[1] != 64 or set(np.unique(a["session"])) != {1, 2}: raise RuntimeError("representation mismatch")
    ids = set(p["role_subjects"]["TRAIN_GEOMETRY"])
    if set(a["subject"]) != ids or len(ids) != 26: raise RuntimeError("TRAIN roles mismatch")
    return a, audit, {"cache_path": str(npz), "cache_sha256": sha(npz), "audit_path": str(meta),
                      "audit_sha256": sha(meta), "checkpoint_sha256": p["checkpoint_sha256"],
                      "split_sha256": p["split_sha256"], "normalizer_sha256": p["normalizer_sha256"],
                      "model_source_sha256": p["model_source_sha256"], "model_state_sha256": p["model_state_sha256_before"],
                      "head_weight_sha256": p["head_weight_sha256"], "role_subjects": p["role_subjects"]}


def ordered_subjects(a): return sorted(set(map(str, a["subject"])), key=int)


def get_rows(a, sub, session):
    ix = (a["subject"] == str(sub)) & (a["session"] == session)
    return np.asarray(a["h"][ix], np.float64), np.asarray(a["y"][ix], np.int64)


def fit_coord(a, subjects):
    mask = np.isin(a["subject"], subjects) & (a["session"] == 1)
    z = StandardScaler().fit(np.asarray(a["h"][mask], np.float64))
    if np.any(z.scale_ <= 0): raise RuntimeError("zero common scale")
    return z


def native_head(a, coord):
    w = np.asarray(a["head_weight"], np.float64)
    b = np.asarray(a["head_bias"], np.float64)
    dw = w[1] - w[0]
    theta = np.r_[dw * coord.scale_, dw @ coord.mean_ + b[1] - b[0]]
    probe = np.asarray(a["h"][:64], np.float64)
    if not np.allclose(margin(theta, coord.transform(probe)), probe @ dw + b[1] - b[0], rtol=1e-5, atol=1e-6):
        raise RuntimeError("native head common-coordinate identity mismatch")
    return theta


def logistic(z, y):
    clf = LogisticRegression(C=1.0, random_state=0, max_iter=2000, solver="lbfgs")
    clf.fit(z, y)
    if not np.array_equal(clf.classes_, np.array([0, 1])): raise RuntimeError("binary class mismatch")
    return np.r_[clf.coef_[0], clf.intercept_[0]].astype(np.float64)


def margin(theta, z): return z @ theta[:-1] + theta[-1]


def scores(theta, z, y):
    p = np.clip(expit(margin(theta, z)), 1e-9, 1 - 1e-9)
    return {"BA": float(balanced_accuracy_score(y, p >= .5)),
            "macro_F1": float(f1_score(y, p >= .5, average="macro", zero_division=0)),
            "NLL": float(log_loss(y, p, labels=[0, 1]))}


def fit_global(a, subjects, coord):
    m = np.isin(a["subject"], subjects) & (a["session"] == 1)
    return logistic(coord.transform(np.asarray(a["h"][m], np.float64)), a["y"][m])


def subject_oracles(a, subjects, coord):
    return {s: logistic(coord.transform(get_rows(a, s, 1)[0]), get_rows(a, s, 1)[1]) for s in subjects}


def choose_global(a, subjects):
    rows = []
    for held in subjects:
        src = [s for s in subjects if s != held]
        coord = fit_coord(a, src); z2 = coord.transform(get_rows(a, held, 2)[0]); y2 = get_rows(a, held, 2)[1]
        for name, theta in (("NATIVE_POPULATION_HEAD", native_head(a, coord)),
                            ("GLOBAL_REFIT_HEAD", fit_global(a, src, coord))):
            rows.append({"subject": held, "head_type": name, **scores(theta, z2, y2)})
    means = {name: (np.mean([r["BA"] for r in rows if r["head_type"] == name]),
                    np.mean([r["NLL"] for r in rows if r["head_type"] == name]))
             for name in ("NATIVE_POPULATION_HEAD", "GLOBAL_REFIT_HEAD")}
    choice = sorted(means, key=lambda n: (-means[n][0], means[n][1], n != "NATIVE_POPULATION_HEAD"))[0]
    return choice, rows, means


def pop_head(a, subjects, coord, choice):
    return native_head(a, coord) if choice == "NATIVE_POPULATION_HEAD" else fit_global(a, subjects, coord)


def covariance_reference(a, subjects, coord):
    m = np.isin(a["subject"], subjects) & (a["session"] == 1)
    z = coord.transform(np.asarray(a["h"][m], np.float64))
    pca = PCA(n_components=8, svd_solver="full").fit(z)
    return pca.components_.T


def descriptor(z, theta, axes, family, p_basis=None):
    mu = z.mean(axis=0); var = np.var(z, axis=0)
    mom = np.r_[mu, np.log(np.sqrt(var) + 1e-6), np.mean(np.linalg.norm(z, axis=1)),
                np.sum(var), np.linalg.norm(mu)]
    cov = np.cov(z, rowvar=False) + 0.1 * np.eye(z.shape[1])
    eig = np.linalg.eigvalsh(cov)[::-1]
    energies = np.einsum("ik,ij,jk->k", axes, cov, axes)
    covf = np.r_[energies, eig[:4], np.trace(cov), np.linalg.slogdet(cov)[1],
                 (eig.sum() ** 2) / np.sum(eig ** 2)]
    m = margin(theta, z); prob = expit(m)
    ent = -(prob * np.log(np.clip(prob, 1e-9, 1)) + (1 - prob) * np.log(np.clip(1 - prob, 1e-9, 1)))
    pred = np.r_[m.mean(), m.std(), ent.mean(), np.quantile(np.abs(prob - .5), [.1, .5, .9]), np.mean(prob >= .5)]
    if family == "CTX_MOMENTS": return mom
    if family == "CTX_COV": return covf
    if family == "CTX_PRED": return pred
    base = np.r_[mom, covf, pred]
    if family == "CTX_COMBINED": return base
    if family == "CTX_COMBINED_PLUS_PC":
        if p_basis is None: raise RuntimeError("missing protected basis")
        # Orthonormal raw-h P is converted to z coordinates through the caller.
        q = p_basis
        if not q.shape[1]: return np.r_[base, np.zeros(7)]
        zp = z @ q; zc = z - zp @ q.T
        ep = np.mean(np.sum(zp * zp, axis=1)); ec = np.mean(np.sum(zc * zc, axis=1))
        return np.r_[base, ep, ec, ep / (ec + 1e-9), np.linalg.norm(zp.mean(axis=0)),
                     np.linalg.norm(zc.mean(axis=0)), np.sum(np.var(zp, axis=0)), np.sum(np.var(zc, axis=0))]
    raise ValueError(family)


def pc_in_coord(a, coord, subjects, audit):
    # Historical P was fit on all TRAIN labels. For strict TRAIN LOSO, using it
    # would leak held-subject labels. The primary analysis never uses it.
    # Secondary P/C is legal only on OUTER where all TRAIN subjects are source.
    if set(subjects) != set(ordered_subjects(a)): return None
    q = np.asarray(a["p_final"], np.float64)
    if not q.shape[1]: return q
    # h = mu + std*z, so raw-h P energy directions map via std in z space.
    q, _ = np.linalg.qr(coord.scale_[:, None] * q)
    return q


def build_episode(a, source, target, choice, budget="ALL", pc=False):
    coord = fit_coord(a, source); pop = pop_head(a, source, coord, choice)
    axes = covariance_reference(a, source, coord)
    oracle = subject_oracles(a, source, coord)
    q = pc_in_coord(a, coord, source, None) if pc else None
    allsub = source + [target]
    desc = {}
    for s in allsub:
        h, _ = get_rows(a, s, 1)
        if budget != "ALL": h = h[:int(budget)]
        if len(h) < 8: raise RuntimeError("context budget too small")
        z = coord.transform(h)
        desc[s] = {fam: descriptor(z, pop, axes, fam, q) for fam in FAMILIES}
        if pc and q is not None: desc[s]["CTX_COMBINED_PLUS_PC"] = descriptor(z, pop, axes, "CTX_COMBINED_PLUS_PC", q)
    z2 = coord.transform(get_rows(a, target, 2)[0]); y2 = get_rows(a, target, 2)[1]
    return {"coord": coord, "pop": pop, "oracle": oracle, "desc": desc, "z2": z2, "y2": y2,
            "source": source, "target": target}


def correction_basis(ep, rank):
    d = np.stack([ep["oracle"][s] - ep["pop"] for s in ep["source"]])
    u, singular, vt = np.linalg.svd(d, full_matrices=False)
    return vt[:rank].T, singular, d


def descriptor_transform(mat, vec):
    sc = StandardScaler().fit(mat)
    mm = sc.transform(mat); vv = sc.transform(vec[None, :])
    n = min(8, len(mat) - 1, mm.shape[1])
    if n > 0:
        p = PCA(n_components=n, svd_solver="full").fit(mm)
        mm, vv = p.transform(mm), p.transform(vv)
    return mm, vv


def predict(ep, family, rank, alpha):
    basis, singular, d = correction_basis(ep, rank)
    mat = np.stack([ep["desc"][s][family] for s in ep["source"]]); vec = ep["desc"][ep["target"]][family]
    mm, vv = descriptor_transform(mat, vec)
    coef = d @ basis
    reg = Ridge(alpha=alpha, fit_intercept=True).fit(mm, coef)
    correction = np.atleast_1d(reg.predict(vv)[0]) @ basis.T
    return ep["pop"] + correction, correction, singular, float(np.linalg.norm(correction))


def predict_grid_episode(ep, families):
    _, singular, d = correction_basis(ep, 8)
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    bases = {r: vt[:r].T for r in RANKS}
    for family in families:
        mat = np.stack([ep["desc"][s][family] for s in ep["source"]])
        vec = ep["desc"][ep["target"]][family]
        mm, vv = descriptor_transform(mat, vec)
        for rank in RANKS:
            basis = bases[rank]
            coef = d @ basis
            for alpha in ALPHAS:
                reg = Ridge(alpha=alpha).fit(mm, coef)
                theta = ep["pop"] + np.atleast_1d(reg.predict(vv)[0]) @ basis.T
                yield (family, rank, alpha), scores(theta, ep["z2"], ep["y2"])


def select_config(a, subjects, choice, families=FAMILIES, nested=False):
    candidate = [(fam, rank, alpha) for fam in families for rank in RANKS for alpha in ALPHAS]
    acc = {c: [] for c in candidate}
    for held in subjects:
        source = [s for s in subjects if s != held]
        ep = build_episode(a, source, held, choice)
        # Descriptor transforms and correction basis are source-only.
        for c, sc in predict_grid_episode(ep, families): acc[c].append(sc)
    def key(c):
        v = acc[c]
        return (-np.mean([x["BA"] for x in v]), np.mean([x["NLL"] for x in v]),
                families.index(c[0]), c[1], -c[2])
    best = min(candidate, key=key)
    table = [{"descriptor": c[0], "rank": c[1], "ridge_alpha": c[2],
              "mean_BA": float(np.mean([x["BA"] for x in acc[c]])),
              "mean_NLL": float(np.mean([x["NLL"] for x in acc[c]]))} for c in candidate]
    return best, table


def train_fold(fold):
    a, audit, proof = load_train(fold)
    subjects = ordered_subjects(a)
    choice, global_rows, global_means = choose_global(a, subjects)
    print("GLOBAL_CHOICE", fold, choice, global_means, flush=True)
    # Fully nested: s is absent from all inner training and selection episodes.
    loso = []; selected = []
    for idx, held in enumerate(subjects):
        source = [s for s in subjects if s != held]
        config, _ = select_config(a, source, choice)
        ep = build_episode(a, source, held, choice)
        theta, correction, singular, _ = predict(ep, *config)
        mean_theta = ep["pop"] + np.mean([ep["oracle"][s] - ep["pop"] for s in source], axis=0)
        rows = [("BEST_GLOBAL", ep["pop"]), ("CONTEXT_PREDICTED", theta), ("MEAN_CORRECTION", mean_theta)]
        for method, rule in rows:
            loso.append({"fold": fold, "subject": held, "method": method, "descriptor": config[0],
                         "rank": config[1], "ridge_alpha": config[2], **scores(rule, ep["z2"], ep["y2"])})
        selected.append({"subject": held, "descriptor": config[0], "rank": config[1], "ridge_alpha": config[2]})
        print("NESTED_LOSO", fold, idx + 1, len(subjects), held, config, flush=True)
    final_config, grid = select_config(a, subjects, choice)
    coord = fit_coord(a, subjects)
    native = native_head(a, coord); global_refit = fit_global(a, subjects, coord)
    pop = pop_head(a, subjects, coord, choice)
    oracle = subject_oracles(a, subjects, coord)
    anchor = []
    for s in subjects:
        h, y = get_rows(a, s, 1)
        for cls in (0, 1): anchor.append(h[np.flatnonzero(y == cls)[:4]])
    anchor = coord.transform(np.concatenate(anchor))
    delta = np.stack([oracle[s] - pop for s in subjects])
    _, singular, vt = np.linalg.svd(delta, full_matrices=False)
    output = RUNTIME / f"fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    save_npz_new(output / "train_seal.npz", scaler_mean=coord.mean_, scaler_scale=coord.scale_,
                 native=native, global_refit=global_refit, pop=pop, oracle=np.stack([oracle[s] for s in subjects]),
                 subjects=np.array(subjects), anchors=anchor, correction_basis=vt,
                 singular=singular)
    save_json_new(output / "TRAIN_SEAL.json", {"schema": "CID_TRAIN_SEAL_V1", "fold": fold,
        "best_global_type": choice, "final_config": {"descriptor": final_config[0], "rank": final_config[1], "ridge_alpha": final_config[2]},
        "nested_loso_choices": selected, "grid": grid, "global_loso_rows": global_rows,
        "global_loso_means": global_means, "train_loso_rows": loso,
        "representation_provenance": proof, "scaler_sha256": digest(coord.mean_, coord.scale_),
        "native_margin_identity_verified": True, "checkpoint_unchanged": True,
        "final_heldout_eeg_reads": 0, "outer_accessed": False,
        "protocol_sha256": sha(EXP / "protocol/PROTOCOL_LOCK.json"),
        "analysis_lock_sha256": sha(EXP / "protocol/ANALYSIS_LOCK.json")})
    print("TRAIN_SEALED", fold, final_config, sha(output / "TRAIN_SEAL.json"), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["train"]); p.add_argument("--fold", type=int, required=True, choices=range(5))
    a = p.parse_args()
    if a.mode == "train": train_fold(a.fold)


if __name__ == "__main__": main()
