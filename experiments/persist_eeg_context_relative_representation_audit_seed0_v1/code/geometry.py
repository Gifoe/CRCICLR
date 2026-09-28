"""Frozen stage geometry; class-relation translation invariance is asserted."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import aggregate as core


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def stage_subject(a1, a2, s, B):
    h1, y1 = core.subject_rows(a1, s)
    h2, y2 = core.subject_rows(a2, s)
    if min(len(h1), len(h2)) <= B:
        raise RuntimeError("insufficient disjoint outcome trials")
    r1, r2 = h1[:B].mean(0), h2[:B].mean(0)
    a, b = h1[B:], h2[B:]
    p, q = y1[B:], y2[B:]
    if set(p) != {0, 1} or set(q) != {0, 1}:
        raise RuntimeError("empty class after context")
    return a, b, p, q, r1, r2


def variance(parts: list[tuple], relative: bool):
    # Balanced subject x session x class moment decomposition. Units are squared
    # Euclidean norm per representation, not percentages of arbitrary coordinates.
    means = {}
    for s, j, h, y, ref in parts:
        z = h - ref if relative else h
        means[(s, j)] = z.mean(0)
    all_means = np.stack(list(means.values()))
    grand = all_means.mean(0)
    ids = sorted(set(s for s, *_ in parts), key=int)
    subject_mean = {s: np.stack([means[(s, j)] for j in (1, 2)]).mean(0) for s in ids}
    subject = float(np.mean([np.sum((subject_mean[s] - grand) ** 2) for s in ids]))
    session = float(np.mean([np.sum((means[(s, j)] - subject_mean[s]) ** 2) for s in ids for j in (1, 2)]))
    # Class effects are measured after a recording mean is removed; otherwise
    # different class proportions contaminate the class variance estimate.
    class_effect, residual, total = [], [], []
    for s, j, h, y, ref in parts:
        z = h - ref if relative else h
        m = z.mean(0)
        c0, c1 = z[y == 0].mean(0), z[y == 1].mean(0)
        class_effect.append(0.5 * (np.sum((c0 - m) ** 2) + np.sum((c1 - m) ** 2)))
        residual.append(float(np.mean(np.sum((z - np.where(y[:, None] == 0, c0, c1)) ** 2, axis=1))))
        total.append(float(np.mean(np.sum((z - grand) ** 2, axis=1))))
    return {"subject_variance": subject, "session_within_subject_variance": session,
            "class_variance": float(np.mean(class_effect)), "residual_trial_variance": float(np.mean(residual)),
            "total_variance": float(np.mean(total))}


def probes(a1, a2, B, relative):
    x, cls, ses, sub, trial = [], [], [], [], []
    ids = core.subjects(a1)
    for s in ids:
        h1, h2, y1, y2, r1, r2 = stage_subject(a1, a2, s, B)
        for j, h, y, r in ((1, h1, y1, r1), (2, h2, y2, r2)):
            z = h - r if relative else h
            x.append(z); cls.extend(y.tolist()); ses.extend([j - 1] * len(z))
            sub.extend([s] * len(z)); trial.extend(np.arange(len(z)).tolist())
    x = np.concatenate(x)
    cls, ses, sub, trial = map(np.asarray, (cls, ses, sub, trial))
    # Even/odd within every subject/session: same identities are present in both
    # halves; probes are TRAIN-only diagnostics, not unseen-subject predictions.
    tr, te = trial % 2 == 0, trial % 2 == 1
    scaler = StandardScaler().fit(x[tr])
    ztr, zte = scaler.transform(x[tr]), scaler.transform(x[te])
    out = {}
    for name, y in (("SESSION", ses), ("SUBJECT", sub), ("CLASS", cls)):
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0)
        clf.fit(ztr, y[tr])
        out[name] = float(balanced_accuracy_score(y[te], clf.predict(zte)))
    return out


def run(fold: int):
    lock_path = core.ROOT / "train_lock" / f"fold{fold}.json"
    if not lock_path.exists():
        raise RuntimeError("TRAIN selection lock required")
    B = json.loads(lock_path.read_text(encoding="utf-8"))["B"]
    dest = core.ROOT / "geometry" / f"fold{fold}"
    if dest.exists():
        raise FileExistsError(dest)
    decomposition, cross, context, layer, info, probe_rows = [], [], [], [], [], []
    for stage in core.LOCK["stages"]:
        a1 = core.read_session(fold, "TRAIN_GEOMETRY", 1, stage)
        a2 = core.read_session(fold, "TRAIN_GEOMETRY", 2, stage)
        parts = []
        relation_abs, relation_rel = [], []
        for s in core.subjects(a1):
            h1, h2, y1, y2, r1, r2 = stage_subject(a1, a2, s, B)
            parts += [(s, 1, h1, y1, r1), (s, 2, h2, y2, r2)]
            within1 = float(np.mean(np.sum((h1 - r1) ** 2, axis=1)))
            within2 = float(np.mean(np.sum((h2 - r2) ** 2, axis=1)))
            shift = float(np.linalg.norm(r1 - r2))
            context.append({"fold": fold, "stage": stage, "role": "TRAIN_GEOMETRY", "subject": s, "B": B,
                            "recording_mean_shift": shift, "within_session1": within1,
                            "within_session2": within2,
                            "mean_shift_to_within_rms": shift / np.sqrt(max((within1 + within2) / 2, 1e-12)),
                            "session1_reference_norm": float(np.linalg.norm(r1)),
                            "session2_reference_norm": float(np.linalg.norm(r2)),
                            "session1_context_class_fraction_diagnostic": float(np.mean(core.subject_rows(a1, s)[1][:B])),
                            "session2_context_class_fraction_diagnostic": float(np.mean(core.subject_rows(a2, s)[1][:B]))})
            cen1 = [h1[y1 == k].mean(0) for k in (0, 1)]
            cen2 = [h2[y2 == k].mean(0) for k in (0, 1)]
            d1, d2 = cen1[1] - cen1[0], cen2[1] - cen2[0]
            # Translation of both classes by the same session reference cannot
            # alter their within-session difference vectors.
            rd1 = (cen1[1] - r1) - (cen1[0] - r1)
            rd2 = (cen2[1] - r2) - (cen2[0] - r2)
            if not np.allclose(d1, rd1, rtol=1e-9, atol=1e-9) or not np.allclose(d2, rd2, rtol=1e-9, atol=1e-9):
                raise RuntimeError("class-relation translation invariant violated")
            ca, cr = cosine(d1, d2), cosine(rd1, rd2)
            relation_abs.append(ca); relation_rel.append(cr)
            cross.append({"fold": fold, "stage": stage, "role": "TRAIN_GEOMETRY", "subject": s, "B": B,
                          "same_class_cosine_abs_0": cosine(cen1[0], cen2[0]),
                          "same_class_cosine_abs_1": cosine(cen1[1], cen2[1]),
                          "same_class_cosine_rel_0": cosine(cen1[0] - r1, cen2[0] - r2),
                          "same_class_cosine_rel_1": cosine(cen1[1] - r1, cen2[1] - r2),
                          "class_relation_cosine_abs": ca, "class_relation_cosine_rel": cr,
                          "class_relation_cosine_delta": cr - ca,
                          "class_relation_drift_abs": float(np.linalg.norm(d1 - d2)),
                          "class_relation_drift_rel": float(np.linalg.norm(rd1 - rd2)),
                          "relation_invariance_expected": True})
        for relative in (False, True):
            arm = "REL_MEAN" if relative else "ABSOLUTE"
            d = variance(parts, relative)
            decomposition.append({"fold": fold, "stage": stage, "arm": arm, "B": B, **d})
        layer.append({"fold": fold, "stage": stage, "B": B,
                      "mean_recording_shift": float(np.mean([r["recording_mean_shift"] for r in context if r["stage"] == stage])),
                      "class_relation_cosine_abs": float(np.mean(relation_abs)),
                      "class_relation_cosine_rel": float(np.mean(relation_rel)),
                      "relation_delta": float(np.mean(relation_rel) - np.mean(relation_abs)),
                      "primary_stage": stage == "EMBEDDING"})
        if stage == "EMBEDDING":
            for rel in (False, True):
                arm = "REL_MEAN" if rel else "ABSOLUTE"
                p = probes(a1, a2, B, rel)
                probe_rows.append({"fold": fold, "stage": stage, "arm": arm,
                                   "session_BA_train_only": p["SESSION"],
                                   "subject_ID_BA_train_only": p["SUBJECT"],
                                   "class_BA_train_only": p["CLASS"]})
                d = variance(parts, rel)
                info.append({"fold": fold, "stage": stage, "arm": arm,
                             "class_variance": d["class_variance"],
                             "class_between_within_ratio": d["class_variance"] / max(d["residual_trial_variance"], 1e-12),
                             "total_variance": d["total_variance"],
                             "class_probe_BA": p["CLASS"]})
    dest.mkdir(parents=True, exist_ok=False)
    tables = {"CONTEXT_REFERENCE_AUDIT": context,
              "REPRESENTATION_VARIANCE_DECOMPOSITION": decomposition,
              "CROSS_SESSION_GEOMETRY": cross,
              "LAYER_LOCALIZATION": layer,
              "SUBJECT_SESSION_CLASS_PROBES": probe_rows,
              "INFORMATION_RETENTION": info}
    for name, rows in tables.items():
        core.write_csv_new(dest / f"{name}.csv", rows)
    core.save_json_new(dest / "AUDIT.json", {"fold": fold, "B": B,
        "train_lock_sha256": core.sha(lock_path), "final_heldout_eeg_reads": 0,
        "class_relation_translation_invariant": True,
        "files": {p.name: core.sha(p) for p in dest.glob("*.csv")}})
    print(json.dumps({"fold": fold, "B": B, "audit_sha256": core.sha(dest / "AUDIT.json")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True, choices=range(5))
    run(ap.parse_args().fold)
