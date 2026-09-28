"""Secondary, predeclared stage-wise frozen linear-decoder localization.

No stage is selected from these OUTER results. The earlier stages use a common
TRAIN-only feature scaler and fixed C=1 within each stage; this is distinct
from the primary raw-EMBEDDING decoder and labelled accordingly.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import aggregate as core


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def run(fold: int):
    for f in range(5):
        if not (core.ROOT / "train_lock" / f"fold{f}.json").is_file():
            raise RuntimeError("all TRAIN locks must precede OUTER localization")
    lock_path = core.ROOT / "train_lock" / f"fold{fold}.json"
    B = json.loads(lock_path.read_text(encoding="utf-8"))["B"]
    dest = core.ROOT / "localization" / f"fold{fold}"
    if dest.exists():
        raise FileExistsError(dest)
    rows, context, cross = [], [], []
    for stage in core.LOCK["stages"]:
        source = core.read_session(fold, "TRAIN_GEOMETRY", 1, stage)
        target = core.read_session(fold, "OUTER_DEVELOPMENT", 2, stage)
        target_s1 = core.read_session(fold, "OUTER_DEVELOPMENT", 1, stage)
        tr, ev = core.split_context(source, B), core.split_context(target, B)
        mu = core.global_mean(source)
        raw_x, _, _ = core.transform(tr, "ABSOLUTE", mu)
        scaler = StandardScaler().fit(raw_x)
        for arm in ("ABSOLUTE", "REL_MEAN"):
            x, y, _ = core.transform(tr, arm, mu)
            v, vy, owner = core.transform(ev, arm, mu)
            x, v = scaler.transform(x), scaler.transform(v)
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300,
                                     solver="liblinear", random_state=0)
            clf.fit(x, y)
            scored = core.scores(clf, v, vy, owner)
            rows.append({"fold": fold, "stage": stage, "arm": arm, "B": B,
                         "OUTER_subject_equal_BA": core.subject_mean(scored),
                         "OUTER_subject_equal_macro_f1": core.subject_mean(scored, "macro_f1"),
                         "OUTER_subject_equal_nll": core.subject_mean(scored, "nll"),
                         "feature_dim": x.shape[1], "TRAIN_only_standardizer": True,
                         "decoder": "balanced logistic liblinear C=1 fixed across earlier stages; secondary descriptive",
                         "primary_stage": stage == "EMBEDDING"})
        for s in core.subjects(target):
            h1, y1 = core.subject_rows(target_s1, s)
            h2, y2 = core.subject_rows(target, s)
            r1, r2 = h1[:B].mean(0), h2[:B].mean(0)
            a, b = h1[B:], h2[B:]
            p, q = y1[B:], y2[B:]
            c1 = [a[p == k].mean(0) for k in (0, 1)]
            c2 = [b[q == k].mean(0) for k in (0, 1)]
            d1, d2 = c1[1] - c1[0], c2[1] - c2[0]
            rd1, rd2 = (c1[1] - r1) - (c1[0] - r1), (c2[1] - r2) - (c2[0] - r2)
            if not np.allclose(d1, rd1, atol=1e-9, rtol=1e-9) or not np.allclose(d2, rd2, atol=1e-9, rtol=1e-9):
                raise RuntimeError("outer class relation invariant violated")
            v1 = float(np.mean(np.sum((a - r1) ** 2, axis=1)))
            v2 = float(np.mean(np.sum((b - r2) ** 2, axis=1)))
            shift = float(np.linalg.norm(r1 - r2))
            context.append({"fold": fold, "stage": stage, "role": "OUTER_DEVELOPMENT", "subject": s, "B": B,
                            "recording_mean_shift": shift, "within_session1": v1, "within_session2": v2,
                            "mean_shift_to_within_rms": shift / np.sqrt(max((v1 + v2) / 2, 1e-12)),
                            "session1_reference_norm": float(np.linalg.norm(r1)),
                            "session2_reference_norm": float(np.linalg.norm(r2)),
                            "session1_context_class_fraction_diagnostic": float(y1[:B].mean()),
                            "session2_context_class_fraction_diagnostic": float(y2[:B].mean())})
            cross.append({"fold": fold, "stage": stage, "role": "OUTER_DEVELOPMENT", "subject": s, "B": B,
                          "same_class_cosine_abs_0": cosine(c1[0], c2[0]),
                          "same_class_cosine_abs_1": cosine(c1[1], c2[1]),
                          "same_class_cosine_rel_0": cosine(c1[0] - r1, c2[0] - r2),
                          "same_class_cosine_rel_1": cosine(c1[1] - r1, c2[1] - r2),
                          "class_relation_cosine_abs": cosine(d1, d2),
                          "class_relation_cosine_rel": cosine(rd1, rd2),
                          "class_relation_cosine_delta": cosine(rd1, rd2) - cosine(d1, d2),
                          "class_relation_drift_abs": float(np.linalg.norm(d1 - d2)),
                          "class_relation_drift_rel": float(np.linalg.norm(rd1 - rd2)),
                          "relation_invariance_expected": True})
    dest.mkdir(parents=True, exist_ok=False)
    core.write_csv_new(dest / "LAYER_LOCALIZATION_OUTER.csv", rows)
    core.write_csv_new(dest / "CONTEXT_REFERENCE_OUTER.csv", context)
    core.write_csv_new(dest / "CROSS_SESSION_GEOMETRY_OUTER.csv", cross)
    core.save_json_new(dest / "AUDIT.json", {"fold": fold, "B": B,
        "train_lock_sha256": core.sha(lock_path), "final_heldout_eeg_reads": 0,
        "outer_stage_selection": False,
        "files": {p.name: core.sha(p) for p in dest.glob("*.csv")}})
    print(json.dumps({"fold": fold, "rows": len(rows),
                      "audit_sha256": core.sha(dest / "AUDIT.json")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, required=True, choices=range(5))
    run(p.parse_args().fold)
