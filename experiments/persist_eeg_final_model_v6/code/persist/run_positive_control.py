"""Synthetic mechanism check for protected/adaptable/harmful separation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import DIAGNOSTICS, PROTOCOL, V6_SEED, write_csv, write_json


def _subject(rng: np.random.Generator, subject: int, trials: int = 200):
    orientation = rng.choice([-1.0, 1.0])
    y_history = np.tile([0, 1], trials // 2)
    y_future = np.tile([0, 1], trials // 2)
    rng.shuffle(y_history); rng.shuffle(y_future)

    def build(labels, future):
        sign = 2.0 * labels - 1.0
        protected = 1.2 * sign + rng.normal(0, 0.8, len(labels))
        adaptable = 1.0 * orientation * sign + rng.normal(0, 0.8, len(labels))
        # This block is prospectively harmful: across model-fit episodes it
        # reverses in the future session and is therefore safe to suppress.
        harmful = (1.4 if not future else -1.4) * sign + rng.normal(0, 0.8, len(labels))
        neutral = rng.normal(0, 1.0, (len(labels), 3))
        return np.column_stack([protected, adaptable, harmful, neutral])

    return build(y_history, False), y_history, build(y_future, True), y_future, orientation


def _transform(x, orientation, mode):
    value = x.copy()
    if mode in {"adapt_only", "persist_selective"}:
        value[:, 1] *= orientation
    if mode == "persist_selective":
        value[:, 2] = 0.0
    if mode == "blanket_erasure":
        value[:, :3] = 0.0
    return value


def run() -> None:
    rng = np.random.default_rng(V6_SEED)
    subjects = [_subject(rng, index) for index in range(60)]
    fit_subjects = subjects[:40]
    outcome_subjects = subjects[40:]
    rows = []
    for mode in ("raw", "adapt_only", "blanket_erasure", "persist_selective"):
        train_x = np.concatenate([_transform(hx, orientation, mode) for hx, _, _, _, orientation in fit_subjects])
        train_y = np.concatenate([hy for _, hy, _, _, _ in fit_subjects])
        scaler = StandardScaler().fit(train_x)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000).fit(scaler.transform(train_x), train_y)
        subject_ba = []
        for _, _, fx, fy, orientation in outcome_subjects:
            prediction = model.predict(scaler.transform(_transform(fx, orientation, mode)))
            subject_ba.append(balanced_accuracy_score(fy, prediction))
        rows.append(
            {
                "method_id": mode,
                "mean_subject_BA": float(np.mean(subject_ba)),
                "protected_coordinate_preserved": mode in {"raw", "adapt_only", "persist_selective"},
                "adaptable_coordinate_aligned": mode in {"adapt_only", "persist_selective"},
                "certified_harmful_coordinate_suppressed": mode == "persist_selective",
                "synthetic_only": True,
                "OUTER_TEST_USED": False,
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    persist_ba = float(table.loc[table.method_id.eq("persist_selective"), "mean_subject_BA"].iloc[0])
    raw_ba = float(table.loc[table.method_id.eq("raw"), "mean_subject_BA"].iloc[0])
    erasure_ba = float(table.loc[table.method_id.eq("blanket_erasure"), "mean_subject_BA"].iloc[0])
    passed = persist_ba > raw_ba + 0.10 and persist_ba > erasure_ba + 0.20
    write_csv(DIAGNOSTICS / "POSITIVE_CONTROL.csv", table)
    write_json(
        PROTOCOL / "POSITIVE_CONTROL_AUDIT.json",
        {
            "status": "PASS" if passed else "FAIL",
            "mechanism": "stable protected label signal; subject-oriented adaptable signal; prospectively sign-reversing harmful signal",
            "claim_scope": "mechanism sensitivity only; not evidence that these components occur in real EEG",
            "synthetic_only": True,
            "OUTER_TEST_USED": False,
        },
    )
    if not passed:
        raise RuntimeError("PERSIST positive control did not separate the mechanisms")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
