"""Recompute the immutable SCST-V1 statistics from stored subject artifacts.

This program is deliberately aggregation-only.  It never opens EEG caches or
future/outer resources and does not retrain a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
V1 = REPO / "experiments" / "persist_eeg_scst_utility_stage1"
EXP = REPO / "experiments" / "persist_eeg_me_hard_scst_v2"
METHODS = ("ERM", "Mixup", "RandomTransport", "SCST-NoConsistency", "Full-SCST")
IMMUTABLE = {
    "ERM": 0.805164,
    "Mixup": 0.806176,
    "RandomTransport": 0.806949,
    "ShuffleSameClass": 0.805928,
    "SCST-NoConsistency": 0.806014,
    "Full-SCST": 0.805410,
    "Full-SCST-ERM": 0.000246,
    "CI95_L": -0.001260,
    "CI95_U": 0.001831,
    "positive_folds": 2,
    "consistency_contribution": -0.000605,
}


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = V1 / "results" / "SCST_PER_SUBJECT_ATCNet-CleanRoom.csv"
    frame = pd.read_csv(source)
    required = {"fold", "seed", "subject_id", "method", "BA"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"V1 artifact missing columns: {sorted(required - set(frame.columns))}")
    if set(frame.method) != set(METHODS):
        raise RuntimeError(f"V1 method grid mismatch: {sorted(set(frame.method))}")

    pivot = frame.pivot_table(index=["fold", "seed", "subject_id"], columns="method", values="BA").reset_index()
    if pivot[list(METHODS)].isna().any().any():
        raise RuntimeError("V1 paired grid is incomplete")
    subject = pivot.groupby("subject_id", as_index=False).mean(numeric_only=True)

    rng = np.random.default_rng(stable_seed("stage1-subject-bootstrap", "ATCNet-CleanRoom"))
    regenerated: dict[str, float | int] = {}
    for method in METHODS:
        values = subject[method].to_numpy(np.float64)
        # Preserve V1 RNG consumption before paired comparisons.
        _ = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
        regenerated[method] = float(values.mean())

    comparisons: dict[str, dict[str, float | int]] = {}
    for method in METHODS[1:]:
        values = (subject[method] - subject["ERM"]).to_numpy(np.float64)
        draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
        fold_delta = pivot.groupby("fold").apply(
            lambda value: float((value[method] - value["ERM"]).mean()),
            include_groups=False,
        )
        comparisons[f"{method}-ERM"] = {
            "delta_BA": float(values.mean()),
            "CI95_L": float(np.quantile(draws, 0.025)),
            "CI95_U": float(np.quantile(draws, 0.975)),
            "positive_folds": int((fold_delta > 0).sum()),
        }

    full = comparisons["Full-SCST-ERM"]
    regenerated.update(
        {
            "Full-SCST-ERM": float(full["delta_BA"]),
            "CI95_L": float(full["CI95_L"]),
            "CI95_U": float(full["CI95_U"]),
            "positive_folds": int(full["positive_folds"]),
            "consistency_contribution": float(regenerated["Full-SCST"] - regenerated["SCST-NoConsistency"]),
            "Full-SCST-Mixup": float(regenerated["Full-SCST"] - regenerated["Mixup"]),
            "Full-SCST-RandomTransport": float(regenerated["Full-SCST"] - regenerated["RandomTransport"]),
        }
    )

    tolerance = 5e-6
    checks: dict[str, dict[str, object]] = {}
    artifact_backed = set(IMMUTABLE) - {"ShuffleSameClass"}
    for key in sorted(artifact_backed):
        observed = regenerated[key]
        expected = IMMUTABLE[key]
        ok = int(observed) == int(expected) if key == "positive_folds" else abs(float(observed) - float(expected)) <= tolerance
        checks[key] = {"expected": expected, "regenerated": observed, "absolute_error": abs(float(observed) - float(expected)), "pass": bool(ok)}

    output = {
        "schema": "ME_HARD_SCST_V2_V1_REPRODUCTION_V1",
        "source_artifact": str(source),
        "source_sha256": sha256(source),
        "rows": int(len(frame)),
        "biological_subjects": int(frame.subject_id.astype(str).nunique()),
        "folds": sorted(frame.fold.astype(int).unique().tolist()),
        "seeds": sorted(frame.seed.astype(int).unique().tolist()),
        "artifact_backed_methods": list(METHODS),
        "regenerated": regenerated,
        "comparisons": comparisons,
        "checks": checks,
        "artifact_backed_reproduction_pass": bool(all(value["pass"] for value in checks.values())),
        "code_path_recovered": True,
        "shuffle_same_class": {
            "immutable_report_value": IMMUTABLE["ShuffleSameClass"],
            "artifact_available": False,
            "code_path_available": False,
            "status": "HISTORICAL_PROMPT_ONLY_NOT_REGENERATED",
        },
        "future_or_outer_data_opened": False,
    }
    if not output["artifact_backed_reproduction_pass"]:
        raise RuntimeError("V1 artifact-backed reproduction mismatch")
    target = EXP / "results" / "V1_REPRODUCTION.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
