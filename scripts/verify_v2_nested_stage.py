#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg/outputs/v2_joint_certified")
FILES = {
    "benefit": ("predictors/BENEFIT_PREDICTOR_OOF.parquet",),
    "calibration": ("nested_dev/ALL_DEV_CALIBRATION_SCORES.parquet", "nested_dev/ALL_DEV_JOINT_BOUNDS.parquet"),
    "decisions": ("nested_dev/ALL_DEV_DECISIONS.parquet",),
    "freeze": ("nested_dev/ALL_DEV_DECISIONS.parquet",),
    "outcomes": ("nested_dev/ALL_DEV_COUNTERFACTUALS.parquet", "nested_dev/DEV_RESULTS_WITH_CI.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=FILES, required=True)
    args = parser.parse_args()
    for relative in FILES[args.stage]:
        path = ROOT / relative
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.stage == "calibration":
        scores = pd.read_parquet(ROOT / FILES[args.stage][0])
        counts = scores.groupby(["dataset", "seed", "outer_fold", "alpha", "subject_id"]).size()
        if not (counts == 1).all():
            raise AssertionError("joint calibration must contain one score per subject")
    if args.stage == "freeze":
        freeze_files = list((ROOT / "nested_dev/decisions").rglob("*.freeze.json"))
        if len(freeze_files) != 100:
            raise AssertionError(f"expected 100 decision freezes, got {len(freeze_files)}")
        for freeze_path in freeze_files:
            payload = json.loads(freeze_path.read_text())
            decision = freeze_path.with_suffix("").with_suffix(".parquet")
            digest = hashlib.sha256(decision.read_bytes()).hexdigest()
            if payload.get("decision_sha256") != digest or payload.get("V_opened") is not True:
                raise AssertionError(f"invalid decision freeze {freeze_path}")
    print(f"verified nested stage: {args.stage}")


if __name__ == "__main__":
    main()
