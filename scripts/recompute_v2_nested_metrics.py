#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.v2.nested_evaluation import _bootstrap_metrics, _metrics


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
BASE = ROOT / "outputs/v2_joint_certified"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    nested = BASE / "nested_dev"
    decisions_path = nested / "ALL_DEV_DECISIONS.parquet"
    counter_path = nested / "ALL_DEV_COUNTERFACTUALS.parquet"
    before = {name: sha(nested / name) for name in ("DEV_RESULTS_BY_FOLD.csv", "DEV_RESULTS_BY_SEED.csv",
                                                     "DEV_RESULTS_SUMMARY.csv", "DEV_RESULTS_WITH_CI.csv")}
    decisions = pd.read_parquet(decisions_path); counters = pd.read_parquet(counter_path)
    rows, ci = [], []
    for keys, selected in decisions.groupby(["dataset", "seed", "outer_fold", "alpha"]):
        dataset, seed, fold, alpha = keys
        counter = counters[(counters.dataset == dataset) & (counters.seed == seed) &
                           (counters.outer_fold == fold) & np.isclose(counters.alpha, alpha)]
        point = _metrics(selected, counter, float(alpha))
        rows.extend({"dataset": dataset, "seed": seed, "outer_fold": fold, "alpha": alpha,
                     "policy": "joint_hsc_tta_v2", "metric": metric, "value": value}
                    for metric, value in point.items())
        rng = np.random.default_rng(81000 + int(seed) * 100 + int(fold) * 10 + int(float(alpha) * 100))
        bootstrap = _bootstrap_metrics(selected, counter, float(alpha), rng, 1000)
        for metric, value in point.items():
            values = bootstrap[metric]
            ci.append({"dataset": dataset, "seed": seed, "outer_fold": fold, "alpha": alpha, "metric": metric,
                       "point_estimate": value, "ci_lower": np.nanquantile(values, .025),
                       "ci_upper": np.nanquantile(values, .975), "n_subjects": selected.subject_id.nunique()})
    fold_frame = pd.DataFrame(rows); fold_frame.to_csv(nested / "DEV_RESULTS_BY_FOLD.csv", index=False)
    by_seed = fold_frame.groupby(["dataset", "seed", "alpha", "policy", "metric"]).value.mean().reset_index()
    by_seed.to_csv(nested / "DEV_RESULTS_BY_SEED.csv", index=False)
    by_seed.groupby(["dataset", "alpha", "policy", "metric"]).value.agg(["mean", "std"]).reset_index().to_csv(
        nested / "DEV_RESULTS_SUMMARY.csv", index=False)
    pd.DataFrame(ci).to_csv(nested / "DEV_RESULTS_WITH_CI.csv", index=False)
    after = {name: sha(nested / name) for name in before}
    freeze = BASE / "freeze/V2_METHOD_FREEZE.json"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "type": "post_freeze_evaluation_only_correction",
               "method_freeze_sha256": sha(freeze), "decision_parquet_sha256": sha(decisions_path),
               "counterfactual_parquet_sha256": sha(counter_path), "method_or_decisions_changed": False,
               "reason": "risk, set size, and singleton metrics must be evaluated at the frozen certified index, not the oracle true critical index",
               "before_sha256": before, "after_sha256": after}
    path = BASE / "provenance/V2_EVALUATION_CORRECTION.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
