#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsc_tta.v3.policy_search import grouped_oof_policy_search

from _common import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--config", required=True)
    parser.add_argument("--main-config", default="configs/v3/main.yaml")
    args = parser.parse_args(); root = project_root(args.root); grid = load_yaml(args.config); main = load_yaml(args.main_config)
    base = root / "outputs/v3_probecert/cross_context_surfaces"
    diagnostics = pd.read_parquet(base / "PROBE_DIAGNOSTICS.parquet")
    outcomes = pd.read_parquet(base / "META_FUTURE_ACTION_OUTCOMES.parquet")
    decisions = []; searches = []
    for (dataset, seed), current in diagnostics.groupby(["dataset", "seed"]):
        for alpha in main["alphas"]:
            future = outcomes[(outcomes.dataset == dataset) & (outcomes.seed == seed) & (outcomes.alpha == alpha)]
            selected, search = grouped_oof_policy_search(current, future, grid, float(main["epsilon"]))
            selected[["dataset", "seed", "alpha"]] = [dataset, int(seed), float(alpha)]
            search[["dataset", "seed", "alpha"]] = [dataset, int(seed), float(alpha)]
            decisions.append(selected); searches.append(search)
    out = root / "outputs/v3_probecert/cross_context_surfaces"; out.mkdir(parents=True, exist_ok=True)
    pd.concat(decisions, ignore_index=True).to_parquet(out / "PROBE_POLICY_OOF_DECISIONS.parquet", index=False)
    pd.concat(searches, ignore_index=True).to_parquet(out / "PROBE_POLICY_THRESHOLD_SEARCH.parquet", index=False)
    combined = pd.concat(decisions); combined["harmful_intervention"] = combined.intervention & (
        combined.classification_degradation > float(main["epsilon"]))
    summary = combined.groupby(["dataset", "seed", "alpha"], as_index=False).agg(
        intervention_rate=("intervention", "mean"), mean_set_size_gain=("set_size_gain", "mean"),
        mean_degradation=("classification_degradation", "mean"),
        harmful_intervention_rate=("harmful_intervention", "mean"))
    summary.to_csv(out / "PROBE_POLICY_OOF_SUMMARY.csv", index=False); print(summary.to_string(index=False))


if __name__ == "__main__": main()
