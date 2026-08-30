"""Apply the preregistered source-only mechanism and recipe-selection rule."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import v2_common as c


def main() -> None:
    path = c.RESULTS / "SOURCE_RECIPE_SEARCH.csv"
    frame = pd.read_csv(path)
    expected = 12 * 2
    if len(frame) != expected or not (frame.units == 15).all():
        raise RuntimeError(f"SOURCE_GRID_INCOMPLETE:{len(frame)} rows")
    rows = []
    for (scope, q, lam), group in frame.groupby(["scope", "q", "lambda_H"]):
        if set(group.dataset) != set(c.DATASETS):
            raise RuntimeError("SOURCE_DOMAIN_MISSING")
        finite = bool(np.isfinite(group[["BA", "ERM_BA", "delta_BA", "valid_candidate_coverage", "hardness_gap", "hardness_gap_CI95_L", "semantic_pass_rate", "bank_stability"]].to_numpy()).all())
        mechanism = bool(
            finite
            and (group.valid_candidate_coverage >= 0.50).all()
            and (group.median_valid_candidates >= 2).all()
            and (group.hardness_gap_CI95_L > 0).all()
            and (group.semantic_pass_rate > 0).all()
            and (group.bank_stability > 0).all()
        )
        deltas = {str(row.dataset): float(row.delta_BA) for row in group.itertuples()}
        minimum = min(deltas.values())
        rows.append({
            "scope": scope, "q": float(q), "lambda_H": float(lam),
            "OpenBMI_delta_BA": deltas["OpenBMI"], "WBCIC_delta_BA": deltas["WBCIC"],
            "minimum_development_delta": minimum, "mechanism_eligible": mechanism,
            "development_positive": bool(minimum > 0), "eligible": bool(mechanism and minimum > 0),
            "coverage_min": float(group.valid_candidate_coverage.min()),
            "hardness_CI95_L_min": float(group.hardness_gap_CI95_L.min()),
            "semantic_pass_min": float(group.semantic_pass_rate.min()),
            "bank_stability_min": float(group.bank_stability.min()),
        })
    ranking = pd.DataFrame(rows)
    ranking["scope_tie"] = ranking.scope.map({"A": 1, "B": 0})
    ranking["lambda_tie"] = -ranking.lambda_H
    ranking["q_tie"] = ranking.q.map({0.50: 1, 0.25: 0})
    ranking = ranking.sort_values(["eligible", "minimum_development_delta", "scope_tie", "lambda_tie", "q_tie"], ascending=False).reset_index(drop=True)
    c.write_csv(c.RESULTS / "SOURCE_RECIPE_RANKING.csv", ranking.drop(columns=["scope_tie", "lambda_tie", "q_tie"]))
    eligible = ranking[ranking.eligible]
    selected = None if eligible.empty else eligible.iloc[0]
    decision = {
        "schema": "ME_HARD_SCST_SOURCE_DECISION_V1",
        "source_results_sha256": c.sha256(path),
        "grid_rows": int(len(frame)),
        "recipes": 12,
        "datasets": list(c.DATASETS),
        "folds": list(c.FOLDS),
        "seeds": list(c.SEEDS),
        "selection_rule": "maximize min(OpenBMI_delta,WBCIC_delta); mechanism gates; positive minimum; ties A, lower lambda, q=0.50",
        "selected": None if selected is None else {
            "scope": str(selected.scope), "q": float(selected.q), "lambda_H": float(selected.lambda_H),
            "OpenBMI_delta_BA": float(selected.OpenBMI_delta_BA), "WBCIC_delta_BA": float(selected.WBCIC_delta_BA),
            "minimum_development_delta": float(selected.minimum_development_delta),
            "coverage_min": float(selected.coverage_min), "hardness_CI95_L_min": float(selected.hardness_CI95_L_min),
            "semantic_pass_min": float(selected.semantic_pass_min), "bank_stability_min": float(selected.bank_stability_min),
        },
        "source_gate_pass": selected is not None,
        "s3_opened": False,
        "outer_or_sealed_opened": False,
        "terminal_if_stop": None if selected is not None else "ME_HARD_SCST_MECHANISM_NOT_REALIZED",
    }
    c.write_json(c.RESULTS / "SOURCE_DECISION.json", decision)
    report = [
        "# Source development report", "",
        "All 12 preregistered recipes were evaluated over OpenBMI session 1 to 2 and WBCIC S1 to S2 using five folds and three seeds.", "",
        f"Source gate: **{'PASS' if selected is not None else 'FAIL'}**.", "",
    ]
    if selected is None:
        report += ["No recipe simultaneously satisfied the mechanism gates and a positive minimum domain delta. WBCIC S3 remains unopened.", "", "Terminal: `ME_HARD_SCST_MECHANISM_NOT_REALIZED`."]
    else:
        report += [
            f"Selected recipe: Scope {selected.scope}, q={selected.q:.2f}, lambda_H={selected.lambda_H:.2f}.",
            f"OpenBMI delta: {selected.OpenBMI_delta_BA:+.6f}; WBCIC S1-to-S2 delta: {selected.WBCIC_delta_BA:+.6f}; minimum: {selected.minimum_development_delta:+.6f}.",
            "The selection was made before any new WBCIC S3 access.",
        ]
    (c.EXP / "SOURCE_DEVELOPMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

