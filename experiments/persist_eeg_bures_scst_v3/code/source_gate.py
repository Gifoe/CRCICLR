"""Evaluate the preregistered V3 source gate and select one recipe.

This module is deliberately fail-closed.  It only consumes source-training
features and source transition validation rows; no WBCIC S3/outer path is
referenced or opened here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as c


def _paired(frame: pd.DataFrame, dataset: str, method: str, control: str, q: float, lam: float) -> tuple[float, float, float, int]:
    left = frame[(frame.dataset == dataset) & (frame.method == method) & (np.isclose(frame.q, q)) & (np.isclose(frame.lambda_T, lam))]
    right = frame[(frame.dataset == dataset) & (frame.method == control)]
    if not len(left) or not len(right):
        return float("nan"), float("nan"), float("nan"), 0
    l = left.groupby("subject_id").BA.mean()
    r = right.groupby("subject_id").BA.mean()
    delta = (l - r).dropna().to_numpy(np.float64)
    return (*c.bootstrap_ci(delta, seed=c.stable_seed("source-gate", dataset, method, control, q, lam)), int(len(delta)))


def _transport_gate(geometry: pd.DataFrame, dataset: str) -> dict[str, object]:
    subset = geometry[(geometry.dataset == dataset) & (geometry.method == "Bures-HardSCST")]
    if not len(subset):
        return {"pass": False, "reason": "NO_BURES_GEOMETRY", "dataset": dataset}
    # Bootstrap biological subjects after averaging fold/seed replicates.
    subject = subset.groupby("subject_id").agg(
        target_distance_improvement=("target_distance_improvement", "mean"),
        target_nll_improvement=("target_nll_improvement", "mean"),
        coverage=("coverage", "mean"),
        class_pass_rate=("class_pass_rate", "mean"),
        median_displacement_ratio=("median_displacement_ratio", "median"),
        median_relative_margin_drop=("median_relative_margin_drop", "median"),
    )
    def ci(column: str) -> tuple[float, float, float]:
        return c.bootstrap_ci(subject[column].to_numpy(float), seed=c.stable_seed("transport-gate", dataset, column))
    distance = ci("target_distance_improvement")
    nll = ci("target_nll_improvement")
    row = {
        "dataset": dataset,
        "subjects": int(len(subject)),
        "target_distance_mean": distance[0], "target_distance_ci95_l": distance[1], "target_distance_ci95_u": distance[2],
        "target_nll_mean": nll[0], "target_nll_ci95_l": nll[1], "target_nll_ci95_u": nll[2],
        "coverage_mean": float(subject.coverage.mean()), "class_pass_rate_mean": float(subject.class_pass_rate.mean()),
        "median_displacement_ratio": float(subject.median_displacement_ratio.median()),
        "median_relative_margin_drop": float(subject.median_relative_margin_drop.median()),
    }
    checks = {
        "target_distance_ci_lower_positive": bool(np.isfinite(distance[1]) and distance[1] > 0),
        "target_nll_ci_lower_positive": bool(np.isfinite(nll[1]) and nll[1] > 0),
        "class_fidelity": bool(row["class_pass_rate_mean"] >= 0.90),
        "coverage": bool(row["coverage_mean"] >= 0.50),
        "displacement": bool(row["median_displacement_ratio"] >= 0.15),
        "margin_drop": bool(row["median_relative_margin_drop"] >= 0.10),
    }
    row["checks"] = checks; row["pass"] = bool(all(checks.values()))
    return row


def main() -> None:
    c.ensure_dirs()
    source_path = c.RESULTS / "SOURCE_PER_SUBJECT.csv"
    geometry_path = c.RESULTS / "GEOMETRY_PER_SUBJECT.csv"
    frame = pd.read_csv(source_path) if source_path.is_file() else pd.DataFrame(columns=["dataset", "method", "q", "lambda_T", "fold", "seed", "subject_id", "BA", "macro_F1"])
    geometry = pd.read_csv(geometry_path) if geometry_path.is_file() else pd.DataFrame()
    transport = {dataset: _transport_gate(geometry, dataset) for dataset in c.DATASETS}
    transport_pass = bool(all(transport[d].get("pass", False) for d in c.DATASETS))
    rows: list[dict[str, object]] = []
    for q, lam in c.RECIPES if hasattr(c, "RECIPES") else ((q, lam) for q in (0.25, 0.50) for lam in (0.25, 0.50, 1.0)):
        open_erm = _paired(frame, "OpenBMI", "Bures-HardSCST", "ERM", q, lam)
        wbcic_erm = _paired(frame, "WBCIC", "Bures-HardSCST", "ERM", q, lam)
        random = _paired(frame, "WBCIC", "Bures-HardSCST", "Bures-HardRandom", q, lam)
        manifold = _paired(frame, "WBCIC", "Bures-HardSCST", "Manifold-Mixup", q, lam)
        clean_ok = bool(np.isfinite(open_erm[0]) and open_erm[0] >= 0.002 and np.isfinite(wbcic_erm[0]) and wbcic_erm[0] >= 0.002 and open_erm[0] >= -0.001 and wbcic_erm[0] >= -0.001)
        checks = {
            "openbmi_delta_ge_002": bool(np.isfinite(open_erm[0]) and open_erm[0] >= 0.002),
            "wbcic_delta_ge_002": bool(np.isfinite(wbcic_erm[0]) and wbcic_erm[0] >= 0.002),
            "openbmi_ci_lower_positive": bool(np.isfinite(open_erm[1]) and open_erm[1] > 0),
            "wbcic_ci_lower_positive": bool(np.isfinite(wbcic_erm[1]) and wbcic_erm[1] > 0),
            "wbcic_vs_random_ci_lower_positive": bool(np.isfinite(random[1]) and random[1] > 0),
            "wbcic_vs_manifold_ci_lower_positive": bool(np.isfinite(manifold[1]) and manifold[1] > 0),
            "clean_degradation_ok": clean_ok,
            "transport_gate": transport_pass,
        }
        rows.append({"q": q, "lambda_T": lam, "openbmi_delta": open_erm[0], "openbmi_ci95_l": open_erm[1], "openbmi_ci95_u": open_erm[2], "wbcic_delta": wbcic_erm[0], "wbcic_ci95_l": wbcic_erm[1], "wbcic_ci95_u": wbcic_erm[2], "wbcic_vs_random_delta": random[0], "wbcic_vs_random_ci95_l": random[1], "wbcic_vs_manifold_delta": manifold[0], "wbcic_vs_manifold_ci95_l": manifold[1], "checks": json.dumps(checks, sort_keys=True), "pass": bool(all(checks.values()))})
    detail = pd.DataFrame(rows)
    c.write_csv(c.RESULTS / "SOURCE_GATE_DETAIL.csv", detail)
    passed = detail[detail["pass"] == True]  # noqa: E712
    selected = None
    if len(passed):
        selected_row = sorted(passed.to_dict("records"), key=lambda row: (-min(float(row["openbmi_delta"]), float(row["wbcic_delta"])), float(row["lambda_T"]), -float(row["q"]))) [0]
        selected = {"q": float(selected_row["q"]), "lambda_T": float(selected_row["lambda_T"]), "selection_rule": "maximise min(OpenBMI,WBCIC delta), tie lower lambda then q=0.50"}
    if not transport_pass:
        terminal = "BURES_SCST_TRANSPORT_NOT_REALIZED"
    elif selected is None:
        terminal = "BURES_SCST_SOURCE_GATE_FAILED"
    else:
        terminal = "SOURCE_GATE_PASSED"
    payload = {"schema": "BURES_SCST_V3_SOURCE_GATE_V1", "source_grid_complete": bool(len(frame) >= len(c.DATASETS) * len(c.FOLDS) * len(c.SEEDS)), "transport": transport, "recipe_rows": rows, "source_gate_pass": bool(selected is not None and transport_pass), "selected": selected, "terminal_if_stop": terminal, "s3_opened": False, "outer_or_sealed_opened": False}
    c.write_json(c.RESULTS / "SOURCE_GATE.json", payload)
    print(json.dumps(payload, indent=2))
    if terminal != "SOURCE_GATE_PASSED":
        print(terminal)


if __name__ == "__main__":
    main()
