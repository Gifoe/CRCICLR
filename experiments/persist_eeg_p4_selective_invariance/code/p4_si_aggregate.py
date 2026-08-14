from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "persist_eeg_p4_selective_invariance"
TASKS = ("mi", "erp", "ssvep")
RUNS = [(fold, seed) for fold in (0, 1, 2) for seed in (0, 1)]


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_results(version: str) -> list[dict[str, Any]]:
    results = []
    for fold, seed in RUNS:
        path = OUT / "development" / version / f"fold-{fold}" / f"seed-{seed}" / "DEVELOPMENT_RESULT.json"
        if not path.exists():
            raise RuntimeError(f"Missing development result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("outer_test_used") is not False:
            raise RuntimeError(f"Outer-test marker is not false: {path}")
        results.append(result)
    return results


def summarize(version: str) -> dict[str, Any]:
    results = load_results(version)
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"version": version, "fold": result["fold"], "seed": result["seed"]}
        for task in TASKS:
            row[f"delta_BA_{task}"] = result["task_performance"]["delta"][task]
            row[f"verifier_delta_AUROC_{task}"] = result["nuisance_subject_verification"]["after_minus_before_AUROC"][task]
            intervention = result["intervention"]["after"]["per_task"][task]
            row[f"protected_harm_{task}"] = -intervention["delta_BA"]["protected"]
            row[f"nuisance_harm_{task}"] = -intervention["delta_BA"]["nuisance"]
            row[f"random_harm_{task}"] = -intervention["delta_BA"]["random"]
            row[f"protected_minus_nuisance_{task}"] = intervention["protected_drop_minus_nuisance_drop"]
            before = result["intervention"]["before"]["per_task"][task]
            row[f"protected_harm_before_{task}"] = -before["delta_BA"]["protected"]
            row[f"protected_harm_retention_{task}"] = (
                (-intervention["delta_BA"]["protected"]) / max(-before["delta_BA"]["protected"], 1e-12)
                if -before["delta_BA"]["protected"] > 0.005
                else None
            )
        row["delta_BA_macro"] = float(np.mean([row[f"delta_BA_{task}"] for task in TASKS]))
        row["verifier_delta_AUROC_macro"] = result["nuisance_subject_verification"]["after_minus_before_AUROC"]["macro"]
        row["protected_minus_nuisance_MI"] = row["protected_minus_nuisance_mi"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    summary: dict[str, Any] = {"version": version, "runs": len(table), "per_task": {}, "checks": {}}
    for metric in [column for column in table.columns if column not in {"version", "fold", "seed"}]:
        values = table[metric].dropna().to_numpy(dtype=np.float64)
        summary.setdefault("metrics", {})[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    delta_columns = [f"delta_BA_{task}" for task in TASKS]
    mean_delta = {task: float(table[f"delta_BA_{task}"].mean()) for task in TASKS}
    catastrophic = {
        task: int((table[f"delta_BA_{task}"] < -0.03).sum()) for task in TASKS
    }
    mi_relevance = table["protected_minus_nuisance_mi"].to_numpy(dtype=np.float64)
    mi_protection_retention = table["protected_harm_retention_mi"].dropna().to_numpy(dtype=np.float64)
    verifier_reduction = -table["verifier_delta_AUROC_macro"].to_numpy(dtype=np.float64)
    protected_random_gap = table["protected_harm_mi"].to_numpy(dtype=np.float64) - table["random_harm_mi"].to_numpy(dtype=np.float64)
    task_relevance = {
        task: {
            "mean_protected_minus_nuisance": float(table[f"protected_minus_nuisance_{task}"].mean()),
            "positive_runs": int((table[f"protected_minus_nuisance_{task}"] > 0).sum()),
        }
        for task in TASKS
    }
    checks = {
        "gate_A_mean_task_delta_at_least_minus_1pp": all(value >= -0.01 for value in mean_delta.values()),
        "gate_A_no_task_catastrophic_in_two_or_more_runs": all(value < 2 for value in catastrophic.values()),
        "gate_B_MI_relevance_mean_at_least_0_005": float(mi_relevance.mean()) >= 0.005,
        "gate_B_MI_relevance_positive_in_at_least_4_of_6": int((mi_relevance > 0.0).sum()) >= 4,
        "gate_B_MI_protected_harm_exceeds_random_mean": float(protected_random_gap.mean()) > 0.0,
        "gate_C_macro_nuisance_reduction_mean_at_least_0_02": float(verifier_reduction.mean()) >= 0.02,
        "gate_C_macro_nuisance_reduction_positive_in_at_least_4_of_6": int((verifier_reduction > 0).sum()) >= 4,
        "gate_D_MI_protected_retention_mean_at_least_0_50": bool(len(mi_protection_retention) and mi_protection_retention.mean() >= 0.50),
        "gate_D_MI_protected_retention_positive_in_at_least_4_of_6": int((mi_protection_retention >= 0.50).sum()) >= 4 if len(mi_protection_retention) else False,
    }
    pass_abcd = all(checks.values())
    mi_mean = mean_delta["mi"]
    mi_positive = int((table["delta_BA_mi"] > 0).sum())
    strong = pass_abcd and mi_mean >= 0.01 and mi_positive >= 5 and all(value >= 0 for value in mean_delta.values())
    viable = pass_abcd and mi_mean >= 0.005 and mi_positive >= 4 and all(value >= 0 for value in mean_delta.values())
    if strong:
        decision = "P4_SI_STRONG"
    elif viable:
        decision = "P4_SI_VIABLE"
    elif checks["gate_A_mean_task_delta_at_least_minus_1pp"] and checks["gate_A_no_task_catastrophic_in_two_or_more_runs"] and checks["gate_B_MI_relevance_mean_at_least_0_005"] and checks["gate_B_MI_relevance_positive_in_at_least_4_of_6"] and checks["gate_C_macro_nuisance_reduction_mean_at_least_0_02"] and checks["gate_C_macro_nuisance_reduction_positive_in_at_least_4_of_6"]:
        decision = "P4_SI_REPRESENTATION_ONLY"
    else:
        decision = "P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED"
    summary.update(
        {
            "decision": decision,
            "mean_task_delta": mean_delta,
            "catastrophic_run_counts_below_minus_3pp": catastrophic,
            "task_relevance": task_relevance,
            "checks": checks,
            "gate_A_to_D_pass": pass_abcd,
            "outer_test_used": False,
            "development_subject_role_limitation": "fold-local held-out subjects only; roles overlap across folds before formal lock",
        }
    )
    output_dir = OUT
    table.to_csv(output_dir / "P4_SI_DEVELOPMENT_SUMMARY.csv", index=False)
    write_json(output_dir / "P4_SI_DEVELOPMENT_SUMMARY.json", summary)
    report = [
        "# P4-SI Development Summary",
        "",
        f"- Version: `{version}`",
        f"- Development runs: `{len(table)}` (folds 0,1,2 × seeds 0,1)",
        f"- Decision: `{decision}`",
        "- Outer-test used: `false`",
        "",
        "## Task deltas",
        "",
        "| Task | Mean SI−reference BA | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        values = table[f"delta_BA_{task}"]
        report.append(f"| {task.upper()} | {values.mean():+.4f} | {values.std(ddof=1):.4f} | {values.min():+.4f} | {values.max():+.4f} |")
    report += [
        "",
        "## Pre-registered checks",
        "",
    ]
    report.extend(f"- `{key}`: `{value}`" for key, value in checks.items())
    report += [
        "",
        "## Interpretation",
        "",
        "The decision is based only on train/validation development artifacts. No outer-test signal, label, embedding, or metric was accessed.",
        "Protected/nuisance intervention sets and independent event probes are reported per run; the random intervention is a same-rank control.",
    ]
    (output_dir / "P4_SI_DEVELOPMENT_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="SI_V0")
    args = parser.parse_args()
    summarize(args.version)


if __name__ == "__main__":
    main()
