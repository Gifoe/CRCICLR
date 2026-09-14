#!/usr/bin/env python3
"""Aggregate the completed TFFormer four-task seed0/1/2 diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path("/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")
EXP = REPO / "experiments/persist_eeg_litebn_tfformer_multiseed_v1"
OUT = EXP / "outputs"
SEED0_SSVEP = Path(
    "/root/rivermind-data/CRCICLR_TFF_WORK/experiments/"
    "persist_eeg_litebn_tfformer_v1/outputs/SEED0_TASK_RESULTS.csv"
)
SEED0_REMAINING = REPO / (
    "experiments/persist_eeg_litebn_tfformer_remaining_tasks_seed0_v1/"
    "outputs/SEED0_HELDOUT_RESULTS.csv"
)
SEED12 = {
    seed: OUT / f"seed{seed}" / f"SEED{seed}_HELDOUT_RESULTS.csv"
    for seed in (1, 2)
}
TASKS = ("OpenBMI_SSVEP", "OpenBMI_ERP", "OpenBMI_MI", "WBCIC_MI")
BENCHMARKS = {
    "OpenBMI_SSVEP": 0.9234,
    "OpenBMI_ERP": 0.8557,
    "OpenBMI_MI": 0.7555,
    "WBCIC_MI": 0.7918,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def markdown_table(frame: pd.DataFrame, decimals: int = 6) -> list[str]:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    seed0_ssvep_all = pd.read_csv(SEED0_SSVEP)
    source = seed0_ssvep_all.loc[
        seed0_ssvep_all["task"] == "OpenBMI_SSVEP"
    ].iloc[0]
    rows = [
        {
            "seed": 0,
            "task": "OpenBMI_SSVEP",
            "TFFormer": float(source["TFFormer"]),
            "LiteBN": float(source["matched_LiteBN"]),
            "delta_vs_LiteBN_pp": 100.0
            * (float(source["TFFormer"]) - float(source["matched_LiteBN"])),
            "benchmark": float(source["benchmark_best"]),
            "delta_vs_benchmark_pp": 100.0
            * (float(source["TFFormer"]) - float(source["benchmark_best"])),
            "PASS": bool(float(source["TFFormer"]) > float(source["benchmark_best"])),
        }
    ]
    seed0_remaining = pd.read_csv(SEED0_REMAINING)
    seed0_remaining.insert(0, "seed", 0)
    rows.extend(seed0_remaining.to_dict("records"))
    for seed, path in SEED12.items():
        frame = pd.read_csv(path)
        if set(frame["seed"].astype(int)) != {seed}:
            raise RuntimeError(f"seed mismatch in {path}")
        rows.extend(frame.to_dict("records"))

    results = pd.DataFrame(rows)
    results["seed"] = results["seed"].astype(int)
    results["PASS"] = results["PASS"].astype(bool)
    results = results.sort_values(["seed", "task"]).reset_index(drop=True)
    expected = {(seed, task) for seed in (0, 1, 2) for task in TASKS}
    actual = set(zip(results["seed"], results["task"]))
    if actual != expected or len(results) != 12:
        raise RuntimeError(f"incomplete result matrix: missing={expected-actual}")
    for row in results.itertuples(index=False):
        if abs(row.delta_vs_LiteBN_pp - 100 * (row.TFFormer - row.LiteBN)) > 1e-9:
            raise RuntimeError(f"LiteBN delta mismatch for seed{row.seed} {row.task}")
        if abs(row.benchmark - BENCHMARKS[row.task]) > 1e-12:
            raise RuntimeError(f"benchmark mismatch for {row.task}")

    task_rows = []
    for task in TASKS:
        frame = results.loc[results["task"] == task]
        candidate_mean = float(frame["TFFormer"].mean())
        task_rows.append(
            {
                "task": task,
                "TFFormer_mean": candidate_mean,
                "TFFormer_std": float(frame["TFFormer"].std(ddof=1)),
                "LiteBN_mean": float(frame["LiteBN"].mean()),
                "LiteBN_std": float(frame["LiteBN"].std(ddof=1)),
                "paired_delta_mean_pp": float(frame["delta_vs_LiteBN_pp"].mean()),
                "paired_delta_std_pp": float(frame["delta_vs_LiteBN_pp"].std(ddof=1)),
                "benchmark": BENCHMARKS[task],
                "delta_mean_vs_benchmark_pp": 100
                * (candidate_mean - BENCHMARKS[task]),
                "benchmark_pass_seeds": int(frame["PASS"].sum()),
            }
        )
    task_aggregate = pd.DataFrame(task_rows)
    seed_aggregate = (
        results.groupby("seed", as_index=False)
        .agg(
            equal_task_TFFormer_mean=("TFFormer", "mean"),
            equal_task_LiteBN_mean=("LiteBN", "mean"),
            equal_task_delta_pp=("delta_vs_LiteBN_pp", "mean"),
            benchmark_pass_tasks=("PASS", "sum"),
        )
        .sort_values("seed")
    )
    equal_task_delta = float(task_aggregate["paired_delta_mean_pp"].mean())
    passed_task_means = int((task_aggregate["delta_mean_vs_benchmark_pp"] > 0).sum())

    results.to_csv(OUT / "THREE_SEED_HELDOUT_RESULTS.csv", index=False)
    task_aggregate.to_csv(OUT / "THREE_SEED_TASK_AGGREGATE.csv", index=False)
    seed_aggregate.to_csv(OUT / "THREE_SEED_SEED_AGGREGATE.csv", index=False)
    provenance = {
        "seed0_OpenBMI_SSVEP": {
            "path": str(SEED0_SSVEP),
            "sha256": sha256(SEED0_SSVEP),
            "training_schedule": "historical full-60-epoch seed0 run",
        },
        "seed0_remaining_tasks": {
            "path": str(SEED0_REMAINING),
            "sha256": sha256(SEED0_REMAINING),
            "training_schedule": "ERP fold0 full 60; later folds min10 patience8",
        },
        **{
            f"seed{seed}": {
                "path": str(path),
                "sha256": sha256(path),
                "training_schedule": "all folds min10 patience8",
            }
            for seed, path in SEED12.items()
        },
        "internal_heldout_diagnostic": True,
        "new_sealed_test_accessed": False,
        "uniform_training_schedule_across_seeds": False,
    }
    write_json(OUT / "THREE_SEED_PROVENANCE.json", provenance)

    decision = {
        "terminal": "STOP_MODEL",
        "reason": (
            "Only WBCIC MI exceeds its hard benchmark in the three-seed mean; "
            "OpenBMI SSVEP, ERP, and MI remain below benchmark. OpenBMI MI also "
            "has a negative mean paired delta versus matched LiteBN."
        ),
        "equal_task_mean_paired_delta_vs_LiteBN_pp": equal_task_delta,
        "task_means_above_hard_benchmark": passed_task_means,
        "total_tasks": 4,
        "protocol_uniform_across_seeds": False,
        "protocol_limitation": (
            "Seed0 SSVEP used the historical full-60-epoch schedule while seed1/2 "
            "used user-requested min10/patience8 early stopping. Treat this as an "
            "internal multiseed diagnostic, not a strict uniform-protocol estimate."
        ),
        "internal_heldout_diagnostic": True,
        "new_sealed_test_accessed": False,
    }
    write_json(OUT / "FINAL_DECISION.json", decision)

    report = [
        "# TFFormer four-task, three-seed diagnostic",
        "",
        "## Per-seed internal-heldout results",
        "",
        *markdown_table(results),
        "",
        "## Three-seed task aggregates",
        "",
        *markdown_table(task_aggregate),
        "",
        "## Equal-task aggregate by seed",
        "",
        *markdown_table(seed_aggregate),
        "",
        f"Equal-task mean paired delta versus matched LiteBN: {equal_task_delta:+.6f} pp.",
        f"Task means above hard benchmark: {passed_task_means}/4.",
        "",
        "## Decision",
        "",
        "`STOP_MODEL`",
        "",
        decision["reason"],
        "",
        "Protocol limitation: " + decision["protocol_limitation"],
        "",
        "These are internal-heldout diagnostics. No new sealed test was accessed.",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
