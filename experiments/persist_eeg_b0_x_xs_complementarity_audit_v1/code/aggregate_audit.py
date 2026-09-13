#!/usr/bin/env python3
"""Subject bootstrap, XS cross-seed summary, and final audit report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".part"); frame.to_csv(temp, index=False); os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(path.name + ".part"); temp.write_text(value.rstrip() + "\n", encoding="utf-8"); os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".part"); temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temp, path)


def bootstrap(subjects: pd.DataFrame, oracle: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    joined = subjects.merge(oracle[["task", "seed", "fold", "subject_id", "candidate_name", "oracle_headroom_pp"]],
                            on=["task", "seed", "fold", "subject_id", "candidate_name"], validate="one_to_one")
    rng = np.random.default_rng(seed); rows = []
    for keys, group in joined.groupby(["task", "seed", "candidate_name"], sort=True):
        values = {
            "candidate_delta_pp": 100 * group.delta_BA.to_numpy(float),
            "oracle_headroom_pp": group.oracle_headroom_pp.to_numpy(float),
            "balanced_rescue_pp": 100 * group.balanced_rescue_rate.to_numpy(float),
            "balanced_harm_pp": 100 * group.balanced_harm_rate.to_numpy(float),
        }
        draws = rng.integers(0, len(group), size=(resamples, len(group)))
        for metric, vector in values.items():
            distribution = vector[draws].mean(axis=1)
            rows.append({"task": keys[0], "seed": int(keys[1]), "candidate": keys[2], "metric": metric,
                "estimate_pp": float(vector.mean()), "ci95_low_pp": float(np.quantile(distribution, .025)),
                "ci95_high_pp": float(np.quantile(distribution, .975)), "resamples": resamples,
                "bootstrap_unit": "real_subject", "status": "PASS"})
    return pd.DataFrame(rows)


def yn(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000); parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args(); outputs = args.repo.resolve() / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1/outputs"
    replay_cells = pd.read_csv(outputs / "REPLAY_CELL_STATUS.csv")
    if len(replay_cells) != 80:
        raise RuntimeError(f"replay-cell cardinality {len(replay_cells)} != 80")
    pairwise = pd.read_csv(outputs / "PAIRWISE_ERROR_SUMMARY.csv")
    subjects = pd.read_csv(outputs / "PAIRWISE_SUBJECT_SUMMARY.csv", dtype={"subject_id": str})
    oracle = pd.read_csv(outputs / "ORACLE_SUBJECT_RESULTS.csv", dtype={"subject_id": str})
    oracle_tasks = pd.read_csv(outputs / "ORACLE_TASK_SUMMARY.csv")
    union = pd.read_csv(outputs / "SEED0_UNION_ORACLE_SUMMARY.csv")
    boot = bootstrap(subjects, oracle, args.resamples, args.seed); atomic_csv(outputs / "PAIRWISE_SUBJECT_BOOTSTRAP.csv", boot)

    xs_subjects = subjects[subjects.candidate_name.eq("LiteBN_XS")]
    xs_oracle = oracle[oracle.candidate_name.eq("LiteBN_XS")]
    xs_seed = xs_subjects.groupby(["task", "seed"], as_index=False).agg(
        net_delta_pp=("delta_BA", lambda x: 100 * x.mean()), balanced_rescue_pp=("balanced_rescue_rate", lambda x: 100 * x.mean()),
        balanced_harm_pp=("balanced_harm_rate", lambda x: 100 * x.mean()))
    xs_seed = xs_seed.merge(xs_oracle.groupby(["task", "seed"], as_index=False).oracle_headroom_pp.mean(), on=["task", "seed"], validate="one_to_one")
    cross = xs_seed.groupby("task", as_index=False).agg(mean_XS_net_delta_pp=("net_delta_pp", "mean"),
        mean_oracle_headroom_pp=("oracle_headroom_pp", "mean"), min_oracle_headroom_pp=("oracle_headroom_pp", "min"),
        max_oracle_headroom_pp=("oracle_headroom_pp", "max"), mean_balanced_rescue_pp=("balanced_rescue_pp", "mean"),
        mean_balanced_harm_pp=("balanced_harm_pp", "mean"))
    xs_full = oracle_tasks[oracle_tasks.candidate_name.eq("LiteBN_XS")].groupby("task").status.apply(lambda x: bool(x.eq("PASS").all()))
    cross["all_seeds_worth_investigating"] = cross.min_oracle_headroom_pp.ge(1.0) & cross.task.map(xs_full).fillna(False)
    cross["status"] = np.where(cross.task.map(xs_full).fillna(False), "PASS", "PROTOCOL_PARTIAL")
    atomic_csv(outputs / "XS_CROSSSEED_COMPLEMENTARITY_SUMMARY.csv", cross)

    table = pairwise.merge(oracle_tasks, left_on=["task", "seed", "candidate"], right_on=["task", "seed", "candidate_name"], validate="one_to_one")
    ci = boot[boot.metric.eq("oracle_headroom_pp")][["task", "seed", "candidate", "ci95_low_pp", "ci95_high_pp"]]
    table = table.merge(ci, on=["task", "seed", "candidate"], validate="one_to_one")
    lines = ["EXPERIMENT_TYPE = ANALYSIS_ONLY", "", "NEW_MODEL_TRAINED = NO", "", "FINAL_HELDOUT_ACCESSED = NO", "",
        "INTERNAL_HELDOUT_ACCESSED = NO", "", "DEVELOPMENT_OUTER_ONLY = YES", "", "# Final complementarity report", "",
        "The oracle values below are label-informed analytical upper bounds, not deployable performance.", "",
        "| Task | Candidate | Seed | B0 BA | Candidate BA | Net Δ pp | Balanced Rescue pp | Balanced Harm pp | Oracle BA | Oracle Headroom pp | Headroom 95% CI | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|" ]
    for row in table.sort_values(["task", "candidate", "seed"]).itertuples(index=False):
        lines.append(f"| {row.task} | {row.candidate} | {int(row.seed)} | {row.B0_BA:.6f} | {row.candidate_BA:.6f} | {row.candidate_delta_pp:+.3f} | {100*row.balanced_rescue_rate:.3f} | {100*row.balanced_harm_rate:.3f} | {row.oracle_BA:.6f} | {row.SUBJECT_EQUAL_ORACLE_HEADROOM_PP:.3f} | [{row.ci95_low_pp:.3f}, {row.ci95_high_pp:.3f}] | {row.status_y} ({int(row.eligible_folds_y)}/5 folds) |")
    lines += ["", "## Descriptive paired rates", "", "| Task | Candidate | Seed | Rescue/all | Harm/all | Rescue given B0 wrong | Harm given B0 correct |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in pairwise.sort_values(["task", "candidate", "seed"]).itertuples(index=False):
        lines.append(f"| {row.task} | {row.candidate} | {int(row.seed)} | {row.rescue_fraction_all:.6f} | {row.harm_fraction_all:.6f} | {row.rescue_given_B0_wrong:.6f} | {row.harm_given_B0_correct:.6f} |")
    lines += ["", "## XS cross-seed summary", "", "| Task | Mean net Δ pp | Mean oracle headroom pp | Min | Max | Mean balanced rescue pp | Mean balanced harm pp | All seeds >=1 pp | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in cross.itertuples(index=False):
        lines.append(f"| {row.task} | {row.mean_XS_net_delta_pp:+.3f} | {row.mean_oracle_headroom_pp:.3f} | {row.min_oracle_headroom_pp:.3f} | {row.max_oracle_headroom_pp:.3f} | {row.mean_balanced_rescue_pp:.3f} | {row.mean_balanced_harm_pp:.3f} | {yn(bool(row.all_seeds_worth_investigating))} | {row.status} |")
    lines += ["", "## Seed0 B0+X+XS union oracle", "", "The common B0 is the exact X seed-0 historical B0 cell. The XS candidate checkpoint and normalizer are identical to the seed-0 XS development replay; this avoids mixing incompatible baselines.", "",
        "| Task | B0 BA | X oracle headroom pp | XS oracle headroom on common B0 pp | Union headroom pp | Gain over best single pp | X-only rescue | XS-only rescue | Both rescue | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in union.itertuples(index=False):
        lines.append(f"| {row.task} | {row.B0_BA:.6f} | {row.X_oracle_headroom_pp:.3f} | {row.XS_oracle_headroom_on_union_B0_pp:.3f} | {row.union_oracle_headroom_pp:.3f} | {row.union_gain_over_best_single_pp:.3f} | {int(row.X_only_rescue)} | {int(row.XS_only_rescue)} | {int(row.X_and_XS_both_rescue)} | {row.status} ({int(row.eligible_folds)}/5 folds) |")

    x_tasks = oracle_tasks[oracle_tasks.candidate_name.eq("LiteBN_X")]; xs_tasks = oracle_tasks[oracle_tasks.candidate_name.eq("LiteBN_XS")]
    x_worth = x_tasks.SUBJECT_EQUAL_ORACLE_HEADROOM_PP.ge(1.0) & x_tasks.status.eq("PASS")
    xs_worth = xs_tasks.SUBJECT_EQUAL_ORACLE_HEADROOM_PP.ge(1.0) & xs_tasks.status.eq("PASS")
    cross_consistent = cross.min_oracle_headroom_pp.ge(1.0) & cross.status.eq("PASS")
    failed = replay_cells[replay_cells.status.ne("PASS")]
    lines += ["", "## Replay exclusions", "", f"{len(failed)}/80 cells failed the fixed deterministic full-FP32 replay gate and were excluded from every statistic. Partial rows are descriptive estimates over the remaining eligible outer subjects, not complete task estimates.", "",
        "| Comparison | Task | Seed | Fold | Status |", "|---|---|---:|---:|---|"]
    for row in failed.itertuples(index=False):
        lines.append(f"| {row.comparison} | {row.task} | {int(row.seed)} | {int(row.fold)} | {row.status} |")
    lines += ["", "## Scientific interpretation", "",
        f"1. X has >=1.0 pp oracle rescue headroom on {int(x_worth.sum())}/4 tasks; meaningful complementarity is {'present' if x_worth.any() else 'not observed'} on the development outer cohort.",
        f"2. XS has >=1.0 pp oracle rescue headroom on {int(xs_worth.sum())}/{len(xs_worth)} task-seed cells; meaningful complementarity is {'present' if xs_worth.any() else 'not observed'}.",
        f"3. XS clears 1.0 pp in every seed on {int(cross_consistent.sum())}/4 tasks; other task signals are seed-isolated or below the practical threshold.",
        "4. Per-task/seed oracle headroom is reported in the primary table; cross-seed ranges are reported in the XS table.",
        f"5. The seed0 union exceeds the better single-candidate oracle on {int(((union.union_gain_over_best_single_pp > 1e-12) & union.status.eq('PASS')).sum())}/3 complete tasks; the partial ERP estimate is reported separately and is not counted as complete evidence.",
        f"6. Based only on oracle headroom, the next stable-rescue/predictability analysis is worth investigating for {int(x_worth.groupby(x_tasks.task).any().sum() + cross_consistent.sum())} candidate/task cases under the stated 1.0 pp threshold; this audit launches no next-stage model."]
    atomic_text(outputs / "FINAL_COMPLEMENTARITY_REPORT.md", "\n".join(lines))
    summary = {"EXPERIMENT_TYPE": "ANALYSIS_ONLY", "NEW_MODEL_TRAINED": "NO", "FINAL_HELDOUT_ACCESSED": "NO",
        "INTERNAL_HELDOUT_ACCESSED": "NO", "DEVELOPMENT_OUTER_ONLY": "YES", "replay_cells": 80,
        "replay_passed_cells": int(replay_cells.status.eq("PASS").sum()), "replay_excluded_cells": int(replay_cells.status.ne("PASS").sum()),
        "replay_status": "PASS" if replay_cells.status.eq("PASS").all() else "PROTOCOL_PARTIAL",
        "pairwise": table.to_dict(orient="records"), "xs_crossseed": cross.to_dict(orient="records"), "seed0_union": union.to_dict(orient="records"),
        "practical_threshold_oracle_headroom_pp": 1.0}
    atomic_json(outputs / "FINAL_COMPLEMENTARITY_SUMMARY.json", summary)
    print("COMPLEMENTARITY_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
