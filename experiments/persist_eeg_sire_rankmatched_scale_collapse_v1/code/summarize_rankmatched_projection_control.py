#!/usr/bin/env python3
"""Biological-subject inference for the frozen rank-matched intervention."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/persist_eeg_sire_rankmatched_scale_collapse_v1"
OUT = EXP / "outputs"
RUNTIME = EXP / "runtime"
BRIDGE = ROOT / "experiments/persist_eeg_sire_diagnostic_design_bridge_v1/outputs/subspace_decomposition/CELL_RESULTS.csv"
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
METRICS = ("BA", "macro_F1", "WS_BA")
CONDITIONS = ("Intact", "ScaleCollapse-fixed", "RandomRank16")


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(content.rstrip() + "\n", encoding="utf-8")
    temp.replace(path)


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap(values, *key):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(stable_seed("rankmatched-bootstrap", *key))
    draws = np.empty(20_000)
    for lo in range(0, len(draws), 2_000):
        hi = min(lo + 2_000, len(draws))
        ix = rng.integers(0, len(values), size=(hi - lo, len(values)))
        draws[lo:hi] = values[ix].mean(1)
    return float(values.mean()), *map(float, np.quantile(draws, [.025, .975]))


def protected_rank_audit():
    frame = pd.read_csv(BRIDGE)
    frame = frame[frame.model.isin(("OFFICIAL_FINAL_LITEBN_REFERENCE", "B1_SAME_SCALE_63"))]
    if len(frame) != 40:
        raise RuntimeError("previous decomposition missing Full/B1 5-fold cells")
    pivot = frame.pivot(index=["task", "fold"], columns="model", values=["active_rank", "protected_rank"])
    rows = []
    for (task, fold), cell in pivot.iterrows():
        full = int(cell[("protected_rank", "OFFICIAL_FINAL_LITEBN_REFERENCE")])
        b1 = int(cell[("protected_rank", "B1_SAME_SCALE_63")])
        af = int(cell[("active_rank", "OFFICIAL_FINAL_LITEBN_REFERENCE")])
        ab = int(cell[("active_rank", "B1_SAME_SCALE_63")])
        rows.append({"task": task, "fold": int(fold), "Full_protected_rank": full,
                     "B1_protected_rank": b1, "Full_active_rank": af, "B1_active_rank": ab,
                     "rank_match": "yes" if full == b1 else "no"})
    csv(OUT / "PROTECTED_RANK_AUDIT.csv", rows)
    by_task = pd.DataFrame(rows).groupby("task")
    lines = ["# Protected-rank audit", "", "Copied from the existing frozen CELL_RESULTS.csv; no decomposition was rerun.", "",
             "| Task | Full ranks | B1 ranks | Mean Full | Mean B1 | Matched folds |",
             "|---|---|---|---:|---:|---:|"]
    for task in TASKS:
        part = by_task.get_group(task).sort_values("fold")
        fr = part.Full_protected_rank.tolist(); br = part.B1_protected_rank.tolist()
        lines.append(f"| {task} | {fr} | {br} | {np.mean(fr):.1f} | {np.mean(br):.1f} | {(part.rank_match == 'yes').sum()}/5 |")
    lines += ["", "WBCIC Full ranks are [3,4,4,4,4] (mean 3.8), whereas B1 ranks are [2,1,2,4,3] (mean 2.4).",
              "B1 does not show higher absolute Protected-only utility under its own selected Protected subspace.",
              "Do not say B1's Protected subspace is intrinsically weaker: Full/B1 selected P are not cross-architecture rank matched.",
              "Within each architecture PSWA is controlled against equal-rank random subspaces; Full PSWA vs B1 PSWA is not cross-architecture rank matched."]
    text(OUT / "PROTECTED_RANK_SUMMARY.md", "\n".join(lines))
    return rows


def collect_cells():
    paths = [RUNTIME / "cells" / task / f"fold{f}_seed{s}.csv"
             for task in TASKS for f in range(5) for s in range(3)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"incomplete frozen intervention: {len(missing)} cell files missing; first={missing[0]}")
    frame = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    if frame.groupby(["task", "fold", "seed"]).size().shape[0] != 60:
        raise RuntimeError("expected 60 frozen model cells")
    for task in TASKS:
        part = frame[frame.task == task]
        expected = (14 if task != "WBCIC_MI" else 10) * (2 if task != "WBCIC_MI" else 3) * 102 * 15
        if len(part) != expected:
            raise RuntimeError(f"incomplete condition rows {task}: {len(part)} != {expected}")
    csv(OUT / "SUBJECT_SESSION_RESULTS.csv", frame)
    return frame


def aggregate_subjects(frame):
    # Fold/seed repetitions first, separately for each fixed projector.
    session = frame.groupby(["task", "subject_id", "session", "condition", "projector_id"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), repetitions=("fold", "count"))
    if not (session.repetitions == 15).all():
        raise RuntimeError("a subject-session/projector is missing fold-seed repetition")
    rows = []
    for (task, subject, condition, projector_id), one in session.groupby(
            ["task", "subject_id", "condition", "projector_id"], sort=True):
        sessions = one.set_index("session")
        required = ("S0", "S1", "S2") if task == "WBCIC_MI" else ("S1", "S2")
        if set(sessions.index) != set(required):
            raise RuntimeError("subject-session coverage mismatch")
        rows.append({"task": task, "subject_id": subject, "condition": condition,
                     "projector_id": projector_id, "BA": float(sessions.loc["S2", "BA"]),
                     "macro_F1": float(sessions.loc["S2", "macro_F1"]),
                     "WS_BA": float(sessions.BA.min())})
    csv(OUT / "SUBJECT_CONDITION_RESULTS.csv", rows)
    return pd.DataFrame(rows)


def summarize(subjects):
    task_rows = []
    effect_rows = []
    random_rows = []
    lines = ["# Same-checkpoint rank-matched ScaleCollapse control", "",
             "Frozen official Full SIRE checkpoints and original train-only normalizers; no neural training or adaptation.",
             "ScaleCollapse-fixed is an intervention on Full, not the separately trained B2 architecture.", "",
             "| Task | Metric | Intact | ScaleCollapse | Random rank-16 | Scale harm pp | Random harm pp | Excess harm pp [95% CI] | Scale harm percentile |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        one = subjects[subjects.task == task]
        intact = one[one.condition == "Intact"].set_index("subject_id")
        scale = one[one.condition == "ScaleCollapse-fixed"].set_index("subject_id")
        random = one[one.condition == "RandomRank16"]
        ids = sorted(intact.index.tolist())
        if len(ids) != (10 if task == "WBCIC_MI" else 14) or len(scale) != len(ids) or len(random) != len(ids) * 100:
            raise RuntimeError("subject/projector mismatch")
        for metric in METRICS:
            random_wide = random.pivot(index="subject_id", columns="projector_id", values=metric).loc[ids]
            i = intact.loc[ids, metric].to_numpy()
            s = scale.loc[ids, metric].to_numpy()
            r = random_wide.to_numpy()
            rmean = r.mean(1)
            excess = 100 * (rmean - s)
            sharm = 100 * (i - s)
            rharm = 100 * (i - rmean)
            mean, low, high = bootstrap(excess, task, metric)
            projector_harms = 100 * (i[:, None] - r).mean(0)
            scale_harm = float(sharm.mean())
            percentile = 100 * (np.mean(projector_harms < scale_harm) + .5 * np.mean(projector_harms == scale_harm))
            q = np.quantile(projector_harms, [.025, .25, .5, .75, .975])
            task_rows.append({"task": task, "metric": metric, "biological_subjects": len(ids),
                              "intact": float(i.mean()), "scale_collapse": float(s.mean()),
                              "random_rank16_mean": float(rmean.mean()),
                              "scale_harm_pp": scale_harm, "random_harm_pp": float(rharm.mean()),
                              "excess_harm_pp": mean, "excess_ci95_low_pp": low, "excess_ci95_high_pp": high,
                              "scale_harm_percentile_among_random": percentile,
                              "random_harm_q2_5_pp": float(q[0]), "random_harm_q25_pp": float(q[1]),
                              "random_harm_q50_pp": float(q[2]), "random_harm_q75_pp": float(q[3]),
                              "random_harm_q97_5_pp": float(q[4])})
            for k, sid in enumerate(ids):
                effect_rows.append({"task": task, "metric": metric, "subject_id": sid,
                                    "intact": i[k], "scale": s[k], "random_mean": rmean[k],
                                    "scale_harm_pp": sharm[k], "random_harm_pp": rharm[k],
                                    "excess_harm_pp": excess[k]})
            for j, harm in enumerate(projector_harms):
                random_rows.append({"task": task, "metric": metric, "projector_id": j,
                                    "random_harm_pp": float(harm), "scale_harm_pp": scale_harm})
            lines.append(f"| {task} | {metric} | {100*i.mean():.2f} | {100*s.mean():.2f} | {100*rmean.mean():.2f} | {scale_harm:+.2f} | {rharm.mean():+.2f} | {mean:+.2f} [{low:+.2f}, {high:+.2f}] | {percentile:.1f} |")
    csv(OUT / "TASK_SUMMARY.csv", task_rows)
    csv(OUT / "PAIRED_EFFECTS.csv", effect_rows)
    csv(OUT / "RANDOM_PROJECTOR_EFFECTS.csv", random_rows)
    lines += ["", "Positive excess harm means the scale-aligned rank-16 collapse damages prediction more than the mean equal-rank random projection under the same frozen decoder.",
              "The empirical percentile is descriptive, not a hypothesis-test p-value.",
              "BA, Macro-F1 and WS-BA are separately reported; WS-BA takes the within-subject session minimum after 15 checkpoint repetitions for each projector.",
              "Previous development replay showed that the diagnostics should not be promoted to an architecture-ranking objective.",
              "PERSIST motivates a general preservation constraint; SIRE instantiates one scale-specific hypothesis.", ""]
    for task in TASKS:
        selected = [r for r in task_rows if r["task"] == task]
        ba = next(r for r in selected if r["metric"] == "BA")
        if ba["excess_harm_pp"] > 0 and ba["excess_ci95_low_pp"] > 0:
            conclusion = "Scale-aligned collapse causes greater predictive harm than generic rank-matched compression under the same frozen decoder; rank reduction alone does not explain the BA effect."
        elif ba["excess_harm_pp"] < 0:
            conclusion = "Scale-aligned collapse has lower mean BA harm than generic rank-matched compression; there is no positive scale-specific evidence from this control."
        else:
            conclusion = "The rank-matched control does not resolve whether collapse harm is specific to scale identity rather than low-rank compression."
        lines.append(f"- {task}: {conclusion}")
    lines += ["", "Protected-rank details are in PROTECTED_RANK_SUMMARY.md. No selector retuning or manuscript edit was made."]
    text(OUT / "FINAL_RANKMATCHED_SCALE_COLLAPSE_REPORT.md", "\n".join(lines))
    return task_rows


def main():
    rank = protected_rank_audit()
    frame = collect_cells()
    subjects = aggregate_subjects(frame)
    summary = summarize(subjects)
    audit = pd.read_csv(OUT / "FULL_REPLAY_AUDIT.csv")
    identity = json.loads((OUT / "IDENTITY_WRAPPER_AUDIT.json").read_text())
    identity_cells = pd.read_csv(OUT / "IDENTITY_WRAPPER_CELLS.csv")
    status = "COMPLETE" if (len(audit) == 12 and audit["pass"].all() and identity["pass"]
        and identity.get("scope") == "all 60 frozen checkpoints, all evaluation trials"
        and len(identity_cells) == 60 and identity_cells["pass"].all()
        and len(summary) == 12 and len(rank) == 20) else "INCOMPLETE"
    text(OUT / "COMPLETION.json", json.dumps({"status": status,
            "official_full_checkpoints": 60, "random_rank16_projectors": 100,
            "tasks": list(TASKS), "subject_bootstrap_draws": 20000,
            "no_neural_retraining": True, "no_selector_retuning": True,
            "identity_max_abs_logit_difference": identity["max_abs_logit_difference"]}, indent=2))
    if status != "COMPLETE":
        raise RuntimeError("completion checks failed")
    print("RANKMATCHED_CONTROL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
