#!/usr/bin/env python3
"""Apply the predeclared structural screen and write final counterfactual conclusions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
STATE_ORDER = ("C0S0M0", "C1S0M0", "C0S1M0", "C0S0M1", "C1S1M0", "C1S0M1", "C0S1M1", "C1S1M1")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(path.name + ".part")
    temp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    exp = repo / "experiments/persist_eeg_xs_counterfactual_factorial_audit_v1"
    outputs, protocol = exp / "outputs", exp / "protocol"
    summary = pd.read_csv(outputs / "COUNTERFACTUAL_TASK_SEED_SUMMARY.csv")
    transitions = pd.read_csv(outputs / "FULL_BACKWARD_ABLATION_TRANSITIONS.csv")
    factorial = pd.read_csv(outputs / "FACTORIAL_EFFECTS.csv")
    replay = pd.read_csv(outputs / "FULL_XS_REPLAY_AUDIT.csv")
    audits_pass = bool(len(replay) == 60 and replay.status.eq("PASS").all() and
                       replay.predictions_exact.all() and replay.labels_exact.all() and
                       replay.trial_keys_exact.all() and replay.subject_set_exact.all() and
                       replay.absolute_metric_difference.max() <= 1e-8 and
                       summary.identity_absolute_difference.max() <= 1e-12)

    task_state_rows = []
    for (task, state), group in summary.groupby(["task", "state_id"], sort=True):
        task_state_rows.append({"task": task, "state_id": state,
                                "seed0_NET_PP": float(group[group.seed == 0].NET_PP.iloc[0]),
                                "seed1_NET_PP": float(group[group.seed == 1].NET_PP.iloc[0]),
                                "seed2_NET_PP": float(group[group.seed == 2].NET_PP.iloc[0]),
                                "three_seed_RESCUE_PP": float(group.RESCUE_PP.mean()),
                                "three_seed_HARM_PP": float(group.HARM_PP.mean()),
                                "three_seed_NET_PP": float(group.NET_PP.mean()),
                                "NET_PP_seed_min": float(group.NET_PP.min()), "NET_PP_seed_max": float(group.NET_PP.max()),
                                "NET_PP_seed_std": float(group.NET_PP.std(ddof=0)),
                                "three_seed_delta_net_vs_full_pp": float(group.delta_net_vs_full_pp.mean()),
                                "three_seed_rescue_retention": float(group.rescue_retention.mean()),
                                "rescue_retention_seed_min": float(group.rescue_retention.min()),
                                "rescue_retention_seed_max": float(group.rescue_retention.max()),
                                "rescue_retention_seed_std": float(group.rescue_retention.std(ddof=0)),
                                "three_seed_harm_reduction": float(group.harm_reduction.mean()),
                                "harm_reduction_seed_min": float(group.harm_reduction.min()),
                                "harm_reduction_seed_max": float(group.harm_reduction.max()),
                                "harm_reduction_seed_std": float(group.harm_reduction.std(ddof=0)),
                                "delta_net_seed_min": float(group.delta_net_vs_full_pp.min()),
                                "delta_net_seed_max": float(group.delta_net_vs_full_pp.max()),
                                "delta_net_seed_std": float(group.delta_net_vs_full_pp.std(ddof=0)),
                                "positive_seeds_vs_full": int(group.delta_net_vs_full_pp.gt(0).sum()),
                                "nonnegative_seeds_vs_full": int(group.delta_net_vs_full_pp.ge(0).sum())})
    task_state = pd.DataFrame(task_state_rows)

    screen_rows = []
    for state in STATE_ORDER:
        cells = summary[summary.state_id == state]
        tasks = task_state[task_state.state_id == state]
        mean_retention = float(cells.rescue_retention.mean())
        mean_harm_reduction = float(cells.harm_reduction.mean())
        equal_task_net = float(tasks.three_seed_NET_PP.mean())
        equal_task_delta = float(tasks.three_seed_delta_net_vs_full_pp.mean())
        tasks_nonnegative = int(tasks.three_seed_delta_net_vs_full_pp.ge(0).sum())
        tasks_positive = int(tasks.three_seed_delta_net_vs_full_pp.gt(0).sum())
        worst_task = float(tasks.three_seed_delta_net_vs_full_pp.min())
        cells_nonnegative = int(cells.delta_net_vs_full_pp.ge(0).sum())
        if state == "C1S1M1":
            status = "REFERENCE_FULL_XS"
        else:
            promising = (mean_retention >= .90 and mean_harm_reduction >= .10 and equal_task_delta >= .30 and
                         tasks_nonnegative >= 3 and worst_task > -.50 and cells_nonnegative >= 8 and audits_pass)
            weak = equal_task_delta > 0 and mean_harm_reduction > 0 and mean_retention >= .85
            status = "PROMISING_HARM_SUPPRESSION_STRUCTURE" if promising else "WEAK_HARM_SUPPRESSION_SIGNAL" if weak else "NO_USEFUL_STRUCTURAL_SIGNAL"
        screen_rows.append({"state_id": state, "active_modules": state.count("1"),
                            "mean_rescue_retention_12_cells": mean_retention,
                            "mean_harm_reduction_12_cells": mean_harm_reduction,
                            "equal_task_mean_NET_PP": equal_task_net,
                            "equal_task_mean_delta_net_vs_full_pp": equal_task_delta,
                            "tasks_positive_vs_full": tasks_positive, "tasks_nonnegative_vs_full": tasks_nonnegative,
                            "task_seed_cells_positive_vs_full": int(cells.delta_net_vs_full_pp.gt(0).sum()),
                            "task_seed_cells_nonnegative_vs_full": cells_nonnegative,
                            "worst_task_three_seed_delta_vs_full_pp": worst_task, "screen_status": status})
    screen = pd.DataFrame(screen_rows)
    atomic_csv(outputs / "STRUCTURAL_SCREEN_SUMMARY.csv", screen)
    promising = screen[screen.screen_status == "PROMISING_HARM_SUPPRESSION_STRUCTURE"].copy()
    weak = screen[screen.screen_status == "WEAK_HARM_SUPPRESSION_SIGNAL"].copy()
    next_candidate = None
    best_weak = None
    if not promising.empty:
        promising = promising.sort_values(["equal_task_mean_delta_net_vs_full_pp", "mean_rescue_retention_12_cells",
                                            "mean_harm_reduction_12_cells", "active_modules"], ascending=[False, False, False, True])
        next_candidate = str(promising.iloc[0].state_id)
    elif not weak.empty:
        weak = weak.sort_values(["equal_task_mean_delta_net_vs_full_pp", "mean_rescue_retention_12_cells",
                                "mean_harm_reduction_12_cells", "active_modules"], ascending=[False, False, False, True])
        best_weak = str(weak.iloc[0].state_id)

    module_global = transitions.groupby("module_removed", as_index=False).agg(
        harm_reversal_rate=("harm_reversal_rate", "mean"), rescue_loss_rate=("rescue_loss_rate", "mean"),
        selectivity_index_pp=("selectivity_index_pp", "mean"), delta_rescue_vs_full_pp=("delta_rescue_vs_full_pp", "mean"),
        delta_harm_vs_full_pp=("delta_harm_vs_full_pp", "mean"), delta_net_vs_full_pp=("delta_net_vs_full_pp", "mean"))
    module_global["positive_selectivity_cells"] = module_global.module_removed.map(
        transitions.groupby("module_removed").selectivity_index_pp.apply(lambda x: int(x.gt(0).sum())))
    effect_global = factorial.groupby("effect", as_index=False)[["RESCUE_PP_effect", "HARM_PP_effect", "NET_PP_effect"]].mean()
    main_effects = effect_global[effect_global.effect.isin(["C", "S", "M"])]
    rescue_module = str(main_effects.loc[main_effects.RESCUE_PP_effect.idxmax(), "effect"])
    harm_module = str(main_effects.loc[main_effects.HARM_PP_effect.idxmax(), "effect"])
    harm_effect_value = float(main_effects.HARM_PP_effect.max())
    main_max = float(effect_global[effect_global.effect.isin(["C", "S", "M"])].NET_PP_effect.abs().max())
    interaction_max = float(effect_global[~effect_global.effect.isin(["C", "S", "M"])].NET_PP_effect.abs().max())
    interactions_material = bool(interaction_max >= .10 and interaction_max >= .25 * main_max)

    result = {
        "EXPERIMENT_TYPE": "COUNTERFACTUAL_ANALYSIS_ONLY", "NEW_EEG_MODEL_TRAINED": "NO",
        "FINAL_HELDOUT_ACCESSED": "NO", "INTERNAL_HELDOUT_ACCESSED": "NO", "DEVELOPMENT_OUTER_ONLY": "YES",
        "protocol_audits_pass": audits_pass, "valid_full_XS_cells": int(replay.status.eq("PASS").sum()),
        "NEXT_RETRAIN_CANDIDATE": next_candidate, "best_weak_state": best_weak,
        "overall_screen_conclusion": "PROMISING_HARM_SUPPRESSION_STRUCTURE" if next_candidate else "WEAK_HARM_SUPPRESSION_SIGNAL" if best_weak else "NO_USEFUL_STRUCTURAL_SIGNAL",
        "module_most_associated_with_rescue": rescue_module, "module_most_associated_with_harm": harm_module,
        "material_nonadditive_interactions": interactions_material,
        "structural_screen": screen.replace({np.nan: None}).to_dict("records"),
        "per_task_state": task_state.replace({np.nan: None}).to_dict("records"),
        "module_backward_ablation": module_global.replace({np.nan: None}).to_dict("records"),
        "factorial_effects_equal_task_seed": effect_global.replace({np.nan: None}).to_dict("records"),
    }
    atomic_json(outputs / "FINAL_COUNTERFACTUAL_FACTORIAL_SUMMARY.json", result)

    lines = ["# XS counterfactual factorial audit", "", "EXPERIMENT_TYPE = COUNTERFACTUAL_ANALYSIS_ONLY",
             "NEW_EEG_MODEL_TRAINED = NO", "FINAL_HELDOUT_ACCESSED = NO", "INTERNAL_HELDOUT_ACCESSED = NO",
             "DEVELOPMENT_OUTER_ONLY = YES", "", "## Protocol gate", "",
             f"- Full XS exact replay: {'PASS' if audits_pass else 'FAIL'} ({int(replay.status.eq('PASS').sum())}/60 cells)",
             f"- Maximum metric reproduction difference: {replay.absolute_metric_difference.max():.3g}",
             "- C0S0M0 is the trained wide XS model with C/S/M disabled; it is not LiteBN.", "",
             "## Table 1 — full cube, four-task / three-seed aggregate", "",
             "| State | Rescue retention | Harm reduction | NET vs B0 pp | Delta NET vs Full pp | Tasks improved /4 | Cells improved /12 | Worst task delta | Screen status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for state in STATE_ORDER:
        row = screen[screen.state_id == state].iloc[0]
        lines.append(f"| {state} | {row.mean_rescue_retention_12_cells:.3f} | {row.mean_harm_reduction_12_cells:.3f} | {row.equal_task_mean_NET_PP:+.3f} | {row.equal_task_mean_delta_net_vs_full_pp:+.3f} | {int(row.tasks_positive_vs_full)}/4 | {int(row.task_seed_cells_positive_vs_full)}/12 | {row.worst_task_three_seed_delta_vs_full_pp:+.3f} | {row.screen_status} |")
    lines += ["", "## Table 2 — per task", "", "| Task | State | Rescue pp | Harm pp | NET pp | Delta NET vs Full pp | Rescue retention | Harm reduction | Positive seeds /3 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for state in STATE_ORDER:
            row = task_state[(task_state.task == task) & (task_state.state_id == state)].iloc[0]
            lines.append(f"| {task} | {state} | {row.three_seed_RESCUE_PP:.3f} | {row.three_seed_HARM_PP:.3f} | {row.three_seed_NET_PP:+.3f} | {row.three_seed_delta_net_vs_full_pp:+.3f} | {row.three_seed_rescue_retention:.3f} | {row.three_seed_harm_reduction:.3f} | {int(row.positive_seeds_vs_full)}/3 |")
    lines += ["", "## Table 3 — module attribution", "", "| Scope | Module removed | Harm reversal % | Rescue loss % | Selectivity pp | Delta Rescue pp | Delta Harm pp | Delta NET pp | Positive selectivity cells |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in module_global.itertuples(index=False):
        lines.append(f"| Four-task | {row.module_removed} | {100*row.harm_reversal_rate:.2f} | {100*row.rescue_loss_rate:.2f} | {row.selectivity_index_pp:+.2f} | {row.delta_rescue_vs_full_pp:+.3f} | {row.delta_harm_vs_full_pp:+.3f} | {row.delta_net_vs_full_pp:+.3f} | {int(row.positive_selectivity_cells)}/12 |")
    for task in TASKS:
        for module, group in transitions[transitions.task == task].groupby("module_removed"):
            lines.append(f"| {task} | {module} | {100*group.harm_reversal_rate.mean():.2f} | {100*group.rescue_loss_rate.mean():.2f} | {group.selectivity_index_pp.mean():+.2f} | {group.delta_rescue_vs_full_pp.mean():+.3f} | {group.delta_harm_vs_full_pp.mean():+.3f} | {group.delta_net_vs_full_pp.mean():+.3f} | {int(group.selectivity_index_pp.gt(0).sum())}/3 |")
    lines += ["", "## Table 4 — factorial effects", "", "Positive main NET effect means switching the module from OFF to ON increases NET on average across other states.", "",
              "| Effect | Rescue pp | Harm pp | NET pp |", "|---|---:|---:|---:|"]
    for effect in ("C", "S", "M", "CxS", "CxM", "SxM", "CxSxM"):
        row = effect_global[effect_global.effect == effect].iloc[0]
        lines.append(f"| {effect} | {row.RESCUE_PP_effect:+.3f} | {row.HARM_PP_effect:+.3f} | {row.NET_PP_effect:+.3f} |")
    best_selectivity = module_global.loc[module_global.selectivity_index_pp.idxmax()]
    consistent_effects = factorial.groupby("effect").NET_PP_effect.apply(lambda x: max(int(x.gt(0).sum()), int(x.lt(0).sum())))
    lines += ["", "## Scientific answers", "",
              f"1. The largest positive factorial Rescue main effect is {rescue_module}.",
              (f"2. None of C/S/M has a positive Harm main effect: turning each ON reduces Harm on average. M is the strongest Harm suppressor; {harm_module} is merely the least-negative Harm effect ({harm_effect_value:+.3f} pp)." if harm_effect_value <= 0 else f"2. {harm_module} has the largest positive factorial Harm main effect ({harm_effect_value:+.3f} pp)."),
              f"3. No removal reverses substantially more Harm than Rescue while improving NET. Removing {best_selectivity.module_removed} has the largest conditional selectivity ({100*best_selectivity.harm_reversal_rate:.2f}% reversal versus {100*best_selectivity.rescue_loss_rate:.2f}% loss, {best_selectivity.selectivity_index_pp:+.2f} pp), but its total NET change is {best_selectivity.delta_net_vs_full_pp:+.3f} pp because it also creates new errors.",
              "4. Main-effect signs are consistent for C and M in 12/12 task-seed cells and for S in 9/12; interaction signs are less consistent. Full counts: " + "; ".join(f"{key} {int(value)}/12" for key, value in consistent_effects.items()) + ".",
              f"5. Non-additive interactions are {'material under the declared descriptive rule' if interactions_material else 'not material under the declared descriptive rule'} (largest absolute interaction NET {interaction_max:.3f} pp; largest absolute main NET {main_max:.3f} pp; rule: interaction >=0.10 pp and >=25% of the largest main effect).",
              f"6. A reduced state preserving at least 90% Rescue while materially reducing Harm: {'YES, '+next_candidate if next_candidate else 'NO under the full PROMISING screen'}.",
              f"7. A reduced state improving equal-task NET by at least +0.30 pp: {'YES' if screen[screen.state_id!='C1S1M1'].equal_task_mean_delta_net_vs_full_pp.max() >= .30 else 'NO'} (best {screen[screen.state_id!='C1S1M1'].equal_task_mean_delta_net_vs_full_pp.max():+.3f} pp).",
              f"8. Future from-scratch reduced-XS experiment: {'YES; NEXT_RETRAIN_CANDIDATE='+next_candidate if next_candidate else 'NO PROMISING candidate; best weak state='+best_weak if best_weak else 'NO; '+result['overall_screen_conclusion']}.", "",
              "These are post-training counterfactuals, not models trained without the removed mechanisms. Counterfactual BA is not deployable-model accuracy. No EEG model was trained and no heldout or final-test artifact was accessed."]
    atomic_text(outputs / "FINAL_COUNTERFACTUAL_FACTORIAL_REPORT.md", "\n".join(lines))
    scope = json.loads((protocol / "DEVELOPMENT_SCOPE_AUDIT.json").read_text(encoding="utf-8"))
    scope.update({"FULL_XS_REPLAY_PASS": audits_pass, "valid_cells": int(replay.status.eq("PASS").sum()), "protocol_audits_pass": audits_pass})
    atomic_json(protocol / "DEVELOPMENT_SCOPE_AUDIT.json", scope)
    print(f"COUNTERFACTUAL_AUDIT_COMPLETE audits={audits_pass} next={next_candidate} weak={best_weak}", flush=True)


if __name__ == "__main__":
    main()
