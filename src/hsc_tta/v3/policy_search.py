from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .action_search import subject_fold
from .probe_policy import ProbePolicy, ProbeThresholds


def evaluate_thresholds(diagnostics: pd.DataFrame, outcomes: pd.DataFrame, thresholds: ProbeThresholds,
                        epsilon: float) -> tuple[pd.DataFrame, dict[str, float]]:
    policy = ProbePolicy(thresholds); rows = []
    outcome_index = outcomes.set_index(["subject_id", "action"])
    for subject, frame in diagnostics.groupby("subject_id", sort=True):
        decision = policy.decide(frame); action = decision["selected_action"]
        reference = outcome_index.loc[(subject, frame.action.iloc[0])]
        if action == "no_tta":
            size = float(reference.source_safe_size); error = float(reference.source_argmax_error); degradation = 0.0
            gain = 0.0; available = True
        else:
            selected = outcome_index.loc[(subject, action)]
            size = float(selected.safe_size); error = float(selected.argmax_error)
            degradation = float(selected.classification_degradation); gain = float(selected.oracle_gain)
            available = bool(selected.action_available)
        rows.append({"subject_id": subject, "selected_action": action, "intervention": action != "no_tta",
                     "average_set_size": size, "argmax_error": error, "classification_degradation": degradation,
                     "set_size_gain": gain, "action_available": available, **decision})
    frame = pd.DataFrame(rows); interventions = frame[frame.intervention]
    metrics = {"mean_set_size": float(frame.average_set_size.mean()), "mean_set_size_gain": float(frame.set_size_gain.mean()),
               "mean_degradation": float(frame.classification_degradation.mean()),
               "intervention_rate": float(frame.intervention.mean()),
               "harmful_intervention_rate": float((interventions.classification_degradation > epsilon).mean()) if len(interventions) else 0.0,
               "selected_nonharm_ppv": float((interventions.classification_degradation <= epsilon).mean()) if len(interventions) else 0.0,
               "positive_selection_precision": float((interventions.set_size_gain > 0).mean()) if len(interventions) else 0.0}
    return frame, metrics


def select_thresholds(diagnostics: pd.DataFrame, outcomes: pd.DataFrame, grid: dict[str, object],
                      epsilon: float) -> tuple[ProbeThresholds, pd.DataFrame]:
    merged = diagnostics.merge(outcomes, on=["dataset", "seed", "subject_id", "action"], suffixes=("", "_future"))
    merged=merged.sort_values(["subject_id","action_cost","action"],kind="stable"); counts=merged.groupby("subject_id").size()
    if counts.nunique()!=1: raise ValueError("each subject must have the same action library")
    n_subjects=len(counts); n_actions=int(counts.iloc[0])
    def matrix(column): return merged[column].to_numpy().reshape(n_subjects,n_actions)
    available=matrix("action_available").astype(bool);r_class=matrix("r_class");update=matrix("normalized_update_magnitude")
    g_set=matrix("g_set");g_aug=matrix("g_aug");positive=matrix("positive_probe_block_fraction");mad=matrix("temporal_mad");drift=matrix("d_src")
    safe_size=matrix("safe_size");source_size=matrix("source_safe_size")[:,0];degradation=matrix("classification_degradation")
    cost=matrix("action_cost");action_rank=np.tile(np.arange(n_actions),(n_subjects,1))

    def fast_metrics(thresholds: ProbeThresholds) -> dict[str, float]:
        eligible=(available&(r_class>=thresholds.tau_class)&(update<=thresholds.tau_update)&(g_set>=thresholds.tau_set)&
                  (g_aug>=-thresholds.tau_aug_margin)&(positive>=thresholds.tau_positive_blocks)&(mad<=thresholds.tau_time_mad)&(drift<=thresholds.tau_drift))
        intervention=eligible.any(1); priority=g_set*1e12-drift*1e6-cost*1e3-action_rank
        chosen=np.where(eligible,priority,-np.inf).argmax(1); index=np.arange(n_subjects)
        selected_size=safe_size[index,chosen];selected_degradation_all=degradation[index,chosen]
        size=np.where(intervention,selected_size,source_size);selected_degradation=np.where(intervention,selected_degradation_all,0.0)
        gain=source_size-size;intervention_degradation=selected_degradation[intervention];selected_gain=gain[intervention]
        return {"mean_set_size": float(size.mean()), "mean_set_size_gain": float(gain.mean()),
                "mean_degradation": float(selected_degradation.mean()), "intervention_rate": float(intervention.mean()),
                "harmful_intervention_rate": float((intervention_degradation > epsilon).mean()) if len(intervention_degradation) else 0.0,
                "selected_nonharm_ppv": float((intervention_degradation <= epsilon).mean()) if len(intervention_degradation) else 0.0,
                "positive_selection_precision": float((selected_gain > 0).mean()) if len(selected_gain) else 0.0}
    update_q = {q: float(diagnostics.normalized_update_magnitude.replace([np.inf, -np.inf], np.nan).dropna().quantile(q))
                for q in grid["tau_update_quantile"]}
    drift_q = {q: float(diagnostics.d_src.quantile(q)) for q in grid["tau_drift_quantile"]}
    mad_q = {q: float(diagnostics.temporal_mad.quantile(q)) for q in grid["tau_time_mad_quantile"]}
    rows = []
    for tau_class, uq, dq, tau_set, aug, blocks, mq in itertools.product(
            grid["tau_class"], grid["tau_update_quantile"], grid["tau_drift_quantile"], grid["tau_set"],
            grid["tau_aug_margin"], grid["tau_positive_blocks"], grid["tau_time_mad_quantile"]):
        thresholds = ProbeThresholds(float(tau_class), update_q[uq], float(tau_set), float(aug), float(blocks), mad_q[mq], drift_q[dq])
        metrics = fast_metrics(thresholds)
        feasible = (metrics["intervention_rate"] >= float(grid["minimum_intervention_rate"]) and
                    metrics["harmful_intervention_rate"] <= float(grid["max_harmful_intervention_rate"]) and
                    metrics["mean_degradation"] <= epsilon)
        rows.append({"threshold_hash": thresholds.config_hash, "thresholds": thresholds, "feasible": feasible,
                     "tau_update_quantile": uq, "tau_drift_quantile": dq, "tau_time_mad_quantile": mq, **metrics})
    search = pd.DataFrame([{**{k: v for k, v in row.items() if k != "thresholds"},
                            **row["thresholds"].__dict__} for row in rows])
    pool = [row for row in rows if row["feasible"]] or rows
    winner = sorted(pool, key=lambda row: (not row["feasible"], row["mean_set_size"],
                                           -row["selected_nonharm_ppv"], -row["intervention_rate"], row["threshold_hash"]))[0]
    return winner["thresholds"], search


def grouped_oof_policy_search(diagnostics: pd.DataFrame, outcomes: pd.DataFrame, grid: dict[str, object],
                              epsilon: float, folds: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = []; searches = []
    subjects = sorted(diagnostics.subject_id.unique())
    assignment = {subject: subject_fold(subject, int(diagnostics.seed.iloc[0]), folds) for subject in subjects}
    for fold in range(folds):
        train = [subject for subject in subjects if assignment[subject] != fold]
        heldout = [subject for subject in subjects if assignment[subject] == fold]
        if not train or not heldout: continue
        train_d = diagnostics[diagnostics.subject_id.isin(train)]; train_o = outcomes[outcomes.subject_id.isin(train)]
        thresholds, search = select_thresholds(train_d, train_o, grid, epsilon); search["inner_fold"] = fold
        selected, _ = evaluate_thresholds(diagnostics[diagnostics.subject_id.isin(heldout)],
                                          outcomes[outcomes.subject_id.isin(heldout)], thresholds, epsilon)
        selected["inner_fold"] = fold; selected["threshold_hash"] = thresholds.config_hash
        decisions.append(selected); searches.append(search)
    return pd.concat(decisions, ignore_index=True), pd.concat(searches, ignore_index=True)
