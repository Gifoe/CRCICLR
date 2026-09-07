from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from build_ensemble_actions import action_menu_mask
from evaluation import by_session_class, concentration_rows, evaluate_prediction
from v3_common import DIAGNOSTICS, FIGURES, pool_mask, write_csv, write_json


ACTION_MENU_IDS = (
    "PROTECTED_SAFE_GLOBAL",
    "FULL_GLOBAL",
    "PROTECTED_SAFE_SINGLE_REPLACEMENT",
    "FULL_SINGLE_REPLACEMENT",
)


def oracle_from_candidates(
    trials: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    method_id: str,
) -> pd.DataFrame:
    labels = trials.outcome_label.to_numpy(dtype=int)
    reference = trials.y_keep_ens.to_numpy(dtype=int)
    selected_id = np.full(len(trials), "B6_ALL_RUN_LOGIT_MEAN", dtype=object)
    selected_family = np.full(len(trials), "KEEP", dtype=object)
    selected_scope = np.full(len(trials), "REFERENCE", dtype=object)
    prediction = reference.copy()

    ordered = candidates.copy()
    if "priority" in ordered:
        ordered = ordered.sort_values(["trial_index", "priority", "candidate_id"])
    else:
        scope_order = pd.Categorical(
            ordered.action_scope,
            categories=["REFERENCE", "GLOBAL", "SINGLE_REPLACEMENT"],
            ordered=True,
        )
        ordered = ordered.assign(_scope_order=scope_order).sort_values(
            ["trial_index", "_scope_order", "candidate_id"]
        )
    for row in ordered.itertuples(index=False):
        index = int(row.trial_index)
        if prediction[index] == labels[index]:
            continue
        if int(row.prediction) == labels[index]:
            prediction[index] = int(row.prediction)
            selected_id[index] = str(row.candidate_id)
            selected_family[index] = str(
                getattr(row, "action_family", getattr(row, "keep_family", "UNKNOWN"))
            )
            selected_scope[index] = str(getattr(row, "action_scope", "KEEP_ONLY"))
    return pd.DataFrame(
        {
            "trial_index": np.arange(len(trials), dtype=int),
            "trial_uid": trials.trial_uid,
            "method_id": method_id,
            "reference_prediction": reference,
            "oracle_prediction": prediction,
            "selected_candidate_id": selected_id,
            "selected_family": selected_family,
            "selected_scope": selected_scope,
            "rescue": (reference != labels) & (prediction == labels),
            "harm": (reference == labels) & (prediction != labels),
            "OUTER_TEST_USED": False,
        }
    )


def action_oracle(trials: pd.DataFrame, candidates: pd.DataFrame, menu_id: str) -> pd.DataFrame:
    mask = action_menu_mask(candidates, menu_id)
    return oracle_from_candidates(
        trials,
        candidates.loc[mask].copy(),
        method_id=f"ORACLE_ACTION_{menu_id}",
    )


def keep_oracle(trials: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    return oracle_from_candidates(trials, candidates, method_id="ORACLE_KEEP_MENU")


def combined_oracle(
    trials: pd.DataFrame,
    keep_candidates: pd.DataFrame,
    action_candidates: pd.DataFrame,
    menu_id: str,
) -> pd.DataFrame:
    action = action_candidates.loc[action_menu_mask(action_candidates, menu_id)].copy()
    action = action[action.candidate_id.ne("KEEP_ENSEMBLE")]
    keep = keep_candidates.copy()
    keep = keep.assign(
        action_family=keep.keep_family,
        action_scope="KEEP_ONLY",
        protected_safe=True,
        uses_erase=False,
    )
    keep_priority = keep.priority.to_numpy(dtype=int)
    action = action.assign(priority=int(keep_priority.max()) + 1)
    columns = sorted(set(keep.columns) & set(action.columns))
    combined = pd.concat([keep[columns], action[columns]], ignore_index=True)
    return oracle_from_candidates(
        trials,
        combined,
        method_id=f"ORACLE_KEEP_PLUS_ACTION_{menu_id}",
    )


def _subset_evaluate(
    trials: pd.DataFrame,
    prediction: np.ndarray,
    method_id: str,
    pool: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    mask = pool_mask(trials, pool)
    subset = trials.loc[mask].reset_index(drop=True)
    return evaluate_prediction(
        subset,
        np.asarray(prediction, dtype=int)[mask],
        subset.y_keep_ens.to_numpy(dtype=int),
        method_id=method_id,
        pool=pool,
    )


def run_headroom_audit(
    trials: pd.DataFrame,
    action_candidates: pd.DataFrame,
    keep_candidates: pd.DataFrame,
    spec: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pools = ("exploration", "holdout", "all_52_exploratory")
    labels = trials.outcome_label.to_numpy(dtype=int)
    b6 = trials.y_keep_ens.to_numpy(dtype=int)

    action_oracles = {
        menu_id: action_oracle(trials, action_candidates, menu_id) for menu_id in ACTION_MENU_IDS
    }
    keep_result = keep_oracle(trials, keep_candidates)
    combined_oracles = {
        menu_id: combined_oracle(trials, keep_candidates, action_candidates, menu_id)
        for menu_id in ACTION_MENU_IDS
    }

    b6_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    keep_rows: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    subject_parts: list[pd.DataFrame] = []
    concentration: list[dict[str, Any]] = []
    stratum_parts: list[pd.DataFrame] = []
    for pool in pools:
        b6_summary, b6_subject = _subset_evaluate(trials, b6, "B6_ALL_RUN_LOGIT_MEAN", pool)
        b6_rows.append(b6_summary)
        subject_parts.append(b6_subject)
        keep_summary, keep_subject = _subset_evaluate(
            trials,
            keep_result.oracle_prediction.to_numpy(dtype=int),
            "ORACLE_KEEP_MENU",
            pool,
        )
        keep_rows.append(keep_summary)
        subject_parts.append(keep_subject)
        for menu_id in ACTION_MENU_IDS:
            method_id = f"ORACLE_ACTION_{menu_id}"
            oracle = action_oracles[menu_id]
            action_summary, action_subject = _subset_evaluate(
                trials,
                oracle.oracle_prediction.to_numpy(dtype=int),
                method_id,
                pool,
            )
            action_rows.append(action_summary)
            subject_parts.append(action_subject)
            combined = combined_oracles[menu_id]
            combined_summary, combined_subject = _subset_evaluate(
                trials,
                combined.oracle_prediction.to_numpy(dtype=int),
                combined.method_id.iloc[0],
                pool,
            )
            subject_parts.append(combined_subject)
            unique_rows.append(
                {
                    "pool": pool,
                    "menu_id": menu_id,
                    "action_oracle_delta_BA_vs_B6": action_summary["mean_subject_delta_BA_vs_B6"],
                    "keep_only_oracle_delta_BA_vs_B6": keep_summary["mean_subject_delta_BA_vs_B6"],
                    "action_oracle_minus_keep_only_oracle_delta_BA": (
                        action_summary["mean_subject_delta_BA_vs_B6"]
                        - keep_summary["mean_subject_delta_BA_vs_B6"]
                    ),
                    "combined_keep_plus_action_delta_BA_vs_B6": combined_summary[
                        "mean_subject_delta_BA_vs_B6"
                    ],
                    "combined_minus_keep_only_oracle_delta_BA": (
                        combined_summary["mean_subject_delta_BA_vs_B6"]
                        - keep_summary["mean_subject_delta_BA_vs_B6"]
                    ),
                    "action_oracle_rescue_trials": int(oracle.rescue.sum()),
                    "keep_only_oracle_rescue_trials": int(keep_result.rescue.sum()),
                    "combined_oracle_rescue_trials": int(combined.rescue.sum()),
                    "unique_action_rescue_trials_beyond_keep_menu": int(
                        np.sum(oracle.rescue.to_numpy(dtype=bool) & ~keep_result.rescue.to_numpy(dtype=bool))
                    ),
                    "OUTER_TEST_USED": False,
                }
            )
            if pool == "all_52_exploratory":
                concentration.extend(
                    concentration_rows(
                        trials,
                        action_subject,
                        oracle.rescue.to_numpy(dtype=int),
                        method_id,
                    )
                )
                stratum_parts.append(
                    by_session_class(
                        trials,
                        oracle.oracle_prediction.to_numpy(dtype=int),
                        b6,
                        method_id,
                    )
                )

    subject_table = pd.concat(subject_parts, ignore_index=True)
    action_table = pd.DataFrame(action_rows)
    keep_table = pd.DataFrame(keep_rows)
    unique_table = pd.DataFrame(unique_rows)
    concentration_table = pd.DataFrame(concentration)
    write_csv(DIAGNOSTICS / "B6_UNIQUE_TRIAL_RESULTS.csv", pd.DataFrame(b6_rows))
    write_csv(DIAGNOSTICS / "RESIDUAL_ORACLE_RESULTS.csv", action_table)
    write_csv(DIAGNOSTICS / "KEEP_ONLY_ORACLE_RESULTS.csv", keep_table)
    write_csv(DIAGNOSTICS / "UNIQUE_ACTION_ORACLE_RESULTS.csv", unique_table)
    write_csv(DIAGNOSTICS / "RESIDUAL_HEADROOM_BY_SUBJECT.csv", subject_table)
    write_csv(DIAGNOSTICS / "RESIDUAL_CONCENTRATION.csv", concentration_table)
    write_csv(
        DIAGNOSTICS / "RESIDUAL_HEADROOM_BY_STRATUM.csv",
        pd.concat(stratum_parts, ignore_index=True),
    )

    keep_rescue = keep_result.rescue.to_numpy(dtype=bool)
    enriched = action_candidates[action_candidates.candidate_id.ne("KEEP_ENSEMBLE")].copy()
    trial_index = enriched.trial_index.to_numpy(dtype=int)
    prediction = enriched.prediction.to_numpy(dtype=int)
    differs = prediction != b6[trial_index]
    rescue = differs & (b6[trial_index] != labels[trial_index]) & (prediction == labels[trial_index])
    harm = differs & (b6[trial_index] == labels[trial_index]) & (prediction != labels[trial_index])
    enriched["differs"] = differs
    enriched["rescue"] = rescue
    enriched["harm"] = harm
    enriched["unique_rescue_beyond_keep"] = rescue & ~keep_rescue[trial_index]
    enriched["overlap_keep_rescue"] = rescue & keep_rescue[trial_index]
    uniqueness_rows = []
    group_columns = ["action_scope", "candidate_id", "action_family", "run_id", "fold_id", "seed_id"]
    for keys, group in enriched.groupby(group_columns, dropna=False, sort=True):
        scope, candidate_id, family, run_id, fold, seed = keys
        differences = int(group.differs.sum())
        rescues = int(group.rescue.sum())
        harms = int(group.harm.sum())
        overlap = int(group.overlap_keep_rescue.sum())
        keep_union = int(np.sum(keep_rescue[group.trial_index.unique()]))
        uniqueness_rows.append(
            {
                "action_scope": scope,
                "candidate_id": candidate_id,
                "action_family": family,
                "run_id": run_id,
                "fold_id": fold,
                "seed_id": seed,
                "eligible_trials": int(len(group)),
                "prediction_differs_from_B6": differences,
                "rescue_count": rescues,
                "harm_count": harms,
                "net_correctness": rescues - harms,
                "rescue_precision": float(rescues / differences) if differences else np.nan,
                "unique_rescue_not_available_from_KEEP_menu": int(group.unique_rescue_beyond_keep.sum()),
                "overlap_with_KEEP_oracle_rescue": overlap,
                "rescue_jaccard_with_KEEP_oracle": (
                    float(overlap / (rescues + keep_union - overlap))
                    if rescues + keep_union - overlap > 0
                    else np.nan
                ),
                "OUTER_TEST_USED": False,
            }
        )
    uniqueness_table = pd.DataFrame(uniqueness_rows)
    write_csv(DIAGNOSTICS / "ACTION_UNIQUENESS.csv", uniqueness_table)

    ensemble_rows = []
    for keys, group in enriched.groupby(group_columns, dropna=False, sort=True):
        scope, candidate_id, family, run_id, fold, seed = keys
        ensemble_rows.append(
            {
                "action_scope": scope,
                "candidate_id": candidate_id,
                "action_family": family,
                "run_id": run_id,
                "fold_id": fold,
                "seed_id": seed,
                "eligible_trials": int(len(group)),
                "prediction_differs_from_B6": int(group.differs.sum()),
                "rescue_count": int(group.rescue.sum()),
                "harm_count": int(group.harm.sum()),
                "net_correctness": int(group.rescue.sum() - group.harm.sum()),
                "rescue_precision": (
                    float(group.rescue.sum() / group.differs.sum()) if group.differs.sum() else np.nan
                ),
                "OUTER_TEST_USED": False,
            }
        )
    write_csv(DIAGNOSTICS / "ACTION_ENSEMBLE_RESULTS.csv", pd.DataFrame(ensemble_rows))

    contribution_rows = []
    for menu_id, oracle in action_oracles.items():
        selected = oracle[oracle.rescue].copy()
        selected["unique_beyond_keep"] = ~keep_rescue[selected.trial_index.to_numpy(dtype=int)]
        for keys, group in selected.groupby(["selected_family", "selected_scope"], dropna=False):
            family, scope = keys
            contribution_rows.append(
                {
                    "menu_id": menu_id,
                    "stratum_type": "action_family",
                    "action_family": family,
                    "action_scope": scope,
                    "run_id": None,
                    "fold_id": np.nan,
                    "seed_id": np.nan,
                    "selected_rescue_count": int(len(group)),
                    "unique_rescue_beyond_KEEP_menu": int(group.unique_beyond_keep.sum()),
                    "OUTER_TEST_USED": False,
                }
            )
        if "SINGLE_REPLACEMENT" in menu_id:
            selected["run_id"] = selected.selected_candidate_id.str.split("->").str[0]
            selected["fold_id"] = selected.run_id.str.extract(r"fold-(\d+)")[0]
            selected["seed_id"] = selected.run_id.str.extract(r"seed-(\d+)")[0]
            for run_id, group in selected.groupby("run_id"):
                contribution_rows.append(
                    {
                        "menu_id": menu_id,
                        "stratum_type": "run",
                        "action_family": None,
                        "action_scope": "SINGLE_REPLACEMENT",
                        "run_id": run_id,
                        "fold_id": int(group.fold_id.iloc[0]),
                        "seed_id": int(group.seed_id.iloc[0]),
                        "selected_rescue_count": int(len(group)),
                        "unique_rescue_beyond_KEEP_menu": int(group.unique_beyond_keep.sum()),
                        "OUTER_TEST_USED": False,
                    }
                )
    contribution_table = pd.DataFrame(contribution_rows)
    write_csv(DIAGNOSTICS / "RESIDUAL_HEADROOM_BY_ACTION.csv", contribution_table)

    all_action = action_table[action_table.pool.eq("all_52_exploratory")].copy()
    maximum = all_action.mean_subject_delta_BA_vs_B6.max()
    strongest_method = sorted(
        all_action[np.isclose(all_action.mean_subject_delta_BA_vs_B6, maximum, atol=1e-15)].method_id
    )[0]
    strongest_menu = strongest_method.removeprefix("ORACLE_ACTION_")
    strongest = all_action[all_action.method_id.eq(strongest_method)].iloc[0]
    unique = unique_table[
        unique_table.pool.eq("all_52_exploratory") & unique_table.menu_id.eq(strongest_menu)
    ].iloc[0]
    strongest_subject = subject_table[
        subject_table.pool.eq("all_52_exploratory") & subject_table.method_id.eq(strongest_method)
    ]
    top20 = concentration_table[
        concentration_table.method_id.eq(strongest_method)
        & concentration_table.unit.eq("subject")
        & np.isclose(concentration_table.top_fraction, 0.20)
    ].fraction_of_positive_gain.iloc[0]
    strongest_oracle = action_oracles[strongest_menu]
    unique_selected = strongest_oracle[
        strongest_oracle.rescue
        & ~keep_rescue[strongest_oracle.trial_index.to_numpy(dtype=int)]
    ].copy()
    family_counts = unique_selected.selected_family.value_counts()
    dominant_action_fraction = float(family_counts.max() / family_counts.sum()) if len(family_counts) else 1.0
    if "SINGLE_REPLACEMENT" in strongest_menu and len(unique_selected):
        run_counts = unique_selected.selected_candidate_id.str.split("->").str[0].value_counts()
        dominant_run_fraction = float(run_counts.max() / run_counts.sum()) if len(run_counts) else 1.0
    else:
        dominant_run_fraction = 0.0
    stratum = pd.concat(stratum_parts, ignore_index=True)
    positive_sessions = int(
        np.sum(
            stratum.method_id.eq(strongest_method)
            & stratum.stratum_type.eq("session")
            & (stratum.delta_metric_vs_B6 > 0)
        )
    )

    thresholds = spec["headroom_gate"]["state_C_all_required"]
    gain = float(strongest.mean_subject_delta_BA_vs_B6)
    direct_unique = float(unique.action_oracle_minus_keep_only_oracle_delta_BA)
    combined_unique = float(unique.combined_minus_keep_only_oracle_delta_BA)
    positive_subjects = int(np.sum(strongest_subject.delta_BA_vs_B6 > 0))
    positive_fraction = float(np.mean(strongest_subject.delta_BA_vs_B6 > 0))
    state_c_checks = {
        "strongest_action_oracle_delta_BA": gain >= thresholds["strongest_action_oracle_delta_BA_min"],
        "action_minus_keep_only_delta_BA": direct_unique
        >= thresholds["action_oracle_minus_keep_only_oracle_delta_BA_min"],
        "combined_minus_keep_only_delta_BA": combined_unique
        >= thresholds["combined_keep_plus_action_minus_keep_oracle_delta_BA_min"],
        "positive_subjects": positive_subjects >= thresholds["positive_subjects_min"],
        "positive_subject_fraction": positive_fraction >= thresholds["positive_subject_fraction_min"],
        "positive_sessions": positive_sessions >= thresholds["positive_sessions_min"],
        "dominant_unique_action_fraction": dominant_action_fraction
        <= thresholds["dominant_unique_action_rescue_fraction_max"],
        "dominant_unique_run_fraction": dominant_run_fraction
        <= thresholds["dominant_unique_run_rescue_fraction_max"],
    }
    if gain < 0.005 or (np.isfinite(top20) and top20 >= 0.90):
        state = "NO_MEANINGFUL_RESIDUAL_HEADROOM"
    elif direct_unique < 0.005 or combined_unique < 0.005:
        state = "RESIDUAL_HEADROOM_IS_GENERIC_DIVERSITY"
    elif all(state_c_checks.values()):
        state = "STRUCTURAL_ACTION_RESIDUAL_EXISTS"
    else:
        state = "NO_MEANINGFUL_RESIDUAL_HEADROOM"

    decision = {
        "state": state,
        "strongest_action_oracle": strongest_method,
        "strongest_menu": strongest_menu,
        "strongest_action_oracle_delta_BA_vs_B6": gain,
        "strongest_action_oracle_CI95": [
            float(strongest.bootstrap_CI95_L),
            float(strongest.bootstrap_CI95_U),
        ],
        "keep_only_oracle_delta_BA_vs_B6": float(unique.keep_only_oracle_delta_BA_vs_B6),
        "action_oracle_minus_keep_only_delta_BA": direct_unique,
        "combined_keep_plus_action_minus_keep_only_delta_BA": combined_unique,
        "positive_subjects": positive_subjects,
        "positive_subject_fraction": positive_fraction,
        "positive_sessions": positive_sessions,
        "top20_subject_gain_concentration": float(top20),
        "dominant_unique_action_rescue_fraction": dominant_action_fraction,
        "dominant_unique_run_rescue_fraction": dominant_run_fraction,
        "state_C_checks": state_c_checks,
        "phase_8_plus_authorized": state == "STRUCTURAL_ACTION_RESIDUAL_EXISTS",
        "OUTER_TEST_USED": False,
    }
    write_json(DIAGNOSTICS / "HEADROOM_DECISION.json", decision)

    plot = all_action.sort_values("mean_subject_delta_BA_vs_B6")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot.method_id.str.replace("ORACLE_ACTION_", "", regex=False), 100 * plot.mean_subject_delta_BA_vs_B6)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label="STATE C headroom threshold")
    ax.set_xlabel("Mean subject Delta BA above B6 (pp)")
    ax.set_title("Residual action oracle above the strong KEEP ensemble")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_oracle_gain.png", dpi=180)
    plt.close(fig)

    pivot = subject_table[
        subject_table.pool.eq("all_52_exploratory")
        & subject_table.method_id.isin((strongest_method, "ORACLE_KEEP_MENU"))
    ].pivot(index="subject_id", columns="method_id", values="delta_BA_vs_B6")
    pivot = pivot.sort_values(strongest_method)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(pivot)), 100 * pivot[strongest_method], marker="o", markersize=3, label=strongest_method)
    ax.plot(np.arange(len(pivot)), 100 * pivot["ORACLE_KEEP_MENU"], marker=".", markersize=3, label="ORACLE_KEEP_MENU")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(pivot)), pivot.index, rotation=90, fontsize=6)
    ax.set_ylabel("Subject Delta BA above B6 (pp)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "subject_residual_headroom.png", dpi=180)
    plt.close(fig)

    action_plot = uniqueness_table.groupby(["action_scope", "action_family"], as_index=False).agg(
        unique_rescue=("unique_rescue_not_available_from_KEEP_menu", "sum"),
        rescue=("rescue_count", "sum"),
        harm=("harm_count", "sum"),
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    labels_plot = action_plot.action_scope + ":" + action_plot.action_family
    x = np.arange(len(action_plot))
    ax.bar(x - 0.25, action_plot.rescue, width=0.25, label="rescue")
    ax.bar(x, -action_plot.harm, width=0.25, label="harm")
    ax.bar(x + 0.25, action_plot.unique_rescue, width=0.25, label="unique rescue beyond KEEP menu")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels_plot, rotation=25, ha="right")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "action_uniqueness.png", dpi=180)
    plt.close(fig)

    conc = concentration_table[concentration_table.method_id.eq(strongest_method)]
    fig, ax = plt.subplots(figsize=(7, 5))
    for unit, group in conc.groupby("unit"):
        ax.plot(100 * group.top_fraction, 100 * group.fraction_of_positive_gain, marker="o", label=unit)
    ax.set_xlabel("Top units selected (%)")
    ax.set_ylabel("Captured positive oracle gain (%)")
    ax.set_title("Residual oracle concentration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_concentration.png", dpi=180)
    plt.close(fig)

    return {
        "decision": decision,
        "action_oracles": action_oracles,
        "keep_oracle": keep_result,
        "combined_oracles": combined_oracles,
        "subject_table": subject_table,
        "action_table": action_table,
        "keep_table": keep_table,
        "unique_table": unique_table,
        "uniqueness_table": uniqueness_table,
        "contribution_table": contribution_table,
    }
