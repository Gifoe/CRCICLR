from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common import DIAGNOSTICS, FIGURES, REPO_ROOT, stable_seed, write_csv, write_json


ROUTER_ACTIONS = ("erase", "amplify", "geometry")


def _subject_ba(group: pd.DataFrame, prediction: np.ndarray | pd.Series) -> float:
    y = group.outcome_label.to_numpy(dtype=int)
    pred = np.asarray(prediction, dtype=int)
    return float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")


def _router_run_subject_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (fold, seed, subject), group in frame.groupby(["fold_id", "seed_id", "subject_id"], sort=True):
        y = group.outcome_label.to_numpy(dtype=int)
        noop = group.pred_noop.to_numpy(dtype=int)
        noop_ba = _subject_ba(group, noop)
        correct = {action: group[f"pred_{action}"].to_numpy(dtype=int) == y for action in ROUTER_ACTIONS}
        noop_correct = noop == y
        oracle = noop.copy()
        unresolved = ~noop_correct
        chosen = np.full(len(group), "NO_OP", dtype=object)
        for action in ROUTER_ACTIONS:
            take = unresolved & correct[action]
            oracle[take] = y[take]
            chosen[take] = action.upper()
            unresolved[take] = False
        oracle_ba = _subject_ba(group, oracle)
        fixed_gains: dict[str, float] = {}
        for action in ROUTER_ACTIONS:
            action_ba = _subject_ba(group, group[f"pred_{action}"].to_numpy(dtype=int))
            pair = noop.copy()
            rescue = (~noop_correct) & correct[action]
            pair[rescue] = y[rescue]
            pair_oracle_ba = _subject_ba(group, pair)
            fixed_gains[action] = action_ba - noop_ba
            rows.append(
                {
                    "family_id": "openbmi_sample_router",
                    "dataset_id": "OpenBMI",
                    "backbone_id": "EEGNet",
                    "fold_id": fold,
                    "seed_id": seed,
                    "subject_id": subject,
                    "block_id": "PROTECTED_UNION",
                    "action": action.upper(),
                    "noop_BA": noop_ba,
                    "action_BA": action_ba,
                    "delta_BA": action_ba - noop_ba,
                    "pair_oracle_BA": pair_oracle_ba,
                    "pair_oracle_gain": pair_oracle_ba - noop_ba,
                    "all_action_oracle_gain": oracle_ba - noop_ba,
                    "decision_units": len(group),
                    "outer_test_used": False,
                }
            )
        best_nontrivial = max(fixed_gains.values())
        rows.append(
            {
                "family_id": "openbmi_sample_router",
                "dataset_id": "OpenBMI",
                "backbone_id": "EEGNet",
                "fold_id": fold,
                "seed_id": seed,
                "subject_id": subject,
                "block_id": "PROTECTED_UNION",
                "action": "ORACLE_ALL_ACTIONS",
                "noop_BA": noop_ba,
                "action_BA": oracle_ba,
                "delta_BA": oracle_ba - noop_ba,
                "pair_oracle_BA": oracle_ba,
                "pair_oracle_gain": oracle_ba - noop_ba,
                "all_action_oracle_gain": oracle_ba - noop_ba,
                "best_nontrivial_fixed_gain": best_nontrivial,
                "best_fixed_including_noop_gain": max(0.0, best_nontrivial),
                "selection_value": oracle_ba - noop_ba - max(0.0, best_nontrivial),
                "decision_units": len(group),
                "outer_test_used": False,
            }
        )
    return rows


def _block_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        effect = float(item.effect_suppress)
        rows.append(
            {
                "family_id": item.family_id,
                "dataset_id": item.dataset_id,
                "backbone_id": item.backbone_id,
                "fold_id": item.fold_id,
                "seed_id": item.seed_id,
                "audit_fold_id": item.audit_fold_id,
                "subject_id": item.subject_id,
                "block_id": item.block_id,
                "action": "SUPPRESS_BLOCK",
                "noop_BA": np.nan,
                "action_BA": np.nan,
                "delta_BA": effect,
                "pair_oracle_BA": np.nan,
                "pair_oracle_gain": max(0.0, effect),
                "all_action_oracle_gain": max(0.0, effect),
                "best_nontrivial_fixed_gain": effect,
                "best_fixed_including_noop_gain": max(0.0, effect),
                "selection_value": 0.0,
                "decision_units": 1,
                "outer_test_used": False,
            }
        )
    return rows


def decomposition_table(data: pd.DataFrame) -> pd.DataFrame:
    router = data[data.family_id == "openbmi_sample_router"].copy()
    block = data[data.family_id != "openbmi_sample_router"].copy()
    result = pd.DataFrame(_router_run_subject_rows(router) + _block_rows(block))
    write_csv(DIAGNOSTICS / "ORACLE_HEADROOM_DECOMPOSITION.csv", result)
    return result


def _router_summary(data: pd.DataFrame) -> dict[str, Any]:
    frame = data[data.family_id == "openbmi_sample_router"].copy()
    subject_rows = pd.DataFrame(_router_run_subject_rows(frame))
    oracle = subject_rows[subject_rows.action == "ORACLE_ALL_ACTIONS"]
    fixed = subject_rows[subject_rows.action.isin([value.upper() for value in ROUTER_ACTIONS])]
    fixed_action = fixed.groupby("action").delta_BA.mean().sort_values(ascending=False)
    pair_oracle = fixed.groupby("action").pair_oracle_gain.mean().sort_values(ascending=False)
    return {
        "family_id": "openbmi_sample_router",
        "decision_units": len(frame),
        "independent_subjects": int(frame.subject_id.nunique()),
        "runs": int(frame[["fold_id", "seed_id"]].drop_duplicates().shape[0]),
        "oracle_action_gain": float(oracle.delta_BA.mean()),
        "best_nontrivial_fixed_action": str(fixed_action.index[0]),
        "best_nontrivial_fixed_action_gain": float(fixed_action.iloc[0]),
        "best_fixed_including_noop_gain": float(max(0.0, fixed_action.iloc[0])),
        "selection_value": float(oracle.delta_BA.mean() - max(0.0, fixed_action.iloc[0])),
        "oracle_gain_by_run": {
            f"fold-{fold}_seed-{seed}": float(group.delta_BA.mean())
            for (fold, seed), group in oracle.groupby(["fold_id", "seed_id"])
        },
        "fixed_gain_by_action": {str(key): float(value) for key, value in fixed_action.items()},
        "pair_oracle_gain_by_action": {str(key): float(value) for key, value in pair_oracle.items()},
    }


def _block_summary(frame: pd.DataFrame) -> dict[str, Any]:
    effect = frame.effect_suppress.to_numpy(dtype=float)
    oracle = np.maximum(effect, 0.0)
    fixed = float(np.mean(effect))
    return {
        "family_id": str(frame.family_id.iloc[0]),
        "decision_units": len(frame),
        "oracle_action_gain": float(np.mean(oracle)),
        "best_nontrivial_fixed_action": "SUPPRESS_BLOCK",
        "best_nontrivial_fixed_action_gain": fixed,
        "best_fixed_including_noop_gain": max(0.0, fixed),
        "selection_value": float(np.mean(oracle) - max(0.0, fixed)),
        "positive_fraction": float(np.mean(effect > 0)),
        "practical_positive_fraction": float(np.mean(effect >= 0.005)),
        "harm_fraction": float(np.mean(effect < 0)),
    }


def _weighted_router_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (fold, seed, subject), group in frame.groupby(["fold_id", "seed_id", "subject_id"], sort=True):
        y = group.outcome_label.to_numpy(dtype=int)
        noop_correct = group.pred_noop.to_numpy(dtype=int) == y
        any_rescue = np.zeros(len(group), dtype=bool)
        rescue_action = np.full(len(group), "NONE", dtype=object)
        for action in ROUTER_ACTIONS:
            correct = group[f"pred_{action}"].to_numpy(dtype=int) == y
            take = (~noop_correct) & (~any_rescue) & correct
            any_rescue[take] = True
            rescue_action[take] = action.upper()
        counts = pd.Series(y).value_counts().to_dict()
        weights = np.asarray([0.5 / counts[int(value)] for value in y], dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "fold_id": np.full(len(group), fold, dtype=int),
                    "seed_id": np.full(len(group), seed, dtype=int),
                    "subject_id": np.full(len(group), str(subject), dtype=object),
                    "manifest_index": group.manifest_index.to_numpy(dtype=int),
                    "action": rescue_action,
                    "oracle_contribution": np.where(any_rescue, weights, 0.0),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def concentration_analysis(data: pd.DataFrame, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    router = data[data.family_id == "openbmi_sample_router"]
    contributions = _weighted_router_contributions(router)
    values = np.sort(contributions.oracle_contribution.to_numpy(dtype=float))[::-1]
    total = float(values.sum())
    top: dict[str, float] = {}
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(np.ceil(fraction * len(values))))
        top[f"top_{int(fraction * 100)}pct_units"] = float(values[:count].sum() / total) if total > 0 else 0.0
    subject_gain = contributions.groupby("subject_id").oracle_contribution.sum().sort_values(ascending=False)
    run_gain = contributions.groupby(["fold_id", "seed_id"]).oracle_contribution.sum().sort_values(ascending=False)
    payload = {
        "openbmi_sample_router": {
            "top_fraction_contribution": top,
            "rescue_unit_fraction": float(np.mean(values > 0)),
            "subjects_with_rescue_fraction": float(np.mean(subject_gain > 0)),
            "largest_subject_share": float(subject_gain.iloc[0] / subject_gain.sum()),
            "largest_run_share": float(run_gain.iloc[0] / run_gain.sum()),
            "subject_shares": {str(key): float(value / subject_gain.sum()) for key, value in subject_gain.items()},
        },
        "family_summaries": summaries,
        "OUTER_TEST_USED": False,
    }
    write_json(DIAGNOSTICS / "ORACLE_CONCENTRATION_ANALYSIS.json", payload)
    return payload


def group_and_action_tables(decomposition: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = decomposition.groupby(["family_id", "action"], as_index=False).agg(
        n=("delta_BA", "size"),
        mean_delta_BA=("delta_BA", "mean"),
        median_delta_BA=("delta_BA", "median"),
        positive_fraction=("delta_BA", lambda value: float(np.mean(np.asarray(value) > 0))),
        practical_positive_fraction=("delta_BA", lambda value: float(np.mean(np.asarray(value) >= 0.005))),
        one_pp_positive_fraction=("delta_BA", lambda value: float(np.mean(np.asarray(value) >= 0.01))),
        harm_fraction=("delta_BA", lambda value: float(np.mean(np.asarray(value) < 0))),
        mean_pair_oracle_gain=("pair_oracle_gain", "mean"),
    )
    groups = decomposition.groupby(
        ["family_id", "dataset_id", "backbone_id", "fold_id", "seed_id", "action"],
        dropna=False,
        as_index=False,
    ).agg(
        units=("decision_units", "sum"),
        mean_delta_BA=("delta_BA", "mean"),
        mean_oracle_gain=("all_action_oracle_gain", "mean"),
    )
    write_csv(DIAGNOSTICS / "ORACLE_HEADROOM_BY_ACTION.csv", actions)
    write_csv(DIAGNOSTICS / "ORACLE_HEADROOM_BY_GROUP.csv", groups)
    return groups, actions


def _figures(data: pd.DataFrame, summaries: dict[str, dict[str, Any]], decomposition: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = list(summaries)
    oracle = [summaries[key]["oracle_action_gain"] for key in labels]
    fixed = [summaries[key]["best_fixed_including_noop_gain"] for key in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, oracle, 0.36, label="Oracle action selection")
    ax.bar(x + 0.18, fixed, 0.36, label="Best fixed incl. NO_OP")
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xticks(x, [value.replace("_", "\n") for value in labels], fontsize=8)
    ax.set_ylabel("Mean subject/group-balanced ΔBA")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_oracle_vs_best_fixed.png", dpi=240)
    fig.savefig(FIGURES / "figure1_oracle_vs_best_fixed.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, family in zip(axes, labels):
        values = decomposition[(decomposition.family_id == family) & (decomposition.action != "ORACLE_ALL_ACTIONS")].delta_BA
        ax.hist(values, bins=30, color="#3b7b9f", alpha=0.85)
        ax.axvline(0, color="#333", linewidth=1)
        ax.set_title(family.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("ΔBA")
    axes[0].set_ylabel("Decision groups")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_gain_distribution.png", dpi=240)
    fig.savefig(FIGURES / "figure2_gain_distribution.pdf")
    plt.close(fig)

    action = decomposition[
        (decomposition.family_id == "openbmi_sample_router")
        & (decomposition.action.isin([value.upper() for value in ROUTER_ACTIONS]))
    ].groupby("action").agg(delta=("delta_BA", "mean"), oracle=("pair_oracle_gain", "mean")).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xx = np.arange(len(action))
    ax.bar(xx - 0.18, action.delta, 0.36, label="Always action")
    ax.bar(xx + 0.18, action.oracle, 0.36, label="Pair oracle")
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xticks(xx, action.index)
    ax.set_ylabel("Subject-balanced ΔBA")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_gain_by_action.png", dpi=240)
    fig.savefig(FIGURES / "figure3_gain_by_action.pdf")
    plt.close(fig)

    oracle_rows = decomposition[
        (decomposition.family_id == "openbmi_sample_router") & (decomposition.action == "ORACLE_ALL_ACTIONS")
    ]
    run = oracle_rows.groupby(["fold_id", "seed_id"]).delta_BA.mean()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([f"F{a}/S{b}" for a, b in run.index], run.values, color="#4f8a70")
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_ylabel("Oracle ΔBA")
    ax.set_title("Oracle headroom across all six runs")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_oracle_by_run.png", dpi=240)
    fig.savefig(FIGURES / "figure4_oracle_by_run.pdf")
    plt.close(fig)

    contributions = _weighted_router_contributions(data[data.family_id == "openbmi_sample_router"])
    values = np.sort(contributions.oracle_contribution.to_numpy(dtype=float))[::-1]
    cumulative = np.cumsum(values) / max(values.sum(), 1e-12)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(np.arange(1, len(values) + 1) / len(values), cumulative, color="#8a4f6f", linewidth=2)
    ax.set_xlabel("Fraction of run-trial decision units")
    ax.set_ylabel("Cumulative fraction of oracle gain")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_oracle_concentration.png", dpi=240)
    fig.savefig(FIGURES / "figure5_oracle_concentration.pdf")
    plt.close(fig)


def run_oracle_analysis(data: pd.DataFrame) -> dict[str, Any]:
    decomposition = decomposition_table(data)
    groups, actions = group_and_action_tables(decomposition)
    summaries = {"openbmi_sample_router": _router_summary(data)}
    for family in ("openbmi_dda_block", "wbcic_development_block"):
        summaries[family] = _block_summary(data[data.family_id == family])
    concentration = concentration_analysis(data, summaries)
    _figures(data, summaries, decomposition)
    report = f"""# Oracle headroom decomposition

The exact historical OpenBMI action menu was reconstructed from the stored
subject-cross-fitted OOF logits. The primary values below average balanced
accuracy within subject before averaging subjects and runs.

## OpenBMI sample router

- Oracle action gain: `{summaries['openbmi_sample_router']['oracle_action_gain']:.6f}` BA.
- Best non-trivial fixed action: `{summaries['openbmi_sample_router']['best_nontrivial_fixed_action']}` with
  `{summaries['openbmi_sample_router']['best_nontrivial_fixed_action_gain']:.6f}` BA.
- Best fixed action including NO_OP: `{summaries['openbmi_sample_router']['best_fixed_including_noop_gain']:.6f}` BA.
- Action-selection value: `{summaries['openbmi_sample_router']['selection_value']:.6f}` BA.
- Subjects with at least one oracle rescue: `{concentration['openbmi_sample_router']['subjects_with_rescue_fraction']:.3f}`.
- Largest subject contribution: `{concentration['openbmi_sample_router']['largest_subject_share']:.3f}` of total oracle gain.
- Largest run contribution: `{concentration['openbmi_sample_router']['largest_run_share']:.3f}`.

All three fixed interventions are net harmful even though each creates some
oracle rescues. This is exactly the rare-rescue/frequent-harm regime; oracle
headroom is not evidence that a legal router can recover it.

## Block families

- DDA oracle gain: `{summaries['openbmi_dda_block']['oracle_action_gain']:.6f}`;
  selection value `{summaries['openbmi_dda_block']['selection_value']:.6f}`.
- WBCIC development oracle gain: `{summaries['wbcic_development_block']['oracle_action_gain']:.6f}`;
  selection value `{summaries['wbcic_development_block']['selection_value']:.6f}`.

The block values are diagnostic upper bounds from choosing suppression only
when its realised held-out consequence is positive. They are not deployable
policies. `OUTER_TEST_USED = false`.
"""
    (DIAGNOSTICS / "ORACLE_HEADROOM_REPORT.md").write_text(report, encoding="utf-8")
    return {"summaries": summaries, "concentration": concentration}


if __name__ == "__main__":
    from common import DATA

    frame = pd.read_csv(DATA / "ACTION_OUTCOME_DATASET.csv", low_memory=False)
    print(json.dumps(run_oracle_analysis(frame), indent=2))
