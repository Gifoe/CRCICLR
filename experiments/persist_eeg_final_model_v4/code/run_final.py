from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from common import (
    DIAGNOSTICS,
    FINAL_LOCK,
    LEADERBOARD,
    OUTPUTS,
    RESEARCH_LOG,
    V4_SEED,
    default_openbmi_cache,
    default_wbcic_repo,
    ensure_directories,
    logit,
    markdown_table,
    sha256_file,
    sigmoid,
    stable_seed,
    write_csv,
    write_json,
)
from datasets import OPENBMI_RUNS, load_openbmi, load_wbcic_development
from make_figures import make as make_figures
from models import linear, stacking
from run_wbcic_keep_search import _raw_features


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[3], text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def _mean_subject_ba(labels: np.ndarray, prediction: np.ndarray, subjects: np.ndarray) -> float:
    values = []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        values.append(balanced_accuracy_score(labels[mask], prediction[mask]))
    return float(np.mean(values))


def _oracle_prediction(labels: np.ndarray, candidate_logits: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    predictions = np.asarray(candidate_logits >= 0, dtype=int)
    correct = predictions == labels[:, None]
    return np.where(correct.any(axis=1), labels, fallback).astype(int)


def _extract_expert_usage(openbmi_cache: Path, wbcic_repo: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = load_openbmi(openbmi_cache)
    raw_logits = np.where(data.keep_run_mask, data.keep_run_logits, data.base_logits[:, None])
    session_values = sorted(np.unique(data.sessions).tolist())
    session_onehot = np.column_stack([(data.sessions == value).astype(float) for value in session_values])
    x = np.column_stack([raw_logits, data.keep_run_mask.astype(float), session_onehot])
    selections = pd.read_csv(DIAGNOSTICS / "FINAL_ABLATION_SELECTIONS.csv")
    selected_flag = selections.selected.astype(str).str.lower().eq("true")
    selected = selections[selections.method_id.eq("A1_DYNAMIC_KEEP_FINAL") & selected_flag]
    if len(selected) != 5:
        raise RuntimeError("Expected one selected A1 configuration per OpenBMI outer fold")
    for item in selected.itertuples(index=False):
        fold_id = int(item.outer_fold)
        fold = next(value for value in data.folds if int(value["outer_fold"]) == fold_id)
        train = np.isin(data.subjects, fold["train_subjects"])
        configuration = json.loads(str(item.configuration))
        model = stacking.build(configuration, stable_seed("FINAL_USAGE", fold_id))
        model.fit(x[train], data.labels[train])
        parameters = np.asarray(model.parameters_, dtype=float)
        theta = parameters[:6]
        weight = np.exp(np.clip(theta, -6.0, 6.0))
        weight /= weight.sum()
        scale = float(np.exp(np.clip(parameters[6], -2.0, 2.0)))
        bias = float(parameters[7])
        for expert, value, raw_theta in zip(OPENBMI_RUNS, weight, theta):
            rows.append(
                {
                    "dataset": "OpenBMI",
                    "model_id": "A1_DYNAMIC_KEEP_FINAL",
                    "outer_fold": fold_id,
                    "expert_or_feature": expert,
                    "usage_kind": "normalized_positive_weight",
                    "value": float(value),
                    "raw_parameter": float(raw_theta),
                    "global_scale": scale,
                    "global_bias": bias,
                    "threshold": float(item.threshold),
                    "OUTER_TEST_USED": False,
                }
            )

    raw_wbcic = load_wbcic_development(
        OUTPUTS / "cache" / "WBCIC_DEV_KEEP_EXPERTS.parquet", wbcic_repo
    )
    expert_logits = raw_wbcic.keep_run_logits[:, :5]
    wbcic = replace(raw_wbcic, base_logits=logit(sigmoid(expert_logits).mean(axis=1)))
    wbcic_x = _raw_features(wbcic)
    wbcic_selection = pd.read_csv(DIAGNOSTICS / "WBCIC_DEV_KEEP_SEARCH_SELECTIONS.csv")
    wbcic_selected_flag = wbcic_selection.selected.astype(str).str.lower().eq("true")
    wbcic_selected = wbcic_selection[
        wbcic_selection.method_id.eq("W1_RAW_LINEAR") & wbcic_selected_flag
    ]
    feature_names = [
        "EEGNet_STABLE_margin",
        "EEGNet_STD_margin",
        "DeepConvNet_margin",
        "EEGConformer_margin",
        "TeCh_margin",
        "B_STRONG_fill_margin",
    ]
    for item in wbcic_selected.itertuples(index=False):
        fold_id = int(item.outer_fold)
        fold = next(value for value in wbcic.folds if int(value["outer_fold"]) == fold_id)
        train = np.isin(wbcic.subjects, fold["train_subjects"])
        configuration = json.loads(str(item.configuration))
        model = linear.build(configuration, stable_seed("WBCIC_USAGE", fold_id))
        model.fit(wbcic_x[train], wbcic.labels[train])
        scaler = model.named_steps["scale"]
        classifier = model.named_steps["classifier"]
        effective = classifier.coef_[0, :6] / np.where(scaler.scale_[:6] == 0, 1.0, scaler.scale_[:6])
        for name, value in zip(feature_names, effective):
            rows.append(
                {
                    "dataset": "WBCIC-development",
                    "model_id": "W1_RAW_LINEAR",
                    "outer_fold": fold_id,
                    "expert_or_feature": name,
                    "usage_kind": "effective_linear_coefficient",
                    "value": float(value),
                    "raw_parameter": np.nan,
                    "global_scale": np.nan,
                    "global_bias": float(classifier.intercept_[0]),
                    "threshold": float(item.threshold),
                    "OUTER_TEST_USED": False,
                }
            )
    frame = pd.DataFrame(rows)
    summaries = []
    for (dataset, model_id, name, kind), group in frame.groupby(
        ["dataset", "model_id", "expert_or_feature", "usage_kind"], sort=False
    ):
        summaries.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "outer_fold": "mean_across_folds",
                "expert_or_feature": name,
                "usage_kind": kind,
                "value": float(group.value.mean()),
                "raw_parameter": float(group.raw_parameter.mean()) if group.raw_parameter.notna().any() else np.nan,
                "global_scale": float(group.global_scale.mean()) if group.global_scale.notna().any() else np.nan,
                "global_bias": float(group.global_bias.mean()),
                "threshold": float(group.threshold.mean()),
                "OUTER_TEST_USED": False,
            }
        )
    frame = pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True)
    write_csv(DIAGNOSTICS / "EXPERT_USAGE.csv", frame)
    return frame


def _development_table(openbmi_cache: Path, wbcic_repo: Path) -> pd.DataFrame:
    ablation = pd.read_csv(OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv")
    open_leader = pd.read_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv")
    wbcic_search = pd.read_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv")
    wbcic_transfer = pd.read_csv(LEADERBOARD / "WBCIC_DEV_MODEL_LEADERBOARD.csv")
    baseline = _read_json(OUTPUTS / "protocol" / "BASELINE_RECONSTRUCTION.json")
    columns = [
        "dataset",
        "table_role",
        "method_id",
        "mean_subject_BA",
        "Delta_BA_vs_B_STRONG",
        "accuracy",
        "macro_f1",
        "NLL",
        "Brier",
        "ECE",
        "positive_subject_fraction",
        "nonnegative_subject_fraction",
        "worst_subject_delta",
        "switch_rate",
        "rescue_precision",
        "harm_rate",
        "CI95_L",
        "CI95_U",
        "prospective",
        "OUTER_TEST_USED",
    ]
    rows: list[dict[str, Any]] = []

    def add_existing(dataset: str, role: str, row: pd.Series, method_id: str | None = None) -> None:
        payload = {column: row.get(column, np.nan) for column in columns}
        payload.update(
            {
                "dataset": dataset,
                "table_role": role,
                "method_id": method_id or str(row.method_id),
                "prospective": True,
                "OUTER_TEST_USED": False,
            }
        )
        rows.append(payload)

    rows.append(
        {
            "dataset": "OpenBMI",
            "table_role": "Single model",
            "method_id": "B0_TARGET_KEEP_historical_pooled",
            "mean_subject_BA": 0.8233173076923076,
            "Delta_BA_vs_B_STRONG": 0.8233173076923076 - float(baseline["B_STRONG_mean_subject_BA"]),
            "accuracy": 0.8233173076923076,
            "macro_f1": 0.8213913190201908,
            "NLL": 0.3871040495524437,
            "Brier": 0.12265613803108705,
            "ECE": 0.03254354883232071,
            "positive_subject_fraction": np.nan,
            "nonnegative_subject_fraction": np.nan,
            "worst_subject_delta": np.nan,
            "switch_rate": np.nan,
            "rescue_precision": np.nan,
            "harm_rate": np.nan,
            "CI95_L": np.nan,
            "CI95_U": np.nan,
            "prospective": True,
            "OUTER_TEST_USED": False,
        }
    )
    role_map = {
        "A0_STATIC_B_STRONG": "Static strong ensemble",
        "A1_DYNAMIC_KEEP_FINAL": "Dynamic KEEP-only / best final",
        "A2_KEEP_ACTION_NO_PERSIST": "KEEP+ACTION without PERSIST",
        "A3_KEEP_ACTION_PERSIST": "KEEP+ACTION+PERSIST",
    }
    for method, role in role_map.items():
        add_existing("OpenBMI", role, ablation[ablation.method_id.eq(method)].iloc[0])
    add_existing(
        "OpenBMI",
        "Best generic stacking control",
        open_leader[open_leader.method_id.eq("M1_DYNAMIC_KEEP_LINEAR")].iloc[0],
    )
    for role, method, ba in (
        ("Oracle KEEP-only", "ORACLE_KEEP_ONLY", float(baseline["KEEP_only_oracle_BA"])),
        ("Oracle KEEP+ACTION", "ORACLE_KEEP_ACTION", float(baseline["complete_KEEP_ACTION_oracle_BA"])),
    ):
        rows.append(
            {
                "dataset": "OpenBMI",
                "table_role": role,
                "method_id": method,
                "mean_subject_BA": ba,
                "Delta_BA_vs_B_STRONG": ba - float(baseline["B_STRONG_mean_subject_BA"]),
                "prospective": False,
                "OUTER_TEST_USED": False,
            }
        )

    stable = wbcic_transfer[wbcic_transfer.method_id.eq("W0_EEGNET_STABLE")].iloc[0]
    strong = wbcic_search[wbcic_search.method_id.eq("W0_B_STRONG_PROBABILITY_MEAN")].iloc[0]
    raw = wbcic_search[wbcic_search.method_id.eq("W1_RAW_LINEAR")].iloc[0]
    direct = wbcic_transfer[wbcic_transfer.method_id.eq("W1_MASKED_POOL_SHRUNK_THR")].iloc[0].copy()
    for old, new in (
        ("Delta_BA_vs_WBCIC_B_STRONG", "Delta_BA_vs_B_STRONG"),
        ("CI95_L_vs_WBCIC_B_STRONG", "CI95_L"),
        ("CI95_U_vs_WBCIC_B_STRONG", "CI95_U"),
    ):
        direct[new] = direct[old]
    add_existing("WBCIC-development", "Single model", stable)
    add_existing("WBCIC-development", "Static strong ensemble", strong)
    add_existing("WBCIC-development", "Direct transferred Dynamic KEEP", direct)
    add_existing("WBCIC-development", "Best generic stacking / development candidate", raw)

    wbcic_data_raw = load_wbcic_development(
        OUTPUTS / "cache" / "WBCIC_DEV_KEEP_EXPERTS.parquet", wbcic_repo
    )
    wbcic_probability = sigmoid(wbcic_data_raw.keep_run_logits[:, :5]).mean(axis=1)
    wbcic_base = (wbcic_probability >= 0.5).astype(int)
    wbcic_oracle = _oracle_prediction(
        wbcic_data_raw.labels, wbcic_data_raw.keep_run_logits[:, :5], wbcic_base
    )
    oracle_ba = _mean_subject_ba(wbcic_data_raw.labels, wbcic_oracle, wbcic_data_raw.subjects)
    rows.append(
        {
            "dataset": "WBCIC-development",
            "table_role": "Oracle KEEP-only",
            "method_id": "WBCIC_ORACLE_KEEP_ONLY",
            "mean_subject_BA": oracle_ba,
            "Delta_BA_vs_B_STRONG": oracle_ba - float(strong.mean_subject_BA),
            "accuracy": float(accuracy_score(wbcic_data_raw.labels, wbcic_oracle)),
            "macro_f1": float(f1_score(wbcic_data_raw.labels, wbcic_oracle, average="macro")),
            "prospective": False,
            "OUTER_TEST_USED": False,
        }
    )
    rows.append(
        {
            "dataset": "WBCIC-development",
            "table_role": "KEEP+ACTION / PERSIST",
            "method_id": "NOT_AVAILABLE_NO_LEGAL_ACTION_EXPERT",
            "prospective": False,
            "OUTER_TEST_USED": False,
        }
    )
    table = pd.DataFrame(rows)
    for column in columns:
        if column not in table:
            table[column] = np.nan
    table = table[columns]
    write_csv(LEADERBOARD / "FINAL_DEVELOPMENT_TABLE.csv", table)
    return table


def _write_cross_benchmark() -> pd.DataFrame:
    open_leader = pd.read_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv")
    wbcic_search = pd.read_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv")
    wbcic_transfer = pd.read_csv(LEADERBOARD / "WBCIC_DEV_MODEL_LEADERBOARD.csv")
    rows = [
        {
            "candidate": "STATIC_STRONG_REFERENCE",
            "OpenBMI_method": "M0_B_STRONG_B6",
            "OpenBMI_Delta_BA": 0.0,
            "WBCIC_method": "W0_B_STRONG_PROBABILITY_MEAN",
            "WBCIC_Delta_BA_vs_static": 0.0,
        },
        {
            "candidate": "DIRECT_MASKED_POSITIVE_POOL_TRANSFER",
            "OpenBMI_method": "M1_MASKED_POOL_SHRUNK_THR",
            "OpenBMI_Delta_BA": float(open_leader[open_leader.method_id.eq("M1_MASKED_POOL_SHRUNK_THR")].iloc[0].Delta_BA_vs_B_STRONG),
            "WBCIC_method": "W1_MASKED_POOL_SHRUNK_THR",
            "WBCIC_Delta_BA_vs_static": float(wbcic_transfer[wbcic_transfer.method_id.eq("W1_MASKED_POOL_SHRUNK_THR")].iloc[0].Delta_BA_vs_WBCIC_B_STRONG),
        },
        {
            "candidate": "BENCHMARK_ADAPTED_GENERIC_LINEAR_STACKING",
            "OpenBMI_method": "M1_DYNAMIC_KEEP_LINEAR",
            "OpenBMI_Delta_BA": float(open_leader[open_leader.method_id.eq("M1_DYNAMIC_KEEP_LINEAR")].iloc[0].Delta_BA_vs_B_STRONG),
            "WBCIC_method": "W1_RAW_LINEAR",
            "WBCIC_Delta_BA_vs_static": float(wbcic_search[wbcic_search.method_id.eq("W1_RAW_LINEAR")].iloc[0].Delta_BA_vs_B_STRONG),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["mean_normalized_gain"] = frame[["OpenBMI_Delta_BA", "WBCIC_Delta_BA_vs_static"]].mean(axis=1)
    frame["worst_benchmark_gain"] = frame[["OpenBMI_Delta_BA", "WBCIC_Delta_BA_vs_static"]].min(axis=1)
    frame["robust_on_both"] = False
    frame["OUTER_TEST_USED"] = False
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_LEADERBOARD.csv", frame)
    return frame


def _write_research_logs() -> None:
    existing_path = RESEARCH_LOG / "ITERATION_SUMMARY.csv"
    existing = pd.read_csv(existing_path) if existing_path.is_file() else pd.DataFrame()
    if "iteration" in existing:
        existing = existing[pd.to_numeric(existing.iteration, errors="coerce").fillna(0) < 100]
    wbcic = pd.read_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv")
    refine = pd.read_csv(LEADERBOARD / "WBCIC_DEV_LINEAR_REFINEMENT.csv")
    transfer = pd.read_csv(LEADERBOARD / "WBCIC_DEV_MODEL_LEADERBOARD.csv")
    a = pd.read_csv(OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv")
    records = [
        (100, "WBCIC_STATIC_AUDIT", "probability mean selected as the strongest static reference", "STATIC ENSEMBLE AUDIT", 0.8036257390983, 0.0, "REFERENCE"),
        (101, "WBCIC_DIRECT_TRANSFER", "OpenBMI masked pool did not transfer above the corrected WBCIC reference", "MASKED POSITIVE POOL", float(transfer[transfer.method_id.eq("W1_MASKED_POOL_SHRUNK_THR")].iloc[0].mean_subject_BA), float(transfer[transfer.method_id.eq("W1_MASKED_POOL_SHRUNK_THR")].iloc[0].Delta_BA_vs_WBCIC_B_STRONG), "ABANDON"),
        (102, "WBCIC_KEEP_SEARCH", "generic raw linear stacking showed a positive but non-robust signal", "LINEAR/HGB/DEEPSETS/POOL SEARCH", float(wbcic[wbcic.method_id.eq("W1_RAW_LINEAR")].iloc[0].mean_subject_BA), float(wbcic[wbcic.method_id.eq("W1_RAW_LINEAR")].iloc[0].Delta_BA_vs_B_STRONG), "MODIFY"),
        (103, "WBCIC_LINEAR_REFINEMENT", "threshold shrinkage and anchored residuals did not make the gain robust", "TARGETED LINEAR/RESIDUAL REFINEMENT", float(refine.iloc[0].mean_subject_BA), float(refine.iloc[0].Delta_BA_vs_B_STRONG), "ABANDON"),
        (104, "FINAL_ABLATIONS", "dynamic KEEP is robust on OpenBMI; ACTION and PERSIST increments are unsupported", "A0-A9 MATCHED ABLATION", float(a[a.method_id.eq("A1_DYNAMIC_KEEP_FINAL")].iloc[0].mean_subject_BA), float(a[a.method_id.eq("A1_DYNAMIC_KEEP_FINAL")].iloc[0].Delta_BA_vs_B_STRONG), "KEEP"),
    ]
    rows = []
    for iteration, model_id, diagnosis, architecture, ba, delta, result in records:
        payload = {
            "iteration": iteration,
            "model_id": model_id,
            "diagnosis": diagnosis,
            "hypothesis": "convert legal frozen expert diversity into grouped unseen-subject gain",
            "architecture": architecture,
            "features": "legal frozen development expert outputs only",
            "loss": "cross-entropy or matched static evaluation",
            "grouped_validation": "five subject-disjoint folds; disjoint inner calibration where trainable",
            "BA": ba,
            "Delta_BA_vs_B_STRONG": delta,
            "Macro_F1": np.nan,
            "NLL": np.nan,
            "Brier": np.nan,
            "switch_rate": np.nan,
            "rescue_count": np.nan,
            "harm_count": np.nan,
            "worst_subject_effect": np.nan,
            "positive_subject_fraction": np.nan,
            "result": result,
            "OUTER_TEST_USED": False,
        }
        rows.append(payload)
        (RESEARCH_LOG / f"ITERATION_{iteration:03d}.md").write_text(
            f"# Iteration {iteration:03d}: {model_id}\n\n"
            f"- Diagnosis: {diagnosis}\n"
            f"- Architecture: `{architecture}`\n"
            f"- Grouped mean subject BA: `{ba:.6f}`\n"
            f"- Delta vs benchmark-specific B_STRONG: `{100*delta:+.3f} pp`\n"
            f"- Decision: `{result}`\n"
            f"- `OUTER_TEST_USED=false`\n",
            encoding="utf-8",
        )
    write_csv(existing_path, pd.concat([existing, pd.DataFrame(rows)], ignore_index=True))


def _write_failure_and_rescue_tables() -> None:
    path = DIAGNOSTICS / "FAILURE_ANALYSIS.csv"
    old = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if "stage" in old:
        old = old[~old.stage.astype(str).str.startswith("final_")]
    extra = pd.DataFrame(
        [
            {"stage": "final_wbcic_transfer", "evidence": "direct masked-pool Delta=-0.001698", "diagnosis": "architecture-specific transfer failed after correcting B_STRONG to probability mean", "next_hypothesis": "do not open outer; benchmark-adapted generic stacking remains exploratory", "OUTER_TEST_USED": False},
            {"stage": "final_wbcic_search", "evidence": "raw linear Delta=+0.003163, CI95=[-0.001837,+0.008173]", "diagnosis": "positive signal lacks subject-level precision and is sensitive to calibration threshold", "next_hypothesis": "stop tuning to limit development overfit", "OUTER_TEST_USED": False},
            {"stage": "final_action", "evidence": "A2-A1=-0.004519, CI95 fully below zero", "diagnosis": "ACTION information destroys the constrained KEEP gain", "next_hypothesis": "exclude ACTION from the selected final discovery model", "OUTER_TEST_USED": False},
            {"stage": "final_persist", "evidence": "A3-A2=-0.000865, CI95=[-0.002981,+0.001250]", "diagnosis": "PERSIST adds neither raw gain nor safety in the matched ladder", "next_hypothesis": "retain PERSIST as diagnostic science, not a performance claim", "OUTER_TEST_USED": False},
        ]
    )
    write_csv(path, pd.concat([old, extra], ignore_index=True))
    open_ablation = pd.read_csv(OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv")
    wbcic = pd.read_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv")
    rescue = pd.concat(
        [
            open_ablation.assign(dataset="OpenBMI"),
            wbcic[wbcic.method_id.isin(["W0_B_STRONG_PROBABILITY_MEAN", "W1_RAW_LINEAR", "W1_DEEPSETS_KEEP"])].assign(dataset="WBCIC-development"),
        ],
        ignore_index=True,
        sort=False,
    )
    keep = ["dataset", "method_id", "switch_rate", "rescue_count", "harm_count", "rescue_precision", "harm_rate", "worst_subject_delta", "OUTER_TEST_USED"]
    write_csv(DIAGNOSTICS / "RESCUE_HARM.csv", rescue[[column for column in keep if column in rescue]])


def _write_figure_contract() -> None:
    (OUTPUTS / "design" / "FIGURE_CONTRACT.md").write_text(
        """# V4 figure contract

## Core conclusion

Constrained dynamic KEEP aggregation produces a robust exploratory OpenBMI gain,
whereas ACTION and PERSIST do not add prospective value and no architecture is
robustly positive on both OpenBMI and WBCIC development.

## Evidence chain

1. Figure 1a: A0-A3 OpenBMI performance relative to the strongest static ensemble.
2. Figure 1b: WBCIC development transfer and adapted controls.
3. Figure 1c: all 52 OpenBMI subject effects, with no subject omitted.
4. Figure 1d: the remaining oracle headroom and the selection bottleneck.
5. Figure 2: paired subject-bootstrap increments for every mandatory component ablation.
6. Figure 3: cross-benchmark gains, separating direct transfer from benchmark adaptation.

## Contract

- Archetype: quantitative grid with a cross-benchmark hero comparison.
- Backend: Python/matplotlib exclusively.
- Exclusions: none; all 52 OpenBMI and 41 WBCIC development subjects are retained.
- Statistics: paired subject bootstrap, 10,000 repetitions; CIs are 95% intervals.
- Export: editable SVG/PDF plus 600-dpi PNG on white background.
- Source data: exact CSVs are stored under `outputs/figures/source_data/`.

## Reviewer risks

OpenBMI is exploratory after repeated development use; WBCIC intervals cross
zero; the initially used WBCIC logit-mean baseline was corrected to the stronger
probability-mean baseline; oracle panels are explicitly non-prospective.
""",
        encoding="utf-8",
    )


def _write_final_spec(expert_usage: pd.DataFrame) -> dict[str, Any]:
    open_summary = expert_usage[
        expert_usage.dataset.eq("OpenBMI") & expert_usage.outer_fold.astype(str).eq("mean_across_folds")
    ][["expert_or_feature", "value"]]
    spec = {
        "status": "FINAL_DEVELOPMENT_RESEARCH_MODEL_FROZEN",
        "terminal_state": "GENERIC_DYNAMIC_ENSEMBLE_WINS",
        "selected_discovery_model": "A1_DYNAMIC_KEEP_FINAL / M1_MASKED_POOL_SHRUNK_THR",
        "architecture": {
            "name": "Masked positive logit pool",
            "inputs": "six frozen KEEP binary margins, availability mask, legal session indicator",
            "pool": "exp(theta_j) positive weights normalized over available experts",
            "output": "learned global positive scale times pooled logit plus bias",
            "action_experts": "excluded by final evidence",
            "PERSIST_features": "excluded from selected performance model",
        },
        "training_protocol": {
            "outer_validation": "five subject-disjoint folds",
            "inner_calibration": "one disjoint subject fold per outer fold",
            "fit_subjects": "remaining subject folds",
            "loss": "binary cross-entropy plus L2 on log-weights",
            "l2_grid": [0.01, 0.1, 1.0],
            "learn_scale": True,
            "session_specific_weights": False,
            "threshold_grid": [0.475, 0.5, 0.525],
            "selection_tiebreak": [
                "mean subject Delta BA",
                "worst-subject Delta BA",
                "lower harm rate",
                "higher rescue precision",
                "lower switch rate",
            ],
            "seed": V4_SEED,
        },
        "mean_openbmi_expert_weights": dict(open_summary.itertuples(index=False, name=None)),
        "benchmark_adaptation": {
            "OpenBMI": "selected constrained pool; exploratory only",
            "WBCIC_development": "best observed model is raw L2 linear stacking; CI crosses zero and it is not authorized for outer evaluation",
        },
        "outer_evaluation_authorized": False,
        "outer_block_reason": "direct transfer is negative and WBCIC adapted gain has a subject-bootstrap CI crossing zero",
        "outer_subject_ids_loaded": False,
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_LOCK / "FINAL_MODEL_SPEC.json", spec)
    weights = "\n".join(f"- `{name}`: `{value:.6f}`" for name, value in spec["mean_openbmi_expert_weights"].items())
    (FINAL_LOCK / "FINAL_MODEL_SPEC.md").write_text(
        f"""# V4 final development research model specification

Terminal state: `{spec['terminal_state']}`

The selected discovery model is a positive, availability-normalized pool over
six frozen KEEP margins. It learns one global scale and bias. L2 is selected
from `{{0.01, 0.1, 1.0}}`; the inner threshold is restricted to
`{{0.475, 0.500, 0.525}}`. All selection is subject-disjoint.

## Mean OpenBMI fold weights

{weights}

ACTION experts and PERSIST inputs are excluded from the selected performance
model because their matched incremental effects are non-positive. This is a
development-research freeze, not an outer-evaluation authorization. The WBCIC
direct transfer was negative and the best adapted generic linear stack had a
confidence interval crossing zero.

- Outer subject IDs loaded: `false`
- `OUTER_TEST_USED=false`
""",
        encoding="utf-8",
    )
    return spec


def _write_report(table: pd.DataFrame, cross: pd.DataFrame, spec: dict[str, Any]) -> None:
    a = pd.read_csv(OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv").set_index("method_id")
    inc = pd.read_csv(OUTPUTS / "ablations" / "PERSIST_INCREMENTAL_VALUE.csv").set_index("comparison")
    open_leader = pd.read_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv").set_index("method_id")
    wbcic = pd.read_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv").set_index("method_id")
    wbcic_transfer = pd.read_csv(LEADERBOARD / "WBCIC_DEV_MODEL_LEADERBOARD.csv").set_index("method_id")
    baseline = _read_json(OUTPUTS / "protocol" / "BASELINE_RECONSTRUCTION.json")
    final = a.loc["A1_DYNAMIC_KEEP_FINAL"]
    wbest = wbcic.loc["W1_RAW_LINEAR"]
    compact = table[["dataset", "table_role", "method_id", "mean_subject_BA", "Delta_BA_vs_B_STRONG", "CI95_L", "CI95_U"]].copy()
    report = f"""# PERSIST-EEG V4 scientific report

## Executive decision

Terminal state: `GENERIC_DYNAMIC_ENSEMBLE_WINS`.

The only robust constructive gain is the generic constrained dynamic KEEP pool
on exploratory OpenBMI: BA `{final.mean_subject_BA:.6f}`, Delta
`{100*final.Delta_BA_vs_B_STRONG:+.3f} pp`, paired subject-bootstrap 95% CI
`[{100*final.CI95_L:+.3f}, {100*final.CI95_U:+.3f}] pp`, with all five folds
positive. ACTION loses `{100*inc.loc['ACTION_increment','Delta_BA']:+.3f} pp`
relative to that model. PERSIST then changes BA by
`{100*inc.loc['PERSIST_increment','Delta_BA']:+.3f} pp` with CI
`[{100*inc.loc['PERSIST_increment','CI95_L']:+.3f}, {100*inc.loc['PERSIST_increment','CI95_U']:+.3f}] pp`
and worsens rather than improves the measured harm/worst-subject endpoints.

WBCIC development does not confirm the architecture. The corrected strongest
static reference is the five-expert probability mean (BA
`{wbcic.loc['W0_B_STRONG_PROBABILITY_MEAN','mean_subject_BA']:.6f}`). Direct
masked-pool transfer is `{100*wbcic_transfer.loc['W1_MASKED_POOL_SHRUNK_THR','Delta_BA_vs_WBCIC_B_STRONG']:+.3f} pp`.
The best adapted generic linear stack reaches BA `{wbest.mean_subject_BA:.6f}`
and `{100*wbest.Delta_BA_vs_B_STRONG:+.3f} pp`, but its CI
`[{100*wbest.CI95_L:+.3f}, {100*wbest.CI95_U:+.3f}] pp` crosses zero. Therefore
`READY_FOR_OUTER_FREEZE` is not justified.

## Final development table

{markdown_table(compact)}

## Answers to the required questions

1. **Strongest static ensemble.** OpenBMI: `B6_ALL_RUN_LOGIT_MEAN`, BA
   `{baseline['B_STRONG_mean_subject_BA']:.6f}`. WBCIC-dev: the five competent
   experts' probability mean, BA `{wbcic.loc['W0_B_STRONG_PROBABILITY_MEAN','mean_subject_BA']:.6f}`.
2. **Best final architecture.** A positive availability-normalized pool over
   frozen KEEP logits, with L2-shrunk log-weights, one scale/bias, and a narrow
   inner-calibrated threshold. It is a discovery model, not an outer-ready model.
3. **Major families tested.** Eight: threshold calibration, generic linear
   stacking, shallow HGB, anchored bounded residuals, positive logit pooling,
   positive probability pooling, contextual pooling, and DeepSets.
4. **Failures.** Trees and DeepSets lacked stable grouped gain; probability
   pooling did not beat its static WBCIC counterpart; residual corrections
   remained unstable; ACTION was harm-dominated; flat PERSIST inputs added no
   independent value; direct cross-benchmark transfer failed.
5. **Final Delta BA.** OpenBMI `{100*final.Delta_BA_vs_B_STRONG:+.3f} pp`.
   WBCIC best adapted exploratory candidate `{100*wbest.Delta_BA_vs_B_STRONG:+.3f} pp`.
6. **Grouped CI.** OpenBMI
   `[{100*final.CI95_L:+.3f}, {100*final.CI95_U:+.3f}] pp`; WBCIC adapted
   `[{100*wbest.CI95_L:+.3f}, {100*wbest.CI95_U:+.3f}] pp`.
7. **Stability.** OpenBMI 5/5 folds positive, 50.0% subjects positive and 82.7%
   nonnegative, worst subject `-1.5 pp`. WBCIC 4/5 folds positive but only 43.9%
   subjects positive, 65.9% nonnegative, worst subject `-3.5 pp`.
8. **Dynamic KEEP value.** Yes on OpenBMI (`+0.452 pp`, LCB > 0); not confirmed
   on WBCIC.
9. **ACTION value.** No. A2-A1 is
   `{100*inc.loc['ACTION_increment','Delta_BA']:+.3f} pp` with a fully negative CI.
10. **PERSIST raw value.** No. A3-A2 is
    `{100*inc.loc['PERSIST_increment','Delta_BA']:+.3f} pp`; CI crosses zero.
11. **PERSIST safety/robustness.** No. It increases harm rate by
    `{100*inc.loc['PERSIST_increment','Delta_harm_rate']:+.3f} pp` and worsens
    the worst-subject endpoint by `{100*inc.loc['PERSIST_increment','Delta_worst_subject']:+.3f} pp`.
12. **PERSIST features.** No category has reliable incremental value. Decision
    dependence and persistence-only increments are small with CIs crossing zero;
    protected inputs have a negative point estimate.
13. **Selected experts.** The winning model uses KEEP experts only. Fold-level
    weights are in `diagnostics/EXPERT_USAGE.csv`; no ACTION expert is selected.
14. **ERASE necessary.** No. It is excluded; WBCIC's existing development audit
    also found no actionable harmful block and large harm from erasing protected structure.
15. **Soft residual vs hard switch.** No evidence of superiority. Bounded
    residual correction remains below B_STRONG, and the V3 hard policies also fail.
16. **Generic stacker.** Yes: the final gain is generic dynamic ensemble
    aggregation, not a PERSIST-aware method. On WBCIC, generic linear stacking
    is also the best point estimate, but not statistically robust.
17. **Frozen representations.** Not evaluated as a final gate. Logit models had
    a real OpenBMI signal, while compatible representations for the full WBCIC
    expert roster were unavailable; adding a single-backbone representation
    would confound architecture and capacity.
18. **Transfer.** Direct transfer fails (`{100*wbcic_transfer.loc['W1_MASKED_POOL_SHRUNK_THR','Delta_BA_vs_WBCIC_B_STRONG']:+.3f} pp`).
    Only benchmark-adapted generic stacking has positive point estimates on both.
19. **Best OpenBMI development performance.** BA `{final.mean_subject_BA:.6f}`.
20. **Best WBCIC development performance.** BA `{wbest.mean_subject_BA:.6f}`,
    exploratory and non-robust.
21. **Gain vs original single model.** OpenBMI final vs historical pooled B0:
    `{100*(final.mean_subject_BA-0.8233173076923076):+.3f} pp`. WBCIC adapted
    model vs EEGNet_STABLE: `{100*(wbest.mean_subject_BA-0.794355):+.3f} pp`.
22. **Gain vs B_STRONG.** OpenBMI `+0.452 pp`; WBCIC adapted `+0.316 pp` with
    CI crossing zero.
23. **Gain decomposition.** OpenBMI single-to-static: approximately `+2.313 pp`;
    static-to-dynamic KEEP: `+0.452 pp`; ACTION beyond dynamic KEEP: `-0.452 pp`;
    PERSIST beyond KEEP+ACTION: `-0.087 pp`.
24. **Outer evaluation justified?** No. OpenBMI is exploratory, direct WBCIC
    transfer is negative, and the adapted WBCIC candidate has LCB < 0.
25. **What must be frozen before outer.** The WBCIC expert checkpoint roster and
    hashes, probability-mean B_STRONG, one model family (not several candidates),
    preprocessing, exact C/regularization, a single calibration/threshold rule,
    seed, legality hashes, and evaluation script. V4 intentionally does not make
    that outer authorization.

## Scientific interpretation

The oracle gaps remain large (`+{100*baseline['KEEP_only_oracle_delta_BA']:.3f} pp`
for KEEP-only and `+{100*baseline['complete_KEEP_ACTION_oracle_delta_BA']:.3f} pp`
for the complete menu), but prospective models recover only a small fraction.
This supports a selection-bottleneck conclusion. It does not support the claim
that PERSIST diagnostics currently improve prediction.

## Legality

- OpenBMI is exploratory only.
- WBCIC uses 41 authorized development subjects and S3 only.
- Sealed outer IDs, labels, logits, metadata, and results were never loaded.
- `OUTER_TEST_USED=false`.
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")


def _write_decision(spec: dict[str, Any]) -> None:
    a = pd.read_csv(OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv").set_index("method_id")
    w = pd.read_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv").set_index("method_id")
    transfer = pd.read_csv(LEADERBOARD / "WBCIC_DEV_MODEL_LEADERBOARD.csv").set_index("method_id")
    final = a.loc["A1_DYNAMIC_KEEP_FINAL"]
    decision = {
        "terminal_state": "GENERIC_DYNAMIC_ENSEMBLE_WINS",
        "selected_discovery_model": spec["selected_discovery_model"],
        "OpenBMI": {
            "role": "exploratory_discovery",
            "BA": float(final.mean_subject_BA),
            "Delta_BA_vs_B_STRONG": float(final.Delta_BA_vs_B_STRONG),
            "CI95": [float(final.CI95_L), float(final.CI95_U)],
            "positive_folds": "5/5",
            "strong_candidate": True,
        },
        "WBCIC_development": {
            "B_STRONG": "W0_B_STRONG_PROBABILITY_MEAN",
            "direct_transfer_Delta_BA": float(transfer.loc["W1_MASKED_POOL_SHRUNK_THR", "Delta_BA_vs_WBCIC_B_STRONG"]),
            "best_adapted_method": "W1_RAW_LINEAR",
            "best_adapted_BA": float(w.loc["W1_RAW_LINEAR", "mean_subject_BA"]),
            "best_adapted_Delta_BA": float(w.loc["W1_RAW_LINEAR", "Delta_BA_vs_B_STRONG"]),
            "best_adapted_CI95": [float(w.loc["W1_RAW_LINEAR", "CI95_L"]), float(w.loc["W1_RAW_LINEAR", "CI95_U"])],
            "strong_candidate": False,
        },
        "ACTION_adds_gain": False,
        "PERSIST_adds_gain": False,
        "PERSIST_improves_safety": False,
        "READY_FOR_OUTER_FREEZE": False,
        "outer_evaluation_authorized": False,
        "outer_subject_ids_loaded": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", decision)


def _write_hashes_and_reproducibility() -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    code_files = sorted((experiment_root / "code").rglob("*.py"))
    key_outputs = [
        OUTPUTS / "protocol" / "BASELINE_RECONSTRUCTION.json",
        OUTPUTS / "protocol" / "DATA_LEGALITY_AUDIT.json",
        OUTPUTS / "protocol" / "WBCIC_EXPERT_TABLE_AUDIT.json",
        OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv",
        OUTPUTS / "ablations" / "PERSIST_INCREMENTAL_VALUE.csv",
        LEADERBOARD / "FINAL_DEVELOPMENT_TABLE.csv",
        LEADERBOARD / "CROSS_BENCHMARK_LEADERBOARD.csv",
        OUTPUTS / "FINAL_DECISION.json",
        OUTPUTS / "SCIENTIFIC_REPORT.md",
        FINAL_LOCK / "FINAL_MODEL_SPEC.json",
    ]
    hashes = {
        "status": "FINAL_MODEL_HASHES_FROZEN",
        "git_branch": _git("branch", "--show-current"),
        "git_starting_commit": _read_json(OUTPUTS / "protocol" / "BASELINE_RECONSTRUCTION.json")["starting_commit"],
        "code": {str(path.relative_to(experiment_root)): sha256_file(path) for path in code_files},
        "key_outputs": {
            str(path.relative_to(experiment_root)): sha256_file(path) for path in key_outputs if path.is_file()
        },
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_LOCK / "FINAL_MODEL_HASHES.json", hashes)
    commands = [
        "python code/reconstruct.py",
        "python code/run_search.py --stage initial",
        "python code/run_keep_refine.py",
        "python code/run_keep_dynamic.py",
        "python code/build_wbcic_experts.py --device cuda --workers 0",
        "python code/audit_wbcic_static.py",
        "python code/run_wbcic_transfer.py",
        "python code/run_wbcic_keep_search.py",
        "python code/run_wbcic_linear_refine.py",
        "python code/run_final_ablations.py",
        "python code/run_final.py",
    ]
    reproducibility = {
        "status": "V4_FINAL_REPRODUCIBILITY_COMPLETE",
        "branch": _git("branch", "--show-current"),
        "working_commit_before_V4_commit": _git("rev-parse", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": V4_SEED,
        "bootstrap_repetitions": 10000,
        "commands": commands,
        "final_hash_file_sha256": sha256_file(FINAL_LOCK / "FINAL_MODEL_HASHES.json"),
        "outer_split_lock_opened": False,
        "outer_subject_ids_loaded": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "REPRODUCIBILITY.json", reproducibility)


def run(openbmi_cache: Path, wbcic_repo: Path) -> None:
    ensure_directories()
    _write_figure_contract()
    expert_usage = _extract_expert_usage(openbmi_cache, wbcic_repo)
    table = _development_table(openbmi_cache, wbcic_repo)
    cross = _write_cross_benchmark()
    _write_research_logs()
    _write_failure_and_rescue_tables()
    spec = _write_final_spec(expert_usage)
    _write_decision(spec)
    _write_report(table, cross, spec)
    make_figures()
    _write_hashes_and_reproducibility()
    print(
        json.dumps(
            {
                "terminal_state": "GENERIC_DYNAMIC_ENSEMBLE_WINS",
                "READY_FOR_OUTER_FREEZE": False,
                "OUTER_TEST_USED": False,
            },
            indent=2,
        )
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--openbmi-cache", type=Path, default=default_openbmi_cache())
    parser.add_argument("--wbcic-repo", type=Path, default=default_wbcic_repo())
    args = parser.parse_args()
    run(args.openbmi_cache, args.wbcic_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
