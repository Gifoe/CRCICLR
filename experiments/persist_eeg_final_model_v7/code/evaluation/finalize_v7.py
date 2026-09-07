from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import balanced_accuracy_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import (
    ABLATIONS,
    BASELINES,
    DIAGNOSTICS,
    EXPERIMENT_ROOT,
    FINAL_CANDIDATE,
    LEADERBOARD,
    OUTPUTS,
    PROTOCOL,
    RESEARCH_LOG,
    V7_SEED,
    ensure_directories,
    sha256_file,
    v6_outputs,
    write_csv,
    write_json,
)


OPEN_ANCHOR = "MI_SPECIFIC_BACKBONE_ADAPTED"
WBCIC_ANCHOR = "V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED"
OPEN_GENERIC = "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD"
WBCIC_GENERIC = "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD"
OPEN_PERSIST = "ANCHOR_BLEND_PERSIST_META"
OPEN_META = "ANCHOR_BLEND_META_GENERIC"
WBCIC_PERSIST = "ANCHOR_PLUS_PERSIST_META_RESIDUAL"
WBCIC_META = "ANCHOR_PLUS_META_GENERIC_RESIDUAL"


def _read_existing(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def _collect_leaderboard(prefix: str, expected_subjects: int) -> pd.DataFrame:
    frames = []
    for path in LEADERBOARD.glob(f"{prefix}_*.csv"):
        if path.name in {"OPENBMI_V7.csv", "WBCIC_DEV_V7.csv"}:
            pass
        frame = pd.read_csv(path)
        if "subjects" in frame and len(frame):
            frame = frame.loc[frame.subjects.astype(int).eq(expected_subjects)].copy()
        if len(frame):
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No complete leaderboard for {prefix}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["mean_subject_BA", "method_id"], ascending=[False, True])
    combined = combined.drop_duplicates("method_id", keep="first").reset_index(drop=True)
    return combined


def _prediction(path: Path, method: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame.method_id.astype(str).eq(method)].copy()
    frame = frame.loc[frame.trial_uid.astype(str).str.startswith(prefix)].copy()
    if frame.trial_uid.duplicated().any() or frame.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError(f"Malformed prediction {path} {method}")
    frame["subject_id"] = frame.subject_id.astype(str)
    return frame


def _subject_metric(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id"):
        rows.append({
            "subject_id": str(subject),
            "BA": float(balanced_accuracy_score(group.label, group.prediction)),
            "NLL": float(log_loss(group.label, np.clip(group.probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
        })
    return pd.DataFrame(rows).set_index("subject_id")


def _paired(source: pd.DataFrame, reference: pd.DataFrame, draws: int = 20_000) -> dict:
    source_metric = _subject_metric(source)
    reference_metric = _subject_metric(reference)
    subjects = sorted(set(source_metric.index) & set(reference_metric.index))
    if len(subjects) != len(source_metric) or len(subjects) != len(reference_metric):
        raise RuntimeError("Paired prediction coverage mismatch")
    delta = source_metric.loc[subjects, "BA"].to_numpy(float) - reference_metric.loc[subjects, "BA"].to_numpy(float)
    generator = np.random.default_rng(V7_SEED)
    bootstrap = np.empty(draws, dtype=float)
    for start in range(0, draws, 1_000):
        stop = min(start + 1_000, draws)
        indices = generator.integers(0, len(delta), size=(stop - start, len(delta)))
        bootstrap[start:stop] = delta[indices].mean(axis=1)
    return {
        "subjects": len(subjects),
        "Delta_BA": float(delta.mean()),
        "CI95_L": float(np.quantile(bootstrap, 0.025)),
        "CI95_U": float(np.quantile(bootstrap, 0.975)),
        "median_subject_delta": float(np.median(delta)),
        "positive_subject_fraction": float(np.mean(delta > 0.0)),
        "nonnegative_subject_fraction": float(np.mean(delta >= 0.0)),
        "worst_subject_delta": float(delta.min()),
        "bootstrap_draws": draws,
    }


def _selected_row(frame: pd.DataFrame, method: str) -> dict:
    selected = frame.loc[frame.method_id.astype(str).eq(method)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {method}, got {len(selected)}")
    return selected.iloc[0].to_dict()


def _utility_summary() -> dict:
    frame = pd.read_csv(DIAGNOSTICS / "UTILITY_PREDICTABILITY.csv")
    result = {}
    for benchmark, group in frame.groupby("benchmark"):
        result[benchmark] = {}
        for mode, part in group.groupby("mode"):
            result[benchmark][mode] = {
                key: float(part[key].mean())
                for key in ("utility_R2", "utility_pearson", "utility_spearman", "utility_sign_accuracy")
            }
    outcome = pd.read_csv(DIAGNOSTICS / "OUTCOME_PROSPECTIVE_UTILITY.csv")
    for (benchmark, mode), group in outcome.groupby(["benchmark", "controller_mode"]):
        target = group.future_ce_gain.to_numpy(float)
        prediction = group.predicted_utility.to_numpy(float)
        result[benchmark][mode]["outcome_utility_pearson"] = float(np.corrcoef(target, prediction)[0, 1])
        result[benchmark][mode]["outcome_utility_sign_accuracy"] = float(np.mean((target > 0) == (prediction > 0)))
    return result


def _component_outputs() -> None:
    future = pd.read_csv(DIAGNOSTICS / "FUTURE_UTILITY_DIAGNOSTICS.csv")
    granularity = future.groupby(["benchmark", "family", "component_id"], as_index=False).agg(
        mean_realized_CE_utility=("future_ce_gain", "mean"),
        mean_realized_BA_utility=("future_ba_gain", "mean"),
        positive_utility_fraction=("future_ce_gain", lambda value: float(np.mean(value > 0))),
        subjects=("subject_id", "nunique"),
    )
    granularity["OUTER_TEST_USED"] = False
    write_csv(ABLATIONS / "ADAPTATION_GRANULARITY.csv", granularity)
    utility = pd.read_csv(DIAGNOSTICS / "UTILITY_PREDICTABILITY.csv")
    write_csv(ABLATIONS / "HISTORY_ENCODER_ABLATION.csv", utility)
    risk = pd.read_csv(ABLATIONS / "META_ABLATION.csv")
    write_csv(ABLATIONS / "RISK_ABLATION.csv", risk)
    pairs = risk.sort_values(
        ["benchmark", "outer_fold", "mode", "mean_subject_BA", "harmful_subject_fraction"],
        ascending=[True, True, True, False, True],
    ).drop_duplicates(["benchmark", "outer_fold", "mode"], keep="first").reset_index(drop=True)
    pairs["capacity_matched"] = True
    write_csv(ABLATIONS / "CAPACITY_MATCHED_CONTROLS.csv", pairs)


def _backbone_ablation(open_table: pd.DataFrame, wbcic_table: pd.DataFrame) -> None:
    rows = []
    family_map = {
        "MI_SPECIFIC": "V6 selected MI-specific EEGNet/Shallow",
        "CAPACITY_MATCHED": "wide EEGNet identity control",
        "HISTORY_EA": "history Euclidean alignment",
        "FBCVARIANCE": "filter-bank log-variance",
        "CONFORMER_NORM": "compact EEG-Conformer",
        "CONFORMER_CCALIGN": "class-conditional session alignment",
        "HYPER": "low-rank history hypernetwork",
    }
    for short, table in (("OpenBMI", open_table), ("WBCIC", wbcic_table)):
        for _, row in table.iterrows():
            method = str(row.method_id)
            family = next((value for key, value in family_map.items() if key in method), "initial utility-controller / anchor")
            rows.append({
                "benchmark": short,
                "method_id": method,
                "family": family,
                "mean_subject_BA": float(row.mean_subject_BA),
                "subjects": int(row.subjects),
                "source_file": row.get("source_file"),
                "OUTER_TEST_USED": False,
            })
    write_csv(ABLATIONS / "BACKBONE_ABLATION.csv", pd.DataFrame(rows).sort_values(["benchmark", "mean_subject_BA"], ascending=[True, False]))


def run() -> None:
    ensure_directories()
    open_table = _collect_leaderboard("OPENBMI", 54)
    wbcic_table = _collect_leaderboard("WBCIC", 41)
    # Ensure initial V7 tables participate even though their names are exact.
    for path, target, expected in ((LEADERBOARD / "OPENBMI_V7.csv", "open", 54), (LEADERBOARD / "WBCIC_DEV_V7.csv", "wbcic", 41)):
        frame = pd.read_csv(path)
        frame = frame.loc[frame.subjects.astype(int).eq(expected)].copy()
        table = open_table if target == "open" else wbcic_table
        table = pd.concat([table, frame.assign(source_file=path.name)], ignore_index=True).sort_values("mean_subject_BA", ascending=False).drop_duplicates("method_id")
        if target == "open": open_table = table
        else: wbcic_table = table
    write_csv(LEADERBOARD / "OPENBMI_V7.csv", open_table)
    write_csv(LEADERBOARD / "WBCIC_DEV_V7.csv", wbcic_table)
    cross = pd.concat([open_table.assign(benchmark_short="OpenBMI"), wbcic_table.assign(benchmark_short="WBCIC")], ignore_index=True, sort=False)
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_V7.csv", cross)
    _component_outputs()
    _backbone_ablation(open_table, wbcic_table)

    initial_predictions = DIAGNOSTICS / "V7_PREDICTIONS.csv"
    open_anchor = _prediction(v6_outputs() / "diagnostics" / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv", OPEN_ANCHOR, "OpenBMI_")
    wbcic_anchor = _prediction(v6_outputs() / "diagnostics" / "WBCIC_FUTURE_SESSION_POPULATION_PREDICTIONS.csv", WBCIC_ANCHOR, "WBCIC_")
    open_generic = _prediction(DIAGNOSTICS / "OPENBMI_CONFORMER_NORM_PREDICTIONS.csv", OPEN_GENERIC, "OpenBMI_")
    wbcic_generic = _prediction(DIAGNOSTICS / "WBCIC_CONFORMER_NORM_PREDICTIONS.csv", WBCIC_GENERIC, "WBCIC_")
    open_persist = _prediction(initial_predictions, OPEN_PERSIST, "OpenBMI_")
    open_meta = _prediction(initial_predictions, OPEN_META, "OpenBMI_")
    wbcic_persist = _prediction(initial_predictions, WBCIC_PERSIST, "WBCIC_")
    wbcic_meta = _prediction(initial_predictions, WBCIC_META, "WBCIC_")
    statistics = {
        "OpenBMI": {
            "strongest_generic_vs_V6_anchor": _paired(open_generic, open_anchor),
            "PERSIST_vs_capacity_matched_META_GENERIC": _paired(open_persist, open_meta),
        },
        "WBCIC": {
            "strongest_generic_vs_V6_anchor": _paired(wbcic_generic, wbcic_anchor),
            "PERSIST_vs_capacity_matched_META_GENERIC": _paired(wbcic_persist, wbcic_meta),
        },
    }
    utility = _utility_summary()
    actions = pd.read_csv(DIAGNOSTICS / "OUTCOME_ACTION_SELECTIONS.csv")
    action_summary = actions.groupby(["benchmark", "mode"], as_index=False).agg(
        adapt_fraction=("action", lambda value: float(np.mean(value == "ADAPT"))),
        harmful_subject_fraction=("Delta_BA", lambda value: float(np.mean(value < 0))),
        worst_subject_delta=("Delta_BA", "min"),
    ).to_dict("records")
    headroom = json.loads((DIAGNOSTICS / "NEW_BACKBONE_HEADROOM.json").read_text(encoding="utf-8"))
    positive_control = json.loads((DIAGNOSTICS / "V7_POSITIVE_CONTROL.json").read_text(encoding="utf-8"))

    open_best = _selected_row(open_table, OPEN_GENERIC)
    wbcic_best = _selected_row(wbcic_table, WBCIC_GENERIC)
    open_persist_row = _selected_row(open_table, OPEN_PERSIST)
    wbcic_persist_row = _selected_row(wbcic_table, WBCIC_PERSIST)
    matched_baselines = pd.DataFrame([
        {"benchmark": "OpenBMI", "method_id": OPEN_GENERIC, "mean_subject_BA": open_best["mean_subject_BA"], "history_matched": True, "OUTER_TEST_USED": False},
        {"benchmark": "WBCIC", "method_id": WBCIC_GENERIC, "mean_subject_BA": wbcic_best["mean_subject_BA"], "history_matched": True, "OUTER_TEST_USED": False},
    ])
    write_csv(BASELINES / "OPENBMI_MATCHED_BASELINES.csv", open_table)
    write_csv(BASELINES / "WBCIC_MATCHED_BASELINES.csv", wbcic_table)
    evolution = pd.DataFrame([
        {"benchmark": "OpenBMI", "stage": "V6", "method_id": OPEN_ANCHOR, "mean_subject_BA": 0.832037037037037},
        {"benchmark": "OpenBMI", "stage": "V7 strongest generic", "method_id": OPEN_GENERIC, "mean_subject_BA": open_best["mean_subject_BA"]},
        {"benchmark": "OpenBMI", "stage": "V7 best PERSIST", "method_id": OPEN_PERSIST, "mean_subject_BA": open_persist_row["mean_subject_BA"]},
        {"benchmark": "WBCIC", "stage": "V6", "method_id": WBCIC_ANCHOR, "mean_subject_BA": 0.8208170115792067},
        {"benchmark": "WBCIC", "stage": "V7 strongest generic", "method_id": WBCIC_GENERIC, "mean_subject_BA": wbcic_best["mean_subject_BA"]},
        {"benchmark": "WBCIC", "stage": "V7 best PERSIST", "method_id": WBCIC_PERSIST, "mean_subject_BA": wbcic_persist_row["mean_subject_BA"]},
    ])
    evolution["OUTER_TEST_USED"] = False
    write_csv(LEADERBOARD / "BASELINE_EVOLUTION.csv", evolution)

    final_results = {
        "OpenBMI": {
            "EEGNet_reference_BA": 0.752962962962963,
            "V6_strong_anchor_BA": 0.832037037037037,
            "strongest_fair_V7_method": OPEN_GENERIC,
            "strongest_fair_V7_BA": float(open_best["mean_subject_BA"]),
            "best_PERSIST_method": OPEN_PERSIST,
            "best_PERSIST_BA": float(open_persist_row["mean_subject_BA"]),
            "target_BA": 0.882037037037037,
            "target_reached": False,
            "plus_5pp_over_EEGNet": float(open_best["mean_subject_BA"]) >= 0.802962962962963,
            **statistics["OpenBMI"],
        },
        "WBCIC": {
            "EEGNet_reference_BA": 0.794354982754373,
            "V6_strong_anchor_BA": 0.8208170115792067,
            "strongest_fair_V7_method": WBCIC_GENERIC,
            "strongest_fair_V7_BA": float(wbcic_best["mean_subject_BA"]),
            "best_PERSIST_method": WBCIC_PERSIST,
            "best_PERSIST_BA": float(wbcic_persist_row["mean_subject_BA"]),
            "target_BA": 0.8708170115792067,
            "target_reached": False,
            "plus_5pp_over_EEGNet": float(wbcic_best["mean_subject_BA"]) >= 0.844354982754373,
            **statistics["WBCIC"],
        },
    }
    write_json(FINAL_CANDIDATE / "DEVELOPMENT_RESULTS.json", final_results)
    write_json(FINAL_CANDIDATE / "DUAL_BENCHMARK_RESULTS.json", {
        "benchmarks": final_results,
        "dual_plus_5pp_over_strongest_matched": False,
        "dual_plus_5pp_over_EEGNet": False,
        "OUTER_TEST_USED": False,
    })
    spec = {
        "provisional_name": "PERSIST-Meta: Future-Utility-Guided Selective Meta-Adaptation",
        "status": "NOT_FROZEN_FOR_OUTER",
        "backbone": "fold-compatible encoder; strongest observed generic control is anchor + compact EEG-Conformer head",
        "adaptation_components": {"OpenBMI": 17, "WBCIC": 18},
        "history_depth": {"OpenBMI": 1, "WBCIC": 2},
        "utility_controller": "capacity-matched Ridge/ExtraTrees grouped-subject cross-fitting",
        "PERSIST_context": ["P", "U", "D", "G", "R"],
        "risk_policy": "mu - kappa*sigma - threshold; PRESERVE when non-positive",
        "suppression_enabled": False,
        "reason_not_frozen": "PERSIST does not beat the strongest generic control and the +5 pp matched target is missed on both benchmarks.",
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_CANDIDATE / "FINAL_MODEL_SPEC.json", spec)
    (FINAL_CANDIDATE / "FINAL_MODEL_SPEC.md").write_text(
        "# PERSIST-Meta provisional specification\n\n"
        "The candidate uses legal history to describe 17 OpenBMI or 18 WBCIC coarse\n"
        "adaptation actions, predicts future-session utility, and applies only the best\n"
        "positive risk-adjusted action. P/U/D/G/R augment an otherwise capacity-matched\n"
        "generic controller. Empirical component-level utility standard deviation is\n"
        "used as the risk surrogate; it is not calibrated predictive uncertainty.\n\n"
        "The candidate is **not frozen for WBCIC outer evaluation** because PERSIST does\n"
        "not beat the strongest generic Conformer ensemble. Suppression remains\n"
        "disabled.\n",
        encoding="utf-8",
    )
    decision = {
        "terminal_state": "V7_SCIENTIFIC_EXHAUSTION",
        "secondary_state": "V7_PERSIST_UTILITY_SIGNAL_FOUND",
        "READY_FOR_OUTER_FREEZE": False,
        "OUTER_TEST_USED": False,
        "PERSIST_utility_signal_found": True,
        "PERSIST_increment_over_capacity_matched_META_GENERIC_point_estimate": True,
        "PERSIST_increment_over_capacity_matched_META_GENERIC_statistically_resolved": False,
        "PERSIST_increment_over_strongest_generic": False,
        "dual_target_reached": False,
        "suppression_used": False,
        "real_harmful_subspace_certified": False,
        "benchmarks": final_results,
        "utility_predictability": utility,
        "action_summary": action_summary,
        "new_expert_headroom": headroom,
        "synthetic_positive_control": positive_control,
        "distinct_structural_families_attempted": 9,
        "development_status": "adaptive_exploratory",
        "development_outcomes_observed_between_structural_iterations": True,
        "within_run_outcome_future_labels_used_for_fit_or_selection": False,
        "one_seed_exploratory": True,
        "scientific_exhaustion_scope": "Within one-seed heavily reused development data and the authorized V1-V7 model/covariance caches. Further tuning would chiefly reuse the same outcomes; credible progress now requires fresh confirmation or a pre-registered new backbone study.",
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", decision)

    questions = {
        "1_strongest_OpenBMI": {"method": OPEN_GENERIC, "BA": open_best["mean_subject_BA"]},
        "2_strongest_WBCIC": {"method": WBCIC_GENERIC, "BA": wbcic_best["mean_subject_BA"]},
        "3_generic_meta_beats_ordinary": "No consistently; Conformer generic ensemble is strongest, initial META-GENERIC is weaker.",
        "4_future_utility_predictable": "Weakly. WBCIC yes at modest correlation; OpenBMI does not transfer reliably to outcome.",
        "5_metrics": utility,
        "6_best_granularity": "coarse calibration/full-gradient on OpenBMI and low-C ridge/prototype on WBCIC; none has target-level oracle headroom",
        "7_PUDG_improve_prediction": True,
        "7_qualification": "Improvement is an average over correlated one-seed adaptive-development configurations; it is not independent confirmation.",
        "8_PERSIST_beats_META_GENERIC": "Point estimates within the initial matched family only; not the strongest generic control.",
        "9_increment_pp": {"OpenBMI_initial_matched": 100 * statistics["OpenBMI"]["PERSIST_vs_capacity_matched_META_GENERIC"]["Delta_BA"], "WBCIC_initial_matched": 100 * statistics["WBCIC"]["PERSIST_vs_capacity_matched_META_GENERIC"]["Delta_BA"]},
        "10_harm_reduction": "No consistent reduction.",
        "11_worst_subject": "No consistent improvement versus strongest generic.",
        "12_OpenBMI_plus5_matched": False,
        "13_WBCIC_plus5_matched": False,
        "14_same_algorithm_K1_K2": True,
        "15_preserve_adapt_fraction": action_summary,
        "16_suppression_needed": False,
        "17_V6_failure_resolved": "Utility prediction gained a PERSIST signal, but the performance failure was not resolved.",
        "18_backbone_gain_pp": {"OpenBMI": 100 * statistics["OpenBMI"]["strongest_generic_vs_V6_anchor"]["Delta_BA"], "WBCIC": 100 * statistics["WBCIC"]["strongest_generic_vs_V6_anchor"]["Delta_BA"]},
        "19_generic_history_adaptation": "Conformer fixed history head is part of the strongest generic ensemble.",
        "20_meta_learning": "Initial selective controller improves its feature base slightly but remains below the strong anchor.",
        "21_unique_PERSIST": "Better utility predictability and small matched-family point estimates; no robust gain over strongest generic.",
        "22_illegal_future_labels": "No future label entered an outcome-subject fit within a run. However, development outcomes were inspected between structural iterations, so every V7 estimate is adaptive/exploratory and not confirmatory.",
        "23_WBCIC_outer_touched": False,
        "24_distinct_families": 9,
        "25_failure_causes": "insufficient component oracle headroom; EA/session alignment erase useful persistent geometry; FBC/hypernetwork overfit meta subjects; generic Conformer diversity helps only modestly",
        "26_exhaustion": decision["scientific_exhaustion_scope"],
    }
    write_json(FINAL_CANDIDATE / "FINAL_QUESTIONS.json", questions)

    report = f"""# PERSIST-EEG V7 scientific report

## Decision

Terminal state: `V7_SCIENTIFIC_EXHAUSTION`. `READY_FOR_OUTER_FREEZE=false` and
`OUTER_TEST_USED=false`.

V7 finds a weak PERSIST signal in adaptive-development future-utility
prediction. It
does **not** establish a PERSIST performance advantage over the strongest
generic control and it misses the requested +5 pp-over-matched-baseline target.

## Strongest fair results

| Benchmark | Strongest method | BA | Delta vs V6 strong anchor |
|---|---|---:|---:|
| OpenBMI S1 -> S2 | `{OPEN_GENERIC}` | {100*float(open_best['mean_subject_BA']):.3f}% | {100*statistics['OpenBMI']['strongest_generic_vs_V6_anchor']['Delta_BA']:+.3f} pp |
| WBCIC S1/S2 -> S3 dev | `{WBCIC_GENERIC}` | {100*float(wbcic_best['mean_subject_BA']):.3f}% | {100*statistics['WBCIC']['strongest_generic_vs_V6_anchor']['Delta_BA']:+.3f} pp |

The paired subject-bootstrap intervals for those generic gains are
`[{100*statistics['OpenBMI']['strongest_generic_vs_V6_anchor']['CI95_L']:+.3f},{100*statistics['OpenBMI']['strongest_generic_vs_V6_anchor']['CI95_U']:+.3f}]` pp on OpenBMI and
`[{100*statistics['WBCIC']['strongest_generic_vs_V6_anchor']['CI95_L']:+.3f},{100*statistics['WBCIC']['strongest_generic_vs_V6_anchor']['CI95_U']:+.3f}]` pp on WBCIC. Neither gain
is statistically resolved.

Both strongest methods are generic fixed anchor/Conformer-history-head blends.
OpenBMI remains more than five points above its old EEGNet reference; WBCIC is
only {100*(float(wbcic_best['mean_subject_BA'])-0.794354982754373):.3f} pp above its EEGNet reference.

## PERSIST-specific evidence

Within the capacity-matched initial controller family, PERSIST context changes
the point estimate by {100*statistics['OpenBMI']['PERSIST_vs_capacity_matched_META_GENERIC']['Delta_BA']:+.3f} pp on OpenBMI and
{100*statistics['WBCIC']['PERSIST_vs_capacity_matched_META_GENERIC']['Delta_BA']:+.3f} pp on WBCIC. These are small and
their paired intervals, `[{100*statistics['OpenBMI']['PERSIST_vs_capacity_matched_META_GENERIC']['CI95_L']:+.3f},{100*statistics['OpenBMI']['PERSIST_vs_capacity_matched_META_GENERIC']['CI95_U']:+.3f}]` and
`[{100*statistics['WBCIC']['PERSIST_vs_capacity_matched_META_GENERIC']['CI95_L']:+.3f},{100*statistics['WBCIC']['PERSIST_vs_capacity_matched_META_GENERIC']['CI95_U']:+.3f}]` pp, include
zero. They are not sufficient to beat the strongest generic Conformer control.
P/U/D/G/R improve mean utility R2/correlation/sign accuracy in the correlated
fold/controller development summaries, particularly on WBCIC, so a
mechanistic signal exists in this analysis but does not translate into
target-level BA.

## Structural diagnosis

Nine genuinely distinct families were examined. History Euclidean alignment
and class-conditional session alignment hurt, consistent with useful persistent
spatial/session structure being erased. Filter-bank variance and the low-rank
hypernetwork fit training episodes but generalized poorly. Compact Conformer
predictions add modest diversity. Even the outcome-only new-expert subject
oracle reaches only {100*headroom['openbmi']['subject_oracle_BA']:.3f}% on OpenBMI and
{100*headroom['wbcic']['subject_oracle_BA']:.3f}% on WBCIC, so another router
cannot reach the target.

## Limits

All development estimates are exploratory: OpenBMI and WBCIC development have
been heavily reused, and observed development outcomes guided later structural
iterations. One seed was used, as requested. No multi-seed robustness
claim is made. Within a run, outcome-subject future labels never fit or select a
model; across runs, observed development outcomes influenced structural
redesign. The WBCIC outer cohort was never opened or enumerated.
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")

    key_files = [
        OUTPUTS / "FINAL_DECISION.json",
        OUTPUTS / "SCIENTIFIC_REPORT.md",
        FINAL_CANDIDATE / "DEVELOPMENT_RESULTS.json",
        FINAL_CANDIDATE / "FINAL_MODEL_SPEC.json",
        LEADERBOARD / "OPENBMI_V7.csv",
        LEADERBOARD / "WBCIC_DEV_V7.csv",
    ]
    hashes = {str(path.relative_to(EXPERIMENT_ROOT)): sha256_file(path) for path in key_files}
    write_json(FINAL_CANDIDATE / "FINAL_MODEL_HASHES.json", hashes)
    code_hashes = {str(path.relative_to(EXPERIMENT_ROOT)): sha256_file(path) for path in sorted((EXPERIMENT_ROOT / "code").rglob("*.py"))}
    reproducibility = {
        "seed": V7_SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "one_seed_exploratory": True,
        "commands": [
            "python code/protocol/bootstrap_protocol.py",
            "python code/backbones/extract_openbmi_episode_cache.py",
            "python code/meta_learning/run_initial_meta.py",
            "python code/backbones/build_raw_ea_cache.py",
            "python code/backbones/train_ea_eegnet.py --benchmark <openbmi|wbcic> --variant <identity|ea>",
            "python code/backbones/train_structural_backbone.py --benchmark <...> --architecture <...>",
            "python code/meta_learning/run_hypernetwork_meta.py",
            "python code/persist/run_positive_control.py",
            "python code/evaluation/finalize_v7.py",
        ],
        "code_hashes": code_hashes,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "REPRODUCIBILITY.json", reproducibility)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    run()
