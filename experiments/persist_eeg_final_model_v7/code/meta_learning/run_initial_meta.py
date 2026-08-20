"""Initial V7 PERSIST-Meta research engine.

This script constructs subject/session episodes, measures realized future
utility for a low-dimensional adaptation bank, cross-fits prospective utility
controllers, and evaluates frozen policies on each outer outcome fold.  The
sealed WBCIC outer cohort is never opened or enumerated.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_bases.components import (
    ComponentSpec,
    action_logits,
    ba,
    base_logit,
    ce,
    component_bank,
    component_descriptors,
    fit_population_state,
    history_context,
    sigmoid,
)
from common import (
    ABLATIONS,
    BASELINES,
    DIAGNOSTICS,
    LEADERBOARD,
    PROTOCOL,
    RESEARCH_LOG,
    V7_SEED,
    ensure_directories,
    stable_seed,
    v6_outputs,
    write_csv,
    write_json,
)
from protocol.datasets import FoldDataset, load_fold
from utility.controller import CONTROLLERS, ControllerConfig, crossfit_predict, fit_predict, utility_metrics


BASE_KINDS = ("raw", "population", "blend25", "blend50", "blend75")
POLICY_KAPPAS = (0.0, 0.5, 1.0)
POLICY_THRESHOLDS = (0.0, 0.0025, 0.005)
POLICY_SCALES = (0.5, 1.0)


def _subject_chunks(subjects: tuple[str, ...], seed: int) -> list[tuple[str, ...]]:
    values = np.asarray(list(map(str, subjects)), dtype=object)
    values = values[np.random.default_rng(seed).permutation(len(values))]
    return [tuple(map(str, part.tolist())) for part in np.array_split(values, min(5, len(values))) if len(part)]


def _fit_state(data: FoldDataset, subjects: tuple[str, ...]):
    mask = data.metadata.subject_id.astype(str).isin(subjects).to_numpy()
    return fit_population_state(data.embeddings[mask], data.metadata.loc[mask, "label"].to_numpy(int))


def _subject_data(data: FoldDataset, subject: str, state, base_kind: str) -> dict[str, np.ndarray]:
    mask = data.metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
    metadata = data.metadata.loc[mask].reset_index(drop=True)
    features = np.asarray(data.embeddings[mask], dtype=np.float64)
    raw = np.asarray(data.logits[mask], dtype=float)
    z = state.transform(features)
    population = np.asarray(state.classifier.decision_function(z), dtype=float)
    base = base_logit(base_kind, raw, population)
    sessions = metadata.session_id.to_numpy(int)
    history = np.isin(sessions, data.history_sessions)
    future = sessions == int(data.future_session)
    if not history.any() or not future.any():
        raise RuntimeError(f"Missing episode side for {data.benchmark} {subject}")
    return {
        "z_history": z[history],
        "y_history": metadata.loc[history, "label"].to_numpy(int),
        "base_history": base[history],
        "sessions_history": sessions[history],
        "z_future": z[future],
        "y_future": metadata.loc[future, "label"].to_numpy(int),
        "base_future": base[future],
        "uid_future": metadata.loc[future, "trial_uid"].astype(str).to_numpy(),
    }


def _base_selection(data: FoldDataset, states: list[tuple[tuple[str, ...], Any]]) -> tuple[str, pd.DataFrame]:
    rows = []
    for base_kind in BASE_KINDS:
        values = []
        nlls = []
        for validation_subjects, state in states:
            for subject in validation_subjects:
                episode = _subject_data(data, subject, state, base_kind)
                values.append(ba(episode["y_future"], episode["base_future"]))
                nlls.append(ce(episode["y_future"], episode["base_future"]))
        rows.append({
            "benchmark": data.benchmark,
            "outer_fold": data.fold,
            "base_kind": base_kind,
            "meta_OOF_mean_subject_BA": float(np.mean(values)),
            "meta_OOF_mean_subject_NLL": float(np.mean(nlls)),
            "subjects": len(values),
            "target_future_labels_used_for_evaluation_subject_fit": False,
            "OUTER_TEST_USED": False,
        })
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(
        ["meta_OOF_mean_subject_BA", "meta_OOF_mean_subject_NLL", "base_kind"],
        ascending=[False, True, True],
    ).iloc[0]
    frame["selected"] = frame.base_kind.eq(selected.base_kind)
    return str(selected.base_kind), frame


def _episode_components(
    data: FoldDataset,
    subject: str,
    state,
    base_kind: str,
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode = _subject_data(data, subject, state, base_kind)
    z_history = episode["z_history"]
    y_history = episode["y_history"]
    base_history = episode["base_history"]
    sessions_history = episode["sessions_history"]
    z_future = episode["z_future"]
    y_future = episode["y_future"]
    base_future = episode["base_future"]
    common = history_context(z_history, y_history, base_history, sessions_history)
    specs = component_bank(state.dimension, include_latest=len(data.history_sessions) > 1)
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for spec in specs:
        candidate_history = action_logits(
            spec, z_history, y_history, base_history,
            z_history, base_history, sessions_history,
        )
        candidate_future = action_logits(
            spec, z_history, y_history, base_history,
            z_future, base_future, sessions_history,
        )
        descriptors = component_descriptors(
            spec, z_history, y_history, base_history,
            sessions_history, candidate_history,
        )
        rows.append({
            "benchmark": data.benchmark,
            "outer_fold": data.fold,
            "subject_id": str(subject),
            "role": role,
            "base_kind": base_kind,
            **spec.payload(),
            **common,
            **descriptors,
            "U_signed_utility_prior": 0.0,
            "future_ce_gain": float(ce(y_future, base_future) - ce(y_future, candidate_future)),
            "future_ba_gain": float(ba(y_future, candidate_future) - ba(y_future, base_future)),
            "meta_query_label_used_for_utility_target": role == "meta_train",
            "evaluation_future_label_used_for_controller_fit": False,
            "OUTER_TEST_USED": False,
        })
        predictions[spec.component_id] = np.asarray(candidate_future, dtype=float)
    bundle = {
        "benchmark": data.benchmark,
        "outer_fold": data.fold,
        "subject_id": str(subject),
        "uid": episode["uid_future"],
        "label": y_future,
        "base_logit": base_future,
        "candidate_logits": predictions,
    }
    return rows, bundle


def _score_logits(label: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    return (
        float(balanced_accuracy_score(label, logits >= 0.0)),
        float(log_loss(label, sigmoid(logits), labels=[0, 1])),
    )


def _apply_policy(
    predicted_rows: pd.DataFrame,
    bundles: dict[str, dict[str, Any]],
    kappa: float,
    threshold: float,
    scale: float,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, np.ndarray]]:
    subject_rows = []
    selected_logits: dict[str, np.ndarray] = {}
    for subject, group in predicted_rows.groupby("subject_id", sort=False):
        subject = str(subject)
        bundle = bundles[subject]
        score = group.predicted_utility.to_numpy(float) - float(kappa) * group.predicted_sigma.to_numpy(float) - float(threshold)
        best_index = int(np.argmax(score))
        best = group.iloc[best_index]
        preserve = bool(score[best_index] <= 0.0)
        if preserve:
            logits = np.asarray(bundle["base_logit"], dtype=float)
            component = "PRESERVE"
        else:
            candidate = np.asarray(bundle["candidate_logits"][str(best.component_id)], dtype=float)
            base = np.asarray(bundle["base_logit"], dtype=float)
            logits = base + float(scale) * (candidate - base)
            component = str(best.component_id)
        selected_logits[subject] = logits
        value, nll = _score_logits(bundle["label"], logits)
        base_value, base_nll = _score_logits(bundle["label"], bundle["base_logit"])
        subject_rows.append({
            "subject_id": subject,
            "selected_component": component,
            "action": "PRESERVE" if preserve else "ADAPT",
            "risk_adjusted_score": float(score[best_index]),
            "predicted_utility": float(best.predicted_utility),
            "predicted_sigma": float(best.predicted_sigma),
            "BA": value,
            "base_BA": base_value,
            "Delta_BA": value - base_value,
            "NLL": nll,
            "base_NLL": base_nll,
        })
    frame = pd.DataFrame(subject_rows)
    metrics = {
        "mean_subject_BA": float(frame.BA.mean()),
        "base_mean_subject_BA": float(frame.base_BA.mean()),
        "Delta_BA": float((frame.BA - frame.base_BA).mean()),
        "mean_subject_NLL": float(frame.NLL.mean()),
        "adapt_fraction": float(frame.action.eq("ADAPT").mean()),
        "harmful_subject_fraction": float((frame.Delta_BA < 0.0).mean()),
        "nonnegative_subject_fraction": float((frame.Delta_BA >= 0.0).mean()),
        "worst_subject_delta": float(frame.Delta_BA.min()),
        "median_subject_delta": float(frame.Delta_BA.median()),
    }
    return metrics, subject_rows, selected_logits


def _select_controller_policy(
    rows: pd.DataFrame,
    bundles: dict[str, dict[str, Any]],
    persist: bool,
    seed: int,
) -> tuple[ControllerConfig, dict[str, float], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    predictability = []
    frames: dict[str, pd.DataFrame] = {}
    for configuration in CONTROLLERS:
        predicted, metrics = crossfit_predict(rows, persist, configuration, seed)
        frames[configuration.controller_id] = predicted
        predictability.append(metrics)
        for kappa in POLICY_KAPPAS:
            for threshold in POLICY_THRESHOLDS:
                for scale in POLICY_SCALES:
                    policy, _, _ = _apply_policy(predicted, bundles, kappa, threshold, scale)
                    candidates.append({
                        "mode": "PERSIST_META" if persist else "META_GENERIC",
                        "controller_id": configuration.controller_id,
                        "kappa": kappa,
                        "threshold": threshold,
                        "scale": scale,
                        **policy,
                    })
    table = pd.DataFrame(candidates)
    selected = table.sort_values(
        ["mean_subject_BA", "harmful_subject_fraction", "worst_subject_delta", "adapt_fraction", "controller_id"],
        ascending=[False, True, False, True, True],
    ).iloc[0].to_dict()
    configuration = next(value for value in CONTROLLERS if value.controller_id == selected["controller_id"])
    selected_frame = frames[configuration.controller_id]
    _, subject_rows, _ = _apply_policy(
        selected_frame,
        bundles,
        float(selected["kappa"]),
        float(selected["threshold"]),
        float(selected["scale"]),
    )
    table["selected"] = (
        table.controller_id.eq(selected["controller_id"])
        & table.kappa.eq(float(selected["kappa"]))
        & table.threshold.eq(float(selected["threshold"]))
        & table.scale.eq(float(selected["scale"]))
    )
    return configuration, selected, table, predictability, subject_rows


def _prediction_frame(
    bundle: dict[str, Any],
    logits: np.ndarray,
    method_id: str,
    history_used: bool,
) -> pd.DataFrame:
    probability = sigmoid(logits)
    return pd.DataFrame({
        "benchmark": bundle["benchmark"],
        "method_id": method_id,
        "trial_uid": bundle["uid"],
        "subject_id": bundle["subject_id"],
        "outer_fold": bundle["outer_fold"],
        "label": bundle["label"],
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int),
        "target_history_labels_used": history_used,
        "target_future_labels_used_for_fit": False,
        "exploratory": True,
        "OUTER_TEST_USED": False,
    })


def _strong_anchor(benchmark: str) -> tuple[pd.DataFrame, str]:
    if benchmark == "openbmi":
        path = v6_outputs() / "diagnostics" / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv"
        method = "MI_SPECIFIC_BACKBONE_ADAPTED"
    else:
        path = v6_outputs() / "diagnostics" / "WBCIC_FUTURE_SESSION_POPULATION_PREDICTIONS.csv"
        method = "V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED"
    frame = pd.read_csv(path)
    frame = frame.loc[frame.method_id.astype(str).eq(method)].copy()
    if frame.trial_uid.duplicated().any() or frame.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError(f"Malformed strong anchor {benchmark}")
    return frame, method


def _summarize_predictions(frame: pd.DataFrame, reference_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows = []
    for (method, subject), group in frame.groupby(["method_id", "subject_id"], sort=False):
        subject_rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "method_id": str(method),
            "subject_id": str(subject),
            "outer_fold": int(group.outer_fold.iloc[0]),
            "BA": float(balanced_accuracy_score(group.label, group.prediction)),
            "NLL": float(log_loss(group.label, np.clip(group.probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
        })
    subjects = pd.DataFrame(subject_rows)
    reference = subjects.loc[subjects.method_id.eq(reference_method), ["subject_id", "BA"]].set_index("subject_id").BA
    rows = []
    for method, group in subjects.groupby("method_id", sort=False):
        aligned = reference.loc[group.subject_id].to_numpy(float)
        delta = group.BA.to_numpy(float) - aligned
        rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "method_id": str(method),
            "subjects": int(len(group)),
            "mean_subject_BA": float(group.BA.mean()),
            "mean_subject_NLL": float(group.NLL.mean()),
            "reference_method_id": reference_method,
            "Delta_BA": float(np.mean(delta)),
            "positive_subject_fraction": float(np.mean(delta > 0.0)),
            "nonnegative_subject_fraction": float(np.mean(delta >= 0.0)),
            "worst_subject_delta": float(np.min(delta)),
            "positive_fold_fraction": float(np.mean(subjects.loc[subjects.method_id.eq(method)].groupby("outer_fold").BA.mean() > subjects.loc[subjects.method_id.eq(reference_method)].groupby("outer_fold").BA.mean())),
            "target_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        })
    return pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False), subjects


def _oracle(bundle: dict[str, Any]) -> dict[str, Any]:
    label = bundle["label"]
    choices = {"PRESERVE": np.asarray(bundle["base_logit"], dtype=float), **bundle["candidate_logits"]}
    scored = [(name, *_score_logits(label, value)) for name, value in choices.items()]
    selected = max(scored, key=lambda row: (row[1], -row[2], row[0]))
    return {"subject_id": bundle["subject_id"], "oracle_component": selected[0], "oracle_BA": selected[1], "oracle_NLL": selected[2]}


def run_benchmark(benchmark: str) -> dict[str, Any]:
    anchor, anchor_method = _strong_anchor(benchmark)
    all_predictions = [anchor]
    all_meta_rows = []
    all_outcome_rows = []
    base_tables = []
    policy_tables = []
    predictability_rows = []
    selection_rows = []
    oracle_rows = []
    component_rows = []
    fold_results = []
    for fold in range(5):
        data = load_fold(benchmark, fold)
        meta_subjects = data.nonoutcome_subjects
        chunks = _subject_chunks(meta_subjects, stable_seed(V7_SEED, benchmark, fold, "meta-chunks"))
        states = []
        for validation in chunks:
            fit_subjects = tuple(subject for subject in meta_subjects if subject not in set(validation))
            states.append((validation, _fit_state(data, fit_subjects)))
        selected_base, base_table = _base_selection(data, states)
        base_tables.append(base_table)
        meta_rows = []
        meta_bundles: dict[str, dict[str, Any]] = {}
        for validation, state in states:
            for subject in validation:
                rows, bundle = _episode_components(data, subject, state, selected_base, "meta_train")
                meta_rows.extend(rows)
                meta_bundles[str(subject)] = bundle
        meta_frame = pd.DataFrame(meta_rows)
        generic_config, generic_policy, generic_table, generic_predictability, generic_subjects = _select_controller_policy(
            meta_frame, meta_bundles, False, stable_seed(V7_SEED, benchmark, fold, "generic-controller")
        )
        persist_config, persist_policy, persist_table, persist_predictability, persist_subjects = _select_controller_policy(
            meta_frame, meta_bundles, True, stable_seed(V7_SEED, benchmark, fold, "persist-controller")
        )
        generic_table["benchmark"] = data.benchmark
        generic_table["outer_fold"] = fold
        persist_table["benchmark"] = data.benchmark
        persist_table["outer_fold"] = fold
        policy_tables.extend([generic_table, persist_table])
        for item in generic_predictability + persist_predictability:
            predictability_rows.append({"benchmark": data.benchmark, "outer_fold": fold, **item})
        final_state = _fit_state(data, meta_subjects)
        outcome_rows = []
        outcome_bundles: dict[str, dict[str, Any]] = {}
        for subject in data.outcome_subjects:
            rows, bundle = _episode_components(data, subject, final_state, selected_base, "outcome_evaluation")
            outcome_rows.extend(rows)
            outcome_bundles[str(subject)] = bundle
            oracle_rows.append({"benchmark": data.benchmark, "outer_fold": fold, **_oracle(bundle), "used_to_tune_policy": False, "OUTER_TEST_USED": False})
        outcome_frame = pd.DataFrame(outcome_rows)
        generic_outcome = fit_predict(
            meta_frame, outcome_frame, False, generic_config,
            stable_seed(V7_SEED, benchmark, fold, "generic-final"),
        )
        persist_outcome = fit_predict(
            meta_frame, outcome_frame, True, persist_config,
            stable_seed(V7_SEED, benchmark, fold, "persist-final"),
        )
        _, generic_selection, generic_logits = _apply_policy(
            generic_outcome, outcome_bundles,
            float(generic_policy["kappa"]), float(generic_policy["threshold"]), float(generic_policy["scale"]),
        )
        _, persist_selection, persist_logits = _apply_policy(
            persist_outcome, outcome_bundles,
            float(persist_policy["kappa"]), float(persist_policy["threshold"]), float(persist_policy["scale"]),
        )
        anchor_index = anchor.set_index("trial_uid")
        for subject, bundle in outcome_bundles.items():
            base = np.asarray(bundle["base_logit"], dtype=float)
            generic = np.asarray(generic_logits[subject], dtype=float)
            persist = np.asarray(persist_logits[subject], dtype=float)
            anchor_probability = anchor_index.loc[bundle["uid"], "probability"].to_numpy(float)
            anchor_logit = np.log(np.clip(anchor_probability, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(anchor_probability, 1e-7, 1 - 1e-7))
            all_predictions.extend([
                _prediction_frame(bundle, base, "V7_FEATURE_BASE", False),
                _prediction_frame(bundle, generic, "META_GENERIC", True),
                _prediction_frame(bundle, persist, "PERSIST_META", True),
                _prediction_frame(bundle, anchor_logit + (generic - base), "ANCHOR_PLUS_META_GENERIC_RESIDUAL", True),
                _prediction_frame(bundle, anchor_logit + (persist - base), "ANCHOR_PLUS_PERSIST_META_RESIDUAL", True),
                _prediction_frame(bundle, 0.5 * (anchor_logit + generic), "ANCHOR_BLEND_META_GENERIC", True),
                _prediction_frame(bundle, 0.5 * (anchor_logit + persist), "ANCHOR_BLEND_PERSIST_META", True),
            ])
        for mode, values, policy, configuration in (
            ("META_GENERIC", generic_selection, generic_policy, generic_config),
            ("PERSIST_META", persist_selection, persist_policy, persist_config),
        ):
            for value in values:
                selection_rows.append({
                    "benchmark": data.benchmark,
                    "outer_fold": fold,
                    "mode": mode,
                    "controller_id": configuration.controller_id,
                    "kappa": policy["kappa"],
                    "threshold": policy["threshold"],
                    "scale": policy["scale"],
                    **value,
                    "target_future_labels_used_for_selection": False,
                    "OUTER_TEST_USED": False,
                })
        meta_frame["outer_fold"] = fold
        outcome_frame["outer_fold"] = fold
        all_meta_rows.append(meta_frame)
        generic_outcome["controller_mode"] = "META_GENERIC"
        persist_outcome["controller_mode"] = "PERSIST_META"
        all_outcome_rows.extend([generic_outcome, persist_outcome])
        fold_results.append({
            "benchmark": data.benchmark,
            "outer_fold": fold,
            "base_kind": selected_base,
            "generic_controller": generic_config.controller_id,
            "generic_meta_OOF_BA": generic_policy["mean_subject_BA"],
            "persist_controller": persist_config.controller_id,
            "persist_meta_OOF_BA": persist_policy["mean_subject_BA"],
            "meta_subjects": len(meta_subjects),
            "outcome_subjects": len(data.outcome_subjects),
            "OUTER_TEST_USED": False,
        })
        component_rows.extend([
            {"benchmark": data.benchmark, "outer_fold": fold, **spec.payload(), "dimension": final_state.dimension, "OUTER_TEST_USED": False}
            for spec in component_bank(final_state.dimension, include_latest=len(data.history_sessions) > 1)
        ])
        print(
            f"[{benchmark} V7] fold={fold} base={selected_base} "
            f"generic={generic_config.controller_id}/{generic_policy['mean_subject_BA']:.4f} "
            f"persist={persist_config.controller_id}/{persist_policy['mean_subject_BA']:.4f}",
            flush=True,
        )
    predictions = pd.concat(all_predictions, ignore_index=True)
    # The anchor was added once in full; every V7 method was added outcome-fold by outcome-fold.
    if predictions.loc[predictions.method_id.eq(anchor_method), "trial_uid"].duplicated().any():
        raise RuntimeError("Anchor prediction duplication")
    leaderboard, subject_results = _summarize_predictions(predictions, anchor_method)
    utility_rows = pd.concat(all_meta_rows, ignore_index=True)
    outcome_utility = pd.concat(all_outcome_rows, ignore_index=True)
    base_frame = pd.concat(base_tables, ignore_index=True)
    policy_frame = pd.concat(policy_tables, ignore_index=True)
    predictability = pd.DataFrame(predictability_rows)
    selection = pd.DataFrame(selection_rows)
    oracle = pd.DataFrame(oracle_rows)
    oracle_summary = {
        "mean_subject_oracle_BA": float(oracle.oracle_BA.mean()),
        "subjects": int(len(oracle)),
        "used_to_tune_policy": False,
    }
    return {
        "predictions": predictions,
        "leaderboard": leaderboard,
        "subject_results": subject_results,
        "meta_utility": utility_rows,
        "outcome_utility": outcome_utility,
        "base_table": base_frame,
        "policy_table": policy_frame,
        "predictability": predictability,
        "selection": selection,
        "oracle": oracle,
        "oracle_summary": oracle_summary,
        "fold_results": pd.DataFrame(fold_results),
        "components": pd.DataFrame(component_rows),
        "anchor_method": anchor_method,
    }


def run() -> None:
    ensure_directories()
    results = {benchmark: run_benchmark(benchmark) for benchmark in ("openbmi", "wbcic")}
    write_csv(LEADERBOARD / "OPENBMI_V7.csv", results["openbmi"]["leaderboard"])
    write_csv(LEADERBOARD / "WBCIC_DEV_V7.csv", results["wbcic"]["leaderboard"])
    cross = pd.concat([
        results["openbmi"]["leaderboard"].assign(benchmark_short="OpenBMI"),
        results["wbcic"]["leaderboard"].assign(benchmark_short="WBCIC"),
    ], ignore_index=True)
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_V7.csv", cross)
    write_csv(DIAGNOSTICS / "ADAPTATION_COMPONENTS.csv", pd.concat([results[key]["components"] for key in results], ignore_index=True))
    meta_utility = pd.concat([results[key]["meta_utility"] for key in results], ignore_index=True)
    write_csv(DIAGNOSTICS / "FUTURE_UTILITY_DIAGNOSTICS.csv", meta_utility)
    write_csv(DIAGNOSTICS / "PERSIST_COMPONENT_DESCRIPTORS.csv", meta_utility[[
        "benchmark", "outer_fold", "subject_id", "component_id", "family",
        "P_persistence", "D_decision_dependence", "G_task_overlap", "R_history_transfer",
        "future_ce_gain", "future_ba_gain", "OUTER_TEST_USED",
    ]])
    write_csv(DIAGNOSTICS / "GRADIENT_TRANSFER_AUDIT.csv", meta_utility.loc[
        meta_utility.family.isin(["meta_sgd", "projected_gradient"]),
        ["benchmark", "outer_fold", "subject_id", "component_id", "P_persistence", "G_task_overlap", "R_history_transfer", "future_ce_gain", "future_ba_gain", "OUTER_TEST_USED"],
    ])
    write_csv(DIAGNOSTICS / "UTILITY_PREDICTABILITY.csv", pd.concat([results[key]["predictability"] for key in results], ignore_index=True))
    write_csv(DIAGNOSTICS / "OUTCOME_PROSPECTIVE_UTILITY.csv", pd.concat([results[key]["outcome_utility"] for key in results], ignore_index=True))
    write_csv(DIAGNOSTICS / "OUTCOME_ACTION_SELECTIONS.csv", pd.concat([results[key]["selection"] for key in results], ignore_index=True))
    write_csv(DIAGNOSTICS / "NEW_COMPONENT_ORACLE.csv", pd.concat([results[key]["oracle"] for key in results], ignore_index=True))
    write_csv(DIAGNOSTICS / "V7_SUBJECT_RESULTS.csv", pd.concat([results[key]["subject_results"] for key in results], ignore_index=True))
    write_csv(DIAGNOSTICS / "V7_PREDICTIONS.csv", pd.concat([results[key]["predictions"] for key in results], ignore_index=True))
    write_csv(BASELINES / "GENERIC_META_BASELINES.csv", pd.concat([results[key]["base_table"] for key in results], ignore_index=True))
    write_csv(BASELINES / "OPENBMI_MATCHED_BASELINES.csv", results["openbmi"]["leaderboard"])
    write_csv(BASELINES / "WBCIC_MATCHED_BASELINES.csv", results["wbcic"]["leaderboard"])
    write_csv(ABLATIONS / "META_ABLATION.csv", pd.concat([results[key]["policy_table"] for key in results], ignore_index=True))
    write_csv(ABLATIONS / "FUTURE_UTILITY_ABLATION.csv", pd.concat([results[key]["predictability"] for key in results], ignore_index=True))
    write_csv(ABLATIONS / "PERSIST_ABLATION.csv", cross.loc[cross.method_id.str.contains("META", regex=False)].copy())
    write_json(PROTOCOL / "INITIAL_META_LEGALITY_AUDIT.json", {
        "meta_population_crossfit": "Population feature heads exclude the meta-query subject group.",
        "controller_crossfit": "Five-fold grouped by subject.",
        "outcome_controller_fit": "Only non-outcome meta episodes.",
        "outcome_future_labels_used_for_fit_or_selection": False,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    write_json(DIAGNOSTICS / "NEW_COMPONENT_ORACLE_SUMMARY.json", {
        key: results[key]["oracle_summary"] for key in results
    })
    summary = {
        key: {
            "leaderboard": results[key]["leaderboard"].to_dict("records"),
            "component_oracle": results[key]["oracle_summary"],
        }
        for key in results
    }
    write_json(RESEARCH_LOG / "ITERATION_001_RESULTS.json", summary)
    (RESEARCH_LOG / "ITERATION_001.md").write_text(
        "# Iteration 001 — coarse future-utility meta-adaptation\n\n"
        "This iteration evaluates calibration, subject heads, shrinkage LDA, prototype transport, "
        "module-wise Meta-SGD, and eight projected-gradient components. Population heads and utility "
        "controllers are subject-group cross-fitted. Outcome future labels never fit the policy.\n\n"
        "## OpenBMI\n\n```text\n" + results["openbmi"]["leaderboard"].to_string(index=False) + "\n```\n\n"
        "## WBCIC development\n\n```text\n" + results["wbcic"]["leaderboard"].to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    print("=== OPENBMI V7 ===", flush=True)
    print(results["openbmi"]["leaderboard"].to_string(index=False), flush=True)
    print("=== WBCIC V7 DEV ===", flush=True)
    print(results["wbcic"]["leaderboard"].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
