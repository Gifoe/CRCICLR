from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

import common as c


MODEL_FEATURES = {
    "M0": [],
    "M1": ["backbone"],
    "M2": ["raw_delta2"],
    "M3": ["certificate_snr"],
    "M4": [],
    "M5": ["representation_stability"],
    "M6": ["decision_stability"],
    "M7": ["adaptation_effect_stability"],
    "M8": [
        "adaptation_effect_stability",
        "decision_stability",
        "certificate_snr",
        "representation_stability",
    ],
}

MECHANISM_MODELS = ("M3", "M5", "M6", "M7", "M8")
SINGLE_MECHANISMS = {
    "M3": "certificate_snr",
    "M5": "representation_stability",
    "M6": "decision_stability",
    "M7": "adaptation_effect_stability",
}
OUTCOMES = ("R_sign", "H", "safe_given_positive_certificate")


def qci(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan
    return float(np.quantile(array, alpha / 2)), float(np.quantile(array, 1 - alpha / 2))


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else math.nan
    prediction = (probability >= 0.5).astype(int)
    return {
        "AUROC": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "Brier": float(brier_score_loss(y, probability)),
    }


def subject_bootstrap_metrics(
    frame: pd.DataFrame,
    outcome: str,
    probability_column: str,
    seed: int,
) -> dict[str, tuple[float, float]]:
    subjects = np.asarray(c.subject_sort(frame.subject_id.unique()))
    grouped = {subject: frame.index[frame.subject_id == subject].to_numpy() for subject in subjects}
    rng = np.random.default_rng(seed)
    draws = {"AUROC": [], "balanced_accuracy": [], "Brier": []}
    for _ in range(c.SUBJECT_BOOTSTRAPS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([grouped[subject] for subject in sampled])
        values = metric_values(
            frame.loc[indices, outcome].to_numpy(int),
            frame.loc[indices, probability_column].to_numpy(float),
        )
        for key, value in values.items():
            if np.isfinite(value):
                draws[key].append(value)
    return {key: qci(value) for key, value in draws.items()}


def _design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for feature in features:
        if feature == "backbone":
            train_parts.append((train.backbone == "EEGConformer").to_numpy(float)[:, None])
            test_parts.append((test.backbone == "EEGConformer").to_numpy(float)[:, None])
            continue
        left = train[feature].to_numpy(float)
        right = test[feature].to_numpy(float)
        mean = float(np.mean(left))
        scale = max(float(np.std(left, ddof=0)), 1e-8)
        train_parts.append(((left - mean) / scale)[:, None])
        test_parts.append(((right - mean) / scale)[:, None])
    if not train_parts:
        return np.empty((len(train), 0)), np.empty((len(test), 0))
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, outcome: str, model: str) -> np.ndarray:
    y = train[outcome].to_numpy(int)
    prevalence = float(np.mean(y)) if len(y) else 0.5
    features = MODEL_FEATURES[model]
    if model in ("M0", "M4") or not features or len(np.unique(y)) < 2:
        return np.full(len(test), prevalence, dtype=float)
    x_train, x_test = _design(train, test, features)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        random_state=0,
    )
    estimator.fit(x_train, y)
    return estimator.predict_proba(x_test)[:, 1]


def cross_validated_predictions(frame: pd.DataFrame, outcome: str, model: str) -> pd.DataFrame:
    if outcome == "safe_given_positive_certificate":
        work = frame[frame.raw_delta2 > 0].copy()
    else:
        work = frame.copy()
    work = work.reset_index(drop=True)
    work["probability"] = np.nan
    if model == "M4":
        return work
    for fold in c.FOLDS:
        train = work[work.fold != fold]
        test = work[work.fold == fold]
        if test.empty:
            continue
        work.loc[test.index, "probability"] = fit_predict(train, test, outcome, model)
    if work.probability.isna().any():
        raise RuntimeError(f"incomplete OOF predictions for {model}/{outcome}")
    return work


def bootstrap_univariate_auc(frame: pd.DataFrame, outcome: str, score: str, seed: int) -> tuple[float, float]:
    subjects = np.asarray(c.subject_sort(frame.subject_id.unique()))
    grouped = {subject: frame.index[frame.subject_id == subject].to_numpy() for subject in subjects}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(c.SUBJECT_BOOTSTRAPS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([grouped[subject] for subject in sampled])
        y = frame.loc[idx, outcome].to_numpy(int)
        if len(np.unique(y)) == 2:
            draws.append(roc_auc_score(y, frame.loc[idx, score].to_numpy(float)))
    return qci(draws)


def univariate_table(frame: pd.DataFrame) -> pd.DataFrame:
    scores = [
        "adaptation_effect_stability",
        "decision_stability",
        "certificate_snr",
        "certificate_lcb90",
        "representation_stability",
        "raw_delta2",
        "s1_parameter_relative_change",
        "s1_anchor_confidence",
    ]
    rows = []
    for score in scores:
        signed_rho = float(stats.spearmanr(frame[score], frame.signed_persistence).statistic)
        for outcome in ("R_sign", "H"):
            auc = float(roc_auc_score(frame[outcome], frame[score]))
            low, high = bootstrap_univariate_auc(
                frame,
                outcome,
                score,
                c.stable_seed("stage05-univariate", score, outcome),
            )
            rows.append(
                {
                    "score": score,
                    "outcome": outcome,
                    "n_subjects": frame.subject_id.nunique(),
                    "n_subject_backbone_rows": len(frame),
                    "spearman_with_signed_persistence": signed_rho,
                    "AUROC_higher_score_predicts_outcome_1": auc,
                    "AUROC_CI95_low": low,
                    "AUROC_CI95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _full_design(frame: pd.DataFrame, score: str, interaction: bool) -> np.ndarray:
    backbone = (frame.backbone == "EEGConformer").to_numpy(float)
    value = frame[score].to_numpy(float)
    value = (value - np.mean(value)) / max(np.std(value), 1e-8)
    columns = [backbone, value]
    if interaction:
        columns.append(backbone * value)
    return np.column_stack(columns)


def _fit_decomposition(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    if len(np.unique(y)) < 2:
        return np.full(x.shape[1], np.nan), math.nan
    estimator = LogisticRegression(C=1e6, solver="liblinear", max_iter=2000, random_state=0)
    estimator.fit(x, y)
    probability = np.clip(estimator.predict_proba(x)[:, 1], 1e-8, 1 - 1e-8)
    return estimator.coef_[0].astype(float), float(2 * len(y) * log_loss(y, probability))


def decomposition_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, score in SINGLE_MECHANISMS.items():
        for outcome in OUTCOMES:
            current = frame[frame.raw_delta2 > 0].copy() if outcome == "safe_given_positive_certificate" else frame.copy()
            y = current[outcome].to_numpy(int)
            backbone = (current.backbone == "EEGConformer").to_numpy(float)[:, None]
            coef0, dev0 = _fit_decomposition(backbone, y)
            x1 = _full_design(current, score, interaction=False)
            coef1, dev1 = _fit_decomposition(x1, y)
            x2 = _full_design(current, score, interaction=True)
            coef2, dev2 = _fit_decomposition(x2, y)
            baseline_backbone = float(coef0[0])
            adjusted_backbone = float(coef1[0])
            reduction = (
                1.0 - abs(adjusted_backbone) / abs(baseline_backbone)
                if abs(baseline_backbone) > 1e-8
                else math.nan
            )
            rows.append(
                {
                    "model": model,
                    "mechanism": score,
                    "outcome": outcome,
                    "n_subjects": current.subject_id.nunique(),
                    "n_rows": len(current),
                    "backbone_only_coefficient": baseline_backbone,
                    "backbone_adjusted_coefficient": adjusted_backbone,
                    "mechanism_main_coefficient": float(coef1[1]),
                    "interaction_coefficient": float(coef2[2]),
                    "absolute_backbone_coefficient_reduction": reduction,
                    "backbone_only_deviance": dev0,
                    "backbone_plus_mechanism_deviance": dev1,
                    "deviance_improvement": dev0 - dev1,
                    "interaction_deviance_improvement": dev1 - dev2,
                }
            )
    return pd.DataFrame(rows)


def choose_training_threshold(train: pd.DataFrame, probability: np.ndarray, denominator: int) -> float:
    if len(train) != len(probability):
        raise ValueError("threshold inputs are not aligned")
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    options = []
    denominator = max(int(denominator), 1)
    for threshold in candidates:
        selected = probability >= threshold
        coverage = float(np.sum(selected) / denominator)
        if coverage + 1e-12 < 0.20 or not np.any(selected):
            continue
        harm = float(np.mean(train.loc[selected, "H"]))
        options.append((harm, -coverage, float(threshold)))
    if not options:
        return float(np.min(probability)) if len(probability) else 1.0
    return min(options)[2]


def nested_reliability_policy(frame: pd.DataFrame, eligible: bool) -> pd.DataFrame:
    policy = frame.copy()
    policy["reliability_probability"] = np.nan
    policy["training_selected_threshold"] = np.nan
    policy["reliability_gate"] = False
    if not eligible:
        policy["policy_execution_status"] = "NOT_EXECUTED_NO_MEANINGFUL_CV_MECHANISM"
        return policy
    for outer_fold in c.FOLDS:
        outer_train_all = policy[policy.fold != outer_fold].copy()
        outer_test = policy[policy.fold == outer_fold].copy()
        train = outer_train_all[outer_train_all.raw_delta2 > 0].copy()
        test = outer_test[outer_test.raw_delta2 > 0].copy()
        if train.empty:
            continue
        train["inner_probability"] = np.nan
        for inner_fold in sorted(train.fold.unique()):
            inner_fit = train[train.fold != inner_fold]
            inner_test = train[train.fold == inner_fold]
            train.loc[inner_test.index, "inner_probability"] = fit_predict(
                inner_fit,
                inner_test,
                "safe_given_positive_certificate",
                "M8",
            )
        if train.inner_probability.isna().any():
            raise RuntimeError(f"incomplete inner OOF predictions for outer fold {outer_fold}")
        threshold = choose_training_threshold(
            train,
            train.inner_probability.to_numpy(float),
            denominator=len(outer_train_all),
        )
        if not test.empty:
            probability = fit_predict(train, test, "safe_given_positive_certificate", "M8")
            policy.loc[test.index, "reliability_probability"] = probability
            policy.loc[test.index, "training_selected_threshold"] = threshold
            policy.loc[test.index, "reliability_gate"] = probability >= threshold
        policy.loc[outer_test.index, "training_selected_threshold"] = threshold
    policy["policy_execution_status"] = "EXECUTED_NESTED_TRAINING_FOLD_THRESHOLD"
    return policy


def policy_summary(policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = policy.copy()
    policy["anchor_policy_BA"] = policy.anchor_S3_BA
    policy["always_adapt_policy_BA"] = policy.adapted_S3_BA
    policy["simple_gate"] = policy.raw_delta2 > 0
    policy["simple_gate_policy_BA"] = np.where(policy.simple_gate, policy.adapted_S3_BA, policy.anchor_S3_BA)
    policy["reliability_gate_policy_BA"] = np.where(
        policy.reliability_gate,
        policy.adapted_S3_BA,
        policy.anchor_S3_BA,
    )

    rows = []
    for name, gate, ba_column in (
        ("Anchor", np.zeros(len(policy), dtype=bool), "anchor_policy_BA"),
        ("Always Adapt", np.ones(len(policy), dtype=bool), "always_adapt_policy_BA"),
        ("Simple S2 Gate", policy.simple_gate.to_numpy(bool), "simple_gate_policy_BA"),
        ("Reliability-Gated S2", policy.reliability_gate.to_numpy(bool), "reliability_gate_policy_BA"),
    ):
        accepted = int(np.sum(gate))
        harm = float(np.mean(policy.loc[gate, "Delta3"] < 0)) if accepted else math.nan
        rows.append(
            {
                "policy": name,
                "coverage": float(accepted / len(policy)),
                "accepted_subject_backbone_rows": accepted,
                "future_harm_given_adaptation": harm,
                "mean_S3_BA": float(policy[ba_column].mean()),
                "n_subjects": policy.subject_id.nunique(),
                "n_subject_backbone_rows": len(policy),
            }
        )
    summary = pd.DataFrame(rows)
    return policy, summary


def bootstrap_policy(policy: pd.DataFrame) -> dict[str, dict[str, float]]:
    subjects = np.asarray(c.subject_sort(policy.subject_id.unique()))
    groups = {subject: policy.index[policy.subject_id == subject].to_numpy() for subject in subjects}
    rng = np.random.default_rng(c.stable_seed("stage05-policy-bootstrap"))
    draws = {
        "simple_harm": [],
        "simple_coverage": [],
        "reliability_harm": [],
        "reliability_coverage": [],
        "anchor_BA": [],
        "always_BA": [],
        "simple_BA": [],
        "reliability_BA": [],
    }
    for _ in range(c.SUBJECT_BOOTSTRAPS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([groups[subject] for subject in sampled])
        current = policy.loc[idx]
        simple = current.simple_gate.to_numpy(bool)
        reliability = current.reliability_gate.to_numpy(bool)
        draws["simple_harm"].append(float(np.mean(current.loc[simple, "Delta3"] < 0)) if np.any(simple) else math.nan)
        draws["simple_coverage"].append(float(np.mean(simple)))
        draws["reliability_harm"].append(float(np.mean(current.loc[reliability, "Delta3"] < 0)) if np.any(reliability) else math.nan)
        draws["reliability_coverage"].append(float(np.mean(reliability)))
        draws["anchor_BA"].append(float(current.anchor_policy_BA.mean()))
        draws["always_BA"].append(float(current.always_adapt_policy_BA.mean()))
        draws["simple_BA"].append(float(current.simple_gate_policy_BA.mean()))
        draws["reliability_BA"].append(float(current.reliability_gate_policy_BA.mean()))
    return {key: {"CI95_low": qci(value)[0], "CI95_high": qci(value)[1]} for key, value in draws.items()}


def main() -> None:
    lock = c.verify_feature_lock(require_committed=True)
    execution = c.read_json(c.RUNTIME / "FEATURE_EXTRACTION_EXECUTION.json")
    feature_path = c.RESULTS / "PER_SUBJECT_FEATURES.csv"
    if not execution.get("complete") or execution.get("S3_signal_rows_read") != 0:
        raise RuntimeError("prospective feature extraction execution is absent or contaminated")
    if execution["feature_protocol_lock_sha256"] != c.sha256(c.PROTOCOL / "RELIABILITY_FEATURE_PROTOCOL_LOCK.json"):
        raise RuntimeError("feature extraction used a different protocol lock")
    if execution["per_subject_features_sha256"] != c.sha256(feature_path):
        raise RuntimeError("feature table changed after extraction")

    features = pd.read_csv(feature_path, dtype={"subject_id": str})
    if len(features) != 82 or features.subject_id.nunique() != 41:
        raise RuntimeError("expected 82 paired subject/backbone feature rows")
    if features.s3_feature_input_accessed.astype(str).str.lower().ne("false").any():
        raise RuntimeError("S3 feature contamination")

    stage0 = pd.read_csv(c.STAGE0_RESULTS / "PER_SUBJECT_UTILITY.csv", dtype={"subject_id": str})
    stage0 = stage0[stage0.scope.isin(c.BACKBONES)].copy()
    outcomes = stage0[[
        "backbone",
        "subject_id",
        "Delta_S2_BA",
        "Delta_S3_BA",
        "anchor_S3_BA",
        "adapted_S3_BA",
    ]].rename(columns={"Delta_S2_BA": "Delta2", "Delta_S3_BA": "Delta3"})
    frame = features.merge(outcomes, on=["backbone", "subject_id"], how="inner", validate="one_to_one")
    if len(frame) != 82 or float(np.max(np.abs(frame.raw_delta2 - frame.Delta2))) > 1e-7:
        raise RuntimeError("feature/outcome alignment failure")
    frame["R_sign"] = (np.sign(frame.Delta2) == np.sign(frame.Delta3)).astype(int)
    frame["R_safe"] = ((frame.Delta2 > 0) & (frame.Delta3 >= 0)).astype(int)
    frame["H"] = ((frame.Delta2 > 0) & (frame.Delta3 < 0)).astype(int)
    frame["safe_given_positive_certificate"] = (frame.Delta3 >= 0).astype(int)
    frame["signed_persistence"] = frame.Delta2 * frame.Delta3
    frame["_subject_sort"] = frame.subject_id.astype(int)
    frame = frame.sort_values(["backbone", "_subject_sort"]).drop(columns="_subject_sort").reset_index(drop=True)
    c.write_csv(
        c.RESULTS / "PER_SUBJECT_RELIABILITY_OUTCOMES.csv",
        frame[[
            "backbone", "fold", "subject_id", "Delta2", "Delta3", "R_sign", "R_safe", "H",
            "safe_given_positive_certificate", "signed_persistence", "anchor_S3_BA", "adapted_S3_BA",
        ]],
    )

    univariate = univariate_table(frame)
    c.write_csv(c.RESULTS / "UNIVARIATE_MECHANISMS.csv", univariate)

    cv_rows = []
    cv_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for outcome in OUTCOMES:
        for model in MODEL_FEATURES:
            if model == "M4":
                cv_rows.append(
                    {
                        "outcome": outcome,
                        "model": model,
                        "status": "UNAVAILABLE_NO_LEGAL_TARGET_LEVEL_IDENTITY",
                        "n_subjects": frame.subject_id.nunique(),
                        "n_rows": math.nan,
                        "prevalence": math.nan,
                        "AUROC": math.nan,
                        "AUROC_CI95_low": math.nan,
                        "AUROC_CI95_high": math.nan,
                        "balanced_accuracy": math.nan,
                        "Brier": math.nan,
                    }
                )
                continue
            prediction = cross_validated_predictions(frame, outcome, model)
            cv_predictions[(outcome, model)] = prediction
            values = metric_values(prediction[outcome], prediction.probability)
            ci = subject_bootstrap_metrics(
                prediction,
                outcome,
                "probability",
                c.stable_seed("stage05-cv", outcome, model),
            )
            cv_rows.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "status": "AVAILABLE_FROZEN",
                    "n_subjects": prediction.subject_id.nunique(),
                    "n_rows": len(prediction),
                    "prevalence": float(prediction[outcome].mean()),
                    **values,
                    "AUROC_CI95_low": ci["AUROC"][0],
                    "AUROC_CI95_high": ci["AUROC"][1],
                    "balanced_accuracy_CI95_low": ci["balanced_accuracy"][0],
                    "balanced_accuracy_CI95_high": ci["balanced_accuracy"][1],
                    "Brier_CI95_low": ci["Brier"][0],
                    "Brier_CI95_high": ci["Brier"][1],
                }
            )
    cv = pd.DataFrame(cv_rows)
    c.write_csv(c.RESULTS / "CROSS_VALIDATED_RELIABILITY.csv", cv)

    decomposition = decomposition_table(frame)
    c.write_csv(c.RESULTS / "BACKBONE_MECHANISM_DECOMPOSITION.csv", decomposition)

    def cv_value(outcome: str, model: str, column: str = "AUROC") -> float:
        row = cv[(cv.outcome == outcome) & (cv.model == model)].iloc[0]
        return float(row[column])

    eligibility_evidence = []
    for outcome in ("R_sign", "safe_given_positive_certificate"):
        comparator = max(cv_value(outcome, "M1"), cv_value(outcome, "M2"))
        for model in MECHANISM_MODELS:
            auc = cv_value(outcome, model)
            eligibility_evidence.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "AUROC": auc,
                    "comparator_AUROC": comparator,
                    "eligible": bool(auc >= 0.55 and auc - comparator >= 0.02),
                }
            )
    policy_eligible = any(item["eligible"] for item in eligibility_evidence)
    policy = nested_reliability_policy(frame, policy_eligible)
    policy, policy_results = policy_summary(policy)
    c.write_csv(c.RESULTS / "PER_SUBJECT_POLICY.csv", policy)
    c.write_csv(c.RESULTS / "RELIABILITY_POLICY_RESULTS.csv", policy_results)
    policy_ci = bootstrap_policy(policy)

    available_rsign = cv[(cv.outcome == "R_sign") & cv.model.isin(("M3", "M5", "M6", "M7"))].copy()
    best_row = available_rsign.sort_values(["AUROC", "model"], ascending=[False, True]).iloc[0]
    best_model = str(best_row.model)
    best_mechanism = SINGLE_MECHANISMS[best_model]

    gate_b_candidates = []
    for outcome in ("R_sign", "safe_given_positive_certificate"):
        baseline = max(cv_value(outcome, "M1"), cv_value(outcome, "M2"))
        for model in SINGLE_MECHANISMS:
            row = cv[(cv.outcome == outcome) & (cv.model == model)].iloc[0]
            gate_b_candidates.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "AUROC": float(row.AUROC),
                    "lower": float(row.AUROC_CI95_low),
                    "improvement": float(row.AUROC - baseline),
                    "pass": bool(row.AUROC >= 0.60 and row.AUROC_CI95_low > 0.50 and row.AUROC - baseline >= 0.03),
                }
            )
    passing_b = [item for item in gate_b_candidates if item["pass"]]
    gate_b = bool(passing_b)

    gate_c = False
    gate_c_evidence = None
    for item in passing_b:
        score = SINGLE_MECHANISMS[item["model"]]
        row = decomposition[(decomposition.mechanism == score) & (decomposition.outcome == item["outcome"])].iloc[0]
        passes = bool(
            row.absolute_backbone_coefficient_reduction >= 0.20
            and row.deviance_improvement >= 2.0
            and abs(row.interaction_coefficient) <= 2 * max(abs(row.mechanism_main_coefficient), 1e-8)
        )
        if passes:
            gate_c = True
            gate_c_evidence = {**item, **row.to_dict()}
            break

    simple = policy_results[policy_results.policy == "Simple S2 Gate"].iloc[0]
    reliability = policy_results[policy_results.policy == "Reliability-Gated S2"].iloc[0]
    anchor = policy_results[policy_results.policy == "Anchor"].iloc[0]
    always = policy_results[policy_results.policy == "Always Adapt"].iloc[0]
    relative_harm_reduction = (
        float((simple.future_harm_given_adaptation - reliability.future_harm_given_adaptation) / simple.future_harm_given_adaptation)
        if np.isfinite(reliability.future_harm_given_adaptation) and simple.future_harm_given_adaptation > 0
        else math.nan
    )
    gates = {
        "A_prospective_observability": bool(
            execution["S3_signal_rows_read"] == 0 and features.s3_feature_input_accessed.astype(str).str.lower().eq("false").all()
        ),
        "B_cross_validated_prediction": gate_b,
        "C_explains_backbone_contrast": gate_c,
        "D_harm_reduction": bool(np.isfinite(relative_harm_reduction) and relative_harm_reduction >= 0.25),
        "E_nontrivial_coverage": bool(reliability.coverage >= 0.20),
        "F_performance": bool(reliability.mean_S3_BA >= anchor.mean_S3_BA - 0.002),
    }
    directional = bool(
        cv[(cv.outcome.isin(["R_sign", "safe_given_positive_certificate"])) & cv.model.isin(MECHANISM_MODELS)].AUROC.max() >= 0.55
    )
    if all(gates.values()):
        terminal = "RELIABILITY_MECHANISM_SUPPORTED"
        authorization = "RELIABILITY_GATED_SCAA_DEVELOPMENT_AUTHORIZED"
    elif gates["A_prospective_observability"] and directional:
        terminal = "RELIABILITY_MECHANISM_PARTIAL"
        authorization = "RELIABILITY_GATED_SCAA_DEVELOPMENT_NOT_AUTHORIZED"
    else:
        terminal = "RELIABILITY_MECHANISM_NOT_SUPPORTED"
        authorization = "RELIABILITY_GATED_SCAA_DEVELOPMENT_NOT_AUTHORIZED"

    best_decomposition = decomposition[
        (decomposition.mechanism == best_mechanism) & (decomposition.outcome == "R_sign")
    ].iloc[0]
    identity_comparison = "Identity I unavailable: prior frozen probe is a model-fit-domain aggregate, not a target-subject score; no target identity model was created."
    strongest = (
        "A pre-S3 reliability mechanism and nested reliability gate satisfy all frozen development gates."
        if terminal == "RELIABILITY_MECHANISM_SUPPORTED"
        else "The audit quantifies whether frozen S1/S2 stability features add out-of-subject information; it does not establish a deployable SCAA rule."
    )
    stronger_unsupported = "No independent reliability confirmation, final SCAA improvement, or outer-subject generalization is established."

    statistics = {
        "schema": "PERSIST_EEG_SCAA_RELIABILITY_STAGE05_STATISTICAL_TESTS_V1",
        "subject_bootstraps": c.SUBJECT_BOOTSTRAPS,
        "feature_bootstraps": c.FEATURE_BOOTSTRAPS,
        "policy_execution_eligibility": policy_eligible,
        "policy_eligibility_evidence": eligibility_evidence,
        "gate_b_candidates": gate_b_candidates,
        "gate_c_evidence": gate_c_evidence,
        "policy_bootstrap": policy_ci,
        "relative_harm_reduction": relative_harm_reduction,
        "gates": gates,
        "terminal": terminal,
        "authorization": authorization,
    }
    c.write_json(c.RESULTS / "STATISTICAL_TESTS.json", statistics)

    final = {
        "schema": "PERSIST_EEG_SCAA_RELIABILITY_STAGE05_FINAL_REPORT_V1",
        "branch": "codex/persist-eeg-scaa-reliability-stage05",
        "stage0_validated_tip": lock["stage0_validated_tip"],
        "development_subjects_only": True,
        "development_subject_count": 41,
        "outer_10_untouched_unenumerated": True,
        "OpenBMI_accessed": False,
        "features_use_only_S1_S2": gates["A_prospective_observability"],
        "feature_definitions_changed_after_S3_association": False,
        "identity_control": identity_comparison,
        "best_reliability_mechanism": best_mechanism,
        "best_reliability_model": best_model,
        "best_R_sign_CV": best_row.to_dict(),
        "raw_Delta2_R_sign_CV": cv[(cv.outcome == "R_sign") & (cv.model == "M2")].iloc[0].to_dict(),
        "backbone_only_R_sign_CV": cv[(cv.outcome == "R_sign") & (cv.model == "M1")].iloc[0].to_dict(),
        "best_backbone_decomposition": best_decomposition.to_dict(),
        "policy_execution_eligible": policy_eligible,
        "policy": {row.policy: row.to_dict() for _, row in policy_results.iterrows()},
        "relative_harm_reduction": relative_harm_reduction,
        "gates": gates,
        "authorization": authorization,
        "terminal": terminal,
        "strongest_supported_claim": strongest,
        "stronger_claim_unsupported": stronger_unsupported,
        "most_serious_limitation": "Only 41 development subjects yield 82 correlated backbone rows and few positive certificates; Identity lacks a legal target-level frozen control and no sealed outer confirmation was opened.",
        "exact_feature_definitions": lock["feature_definitions"],
    }
    c.write_json(c.EXP / "RELIABILITY_STAGE05_FINAL_REPORT.json", final)

    report = f"""# PERSIST-EEG SCAA Reliability Stage-0.5 final report

## Frozen answers

1. Only WBCIC 41 development subjects used: **yes**.
2. Outer 10 untouched and unenumerated: **yes**.
3. All reliability features computed without S3: **yes**; feature extraction read signal sessions S1/S2 only.
4. Decision metric: `{lock['feature_definitions']['decision_stability']}`
5. Representation metric: `{lock['feature_definitions']['representation_stability']}`
6. Adaptation-effect metric: `{lock['feature_definitions']['adaptation_effect_stability']}`
7. Certificate precision: `{lock['feature_definitions']['certificate_precision']}`
8. Feature definitions changed after S3 association: **no**.
9. Best sign-persistence feature: **{best_mechanism}**.
10. Harmful-certificate predictors: see `CROSS_VALIDATED_RELIABILITY.csv`; no outcome-driven feature selection was used.
11. Best R_sign CV AUROC: **{best_row.AUROC:.4f}**.
12. Subject-bootstrap 95% CI: **[{best_row.AUROC_CI95_low:.4f}, {best_row.AUROC_CI95_high:.4f}]**.
13. Raw Delta2 R_sign AUROC: **{cv_value('R_sign', 'M2'):.4f}**.
14. Identity comparison: **unavailable rather than fabricated**.
15. Backbone-only R_sign AUROC: **{cv_value('R_sign', 'M1'):.4f}**.
16. Explains EEGConformer-vs-EEGNet: **{gates['C_explains_backbone_contrast']}** under the frozen Gate C.
17. Backbone coefficient: **{best_decomposition.backbone_only_coefficient:.4f} -> {best_decomposition.backbone_adjusted_coefficient:.4f}** after {best_mechanism}.
18. Cross-backbone consistency: interaction/main absolute ratio **{abs(best_decomposition.interaction_coefficient) / max(abs(best_decomposition.mechanism_main_coefficient), 1e-8):.4f}**.
19. Simple S2-gate coverage: **{simple.coverage:.4f}**.
20. Simple S2-gate S3 harm: **{simple.future_harm_given_adaptation:.4f}**.
21. Reliability-gated coverage: **{reliability.coverage:.4f}**.
22. Reliability-gated S3 harm: **{reliability.future_harm_given_adaptation if np.isfinite(reliability.future_harm_given_adaptation) else 'not estimable'}**.
23. Relative harm reduction: **{relative_harm_reduction if np.isfinite(relative_harm_reduction) else 'not estimable'}**.
24. Anchor S3 BA: **{anchor.mean_S3_BA:.6f}**.
25. Always-Adapt S3 BA: **{always.mean_S3_BA:.6f}**.
26. Simple S2-gated BA: **{simple.mean_S3_BA:.6f}**.
27. Reliability-gated BA: **{reliability.mean_S3_BA:.6f}**.
28. Nontrivial adaptation rate: **{gates['E_nontrivial_coverage']}**.
29. Gates A-F: `{gates}`.
30. Authorization: **{authorization}**.
31. Strongest justified claim: {strongest}
32. Stronger unsupported claim: {stronger_unsupported}
33. Terminal: **{terminal}**.

The subject is the statistical unit throughout: the two backbone rows are held out and bootstrapped together.
"""
    c.write_text(c.EXP / "RELIABILITY_STAGE05_FINAL_REPORT.md", report)

    c.write_text(
        c.EXP / "MECHANISM_AUDIT.md",
        f"""# Mechanism audit

All features were frozen before the S3 outcome merge and use only S1-validation/S2. The highest frozen single-family R_sign OOF AUROC was `{best_mechanism}` ({best_row.AUROC:.4f}, subject-bootstrap CI [{best_row.AUROC_CI95_low:.4f}, {best_row.AUROC_CI95_high:.4f}]). Raw Delta2 was {cv_value('R_sign', 'M2'):.4f}; backbone-only was {cv_value('R_sign', 'M1'):.4f}. {identity_comparison}

The conceptual priority was not changed after seeing outcomes. All higher-priority failures remain in the result tables.
""",
    )
    c.write_text(
        c.EXP / "POLICY_AUDIT.md",
        f"""# Policy audit

Policy eligibility under the frozen rule: **{policy_eligible}**. Thresholds were selected only from inner OOF predictions and outcomes within each outer training split. The held-out subject and both of its backbone rows were absent from fitting and threshold selection.

Simple S2 gate: coverage {simple.coverage:.4f}, harm {simple.future_harm_given_adaptation:.4f}, BA {simple.mean_S3_BA:.6f}. Reliability gate: coverage {reliability.coverage:.4f}, harm {reliability.future_harm_given_adaptation}, BA {reliability.mean_S3_BA:.6f}.
""",
    )
    c.write_text(
        c.EXP / "CLAIM_AUDIT.md",
        f"""# Claim audit

Terminal: **{terminal}**  
Authorization: **{authorization}**

Strongest supported claim: {strongest}

Unsupported stronger claim: {stronger_unsupported}
""",
    )
    print(terminal)
    print(authorization)


if __name__ == "__main__":
    main()
