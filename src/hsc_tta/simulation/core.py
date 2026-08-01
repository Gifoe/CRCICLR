from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.certification import (
    ACTIONS,
    apply_critical_index_certificate,
    fit_actionwise_simultaneous_quantile,
)


def _mcse(values: np.ndarray) -> float:
    p = float(np.mean(values))
    return math.sqrt(max(p * (1.0 - p), 0.0) / len(values))


def evaluate_cpu_go(raw: pd.DataFrame, repetitions: int, n_nontrivial: int) -> dict[str, float]:
    """Evaluate the frozen CPU GO criteria without inspecting real final-test data."""
    if repetitions < 500:
        raise RuntimeError("CPU GO requires at least 500 Monte Carlo repetitions")
    feasible = raw[raw.alpha == 0.20]
    if feasible.empty:
        raise RuntimeError("CPU GO failed: alpha=0.20 feasible simulation is missing")
    csr = float(feasible.csr_nonfull.mean())
    coverage = float(feasible.selected_coverage.mean())
    tolerance = 0.01 + 2.0 * float(
        feasible.selected_coverage.std(ddof=1) / math.sqrt(repetitions)
    )
    saturation = float(feasible.q_saturated.mean())
    if csr <= 0:
        raise RuntimeError("CPU GO failed: feasible alpha=0.20 CSR is zero")
    if coverage < 0.90 - tolerance:
        raise RuntimeError("CPU GO failed: selected-risk coverage below tolerance")
    if saturation > 0.05:
        raise RuntimeError("CPU GO failed: critical-index quantile frequently saturates")
    return {
        "csr_nonfull_alpha_0_20": csr,
        "selected_coverage_alpha_0_20": coverage,
        "coverage_tolerance": tolerance,
        "q_saturation_rate": saturation,
        "n_nontrivial_lambdas": float(n_nontrivial),
    }


def _calibration_frame(
    truth: np.ndarray, prediction: np.ndarray, *, alpha: float, repetition: int
) -> pd.DataFrame:
    rows = []
    for subject in range(truth.shape[0]):
        for action_index, action in enumerate(ACTIONS):
            rows.append(
                {
                    "dataset": "synthetic",
                    "seed": repetition,
                    "episode_id": f"synthetic:{repetition}:{subject}",
                    "subject_id": f"s{subject:03d}",
                    "alpha": alpha,
                    "action": action,
                    "critical_index": int(truth[subject, action_index]),
                    "predicted_critical_index": float(prediction[subject, action_index]),
                }
            )
    return pd.DataFrame(rows)


def _draw_indices(
    rng: np.random.Generator,
    n: int,
    *,
    alpha: float,
    predictor_bias: float,
    n_nontrivial: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_difficulty = rng.normal(0, 2.0, size=(n, 1))
    action_shift = np.asarray([0.0, -0.8, 0.7])[None, :]
    center = 7.0 if np.isclose(alpha, 0.20) else 10.5
    truth = np.clip(
        np.rint(center + subject_difficulty + action_shift + rng.normal(0, 0.6, (n, 3))),
        0,
        n_nontrivial,
    ).astype(int)
    u_signal = rng.normal(0, 0.9, (n, 3))
    residual_noise = 1.8 * u_signal + rng.normal(0, 1.4, (n, 3))
    prediction = np.clip(truth - residual_noise + predictor_bias, 0, n_nontrivial)
    return truth, prediction, u_signal


def _run_repetition(
    rng: np.random.Generator,
    repetition: int,
    *,
    alpha: float,
    n_calibration: int,
    n_test: int,
    n_nontrivial: int,
    predictor_bias: float = 0.0,
) -> dict[str, float | bool]:
    cal_truth, cal_prediction, _ = _draw_indices(
        rng,
        n_calibration,
        alpha=alpha,
        predictor_bias=predictor_bias,
        n_nontrivial=n_nontrivial,
    )
    quantile = fit_actionwise_simultaneous_quantile(
        _calibration_frame(cal_truth, cal_prediction, alpha=alpha, repetition=repetition),
        delta=0.10,
        n_nontrivial_lambdas=n_nontrivial,
    )
    test_truth, test_prediction, u_signal = _draw_indices(
        rng,
        n_test,
        alpha=alpha,
        predictor_bias=predictor_bias,
        n_nontrivial=n_nontrivial,
    )
    certified = apply_critical_index_certificate(test_prediction.ravel(), quantile).reshape(n_test, 3)
    actionwise_valid = np.all(test_truth <= certified, axis=1)
    nontrivial = certified < n_nontrivial
    context_set_size = 1.0 + 4.0 * certified / n_nontrivial + np.asarray([0.0, 0.05, 0.10])
    context_set_size[~nontrivial] = 5.0
    # U-only selector proxy: utility is context set size plus a small observed diagnostic.
    utility = context_set_size + 0.03 * u_signal + np.asarray([0.0, 0.01, 0.02])
    utility[~nontrivial] = np.inf
    any_nontrivial = np.any(nontrivial, axis=1)
    selected_action = np.argmin(utility, axis=1)
    selected_index = certified[np.arange(n_test), selected_action]
    selected_truth = test_truth[np.arange(n_test), selected_action]
    selected_valid = np.where(any_nontrivial, selected_truth <= selected_index, True)
    average_selected_size = np.where(
        any_nontrivial,
        context_set_size[np.arange(n_test), selected_action],
        5.0,
    )
    # Pointwise action calibration, then adversarial U-only selection correlated with residual.
    residual_cal = cal_truth - cal_prediction
    k = int(math.ceil((n_calibration + 1) * 0.90))
    point_q = np.sort(residual_cal, axis=0)[k - 1]
    point_certified = np.clip(np.ceil(test_prediction + point_q), 0, n_nontrivial)
    adversarial_action = np.argmax(u_signal, axis=1)
    point_selected_valid = test_truth[np.arange(n_test), adversarial_action] <= point_certified[
        np.arange(n_test), adversarial_action
    ]
    simultaneous_selected_valid = test_truth[
        np.arange(n_test), adversarial_action
    ] <= certified[np.arange(n_test), adversarial_action]
    return {
        "alpha": alpha,
        "q_alpha": quantile.q_alpha,
        "actionwise_coverage": float(np.mean(actionwise_valid)),
        "selected_coverage": float(np.mean(selected_valid)),
        "csr_nonfull": float(np.mean(any_nontrivial)),
        "full_set_fallback_rate": float(np.mean(~any_nontrivial)),
        "mean_context_set_size": float(np.mean(average_selected_size)),
        "pointwise_adversarial_coverage": float(np.mean(point_selected_valid)),
        "simultaneous_adversarial_coverage": float(np.mean(simultaneous_selected_valid)),
        "q_saturated": bool(quantile.q_alpha >= n_nontrivial),
    }


def _adversarial_selection_simulation(
    rng: np.random.Generator,
    *,
    repetitions: int,
    n_calibration: int,
    n_test: int,
) -> pd.DataFrame:
    """Contrast marginal action calibration with calibration of the actionwise maximum."""
    pointwise_rates: list[float] = []
    simultaneous_rates: list[float] = []
    k = int(math.ceil((n_calibration + 1) * 0.90))
    for _ in range(repetitions):
        calibration_scores = rng.normal(size=(n_calibration, len(ACTIONS)))
        pointwise_q = np.sort(calibration_scores, axis=0)[k - 1]
        simultaneous_q = float(np.sort(calibration_scores.max(axis=1))[k - 1])
        # The selector sees a U_s diagnostic correlated with each action's future residual.
        context_signal = rng.normal(size=(n_test, len(ACTIONS)))
        future_residual = context_signal + rng.normal(0, 0.05, context_signal.shape)
        selected = np.argmax(context_signal, axis=1)
        selected_residual = future_residual[np.arange(n_test), selected]
        pointwise_rates.append(float(np.mean(selected_residual <= pointwise_q[selected])))
        simultaneous_rates.append(float(np.mean(selected_residual <= simultaneous_q)))
    return pd.DataFrame(
        [
            {
                "method": "pointwise_action_calibration",
                "coverage": float(np.mean(pointwise_rates)),
                "monte_carlo_se": float(np.std(pointwise_rates, ddof=1) / math.sqrt(repetitions)),
            },
            {
                "method": "actionwise_simultaneous_calibration",
                "coverage": float(np.mean(simultaneous_rates)),
                "monte_carlo_se": float(np.std(simultaneous_rates, ddof=1) / math.sqrt(repetitions)),
            },
        ]
    )


def run_simulations(
    output_dir: str | Path,
    *,
    seed: int = 0,
    repetitions: int = 500,
    n_calibration: int = 20,
    n_test: int = 80,
    n_nontrivial: int = 20,
    enforce_go: bool = False,
) -> dict[str, pd.DataFrame]:
    if repetitions < 1 or n_calibration < 2 or n_test < 1:
        raise ValueError("positive repetitions/test size and at least two calibration subjects required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = [
        _run_repetition(
            rng,
            repetition,
            alpha=alpha,
            n_calibration=n_calibration,
            n_test=n_test,
            n_nontrivial=n_nontrivial,
        )
        for repetition in range(repetitions)
        for alpha in (0.10, 0.20)
    ]
    raw = pd.DataFrame(records)
    summary_rows = []
    for alpha, group in raw.groupby("alpha", sort=True):
        for metric in (
            "actionwise_coverage",
            "selected_coverage",
            "csr_nonfull",
            "full_set_fallback_rate",
            "mean_context_set_size",
        ):
            values = group[metric].to_numpy(float)
            summary_rows.append(
                {
                    "simulation": "B/C/D/F",
                    "alpha": alpha,
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "monte_carlo_se": float(values.std(ddof=1) / math.sqrt(repetitions)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    simulation_a = pd.DataFrame(
        [
            {"n_subjects": subjects, "windows_per_subject": windows, "independent_units": subjects}
            for subjects in (10, 30, 60)
            for windows in (20, 200, 2000)
        ]
    )
    simulation_e_rows = []
    for bias in (-2.0, 0.0, 2.0):
        values = [
            _run_repetition(
                rng,
                repetitions + index,
                alpha=0.20,
                n_calibration=n_calibration,
                n_test=n_test,
                n_nontrivial=n_nontrivial,
                predictor_bias=bias,
            )
            for index in range(repetitions)
        ]
        coverage = np.asarray([row["selected_coverage"] for row in values], float)
        qs = np.asarray([row["q_alpha"] for row in values], float)
        simulation_e_rows.append(
            {
                "predictor_bias": bias,
                "selected_coverage": float(coverage.mean()),
                "monte_carlo_se": float(coverage.std(ddof=1) / math.sqrt(repetitions)),
                "mean_q_alpha": float(qs.mean()),
            }
        )
    simulation_e = pd.DataFrame(simulation_e_rows)
    simulation_g = _adversarial_selection_simulation(
        rng,
        repetitions=repetitions,
        n_calibration=n_calibration,
        n_test=n_test,
    )
    outputs = {
        "simulation_summary": summary,
        "simulation_a_subject_vs_window": simulation_a,
        "simulation_b_to_f_repetitions": raw,
        "simulation_e_misspecification": simulation_e,
        "simulation_g_adversarial_selection": simulation_g,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    if enforce_go:
        evaluate_cpu_go(raw, repetitions, n_nontrivial)
    return outputs
