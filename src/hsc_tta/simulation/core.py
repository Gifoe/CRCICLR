from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from hsc_tta.certification import empirical_bernstein_bound, fit_simultaneous_quantile, apply_certificate
from hsc_tta.selection import select_safe_action


ACTIONS = ("no_tta", "t3a", "entropy_adapter")


def generate_subject_surface(n_subjects: int = 120, lambdas: np.ndarray | None = None, seed: int = 0, split_role: str = "mixed") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lambdas = np.asarray(lambdas if lambdas is not None else np.linspace(0.50, 0.99, 20))
    rows = []
    for s in range(n_subjects):
        latent = rng.normal()
        role = ("meta_risk_train" if s < n_subjects // 2 else "conformal_calibration" if s < 3 * n_subjects // 4 else "final_test") if split_role == "mixed" else split_role
        for ai, action in enumerate(ACTIONS):
            harm = (0.035 * ai * np.tanh(latent))
            base = np.clip(0.30 + 0.08 * latent + harm, 0.03, 0.75)
            for lam in lambdas:
                true_risk = float(np.clip(base - 0.24 * (lam - 0.5) + rng.normal(0, 0.012), 0, 1))
                blocks = np.clip(rng.normal(true_risk, 0.05, 12), 0, 1)
                bound = empirical_bernstein_bound(blocks)
                predicted = float(np.clip(true_risk + rng.normal(-0.035, 0.035), 0, 1))
                rows.append({"subject_id": f"synthetic:{s:04d}", "split_role": role, "action": action, "lambda": float(lam), "predicted_risk": predicted, "upper_risk": bound["upper_risk"], "future_risk": true_risk, "average_set_size": float(np.clip(5 - 6 * (lam - 0.5) + 0.15 * ai, 1, 5)), "singleton_rate": float(np.clip(1.4 * (lam - 0.5) - 0.05 * ai, 0, 1)), "n_classes": 5, "argmax_error": float(np.clip(base + 0.02 * ai, 0, 1))})
    return pd.DataFrame(rows)


def run_simulations(output_dir: str | Path, seed: int = 0, n_subjects: int = 120) -> dict[str, pd.DataFrame]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    surface = generate_subject_surface(n_subjects=n_subjects, seed=seed)
    cal = surface[surface.split_role == "conformal_calibration"]
    test = surface[surface.split_role == "final_test"].copy()
    quantile = fit_simultaneous_quantile(cal, delta=0.10)
    test["certified_upper_bound"] = apply_certificate(test.predicted_risk, quantile)
    decisions = []
    for subject_id, group in test.groupby("subject_id"):
        choice = select_safe_action(group, alpha=0.20)
        if choice["status"] == "certified":
            row = choice["selected_row"]
            decisions.append({"subject_id": subject_id, "status": "certified", "selected_action": row["action"], "selected_lambda": row["lambda"], "true_future_risk": row["future_risk"], "certified_upper_bound": row["certified_upper_bound"], "average_set_size": row["average_set_size"], "singleton_rate": row["singleton_rate"]})
        else:
            decisions.append({"subject_id": subject_id, "status": "uncertified", "selected_action": None, "selected_lambda": np.nan, "true_future_risk": np.nan, "certified_upper_bound": 1.0, "average_set_size": 5.0, "singleton_rate": 0.0})
    decisions = pd.DataFrame(decisions)
    # Simulation A: windows do not increase the number of independent subjects.
    pseudo = pd.DataFrame([{"n_subjects": n, "windows_per_subject": w, "effective_subject_units": n} for n in (10, 30, 60) for w in (20, 200, 2000)])
    coverage = test.groupby("subject_id").apply(lambda g: bool((g.future_risk <= g.certified_upper_bound).all()), include_groups=False).rename("surface_valid").reset_index()
    post = pd.DataFrame({"method": ["pointwise_proxy", "simultaneous"], "coverage": [float((test.future_risk <= np.clip(test.predicted_risk + np.quantile(cal.upper_risk - cal.predicted_risk, 0.9), 0, 1)).mean()), float(coverage.surface_valid.mean())]})
    safety = decisions.groupby("status").agg(n_subjects=("subject_id", "count"), mean_set_size=("average_set_size", "mean")).reset_index()
    misspec = pd.DataFrame({"predictor_bias": [-0.10, -0.05, 0.0], "conformal_q": [float(quantile.q + 0.10), float(quantile.q + 0.05), float(quantile.q)]})
    summary = pd.DataFrame([{"seed": seed, "n_subjects": n_subjects, "n_calibration": quantile.n_calibration_subjects, "q": quantile.q, "surface_coverage": float(coverage.surface_valid.mean()), "certified_subject_rate": float((decisions.status == "certified").mean())}])
    outputs = {"simulation_summary": summary, "certificate_coverage": coverage, "pseudo_sample_size": pseudo, "post_selection_validity": post, "safety_utility": safety, "risk_predictor_misspecification": misspec}
    for name, frame in outputs.items(): frame.to_csv(output / f"{name}.csv", index=False)
    return outputs

