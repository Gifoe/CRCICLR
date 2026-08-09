from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.v2.joint_certificate import finite_sample_quantile
from hsc_tta.v2.selector_v2 import select_joint_action


CALIBRATION_COUNTS = (10, 15, 20, 25, 30, 50, 75, 100)
ACTION_COUNTS = (2, 3, 5, 10)


def _apply_q(bounds: pd.DataFrame, features: pd.DataFrame, outcomes: pd.DataFrame, q: float) -> dict[str, float]:
    current = bounds.copy()
    current["index"] = np.clip(np.ceil(current.predicted_critical_index + q * current.c_j), 0, 20).astype(int)
    current["lower"] = current.predicted_benefit - q * current.c_delta
    current.loc[current.action == "no_tta", "lower"] = -np.inf
    current = current.merge(features[["dataset", "seed", "subject_id", "action", "action_cost"] +
                                     [f"context_set_size_j{i}" for i in range(21)]],
                            on=["dataset", "seed", "subject_id", "action"], validate="one_to_one")
    chosen = []
    for subject, group in current.groupby("subject_id"):
        candidates = pd.DataFrame({"action": group.action, "available": group.available,
                                   "certified_critical_index": group["index"], "benefit_lower": group.lower,
                                   "context_average_set_size": [r[f"context_set_size_j{int(r['index'])}"] for _, r in group.iterrows()],
                                   "adaptation_cost": group.action_cost})
        decision = select_joint_action(candidates, sentinel_index=20)
        chosen.append({"subject_id": subject, "action": decision["selected_action"],
                       "selected_index": decision["certified_critical_index"]})
    selected = pd.DataFrame(chosen).merge(outcomes, on=["subject_id", "action"], validate="one_to_one")
    risks = np.asarray([r[f"risk_j{int(r.selected_index)}"] for _, r in selected.iterrows()])
    sizes = np.asarray([r[f"set_size_j{int(r.selected_index)}"] for _, r in selected.iterrows()])
    tta = selected.action != "no_tta"
    certified = selected.selected_index < 20
    safe_oracle = outcomes[(outcomes.action != "no_tta") & (outcomes.true_critical_index < 20) &
                           (outcomes.true_benefit > 0)].groupby("subject_id").true_benefit.max()
    oracle_total = float(safe_oracle.sum())
    captured = float(selected.set_index("subject_id").true_benefit.reindex(safe_oracle.index).clip(lower=0).sum() / oracle_total) if oracle_total else np.nan
    return {"csr": float(certified.mean()), "full_set_fallback": float((~certified).mean()),
            "positive_benefit_certification_rate": float(tta.mean()),
            "joint_validity": float(np.mean((risks <= float(outcomes.alpha.iloc[0])) & (~tta | (selected.true_benefit >= 0)))),
            "average_set_size": float(sizes.mean()), "safe_oracle_headroom_captured": captured}


def run_certifiability(root: str | Path, repetitions: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    base = root / "outputs/v2_joint_certified"
    scores = pd.read_parquet(base / "nested_dev/ALL_DEV_CALIBRATION_SCORES.parquet")
    bounds = pd.read_parquet(base / "nested_dev/ALL_DEV_JOINT_BOUNDS.parquet")
    outcomes = pd.read_parquet(base / "nested_dev/ALL_DEV_COUNTERFACTUALS.parquet")
    features = pd.read_parquet(base / "actions/DEVELOPMENT_CONTEXT_FEATURES.parquet")
    sample_rows: list[dict[str, float | int | str]] = []
    action_rows: list[dict[str, float | int | str]] = []
    keys_columns = ["dataset", "seed", "outer_fold", "alpha"]
    for keys, current_scores in scores.groupby(keys_columns):
        dataset, seed, fold, alpha = keys
        current_bounds = bounds[(bounds.dataset == dataset) & (bounds.seed == seed) &
                                (bounds.outer_fold == fold) & np.isclose(bounds.alpha, alpha)]
        ids = current_bounds.subject_id.unique()
        current_outcomes = outcomes[(outcomes.dataset == dataset) & (outcomes.seed == seed) &
                                    (outcomes.outer_fold == fold) & np.isclose(outcomes.alpha, alpha)]
        current_features = features[(features.dataset == dataset) & (features.seed == seed) & features.subject_id.isin(ids)]
        observed = current_scores.joint_score.to_numpy(float)
        rng = np.random.default_rng(92000 + int(seed) * 100 + int(fold) * 10 + int(float(alpha) * 100))
        for m in CALIBRATION_COUNTS:
            for rep in range(repetitions):
                sampled = rng.choice(observed, m, replace=m > len(observed))
                q, _ = finite_sample_quantile(sampled, 0.1)
                metrics = _apply_q(current_bounds, current_features, current_outcomes, q)
                sample_rows.append({"dataset": dataset, "seed": seed, "outer_fold": fold, "alpha": alpha,
                                    "calibration_subject_count": m, "repetition": rep, "estimated_q": q, **metrics})
        for action_count in ACTION_COUNTS:
            # The observed library has three actions. For larger hypothetical libraries, use a
            # conservative multiplicity stress test (Bonferroni delta scaling), not fabricated actions.
            delta_effective = min(0.1, 0.1 * 3.0 / action_count)
            q, _ = finite_sample_quantile(observed, delta_effective)
            metrics = _apply_q(current_bounds, current_features, current_outcomes, q)
            action_rows.append({"dataset": dataset, "seed": seed, "outer_fold": fold, "alpha": alpha,
                                "action_count": action_count, "observed_action_count": 3,
                                "delta_effective": delta_effective, "estimated_q": q, **metrics})
    sample_frame, action_frame = pd.DataFrame(sample_rows), pd.DataFrame(action_rows)
    out = base / "certifiability"
    out.mkdir(exist_ok=True)
    sample_frame.to_csv(out / "CERTIFIABILITY_SAMPLE_SIZE.csv", index=False)
    action_frame.to_csv(out / "CERTIFIABILITY_ACTION_COUNT.csv", index=False)
    summary = sample_frame.groupby(["dataset", "alpha", "calibration_subject_count"])[
        ["estimated_q", "csr", "full_set_fallback", "positive_benefit_certification_rate", "joint_validity",
         "average_set_size", "safe_oracle_headroom_captured"]].mean().reset_index()
    lines = ["# Certifiability and sample-size audit", "",
             "Counts above the observed calibration budget are residual-bootstrap projections, not new independent subjects. Action counts 5 and 10 use conservative multiplicity stress scaling while evaluating the observed three-action library; no synthetic TTA utility is claimed.", "",
             "| Dataset | alpha | m | q | CSR | fallback | positive TTA | joint validity |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.alpha:.2f} | {row.calibration_subject_count} | {row.estimated_q:.3f} | {row.csr:.3f} | {row.full_set_fallback:.3f} | {row.positive_benefit_certification_rate:.3f} | {row.joint_validity:.3f} |")
    lines.extend(["", "## Interpretation", "",
                  "The current m=12/14 nested calibration folds are sufficient for conservative validity but not for certifying positive adaptation: the empirical positive-TTA rate is zero. The requested m=15/20/25 range improves risk CSR in bootstrap projections but cannot solve weak U-to-benefit predictability by sample size alone."])
    (out / "CERTIFIABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return sample_frame, action_frame
