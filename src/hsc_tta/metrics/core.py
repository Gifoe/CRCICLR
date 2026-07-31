from __future__ import annotations

import numpy as np
import pandas as pd


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def subject_metrics(decisions: pd.DataFrame, alpha: float, n_classes: int) -> dict[str, float]:
    required = {"subject_id", "status", "true_future_risk", "certified_upper_bound", "average_set_size", "singleton_rate", "selected_action", "selected_error", "no_tta_error"}
    if not required.issubset(decisions.columns):
        raise ValueError(f"missing columns: {sorted(required - set(decisions.columns))}")
    d = decisions.drop_duplicates("subject_id").copy()
    cert = d.status == "certified"
    adapted = cert & d.selected_action.isin(["t3a", "entropy_adapter"])
    return {
        "n_subjects": float(len(d)),
        "selected_certificate_validity_all": _mean_or_nan(d.true_future_risk <= d.certified_upper_bound),
        "selected_certificate_validity_certified": _mean_or_nan((d.loc[cert, "true_future_risk"] <= d.loc[cert, "certified_upper_bound"])),
        "subject_risk_violation_all": _mean_or_nan(d.true_future_risk > alpha),
        "subject_risk_violation_certified": _mean_or_nan(d.loc[cert, "true_future_risk"] > alpha),
        "csr": float(cert.mean()),
        "csr_nonfull": float((cert & (d.average_set_size < n_classes)).mean()),
        "csr_at_2": float((cert & (d.average_set_size <= 2)).mean()),
        "mean_excess_risk": _mean_or_nan(np.maximum(d.true_future_risk - alpha, 0)),
        "bound_gap": _mean_or_nan(d.certified_upper_bound - d.true_future_risk),
        "negative_adaptation_rate": _mean_or_nan(d.loc[adapted, "selected_error"] > d.loc[adapted, "no_tta_error"]),
        "average_set_size": _mean_or_nan(d.loc[cert, "average_set_size"]),
        "singleton_rate": _mean_or_nan(d.loc[cert, "singleton_rate"]),
        "uncertified_rate": float((~cert).mean()),
    }


def subject_bootstrap_ci(decisions: pd.DataFrame, statistic, replicates: int = 1000, seed: int = 2027) -> tuple[float, float]:
    subjects = decisions.subject_id.drop_duplicates().to_numpy()
    if subjects.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = []
    indexed = decisions.set_index("subject_id", drop=False)
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=subjects.size, replace=True)
        frame = pd.concat([indexed.loc[[s]] for s in sampled], ignore_index=True)
        values.append(float(statistic(frame)))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())

