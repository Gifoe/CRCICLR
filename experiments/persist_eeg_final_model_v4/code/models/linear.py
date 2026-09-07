from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def configurations() -> list[dict[str, Any]]:
    return [
        {"C": c, "penalty": penalty}
        for penalty in ("l2", "l1")
        for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    ]


def build(configuration: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(configuration["C"]),
                    penalty=str(configuration["penalty"]),
                    solver="liblinear",
                    max_iter=3000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
