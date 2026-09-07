from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def configurations() -> list[dict[str, Any]]:
    return [
        {"max_depth": depth, "min_samples_leaf": leaf, "l2": l2}
        for depth in (2, 3)
        for leaf in (50, 100, 200)
        for l2 in (1.0, 10.0)
    ]


def build(configuration: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=150,
                    max_depth=int(configuration["max_depth"]),
                    min_samples_leaf=int(configuration["min_samples_leaf"]),
                    l2_regularization=float(configuration["l2"]),
                    random_state=int(seed),
                ),
            ),
        ]
    )
