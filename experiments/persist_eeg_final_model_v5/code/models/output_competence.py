from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build(configuration: dict[str, Any], seed: int):
    family = str(configuration["family"])
    if family == "logistic":
        steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
        components = configuration.get("pca_components")
        if components is not None:
            steps.append(("pca", PCA(n_components=int(components), random_state=seed)))
        steps.append(
            (
                "model",
                LogisticRegression(
                    C=float(configuration["C"]),
                    solver="lbfgs",
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        )
        return Pipeline(steps)
    if family == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=float(configuration["learning_rate"]),
            max_leaf_nodes=int(configuration["max_leaf_nodes"]),
            max_iter=int(configuration.get("max_iter", 200)),
            l2_regularization=float(configuration["l2"]),
            min_samples_leaf=int(configuration.get("min_samples_leaf", 40)),
            random_state=seed,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(configuration.get("n_estimators", 500)),
            max_depth=configuration.get("max_depth"),
            min_samples_leaf=int(configuration.get("min_samples_leaf", 10)),
            max_features=configuration.get("max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(family)


def fit(model, x, y, sample_weight):
    if isinstance(model, Pipeline):
        return model.fit(x, y, model__sample_weight=sample_weight)
    return model.fit(x, y, sample_weight=sample_weight)
