"""Train-only local-neighbourhood error competence."""

from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build(configuration: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    components = configuration.get("pca_components")
    if components is not None:
        steps.append(("pca", PCA(n_components=int(components), random_state=0)))
    steps.append(
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=int(configuration["n_neighbors"]),
                weights=str(configuration.get("weights", "distance")),
                metric="euclidean",
                n_jobs=4,
            ),
        )
    )
    return Pipeline(steps)
