from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .basis import CanonicalBasis, normalize_electrode_name


@dataclass(frozen=True)
class MeasurementFunctional:
    name: str
    terms: tuple[tuple[str, float], ...]
    reference_definition: str

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("measurement functional requires at least one term")
        if not all(np.isfinite(weight) and weight != 0 for _, weight in self.terms):
            raise ValueError("measurement weights must be finite and nonzero")

    @classmethod
    def bipolar(cls, active: str, reference: str) -> "MeasurementFunctional":
        return cls(f"{active}-{reference}", ((active, 1.0), (reference, -1.0)), reference)

    def normalized(self, aliases: Mapping[str, str] | None = None) -> "MeasurementFunctional":
        merged: dict[str, float] = {}
        for electrode, weight in self.terms:
            key = normalize_electrode_name(electrode, aliases)
            merged[key] = merged.get(key, 0.0) + float(weight)
        terms = tuple((key, value) for key, value in sorted(merged.items()) if abs(value) > 1e-14)
        if not terms:
            raise ValueError(f"functional {self.name} cancels to zero")
        return MeasurementFunctional(self.name, terms, self.reference_definition)


def build_measurement_matrix(
    functionals: Iterable[MeasurementFunctional],
    coordinates: Mapping[str, np.ndarray],
    basis: CanonicalBasis,
    aliases: Mapping[str, str] | None = None,
) -> np.ndarray:
    rows = []
    normalized_coordinates = {normalize_electrode_name(k, aliases): np.asarray(v, float) for k, v in coordinates.items()}
    for functional in functionals:
        row = np.zeros(basis.dimension, dtype=float)
        for electrode, weight in functional.normalized(aliases).terms:
            if electrode not in normalized_coordinates:
                raise ValueError(f"coordinate missing for electrode {electrode}")
            row += weight * basis.evaluate(normalized_coordinates[electrode])[0]
        rows.append(row)
    return np.asarray(rows, dtype=float)


def coefficient_functionals(names: list[str], coefficients: np.ndarray, reference: str) -> list[MeasurementFunctional]:
    matrix = np.asarray(coefficients, float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("coefficient matrix shape mismatch")
    output = []
    for row_index, row in enumerate(matrix):
        terms = tuple((name, float(weight)) for name, weight in zip(names, row) if abs(weight) > 1e-12)
        output.append(MeasurementFunctional(f"derived_{row_index:02d}", terms, reference))
    return output
