from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np

from .lifting import lifting_operators


@dataclass(frozen=True)
class OperatorView:
    operator_id: str
    operator_family: str
    source_operator_id: str
    A: np.ndarray
    B: np.ndarray
    electrode_coefficients: np.ndarray
    channel_definition: tuple[str, ...]
    reference_definition: str
    split: str
    is_legal: bool = True
    legality_reason: str = "linear transformation of audited source observation"

    def audit_row(self, source_b: np.ndarray) -> dict[str, object]:
        residual = float(np.linalg.norm(self.B - self.A @ source_b) / (np.linalg.norm(self.B) + 1e-15))
        singular = np.linalg.svd(self.B, compute_uv=False)
        positive = singular[singular > np.finfo(float).eps * max(self.B.shape) * (singular[0] if len(singular) else 1.0)]
        return {
            "operator_id": self.operator_id, "operator_family": self.operator_family,
            "source_operator_id": self.source_operator_id, "channel_definition": "|".join(self.channel_definition),
            "electrode_terms": int(np.count_nonzero(self.electrode_coefficients)),
            "signed_weights": bool(np.any(self.electrode_coefficients < 0) and np.any(self.electrode_coefficients > 0)),
            "reference_definition": self.reference_definition, "is_legal": bool(self.is_legal and residual < 1e-10),
            "legality_residual": residual, "rank": int(np.linalg.matrix_rank(self.B)),
            "condition_number": float(positive.max() / positive.min()) if len(positive) else float("inf"),
            "split": self.split, "a_sha256": hashlib.sha256(np.ascontiguousarray(self.A).tobytes()).hexdigest(),
            "b_sha256": hashlib.sha256(np.ascontiguousarray(self.B).tobytes()).hexdigest(),
            "legality_reason": self.legality_reason,
        }

    def observability(self, alpha: float, dimension: int) -> dict[str, float | str]:
        values = lifting_operators(self.B, alpha)
        return {
            "operator_id": self.operator_id,
            "O_dim": float(values["rank"] / dimension),
            "O_eff": float(np.trace(values["R"]) / dimension),
        }


def _view(identifier: str, family: str, split: str, a: np.ndarray, source_b: np.ndarray, source_coefficients: np.ndarray, definitions: Iterable[str], reference: str) -> OperatorView:
    a = np.asarray(a, float)
    return OperatorView(identifier, family, "eegmmidb_car64", a, a @ source_b, a @ source_coefficients, tuple(definitions), reference, split)


def _selector(indices: list[int], size: int) -> np.ndarray:
    output = np.zeros((len(indices), size), float)
    output[np.arange(len(indices)), indices] = 1.0
    return output


def _bipolar(pairs: list[tuple[int, int]], size: int) -> np.ndarray:
    output = np.zeros((len(pairs), size), float)
    for row, (left, right) in enumerate(pairs):
        output[row, left], output[row, right] = 1.0, -1.0
    return output


def generate_eegmmidb_operators(channel_names: list[str], source_b: np.ndarray) -> list[OperatorView]:
    """Fixed legal transforms. Test topologies are disjoint, not permutations."""
    count = len(channel_names)
    if source_b.shape[0] != count:
        raise ValueError("source B/channel mismatch")
    car = np.eye(count) - np.ones((count, count)) / count
    source_coefficients = car.copy()
    index = {name: position for position, name in enumerate(channel_names)}

    def pick(names: list[str]) -> list[int]:
        missing = sorted(set(names) - set(index))
        if missing:
            raise ValueError(f"operator electrodes missing: {missing}")
        return [index[name] for name in names]

    dense_a = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "PZ", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "O1", "OZ", "O2"]
    dense_b = ["AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C5", "C1", "C2", "C6", "TP7", "CP3", "CP4", "TP8", "P5", "P1", "P2", "P6", "T9", "T10", "POZ", "O1", "O2", "IZ", "C3", "C4"]
    sparse_a = ["FP1", "FP2", "F7", "F3", "F4", "F8", "T7", "C3", "C4", "T8", "P7", "P3", "P4", "P8", "O1", "O2"]
    sparse_b = ["AF3", "AF4", "F5", "F1", "F2", "F6", "FC3", "FC4", "CP3", "CP4", "P5", "P1", "P2", "P6", "PO3", "PO4"]
    sparse_test = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sparse_test_12 = ["FP1", "FP2", "F3", "F4", "FC3", "FC4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sparse_test_4 = ["C3", "C4", "O1", "O2"]
    bipolar_train = [("F3", "C3"), ("C3", "P3"), ("P3", "O1"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("F7", "T7"), ("F8", "T8")]
    bipolar_val = [("FC3", "CP3"), ("CP3", "PO3"), ("FC4", "CP4"), ("CP4", "PO4"), ("C5", "P5"), ("C6", "P6")]
    bipolar_test = [("F3", "P3"), ("C3", "O1"), ("F4", "P4"), ("C4", "O2"), ("FP1", "O1"), ("FP2", "O2")]
    bipolar_test_2 = [("FC5", "CP5"), ("C5", "P5"), ("FC6", "CP6"), ("C6", "P6"), ("AF3", "PO3"), ("AF4", "PO4")]
    bipolar_test_3 = [("F7", "P7"), ("T7", "O1"), ("F8", "P8"), ("T8", "O2"), ("F1", "P1"), ("F2", "P2")]
    views = []
    for identifier, split, names in (
        ("dense32_a", "train", dense_a), ("dense32_b", "train", dense_b),
        ("sparse16_a", "train", sparse_a), ("sparse16_b", "validation", sparse_b),
        ("sparse12_heldout", "test", sparse_test_12), ("sparse8_heldout", "test", sparse_test),
        ("sparse4_heldout", "test", sparse_test_4),
    ):
        ids = pick(names); a = _selector(ids, count)
        views.append(_view(identifier, "dense_subset" if len(ids) == 32 else "sparse_subset", split, a, source_b, source_coefficients, names, "CAR64-derived referential subset"))
    for identifier, split, pairs in (
        ("bipolar_train", "train", bipolar_train), ("bipolar_validation", "validation", bipolar_val),
        ("bipolar_heldout_topology", "test", bipolar_test),
        ("bipolar_heldout_topology_2", "test", bipolar_test_2),
        ("bipolar_heldout_topology_3", "test", bipolar_test_3),
    ):
        numeric = [(index[a], index[b]) for a, b in pairs]
        a = _bipolar(numeric, count)
        views.append(_view(identifier, "bipolar", split, a, source_b, source_coefficients, [f"{a}-{b}" for a, b in pairs], "reference cancels within explicit derivation"))
    polarity_source = next(view for view in views if view.operator_id == "bipolar_heldout_topology")
    polarity_a = -polarity_source.A
    views.append(_view("polarity_heldout", "polarity", "test", polarity_a, source_b, source_coefficients, [f"reverse({x})" for x in polarity_source.channel_definition], "explicit sign reversal"))
    polarity_source_2 = next(view for view in views if view.operator_id == "bipolar_heldout_topology_2")
    views.append(_view("polarity_heldout_2", "polarity", "test", -polarity_source_2.A, source_b, source_coefficients, [f"reverse({x})" for x in polarity_source_2.channel_definition], "explicit sign reversal"))
    sparse_ids = pick(sparse_test)
    local_car = (np.eye(len(sparse_ids)) - np.ones((len(sparse_ids), len(sparse_ids))) / len(sparse_ids)) @ _selector(sparse_ids, count)
    views.append(_view("rereference_car8_heldout", "rereference", "test", local_car, source_b, source_coefficients, [f"{name}-mean(CAR8)" for name in sparse_test], "explicit CAR8 derived from available channels"))
    return views
