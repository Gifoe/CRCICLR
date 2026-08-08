from __future__ import annotations

import numpy as np


def scale_aware_lambda(matrix: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    singular = np.linalg.svd(np.asarray(matrix, float), compute_uv=False)
    positive = singular[singular > np.finfo(float).eps * max(matrix.shape) * (singular[0] if len(singular) else 1.0)]
    if not len(positive):
        raise ValueError("measurement matrix has zero numerical rank")
    return float(alpha * np.median(positive ** 2)), singular


def lifting_operators(matrix: np.ndarray, alpha: float = 1e-2) -> dict[str, np.ndarray | float | int]:
    b = np.asarray(matrix, dtype=float)
    if b.ndim != 2:
        raise ValueError("B must be a matrix")
    regularization, singular = scale_aware_lambda(b, alpha)
    gram = b @ b.T + regularization * np.eye(b.shape[0])
    lifting = np.linalg.solve(gram, b).T
    projector = np.linalg.pinv(b) @ b
    resolution = lifting @ b
    rank = int(np.linalg.matrix_rank(b))
    positive = singular[singular > np.finfo(float).eps * max(b.shape) * singular[0]]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    return {
        "L": lifting, "Q": projector, "R": resolution,
        "lambda": regularization, "alpha": float(alpha), "singular_values": singular,
        "rank": rank, "condition_number": condition,
    }


def numerical_audit(matrix: np.ndarray, alpha: float = 1e-2) -> dict[str, object]:
    values = lifting_operators(matrix, alpha)
    q, r = values["Q"], values["R"]
    eigenvalues = np.linalg.eigvalsh((r + r.T) / 2)
    return {
        "rank": values["rank"], "condition_number": values["condition_number"],
        "alpha": values["alpha"], "lambda": values["lambda"],
        "singular_values": values["singular_values"].tolist(),
        "q_symmetry_relative_error": float(np.linalg.norm(q - q.T) / (np.linalg.norm(q) + 1e-15)),
        "q_idempotence_relative_error": float(np.linalg.norm(q @ q - q) / (np.linalg.norm(q) + 1e-15)),
        "r_symmetry_relative_error": float(np.linalg.norm(r - r.T) / (np.linalg.norm(r) + 1e-15)),
        "r_eigenvalue_min": float(eigenvalues.min()), "r_eigenvalue_max": float(eigenvalues.max()),
    }
