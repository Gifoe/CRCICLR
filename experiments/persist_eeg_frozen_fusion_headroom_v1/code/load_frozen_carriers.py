"""Public provenance helper for the frozen fusion audit.

Checkpoint loading is implemented in evaluate_fusion.py so that strict loading,
parameter freezing, and evaluation mode are one atomic operation.
"""
from evaluate_fusion import load_model, sha256  # re-exported for audit review

__all__ = ["load_model", "sha256"]
