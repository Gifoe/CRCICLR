"""Aggregation is intentionally invoked by evaluate_fusion after reproduction.

Keeping the entry point small makes clear that it cannot be run on arbitrary
or tuned prediction files as a post-hoc method-selection step.
"""
from evaluate_fusion import bootstrap_pp, primary_subject_frame

__all__ = ["bootstrap_pp", "primary_subject_frame"]
