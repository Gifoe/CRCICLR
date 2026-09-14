"""Frozen-cache loading for SEARCH cells.

This module is intentionally imported only by ``run_search.py`` after the
committed protocol freeze.  It has no fixed-held-out path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(os.environ["SEVEN_REPO"]).resolve()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    try: spec.loader.exec_module(value)
    except Exception:
        sys.modules.pop(name, None); raise
    return value


def _sources() -> tuple[Any, Any]:
    modern = _load_module("seven_search_modern_common", REPO / "experiments" / "persist_eeg_outcome_blind_modern_backbone_seed0_v1" / "code" / "modern_common.py")
    tasks = _load_module("seven_search_task_datasets", REPO / "experiments" / "persist_eeg_openbmi_task_generality_v1" / "code" / "task_datasets.py")
    return modern, tasks


def _normalise(train: np.ndarray, *other: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    observations = train.shape[0] * train.shape[2]
    total = train.sum(axis=(0, 2), dtype=np.float64)
    square = np.square(train, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
    mean = (total / observations).astype(np.float32)
    std = np.sqrt(np.maximum(square / observations - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    apply = lambda value: ((value - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    return apply(train), [apply(value) for value in other], {"mean_std_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest(), "samples_per_channel": int(observations)}


def _extract_openbmi(inner: Any, root: Path, subjects: list[str], sessions: tuple[int, ...], cache_name: str, mapping: dict[int, int] | None = None):
    return inner._openbmi_rows(root, subjects, sessions, cache_name, mapping)


def _extract_wbcic(inner: Any, root: Path, subjects: list[str], sessions: tuple[int, ...], mapping: dict[int, int] | None = None):
    return inner._wbcic_rows(root, subjects, sessions, mapping)


def load_search_fold(task: str, fold_id: int) -> dict[str, Any]:
    """Load one frozen inner train/val/outer SEARCH fold; never held-out data."""
    inner = _load_module("seven_search_inner_loader", REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code" / "tech_recipe_selection.py")
    modern, task_code = _sources()
    if task in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = "OpenBMI" if task == "OpenBMI_MI" else "WBCIC"
        folds, _, split_hash = modern.load_split()
        fold = next(value for value in folds[dataset] if int(value["fold_id"]) == int(fold_id))
        source_sessions, future = modern.SOURCE_SESSIONS[dataset], (modern.EVAL_SESSION,)
        if dataset == "OpenBMI":
            root = Path(os.environ["FULL_OPENBMI_CACHE"]).resolve()
            train_x, train_y, train_s, mapping = _extract_openbmi(inner, root, fold["inner_train_subjects"], source_sessions, "mi")
            val_x, val_y, val_s, _ = _extract_openbmi(inner, root, fold["inner_val_subjects"], future, "mi", mapping)
            outer_x, outer_y, outer_s, _ = _extract_openbmi(inner, root, fold["outer_dev_subjects"], future, "mi", mapping)
        else:
            root = Path(os.environ["FULL_WBCIC_CACHE"]).resolve()
            train_x, train_y, train_s, mapping = _extract_wbcic(inner, root, fold["inner_train_subjects"], source_sessions)
            val_x, val_y, val_s, _ = _extract_wbcic(inner, root, fold["inner_val_subjects"], future, mapping)
            outer_x, outer_y, outer_s, _ = _extract_wbcic(inner, root, fold["outer_dev_subjects"], future, mapping)
        classes = int(np.unique(train_y).size)
        weighted = False
    else:
        task_name = "ERP" if task == "OpenBMI_ERP" else "SSVEP"
        search, _, reference, _ = task_code.split_reference()
        fold = next(value for value in reference["folds"] if int(value["fold_id"]) == int(fold_id))
        spec = task_code.TASKS[task_name]; root = Path(os.environ["FULL_OPENBMI_CACHE"]).resolve()
        train_x, train_y, train_s, mapping = _extract_openbmi(inner, root, fold["inner_train_subjects"], (spec["source_session"],), spec["cache_name"])
        val_x, val_y, val_s, _ = _extract_openbmi(inner, root, fold["inner_val_subjects"], (spec["future_session"],), spec["cache_name"], mapping)
        outer_x, outer_y, outer_s, _ = _extract_openbmi(inner, root, fold["outer_dev_subjects"], (spec["future_session"],), spec["cache_name"], mapping)
        split_hash, classes, weighted = reference["source_sha256"], int(spec["classes"]), bool(spec["weighted_ce"])
    train_x, values, norm = _normalise(train_x, val_x, outer_x)
    val_x, outer_x = values
    if any(set(np.unique(labels)) != set(range(classes)) for labels in (train_y, val_y, outer_y)):
        raise RuntimeError(f"non-contiguous label mapping for {task}/fold{fold_id}")
    if set(fold["inner_train_subjects"]) & set(fold["inner_val_subjects"]) or set(fold["outer_dev_subjects"]) & (set(fold["inner_train_subjects"]) | set(fold["inner_val_subjects"])):
        raise RuntimeError("frozen SEARCH fold overlap")
    return {"task": task, "dataset": dataset if task in ("OpenBMI_MI", "WBCIC_MI") else "OpenBMI", "fold": fold,
            "split_sha256": split_hash, "classes": classes, "channels": int(train_x.shape[1]), "samples": int(train_x.shape[2]),
            "train_x": train_x, "train_y": train_y, "train_subjects": train_s,
            "val_x": val_x, "val_y": val_y, "val_subjects": val_s,
            "outer_x": outer_x, "outer_y": outer_y, "outer_subjects": outer_s,
            "normalizer": norm, "weighted_cross_entropy": weighted}
