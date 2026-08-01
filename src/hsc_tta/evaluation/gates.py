from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hsc_tta.freeze import file_sha256, verify_freeze_manifest
from hsc_tta.schemas import PreOutcomeDecisionRow


JOIN_KEY = ["dataset", "seed", "episode_id", "subject_id", "alpha"]
FORBIDDEN_CONTEXT_FIELDS = {
    "true_future_risk",
    "future_risk",
    "future_average_set_size",
    "future_singleton_rate",
    "macro_f1",
    "balanced_accuracy",
    "harmful_adaptation",
    "selected_error",
    "no_tta_error",
}


def _reject_future(frame: pd.DataFrame) -> None:
    forbidden = sorted(
        column
        for column in frame.columns
        if column.startswith("future_") or column in FORBIDDEN_CONTEXT_FIELDS
    )
    if forbidden:
        raise ValueError(f"U_s-only table contains future outcome fields: {forbidden}")


def write_pre_outcome_decisions(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    freeze_manifest_path: str | Path,
    lock_path: str | Path,
) -> dict[str, str]:
    manifest = verify_freeze_manifest(freeze_manifest_path)
    _reject_future(frame)
    if frame.duplicated(JOIN_KEY).any():
        raise ValueError("duplicate pre-outcome decision keys")
    for row in frame.to_dict(orient="records"):
        PreOutcomeDecisionRow.model_validate(row)
    if set(frame["freeze_hash"].astype(str)) != {str(manifest["manifest_hash"])}:
        raise ValueError("decision rows do not reference the active freeze manifest hash")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    lock = {
        "freeze_manifest_hash": str(manifest["manifest_hash"]),
        "decision_sha256": file_sha256(output),
    }
    target_lock = Path(lock_path)
    target_lock.parent.mkdir(parents=True, exist_ok=True)
    temporary_lock = target_lock.with_suffix(target_lock.suffix + ".tmp")
    temporary_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_lock.replace(target_lock)
    return lock


def verify_final_test_gate(
    freeze_manifest_path: str | Path,
    decision_path: str | Path,
    decision_lock_path: str | Path,
) -> None:
    manifest = verify_freeze_manifest(freeze_manifest_path)
    decision = Path(decision_path)
    lock_path = Path(decision_lock_path)
    if not decision.is_file() or not lock_path.is_file():
        raise RuntimeError("final-test blocked: pre-outcome decisions are not frozen")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("freeze_manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("final-test blocked: decision/config freeze hashes differ")
    if lock.get("decision_sha256") != file_sha256(decision):
        raise RuntimeError("final-test blocked: pre-outcome decision hash mismatch")


def join_decisions_and_outcomes(decisions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if decisions.duplicated(JOIN_KEY).any() or outcomes.duplicated(JOIN_KEY).any():
        raise ValueError("decision/outcome tables must be one-to-one on the full key")
    _reject_future(decisions)
    joined = decisions.merge(outcomes, on=JOIN_KEY, how="inner", validate="one_to_one", suffixes=("", "_outcome"))
    if len(joined) != len(decisions) or len(joined) != len(outcomes):
        raise ValueError("decision/outcome keys do not match exactly")
    return joined
