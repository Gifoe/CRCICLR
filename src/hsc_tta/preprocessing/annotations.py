from __future__ import annotations

import re

SLEEP_LABEL_MAP = {
    "W": 0, "WAKE": 0, "SLEEP STAGE W": 0,
    "N1": 1, "S1": 1, "1": 1, "SLEEP STAGE 1": 1, "SLEEP STAGE N1": 1,
    "N2": 2, "S2": 2, "2": 2, "SLEEP STAGE 2": 2, "SLEEP STAGE N2": 2,
    "N3": 3, "S3": 3, "S4": 3, "3": 3, "4": 3, "SLEEP STAGE 3": 3, "SLEEP STAGE 4": 3, "SLEEP STAGE N3": 3,
    "R": 4, "REM": 4, "SLEEP STAGE R": 4,
}
INVALID_SLEEP = {"?", "UNKNOWN", "UNSCORED", "MOVEMENT", "M", "SLEEP STAGE ?"}


def map_sleep_label(raw_label: object, dataset: str) -> int | None:
    if dataset.lower() not in {"hmc", "cap"}:
        raise ValueError("dataset must be hmc or cap")
    key = re.sub(r"\s+", " ", str(raw_label).strip().upper())
    if key in INVALID_SLEEP or not key:
        return None
    return SLEEP_LABEL_MAP.get(key)


def map_mi_event(run_id: int, annotation: str) -> int | None:
    label = annotation.strip().upper()
    if label == "T0":
        return None
    if run_id in {4, 8, 12}:
        return {"T1": 0, "T2": 1}.get(label)
    if run_id in {6, 10, 14}:
        return {"T1": 2, "T2": 3}.get(label)
    raise ValueError(f"run {run_id} is not a configured motor-imagery run")

