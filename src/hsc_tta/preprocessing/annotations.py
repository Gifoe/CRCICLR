from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
import csv

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


def load_sleep_annotations(signal_path: str | Path, dataset: str, recording_start: datetime | None) -> list[dict[str, object]]:
    path = Path(signal_path)
    if dataset == "hmc":
        import mne
        sidecar = path.with_name(path.stem + "_sleepscoring.edf")
        annotations = mne.read_annotations(sidecar)
        return [{"onset": float(a["onset"]), "duration": float(a["duration"]), "description": str(a["description"])} for a in annotations]
    if dataset != "cap": raise ValueError("dataset must be hmc or cap")
    if recording_start is None: raise ValueError("CAP alignment requires EDF recording start time")
    sidecar = path.with_suffix(".txt")
    lines = sidecar.read_text(encoding="latin-1").splitlines()
    header_index = next((i for i,line in enumerate(lines) if line.startswith("Sleep Stage\t")), None)
    if header_index is None: raise ValueError(f"CAP sleep-stage table missing: {sidecar}")
    reader = csv.DictReader(lines[header_index:], delimiter="\t")
    output=[]
    for row in reader:
        event=(row.get("Event") or "").strip()
        if not event.startswith("SLEEP-"): continue
        # CAP exports are inconsistent across records: older files use
        # ``HH.MM.SS`` while most current files use ``HH:MM:SS``.
        clock_text = row["Time [hh:mm:ss]"].strip().replace(".", ":")
        clock=datetime.strptime(clock_text, "%H:%M:%S").time()
        timestamp=datetime.combine(recording_start.date(),clock,tzinfo=recording_start.tzinfo)
        if timestamp < recording_start: timestamp += timedelta(days=1)
        output.append({"onset": (timestamp-recording_start).total_seconds(), "duration": float(row.get("Duration[s]") or 30), "description": (row.get("Sleep Stage") or event.removeprefix("SLEEP-S"))})
    return output
