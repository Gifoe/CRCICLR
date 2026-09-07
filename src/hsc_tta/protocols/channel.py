from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py


def _normalize(name: str) -> str:
    value = re.sub(r"\s+", "", name.upper()).replace("_", "-")
    value = re.sub(r"^EEG", "", value)
    return value.replace("M1", "A1").replace("M2", "A2")


TARGETS = {"C3": "C3-A2", "C4": "C4-A1"}


def scan_sleep_channel_availability(processed_root: str | Path) -> dict[str, dict[str, list[str]]]:
    """Read channel names only; labels and outcomes are never opened."""
    root = Path(processed_root)
    availability: dict[str, dict[str, list[str]]] = {}
    for dataset in ("hmc", "cap"):
        per_channel = {"C3": [], "C4": []}
        for path in sorted((root / dataset).glob("*.h5")):
            with h5py.File(path, "r") as handle:
                names = {
                    _normalize(value.decode() if isinstance(value, bytes) else str(value))
                    for value in handle["channel_names"][:]
                }
                metadata = json.loads(handle.attrs["metadata_json"])
                subject_id = str(metadata["subject_id"])
            for channel, target in TARGETS.items():
                if _normalize(target) in names:
                    per_channel[channel].append(subject_id)
        availability[dataset] = {
            channel: sorted(subjects) for channel, subjects in per_channel.items()
        }
    return availability


def choose_common_central_channel(
    availability: dict[str, dict[str, list[str]]]
) -> dict[str, object]:
    if set(availability) != {"hmc", "cap"}:
        raise ValueError("availability must contain exactly HMC and CAP")
    rows: dict[str, dict[str, int]] = {}
    for channel in ("C3", "C4"):
        hmc = len(set(availability["hmc"].get(channel, [])))
        cap = len(set(availability["cap"].get(channel, [])))
        rows[channel] = {
            "hmc_subjects": hmc,
            "cap_subjects": cap,
            "combined_subjects": hmc + cap,
            "minimum_site_subjects": min(hmc, cap),
        }
    viable = [channel for channel in ("C4", "C3") if rows[channel]["minimum_site_subjects"] > 0]
    if not viable:
        raise ValueError("no central derivation is available in both HMC and CAP")
    selected = max(
        viable,
        key=lambda channel: (
            rows[channel]["combined_subjects"],
            rows[channel]["minimum_site_subjects"],
            1 if channel == "C4" else 0,
        ),
    )
    payload: dict[str, object] = {
        "protocol_version": "hmc-cap-common-single-central-v1",
        "selection_basis": "channel_availability_only",
        "tie_break": "C4_then_C3",
        "selected_channel": selected,
        "normalized_derivation": TARGETS[selected],
        "counts": rows,
        "hmc_subject_ids": availability["hmc"][selected],
        "cap_subject_ids": availability["cap"][selected],
        "prohibitions": [
            "no channel duplication",
            "no performance-based channel selection",
            "same single-channel protocol for HMC internal and HMC-to-CAP experiments",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload
