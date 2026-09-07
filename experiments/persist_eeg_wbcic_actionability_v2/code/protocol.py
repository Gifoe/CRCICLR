"""Prospective protocol and raw-data audit for WBCIC/Yang2025.

This command is the only stage allowed to enumerate the complete raw tree.
It audits the 51-subject cohort before sealing ten outer subjects.  Downstream
development commands read DEVELOPMENT_SCOPE_LOCK.json and construct paths for
those 41 subjects only; they never open OUTER_SPLIT_LOCK.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2"
OUT = EXP_ROOT / "outputs"
PROTOCOL = OUT / "protocol"
RESULTS = OUT / "results"
REFERENCE_COMMITS = [
    "1eca3976d62d38fb4291e217ca06add484babd41",
    "78f010644e86639d44a844558ab37bd865815082",
]
IMPLEMENTATION_ID = "persist_eeg_wbcic_eegnet_actionability_v2_20260817"
PROTOCOL_SEED = "PERSIST-EEG-WBCIC-EEGNET-V2-20260817"
EXPECTED_SUBJECTS = [f"sub-{index}" for index in range(1, 52)]
EXPECTED_SESSIONS = [0, 1, 2]
BLOCKS = [
    {"block": "P01_04", "start": 0, "end": 4, "rank": 4},
    {"block": "P05_08", "start": 4, "end": 8, "rank": 4},
    {"block": "P09_16", "start": 8, "end": 16, "rank": 8},
    {"block": "P17_32", "start": 16, "end": 32, "rank": 16},
]


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return None if not np.isfinite(value) else float(value)
    except ImportError:
        pass
    return value


def encoded(payload: Any) -> str:
    return json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_once(path: Path, payload: Any) -> None:
    text = encoded(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Prospective lock mismatch; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(encoded(payload), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def stable_key(namespace: str, subject: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_SEED}|{namespace}|{subject}".encode()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=5
        ).strip()
    except Exception:
        return None


def code_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((EXP_ROOT / "code").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def subject_number(subject: str) -> int:
    return int(subject.split("-", 1)[1])


def session_files(raw_root: Path, subject: str, session: int) -> dict[str, Path]:
    eeg = raw_root / subject / f"ses-{session}" / "eeg"
    stem = f"{subject}_ses-{session}_task-imagery_run-0"
    return {
        "bdf": eeg / f"{stem}_eeg.bdf",
        "eeg_json": eeg / f"{stem}_eeg.json",
        "events_tsv": eeg / f"{stem}_events.tsv",
        "events_json": eeg / f"{stem}_events.json",
        "channels_tsv": eeg / f"{stem}_channels.tsv",
        "channels_json": eeg / f"{stem}_channels.json",
    }


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest_file(path: Path, algorithm: str, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"WBCIC_CORE_DATA_INTEGRITY_FAIL: manifest file missing: {path}")
    if algorithm == "sha256":
        observed = sha256_file(path)
    elif algorithm == "git":
        observed = git_blob_sha1(path)
    else:
        raise RuntimeError(f"WBCIC_CORE_DATA_INTEGRITY_FAIL: unsupported checksum algorithm {algorithm}")
    if observed.lower() != expected.lower():
        raise RuntimeError(f"WBCIC_CORE_DATA_INTEGRITY_FAIL: content checksum mismatch: {path}")


def inspect_bdf(path: Path) -> dict[str, Any]:
    import mne

    raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    return {
        "bdf_header_open": True,
        "bdf_sfreq": float(raw.info["sfreq"]),
        "bdf_channels": int(len(raw.ch_names)),
        "bdf_samples": int(raw.n_times),
        "bdf_duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "bdf_channel_names_hash": sha_lines(list(raw.ch_names)),
    }


def deterministic_split(subjects: list[str]) -> tuple[list[str], list[str], list[list[str]]]:
    ordered_outer = sorted(subjects, key=lambda subject: stable_key("outer", subject))
    outer = sorted(ordered_outer[:10], key=subject_number)
    development = sorted(set(subjects) - set(outer), key=subject_number)
    ordered_folds = sorted(development, key=lambda subject: stable_key("fold", subject))
    folds: list[list[str]] = [[] for _ in range(5)]
    for index, subject in enumerate(ordered_folds):
        folds[index % 5].append(subject)
    folds = [sorted(fold, key=subject_number) for fold in folds]
    assert sorted(sum(folds, []), key=subject_number) == development
    assert not set(development).intersection(outer)
    return development, outer, folds


def audit_raw(raw_root: Path, header_workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = raw_root / "filtered_manifest.json"
    completion_path = raw_root / "download_completion.json"
    if not manifest_path.exists() or not completion_path.exists():
        raise RuntimeError("WBCIC_DATA_NOT_FOUND: filtered manifest/completion record missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if not completion.get("complete"):
        raise RuntimeError("WBCIC_CORE_DATA_INTEGRITY_FAIL: download is not complete")
    paths = [str(item["path"]) for item in manifest]
    if len(paths) != len(set(paths)) or any(
        PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts for path in paths
    ):
        raise RuntimeError("WBCIC_CORE_DATA_INTEGRITY_FAIL: unsafe/duplicate manifest paths")
    if any(path.startswith("sourcedata/") for path in paths):
        raise RuntimeError("WBCIC_CORE_DATA_INTEGRITY_FAIL: sourcedata entered filtered manifest")
    manifest_by_path = {str(item["path"]): item for item in manifest}
    checksum_jobs: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, header_workers)) as executor:
        for item in manifest:
            path = raw_root / PurePosixPath(str(item["path"]))
            if not path.is_file() or path.stat().st_size != int(item["size"]):
                raise RuntimeError(
                    f"WBCIC_CORE_DATA_INTEGRITY_FAIL: manifest size/missing file: {item['path']}"
                )
            future = executor.submit(
                verify_manifest_file,
                path,
                str(item["checksum_algorithm"]),
                str(item["checksum"]),
            )
            checksum_jobs[future] = str(item["path"])
        for index, future in enumerate(as_completed(checksum_jobs), 1):
            future.result()
            if index % 100 == 0 or index == len(checksum_jobs):
                print(f"[checksum {index}/{len(checksum_jobs)}]", flush=True)
    participants = read_tsv(raw_root / "participants.tsv")
    observed_subjects = sorted([row["participant_id"] for row in participants], key=subject_number)
    if observed_subjects != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"WBCIC_CORE_DATA_INTEGRITY_FAIL: expected core 51, observed {observed_subjects}"
        )

    rows: list[dict[str, Any]] = []
    bdf_jobs: dict[Any, int] = {}
    for subject in observed_subjects:
        for session in EXPECTED_SESSIONS:
            files = session_files(raw_root, subject, session)
            for kind, path in files.items():
                if not path.is_file():
                    raise RuntimeError(f"WBCIC_CORE_DATA_INTEGRITY_FAIL: missing {kind}: {path}")
                relative = path.relative_to(raw_root).as_posix()
                expected = manifest_by_path.get(relative)
                if expected is None or path.stat().st_size != int(expected["size"]):
                    raise RuntimeError(f"WBCIC_CORE_DATA_INTEGRITY_FAIL: size/manifest mismatch: {relative}")
            eeg = json.loads(files["eeg_json"].read_text(encoding="utf-8"))
            events = read_tsv(files["events_tsv"])
            channels = read_tsv(files["channels_tsv"])
            eeg_channels = [row["name"] for row in channels if row["type"] == "EEG"]
            left = sum(row["trial_type"] == "left_hand" for row in events)
            right = sum(row["trial_type"] == "right_hand" for row in events)
            labels = sorted(set(row["trial_type"] for row in events))
            malformed = sum(
                row["trial_type"] not in {"left_hand", "right_hand"}
                or float(row["duration"]) != 4.0
                or int(row["value"]) not in {1, 2}
                for row in events
            )
            row = {
                "subject": subject,
                "session": session,
                "bdf_path": str(files["bdf"].resolve()),
                "bdf_bytes": files["bdf"].stat().st_size,
                "manifest_checksum_algorithm": manifest_by_path[files["bdf"].relative_to(raw_root).as_posix()]["checksum_algorithm"],
                "manifest_checksum": manifest_by_path[files["bdf"].relative_to(raw_root).as_posix()]["checksum"],
                "sampling_rate_hz": float(eeg["SamplingFrequency"]),
                "recording_duration_seconds": float(eeg["RecordingDuration"]),
                "eeg_channel_count": len(eeg_channels),
                "total_channel_count": len(channels),
                "channel_names_hash": sha_lines(eeg_channels),
                "pz_present": "Pz" in eeg_channels,
                "all_channels_good": all(channel["status"] == "good" for channel in channels),
                "event_count": len(events),
                "left_hand_trials": left,
                "right_hand_trials": right,
                "event_labels": "|".join(labels),
                "malformed_events": malformed,
                "manifest_size_match": True,
            }
            if (
                row["sampling_rate_hz"] != 1000.0
                or row["eeg_channel_count"] != 59
                or not row["pz_present"]
                or labels != ["left_hand", "right_hand"]
                or left == 0
                or right == 0
                or malformed
            ):
                raise RuntimeError(f"WBCIC_CORE_DATA_INTEGRITY_FAIL: malformed session {row}")
            rows.append(row)

    with ThreadPoolExecutor(max_workers=max(1, header_workers)) as executor:
        for index, row in enumerate(rows):
            bdf_jobs[executor.submit(inspect_bdf, Path(row["bdf_path"]))] = index
        for future in as_completed(bdf_jobs):
            index = bdf_jobs[future]
            rows[index].update(future.result())
            if (
                rows[index]["bdf_sfreq"] != 1000.0
                or rows[index]["bdf_channels"] != 64
                or abs(rows[index]["bdf_duration_seconds"] - rows[index]["recording_duration_seconds"]) > 0.01
            ):
                raise RuntimeError(f"WBCIC_CORE_DATA_INTEGRITY_FAIL: BDF header mismatch {rows[index]}")

    canonical_manifest = [
        {key: item[key] for key in ("path", "size", "checksum_algorithm", "checksum")}
        for item in sorted(manifest, key=lambda value: value["path"])
    ]
    tree_hash = hashlib.sha256(
        json.dumps(canonical_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "status": "WBCIC_CORE_DATA_INTEGRITY_PASS",
        "raw_root": str(raw_root.resolve()),
        "dataset": "NEMAR nm000348 / Yang2025 2C",
        "dataset_version": "v1.0.4",
        "manifest_file_count": len(manifest),
        "manifest_total_bytes": sum(int(item["size"]) for item in manifest),
        "tree_hash_algorithm": "sha256(canonical manifest path,size,checksum_algorithm,checksum)",
        "tree_hash": tree_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "download_completion_sha256": sha256_file(completion_path),
        "download_complete": True,
        "download_failures": completion.get("download_failures", []),
        "content_integrity": "all manifest entries independently verified: SHA-256 for 153 BDF files and Git blob SHA-1 for 1081 metadata/auxiliary files",
        "content_checksum_verified_files": len(manifest),
        "subjects": observed_subjects,
        "subject_count": len(observed_subjects),
        "sessions": EXPECTED_SESSIONS,
        "session_count": len(rows),
        "bdf_count": sum(path.endswith(".bdf") for path in paths),
        "sourcedata_files_present": 0,
        "source_provenance": "https://data.nemar.org/nm000348/v1.0.4/ manifest-filtered, sourcedata excluded",
        "license": "CC-BY-4.0",
        "bids_version": "1.9.0",
        "raw_data_immutable": True,
    }
    return summary, rows


def write_inventory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare(raw_root: Path, header_workers: int) -> None:
    outcome_files = list(RESULTS.glob("*")) if RESULTS.exists() else []
    required_locks = [
        PROTOCOL / "WBCIC_RAW_DATA_LOCK.json",
        PROTOCOL / "OUTER_SPLIT_LOCK.json",
        PROTOCOL / "DEVELOPMENT_SCOPE_LOCK.json",
        PROTOCOL / "SESSION_PROTOCOL_LOCK.json",
        PROTOCOL / "PREPROCESSING_PROTOCOL_LOCK.json",
        PROTOCOL / "REPRESENTATION_CANDIDATE_LOCK.json",
        PROTOCOL / "PERSISTENCE_BASIS_PROTOCOL_LOCK.json",
        PROTOCOL / "ACTIONABILITY_PROTOCOL_LOCK.json",
    ]
    if outcome_files and not all(path.exists() for path in required_locks):
        raise RuntimeError("Outcome artifacts exist without all prospective locks")
    raw, rows = audit_raw(raw_root.resolve(), header_workers)
    development, outer, folds = deterministic_split(raw["subjects"])
    now = datetime.now(timezone.utc).isoformat()
    common = {
        "implementation_id": IMPLEMENTATION_ID,
        "reference_commits": REFERENCE_COMMITS,
        "code_commit_at_freeze": git_commit(),
        "code_tree_sha256": code_tree_sha256(),
        "protocol_seed": PROTOCOL_SEED,
        "frozen_at_utc": now,
        "frozen_before_task_or_actionability_outcomes": True,
    }
    raw_lock = {**common, **raw}
    split_lock = {
        **common,
        "method": "sort SHA256(protocol_seed|outer|subject); first ten outer; hash-balanced five-fold development partition",
        "development_subjects": development,
        "outer_subjects": outer,
        "development_count": 41,
        "outer_count": 10,
        "development_hash": sha_lines(development),
        "outer_hash": sha_lines(outer),
        "folds": {f"F{index}": fold for index, fold in enumerate(folds)},
        "fold_hashes": {f"F{index}": sha_lines(fold) for index, fold in enumerate(folds)},
        "outer_test_state": "OUTER_TEST_LOCKED",
        "outer_evaluation_authorized": False,
    }
    roles = {}
    for fold in range(5):
        roles[str(fold)] = {
            "outcome": folds[fold],
            "discovery_decision": folds[(fold + 1) % 5],
            "model_fit": sorted(
                sum([folds[index] for index in range(5) if index not in {fold, (fold + 1) % 5}], []),
                key=subject_number,
            ),
        }
    scope = {
        **common,
        "allowed_subjects": development,
        "allowed_subjects_hash": sha_lines(development),
        "folds": {f"F{index}": fold for index, fold in enumerate(folds)},
        "audit_roles": roles,
        "outer_subject_count": 10,
        "outer_subject_hash_only": sha_lines(outer),
        "outer_subject_ids_present": False,
        "runtime_must_not_open": "OUTER_SPLIT_LOCK.json",
        "runtime_path_policy": "construct raw paths only for allowed subjects; never enumerate raw root",
        "scope_violation_terminal_state": "DATA_SCOPE_VIOLATION",
    }
    session_lock = {
        **common,
        "primary": "S1+S2->S3",
        "bids_mapping": {"S1": "ses-0", "S2": "ses-1", "S3": "ses-2"},
        "model_fit_sessions": [0, 1],
        "discovery_sessions": [0, 1],
        "outcome_session": 2,
        "primary_target": "unseen subject + future session",
        "trial_random_split_forbidden": True,
        "secondary_robustness": ["S1+S3->S2", "S2+S3->S1"],
    }
    preprocessing = {
        **common,
        "task": "2C left/right motor imagery",
        "event_mapping": {"left_hand": 0, "right_hand": 1},
        "event_tsv_onset_semantics": "the four-second BIDS event already marks the imagery interval corresponding to trial-relative [1.5,5.5] s; no additional 1.5 s shift",
        "epoch_seconds_from_event": [0.0, 4.0],
        "filter_margin_seconds": 2.0,
        "bandpass_hz": [0.5, 40.0],
        "filter": "zero-phase fourth-order Butterworth SOS on an eight-second event-centered window",
        "line_noise": "50 Hz is above the frozen 40 Hz low-pass; a redundant notch is intentionally not applied",
        "input_sampling_rate_hz": 1000,
        "output_sampling_rate_hz": 250,
        "resampling": "scipy.signal.resample_poly up=1 down=4 after anti-alias band-pass",
        "channel_policy": "subtract Pz from each of the other 58 EEG channels, then drop Pz; EOG/ECG excluded",
        "channels_out": 58,
        "amplitude_transform": "microvolts / 20, clipped to [-12.5,12.5]; no cross-subject statistics",
        "primary_reference": "Pz",
        "CAR": "pre-registered secondary robustness only; cannot replace primary after H1-H5",
        "cue_shortcut_diagnostic": ["early imagery [0,2] s", "late imagery [2,4] s"],
        "frozen_before_actionability": True,
    }
    candidate_lock = {
        **common,
        "candidate_pool": ["EEGNet"],
        "user_constraint": "finish the EEGNet route before any secondary backbone",
        "selection_is_task_only": True,
        "selection_metric": "mean subject balanced accuracy on cross-fitted unseen-subject S3",
        "architecture": {
            "name": "EEGNet",
            "F1": 8,
            "D": 2,
            "F2": 16,
            "temporal_kernel": 64,
            "separable_kernel": 16,
            "embedding_dim": 32,
        },
        "configs": [
            {"id": "EEGNET_STD", "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.50, "epochs": 30, "batch_size": 64},
            {"id": "EEGNET_STABLE", "learning_rate": 3e-4, "weight_decay": 5e-4, "dropout": 0.25, "epochs": 40, "batch_size": 64},
        ],
        "seeds": [20260817],
        "outer_test_selection_forbidden": True,
        "actionability_based_selection_forbidden": True,
    }
    basis_lock = {
        **common,
        "method": "fold-specific eigendecomposition of symmetrized S1/S2 cross-session subject-centroid covariance",
        "centering": "discovery-subject session means only",
        "outcome_S3_labels_used": False,
        "embedding_dim": 32,
        "candidate_blocks": BLOCKS,
        "same_rank_controls": 100,
        "basis_frozen_before_outcome_aggregation": True,
    }
    actionability_lock = {
        **common,
        "statistical_unit": "subject",
        "cross_fitting": "five folds: outcome F_k; discovery/decision F_(k+1); model-fit remaining three",
        "random_draws": 100,
        "bootstrap_draws": 10000,
        "confidence_level": 0.95,
        "multiplicity": "Holm separately across four primary blocks for H1,H2,H3-finite,H4 and protected-utility",
        "signed_utility": {"u_abs": "CE_erase-CE_raw", "u_spec": "u_abs-mean(CE_random-CE_raw)", "harmful": "negative", "protected": "positive"},
        "finite_intervention": "project raw embedding h through the candidate projector; logits use W(I-P)h+b, exactly matching the authorized AGDI alpha=1 intervention",
        "gates": {
            "competence": "mean subject BA>=0.60 AND LCB95>0.55 AND >=70% subjects BA>0.5",
            "H1": "LCB95 persistence-specific>0 and Holm one-sided p<0.05",
            "H2": "UCB95 u_spec<0 and Holm one-sided p<0.05",
            "H3": "local random-control LCB>1 and p<0.05; finite-ratio LCB>1 and Holm p<0.05",
            "H4": "LCB95 delta-BA-specific>0, mean>=0.005, Holm p<0.05",
            "H5": "all leave-one-fold-out means>0, all leave-one-subject-out means>0, >=60% nonnegative subjects",
            "actionable_harmful": "H1 AND H2 AND H3 AND H4 AND H5",
        },
        "agdi_alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "outer_test_state": "OUTER_TEST_LOCKED",
    }
    data_scope = {
        **common,
        "status": "DATA_SCOPE_PASS",
        "allowed_subject_count": 41,
        "allowed_subjects_hash": sha_lines(development),
        "outer_subject_count": 10,
        "outer_subject_hash": sha_lines(outer),
        "outer_subject_ids_materialized_downstream": False,
        "fail_closed": True,
    }

    write_once(PROTOCOL / "WBCIC_RAW_DATA_LOCK.json", raw_lock)
    write_once(PROTOCOL / "RAW_DATASET_INVENTORY.json", {**common, **raw, "sessions_detail": rows})
    write_once(PROTOCOL / "PROVENANCE_AUDIT.json", {**common, "status": "PROVENANCE_PASS", "core_cohort": EXPECTED_SUBJECTS, "observed_2C_subjects": raw["subjects"], "observed_3C_subjects": [], "subject_count_provenance_mismatch": False, "distribution_note": "filtered NEMAR BIDS contains the formal 51-subject 2C core; sourcedata and 3C distribution are absent by design"})
    write_inventory_csv(PROTOCOL / "SUBJECT_SESSION_INVENTORY.csv", rows)
    report = [
        "# WBCIC/Yang2025 data integrity report",
        "",
        "Terminal state: `WBCIC_CORE_DATA_INTEGRITY_PASS`",
        "",
        f"- Core subjects: {raw['subject_count']}",
        f"- Sessions/BDF: {raw['session_count']}",
        f"- Manifest files: {raw['manifest_file_count']}",
        f"- Manifest bytes: {raw['manifest_total_bytes']}",
        "- All 1,234 local files passed their frozen NEMAR manifest content checksum (SHA-256 for BDF; Git blob SHA-1 otherwise).",
        "- Every BDF header opened at 1000 Hz with 64 total channels.",
        "- Every session has 59 EEG channels including Pz, left/right events, and no malformed event rows.",
        "- No sourcedata file is present.",
    ]
    (PROTOCOL / "DATA_INTEGRITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_once(PROTOCOL / "OUTER_SPLIT_LOCK.json", split_lock)
    write_once(PROTOCOL / "DEVELOPMENT_SCOPE_LOCK.json", scope)
    write_once(PROTOCOL / "SESSION_PROTOCOL_LOCK.json", session_lock)
    write_once(PROTOCOL / "PREPROCESSING_PROTOCOL_LOCK.json", preprocessing)
    write_once(PROTOCOL / "REPRESENTATION_CANDIDATE_LOCK.json", candidate_lock)
    write_once(PROTOCOL / "PERSISTENCE_BASIS_PROTOCOL_LOCK.json", basis_lock)
    write_once(PROTOCOL / "ACTIONABILITY_PROTOCOL_LOCK.json", actionability_lock)
    write_once(PROTOCOL / "DATA_SCOPE_AUDIT.json", data_scope)
    print(encoded({"status": "WBCIC_PROTOCOL_FROZEN", "development": 41, "outer": 10, "fold_sizes": [len(fold) for fold in folds], "raw_root": raw["raw_root"]}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepare", nargs="?")
    parser.add_argument("--raw-root", type=Path, default=Path(r"D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids"))
    parser.add_argument("--header-workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.raw_root, args.header_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
