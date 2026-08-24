"""Strict evaluation-only source-model diagnostic for frozen PERSIST-Net.

This program never trains or adapts a model.  It first audits the exact final-v1
artifacts, then replays B0/B1, and only after a persisted replay PASS evaluates
the already-saved dual-path source checkpoints on outcome Session 2.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
BASE_EXPERIMENT = REPO / "experiments" / "persist_eeg_persist_net_final_v1"
BASE_CODE = BASE_EXPERIMENT / "code"
if str(BASE_CODE) not in sys.path:
    sys.path.insert(0, str(BASE_CODE))
import core  # noqa: E402


DEFAULT_SOURCE_EXPERIMENT = Path(
    os.environ.get(
        "PERSIST_FINAL_V1_EXPERIMENT",
        str(BASE_EXPERIMENT),
    )
)
RESULTS = EXPERIMENT / "results"
RUNTIME = EXPERIMENT / "runtime"
AUDIT_JSON = RUNTIME / "INPUT_AUDIT.json"
REPLAY_JSON = RUNTIME / "REPLAY_PASS.json"

CHECKPOINT_KEYS = (
    "B0_VANILLA_EEGNET",
    "B1_STRONG_EEGNET",
    "A2_DUAL_CONTROL",
    "PUD_SOURCE",
    "A7_IDENTITY_PROTECTED",
    "A8_RANDOM_PROTECTED",
)
DUAL_OUTPUT_NAMES = {
    "A2_DUAL_CONTROL": "A2_SOURCE_ONLY",
    "PUD_SOURCE": "PUD_SOURCE_ONLY",
    "A7_IDENTITY_PROTECTED": "IDENTITY_SOURCE_ONLY",
    "A8_RANDOM_PROTECTED": "RANDOM_SOURCE_ONLY",
}
TOL = 1e-12


class DiagnosticBlocked(RuntimeError):
    pass


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    """Hash text semantics while neutralizing Git's Windows CRLF checkout."""
    value = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sorted_subjects(values: Iterable[str]) -> list[str]:
    return sorted(map(str, values), key=lambda value: int(value) if value.isdigit() else value)


def static_evaluation_only_audit() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    called: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.append(node.func.attr)
    forbidden = {
        "adapt_dual",
        "adapt_single",
        "train_dual",
        "train_single",
        "fit_certificate",
        "backward",
        "step",
    }
    observed = sorted(forbidden.intersection(called))
    return {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "forbidden_training_or_adaptation_calls": observed,
        "pass": not observed,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_path(entry: Mapping[str, Any]) -> Path:
    return Path(str(entry["path"]))


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def audit_inputs(source_experiment: Path) -> dict[str, Any]:
    ensure_dirs()
    source_experiment = source_experiment.resolve()
    static_audit = static_evaluation_only_audit()
    issues: list[str] = []
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []

    if not static_audit["pass"]:
        issues.append(f"diagnostic script contains forbidden calls: {static_audit['forbidden_training_or_adaptation_calls']}")

    source_code = source_experiment / "code"
    implementation_files = {
        "core.py": source_code / "core.py",
        "run_experiment.py": source_code / "run_experiment.py",
        "PROTOCOL_FROZEN.json": source_experiment / "PROTOCOL_FROZEN.json",
    }
    implementation_hashes: dict[str, str] = {}
    loader_normalized_hashes: dict[str, str] = {}
    for name, path in implementation_files.items():
        if not path.is_file():
            missing.append(str(path))
        else:
            implementation_hashes[name] = sha256_file(path)
            loader_path = BASE_EXPERIMENT / ("code" if name.endswith(".py") else "") / name
            if not loader_path.is_file():
                missing.append(str(loader_path))
            elif normalized_text_sha256(loader_path) != normalized_text_sha256(path):
                issues.append(f"diagnostic loader differs from frozen source implementation: {name}")
            else:
                loader_normalized_hashes[name] = normalized_text_sha256(loader_path)

    for fold in range(5):
        for seed in range(3):
            run_dir = source_experiment / "runtime" / "runs" / f"fold-{fold}" / f"seed-{seed}"
            lock_path = run_dir / "RUN_LOCK.json"
            subject_path = run_dir / "SUBJECT_RESULTS.csv"
            if not lock_path.is_file():
                missing.append(str(lock_path))
                continue
            if not subject_path.is_file():
                missing.append(str(subject_path))
                continue
            lock = load_json(lock_path)
            if int(lock.get("fold", -1)) != fold or int(lock.get("seed", -1)) != seed:
                issues.append(f"fold-{fold}/seed-{seed}: lock identity mismatch")
            if lock.get("internal_holdout_accessed") is not False:
                issues.append(f"fold-{fold}/seed-{seed}: internal holdout flag is not false")
            if lock.get("outer_test_used") is not False:
                issues.append(f"fold-{fold}/seed-{seed}: outer-test flag is not false")
            if lock.get("target_future_labels_used") is not False:
                issues.append(f"fold-{fold}/seed-{seed}: future-label flag is not false")
            for name, observed_hash in implementation_hashes.items():
                expected_hash = str(lock.get("implementation_sha256", {}).get(name, ""))
                if observed_hash != expected_hash:
                    issues.append(f"fold-{fold}/seed-{seed}: implementation hash mismatch for {name}")

            normalizer = Path(str(lock.get("normalizer", "")))
            normalizer_exists = normalizer.is_file()
            normalizer_actual = sha256_file(normalizer) if normalizer_exists else ""
            normalizer_expected = str(lock.get("normalizer_sha256", ""))
            if not normalizer_exists:
                missing.append(str(normalizer))
            elif normalizer_actual != normalizer_expected:
                issues.append(f"fold-{fold}/seed-{seed}: normalizer hash mismatch")
            if normalizer_exists and not _within(normalizer, source_experiment / "runtime" / "cache"):
                issues.append(f"fold-{fold}/seed-{seed}: normalizer is outside authorized cache")

            hashes = lock.get("checkpoint_hashes", {})
            b0_entry = hashes.get("B0_VANILLA_EEGNET", {})
            b1_entry = hashes.get("B1_STRONG_EEGNET", {})
            b0_path = checkpoint_path(b0_entry) if b0_entry else Path()
            b1_path = checkpoint_path(b1_entry) if b1_entry else Path()
            b0_alias = bool(b0_entry and b1_entry and b0_path == b1_path)
            alias_valid = bool(
                b0_alias
                and lock.get("baseline_configuration", {}).get("id") == "EEGNET_F8"
                and int(lock.get("B0_seed", -1)) == int(lock.get("teacher_seed", -2))
                and int(lock.get("B0_epochs", -1)) == int(lock.get("baseline_epochs", -2))
                and b0_entry.get("sha256") == b1_entry.get("sha256")
                and int(b0_entry.get("parameters", -1)) == int(b1_entry.get("parameters", -2))
            )
            if b0_alias and not alias_valid:
                issues.append(f"fold-{fold}/seed-{seed}: ambiguous B0-to-B1 alias")

            for method in CHECKPOINT_KEYS:
                entry = hashes.get(method)
                if not isinstance(entry, dict):
                    missing.append(f"{lock_path}::{method}")
                    continue
                path = checkpoint_path(entry)
                exists = path.is_file()
                actual = sha256_file(path) if exists else ""
                expected = str(entry.get("sha256", ""))
                hash_match = bool(exists and actual == expected)
                in_scope = bool(exists and _within(path, run_dir / "checkpoints"))
                if not exists:
                    missing.append(str(path))
                elif not hash_match:
                    issues.append(f"fold-{fold}/seed-{seed}/{method}: checkpoint hash mismatch")
                if exists and not in_scope:
                    issues.append(f"fold-{fold}/seed-{seed}/{method}: checkpoint outside frozen run")
                named_b0_exists = (run_dir / "checkpoints" / "B0_VANILLA_EEGNET.pt").is_file()
                row = {
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "path": str(path),
                    "exists": exists,
                    "sha256_expected": expected,
                    "sha256_actual": actual,
                    "hash_match": hash_match,
                    "parameters": int(entry.get("parameters", -1)),
                    "within_frozen_run": in_scope,
                    "B0_aliases_B1": b0_alias if method == "B0_VANILLA_EEGNET" else False,
                    "B0_alias_unambiguous": alias_valid if method == "B0_VANILLA_EEGNET" else False,
                    "named_B0_file_exists": named_b0_exists if method == "B0_VANILLA_EEGNET" else None,
                }
                rows.append(row)
                fingerprints.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "sha256": actual,
                        "path": str(path),
                    }
                )

            certificate = run_dir / "certificate" / "PUD_CERTIFICATE.npz"
            certificate_audit = run_dir / "certificate" / "PUD_CERTIFICATION_AUDIT.json"
            if not certificate.is_file():
                missing.append(str(certificate))
            if not certificate_audit.is_file():
                missing.append(str(certificate_audit))
            run_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "baseline_configuration": lock.get("baseline_configuration", {}).get("id"),
                    "B0_aliases_B1": b0_alias,
                    "B0_alias_unambiguous": alias_valid if b0_alias else None,
                    "normalizer_sha256": normalizer_actual,
                    "source_subject_count": len(lock.get("source_subjects", [])),
                    "source_sessions": lock.get("source_training_sessions", []),
                    "future_session": lock.get("future_evaluation_session"),
                }
            )
            fingerprints.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "normalizer_sha256": normalizer_actual,
                    "lock_sha256": sha256_file(lock_path),
                    "subject_results_sha256": sha256_file(subject_path),
                    "certificate_sha256": sha256_file(certificate) if certificate.is_file() else "",
                }
            )

    cache = source_experiment / "runtime" / "cache"
    cache_paths = {
        "signals": cache / "OPENBMI_V8_SEARCH_MI_RAW.npy",
        "metadata": cache / "OPENBMI_V8_SEARCH_MI_METADATA.parquet",
        "purity_audit": source_experiment / "protocol" / "HOLDOUT_RUNTIME_AUDIT.json",
    }
    for name, path in cache_paths.items():
        if not path.is_file():
            missing.append(str(path))
        elif not _within(path, source_experiment):
            issues.append(f"{name}: path outside source experiment")
    purity = load_json(cache_paths["purity_audit"]) if cache_paths["purity_audit"].is_file() else {}
    cache_integrity: dict[str, Any] = {}
    if purity:
        if purity.get("all_checks_passed") is not True:
            issues.append("authoritative cache purity audit did not pass")
        if purity.get("materialized_subjects_intersect_holdout") is not False:
            issues.append("authoritative cache intersects internal holdout")
        if purity.get("holdout_eeg_materialized") is not False:
            issues.append("authoritative cache reports holdout EEG materialization")
        if purity.get("holdout_labels_materialized") is not False:
            issues.append("authoritative cache reports holdout label materialization")
        if purity.get("outer_test_used") is not False:
            issues.append("authoritative cache reports outer-test use")
        for label, path, expected_key in (
            ("signals", cache_paths["signals"], "cache_sha256"),
            ("metadata", cache_paths["metadata"], "metadata_sha256"),
        ):
            if path.is_file():
                actual_hash = sha256_file(path)
                expected_hash = str(purity.get(expected_key, ""))
                cache_integrity[label] = {
                    "path": str(path),
                    "sha256_expected": expected_hash,
                    "sha256_actual": actual_hash,
                    "hash_match": actual_hash == expected_hash,
                }
                fingerprints.append({"authorized_cache": label, "sha256": actual_hash})
                if actual_hash != expected_hash:
                    issues.append(f"authorized {label} cache hash mismatch")

    fingerprints.append(
        {
            "diagnostic_script_sha256": static_audit["script_sha256"],
            "evaluation_only_static_audit_pass": static_audit["pass"],
        }
    )

    input_fingerprint = canonical_sha256(sorted(fingerprints, key=lambda row: json.dumps(row, sort_keys=True)))
    payload = {
        "status": "AUDIT_PASS" if not missing and not issues else "AUDIT_FAIL",
        "pass": bool(not missing and not issues),
        "base_commit": "12ab811c2a6194192b430f9c010781acd1c0379f",
        "source_experiment": str(source_experiment),
        "static_evaluation_only_audit": static_audit,
        "implementation_hashes": implementation_hashes,
        "loader_normalized_text_hashes": loader_normalized_hashes,
        "runs": run_rows,
        "checkpoint_rows": len(rows),
        "B0_named_files_absent": int(
            sum(
                row["method"] == "B0_VANILLA_EEGNET" and not bool(row["named_B0_file_exists"])
                for row in rows
            )
        ),
        "B0_unambiguous_aliases": int(
            sum(
                row["method"] == "B0_VANILLA_EEGNET" and bool(row["B0_alias_unambiguous"])
                for row in rows
            )
        ),
        "cache_paths": cache_paths,
        "cache_integrity": cache_integrity,
        "cache_purity_audit": purity,
        "missing": sorted(set(missing)),
        "issues": sorted(set(issues)),
        "input_fingerprint": input_fingerprint,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
        "audited_at_unix": time.time(),
    }
    write_csv(RESULTS / "checkpoint_audit.csv", pd.DataFrame(rows))
    write_csv(RESULTS / "run_provenance.csv", pd.DataFrame(run_rows))
    write_json(AUDIT_JSON, payload)
    if missing:
        print("SOURCE_ONLY_DIAGNOSTIC_BLOCKED_BY_MISSING_CHECKPOINT", flush=True)
        raise DiagnosticBlocked("missing frozen source artifact(s): " + "; ".join(sorted(set(missing))))
    if issues:
        raise RuntimeError("input audit failed: " + "; ".join(sorted(set(issues))))
    print("CHECKPOINT_AND_CACHE_AUDIT_PASS", flush=True)
    return payload


def load_authorized_s2_data(source_experiment: Path) -> tuple[core.DevelopmentData, dict[str, Any]]:
    cache = source_experiment / "runtime" / "cache"
    x_path = cache / "OPENBMI_V8_SEARCH_MI_RAW.npy"
    metadata_path = cache / "OPENBMI_V8_SEARCH_MI_METADATA.parquet"
    base = pd.read_parquet(metadata_path, columns=["subject_id", "session_id"], engine="pyarrow")
    base["subject_id"] = base.subject_id.astype(str)
    base["session_id"] = base.session_id.astype(int)
    s2 = pd.read_parquet(
        metadata_path,
        columns=["subject_id", "session_id", "label"],
        filters=[("session_id", "==", 2)],
        engine="pyarrow",
    ).reset_index(drop=True)
    s2["subject_id"] = s2.subject_id.astype(str)
    s2["session_id"] = s2.session_id.astype(int)
    positions = np.flatnonzero(base.session_id.to_numpy() == 2)
    expected = base.iloc[positions].reset_index(drop=True)
    if len(s2) != len(positions) or not np.array_equal(s2.subject_id.to_numpy(), expected.subject_id.to_numpy()):
        raise RuntimeError("predicate-filtered Session-2 metadata does not preserve cache row order")
    if set(s2.label.astype(int)) != {0, 1}:
        raise RuntimeError("unexpected Session-2 labels")
    metadata = base.copy()
    metadata["label"] = -1
    metadata.loc[positions, "label"] = s2.label.to_numpy(dtype=np.int64)
    metadata["label"] = metadata.label.astype(int)
    subjects = tuple(sorted_subjects(base.subject_id.unique()))
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    cell_counts = s2.groupby(["subject_id", "label"]).size()
    valid = bool(
        len(base) == 8000
        and len(x) == 8000
        and tuple(x.shape[1:]) == (62, 1000)
        and len(subjects) == 40
        and set(base.session_id) == {1, 2}
        and len(s2) == 4000
        and set(cell_counts.astype(int)) == {50}
        and np.isfinite(np.asarray(x[positions[::97]], dtype=np.float32)).all()
    )
    if not valid:
        raise RuntimeError("authorized V8_SEARCH Session-2 cache validation failed")
    data = core.DevelopmentData(x=x, metadata=metadata, search_subjects=subjects, holdout_count=14)
    audit = {
        "signal_path": str(x_path),
        "metadata_path": str(metadata_path),
        "signal_shape": list(x.shape),
        "subjects": len(subjects),
        "sessions_in_identifier_metadata": sorted(base.session_id.unique().tolist()),
        "Session2_rows": len(s2),
        "Session2_per_subject_class_counts": sorted(set(map(int, cell_counts.tolist()))),
        "target_history_labels_materialized": False,
        "target_history_labels_used": False,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
        "pass": True,
    }
    return data, audit


def load_normalizer(lock: Mapping[str, Any], source_subjects: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(lock["normalizer"]))
    if sha256_file(path) != str(lock["normalizer_sha256"]):
        raise RuntimeError(f"normalizer changed: {path}")
    values = np.load(path, allow_pickle=False)
    observed_subjects = sorted_subjects(values["subjects"].astype(str).tolist())
    expected_subjects = sorted_subjects(source_subjects)
    if observed_subjects != expected_subjects or values["sessions"].astype(int).tolist() != [1, 2]:
        raise RuntimeError(f"normalizer provenance mismatch: {path}")
    return values["mean"].astype(np.float32), values["std"].astype(np.float32)


def eegnet_f8_config() -> dict[str, Any]:
    return dict(next(row for row in core.protocol()["baseline_candidates"] if row["id"] == "EEGNET_F8"))


def instantiate_model(method: str, lock: Mapping[str, Any]) -> torch.nn.Module:
    if method == "B0_VANILLA_EEGNET":
        return core.EEGNetClassifier(eegnet_f8_config())
    if method == "B1_STRONG_EEGNET":
        return core.EEGNetClassifier(dict(lock["baseline_configuration"]))
    if method in DUAL_OUTPUT_NAMES:
        return core.DualPathEEGNet(dict(lock["dual_width"]))
    raise KeyError(method)


def restore_model(
    method: str,
    lock: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Path, str, str]:
    entry = lock["checkpoint_hashes"][method]
    path = checkpoint_path(entry)
    file_before = sha256_file(path)
    if file_before != str(entry["sha256"]):
        raise RuntimeError(f"checkpoint hash mismatch before restore: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"invalid checkpoint payload: {path}")
    model = instantiate_model(method, lock)
    model.load_state_dict(payload["state_dict"], strict=True)
    state_before = tensor_state_sha256(model)
    model.to(device)
    model.eval()
    return model, path, file_before, state_before


def metric_rows(
    method: str,
    fold: int,
    seed: int,
    labels: np.ndarray,
    logits: np.ndarray,
    subjects: np.ndarray,
) -> list[dict[str, Any]]:
    prediction = np.asarray(logits).argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for subject in sorted_subjects(set(map(str, subjects))):
        mask = np.asarray(subjects, dtype=str) == subject
        rows.append(
            {
                "method": method,
                "fold": fold,
                "seed": seed,
                "subject_id": subject,
                "BA": float(balanced_accuracy_score(labels[mask], prediction[mask])),
                "macro_f1": float(f1_score(labels[mask], prediction[mask], average="macro")),
                "n_trials": int(mask.sum()),
                "evaluation_session": 2,
                "target_history_labels_used": False,
                "internal_holdout_used": False,
                "WBCIC_outer_used": False,
            }
        )
    return rows


def run_replay(source_experiment: Path) -> dict[str, Any]:
    audit = audit_inputs(source_experiment)
    source_experiment = source_experiment.resolve()
    data, data_audit = load_authorized_s2_data(source_experiment)
    roles = core.outer_folds(data.search_subjects)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core.set_determinism(core.stable_seed("source-only-replay"))
    replay_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    started = time.time()

    for fold in range(5):
        role = roles[fold]
        for seed in range(3):
            run_dir = source_experiment / "runtime" / "runs" / f"fold-{fold}" / f"seed-{seed}"
            lock = load_json(run_dir / "RUN_LOCK.json")
            if sorted_subjects(lock["source_subjects"]) != sorted_subjects(role["source"]):
                raise RuntimeError(f"fold role mismatch for fold {fold}, seed {seed}")
            mean, std = load_normalizer(lock, role["source"])
            authoritative = pd.read_csv(run_dir / "SUBJECT_RESULTS.csv")
            authoritative["subject_id"] = authoritative.subject_id.astype(str)

            for method in ("B0_VANILLA_EEGNET", "B1_STRONG_EEGNET"):
                model, checkpoint, file_before, state_before = restore_model(method, lock, device)
                actual_parts: list[pd.DataFrame] = []
                # The authoritative final-v1 loop evaluated one outcome subject
                # (100 trials) per call. Matching that batch geometry is needed
                # for bit-stable AMP boundary predictions.
                for subject in role["outcome"]:
                    future_indices = core.row_indices(data.metadata, (subject,), (2,))
                    evaluation = core.evaluate_single(
                        model,
                        data,
                        future_indices,
                        device,
                        mean,
                        std,
                        include_features=False,
                        batch_size=512,
                    )
                    actual_parts.append(
                        pd.DataFrame(
                            metric_rows(
                                method,
                                fold,
                                seed,
                                evaluation.labels,
                                evaluation.logits,
                                evaluation.subjects,
                            )
                        )
                    )
                actual = pd.concat(actual_parts, ignore_index=True)
                expected = authoritative.loc[
                    authoritative.method.eq(method), ["subject_id", "BA", "macro_f1", "n_trials"]
                ].copy()
                expected["subject_id"] = expected.subject_id.astype(str)
                merged = actual.merge(expected, on="subject_id", suffixes=("_replay", "_authoritative"), validate="one_to_one")
                if len(merged) != 8:
                    raise RuntimeError(f"incomplete authoritative replay rows for {fold}/{seed}/{method}")
                merged["method"] = method
                merged["fold"] = fold
                merged["seed"] = seed
                merged["BA_abs_error"] = np.abs(merged.BA_replay - merged.BA_authoritative)
                merged["macro_f1_abs_error"] = np.abs(
                    merged.macro_f1_replay - merged.macro_f1_authoritative
                )
                merged["n_trials_match"] = merged.n_trials_replay == merged.n_trials_authoritative
                comparison_rows.extend(merged.to_dict("records"))
                replay_rows.extend(actual.to_dict("records"))

                state_after = tensor_state_sha256(model)
                file_after = sha256_file(checkpoint)
                integrity_rows.append(
                    {
                        "stage": "replay",
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256_before": file_before,
                        "checkpoint_sha256_after": file_after,
                        "checkpoint_unchanged": file_before == file_after,
                        "state_sha256_before": state_before,
                        "state_sha256_after": state_after,
                        "parameters_and_buffers_unchanged": state_before == state_after,
                        "model_eval_mode": not model.training,
                    }
                )
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[replay] fold={fold} seed={seed}", flush=True)

    comparison = pd.DataFrame(comparison_rows)
    integrity = pd.DataFrame(integrity_rows)
    max_ba = float(comparison.BA_abs_error.max())
    max_f1 = float(comparison.macro_f1_abs_error.max())
    replay_pass = bool(
        len(comparison) == 240
        and max_ba <= TOL
        and max_f1 <= TOL
        and comparison.n_trials_match.all()
        and integrity.checkpoint_unchanged.all()
        and integrity.parameters_and_buffers_unchanged.all()
        and integrity.model_eval_mode.all()
    )
    payload = {
        "status": "REPLAY_VALIDATION_PASS" if replay_pass else "REPLAY_VALIDATION_FAIL",
        "pass": replay_pass,
        "input_fingerprint": audit["input_fingerprint"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "fold_seed_runs": 15,
        "evaluation_batch_geometry": "one outcome subject (100 trials) per inference call, matching final-v1",
        "methods": ["B0_VANILLA_EEGNET", "B1_STRONG_EEGNET"],
        "subject_method_rows": len(comparison),
        "max_abs_BA_error": max_ba,
        "max_abs_macro_f1_error": max_f1,
        "all_trial_counts_match": bool(comparison.n_trials_match.all()),
        "all_checkpoint_files_unchanged": bool(integrity.checkpoint_unchanged.all()),
        "all_parameters_and_buffers_unchanged": bool(integrity.parameters_and_buffers_unchanged.all()),
        "data_audit": data_audit,
        "runtime_s": time.time() - started,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    write_csv(RESULTS / "replay_per_subject.csv", pd.DataFrame(replay_rows))
    write_csv(RESULTS / "replay_comparison.csv", comparison)
    write_csv(RESULTS / "evaluation_integrity_replay.csv", integrity)
    write_json(REPLAY_JSON, payload)
    if not replay_pass:
        raise RuntimeError(f"B0/B1 replay validation failed: {payload}")
    print("REPLAY_VALIDATION_PASS", flush=True)
    return payload


def load_certificate(run_dir: Path) -> core.Certificate:
    path = run_dir / "certificate" / "PUD_CERTIFICATE.npz"
    values = np.load(path, allow_pickle=False)
    bases = {
        name.removeprefix("basis_"): np.asarray(values[name], dtype=np.float32)
        for name in values.files
        if name.startswith("basis_")
    }
    rows = pd.read_csv(run_dir / "certificate" / "PUD_CERTIFICATION.csv")
    audit = load_json(run_dir / "certificate" / "PUD_CERTIFICATION_AUDIT.json")
    certificate = core.Certificate(
        mean=np.asarray(values["mean"], dtype=np.float32),
        whitener=np.asarray(values["whitener"], dtype=np.float32),
        dewhitener=np.asarray(values["dewhitener"], dtype=np.float32),
        directions=np.asarray(values["directions"], dtype=np.float32),
        rho=np.asarray(values["rho"], dtype=np.float32),
        rows=rows,
        bases=bases,
        audit=audit,
    )
    if certificate.rank != int(audit["PUD_rank"]):
        raise RuntimeError(f"certificate rank mismatch: {path}")
    return certificate


def run_source_only_evaluation(source_experiment: Path) -> dict[str, Any]:
    if not REPLAY_JSON.is_file():
        raise RuntimeError("PUD source-only evaluation is forbidden until replay has run")
    replay = load_json(REPLAY_JSON)
    if replay.get("pass") is not True:
        raise RuntimeError("PUD source-only evaluation is forbidden because replay did not pass")
    audit = audit_inputs(source_experiment)
    if audit["input_fingerprint"] != replay.get("input_fingerprint"):
        raise RuntimeError("frozen inputs changed after replay")

    source_experiment = source_experiment.resolve()
    data, data_audit = load_authorized_s2_data(source_experiment)
    roles = core.outer_folds(data.search_subjects)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core.set_determinism(core.stable_seed("source-only-evaluation"))
    rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    adapted_rows: list[dict[str, Any]] = []
    started = time.time()

    for fold in range(5):
        role = roles[fold]
        for seed in range(3):
            run_dir = source_experiment / "runtime" / "runs" / f"fold-{fold}" / f"seed-{seed}"
            lock = load_json(run_dir / "RUN_LOCK.json")
            mean, std = load_normalizer(lock, role["source"])
            authoritative = pd.read_csv(run_dir / "SUBJECT_RESULTS.csv")
            authoritative["subject_id"] = authoritative.subject_id.astype(str)

            teacher, teacher_path, teacher_file_before, teacher_state_before = restore_model(
                "B1_STRONG_EEGNET", lock, device
            )
            certificate = load_certificate(run_dir)
            restored: dict[str, tuple[torch.nn.Module, Path, str, str]] = {}
            for checkpoint_method, output_method in DUAL_OUTPUT_NAMES.items():
                restored[output_method] = restore_model(checkpoint_method, lock, device)

            # Reproduce final-v1's exact subject-at-a-time evaluation geometry.
            for subject in role["outcome"]:
                future_indices = core.row_indices(data.metadata, (subject,), (2,))
                teacher_eval = core.evaluate_single(
                    teacher,
                    data,
                    future_indices,
                    device,
                    mean,
                    std,
                    include_features=True,
                    batch_size=512,
                )
                teacher_target = np.asarray(
                    core.teacher_targets(teacher, certificate, teacher_eval, "PUD")["protected"],
                    dtype=np.float64,
                )
                for output_method, (model, _, _, _) in restored.items():
                    evaluation = core.evaluate_dual(
                        model,
                        data,
                        future_indices,
                        device,
                        mean,
                        std,
                        batch_size=512,
                    )
                    if not np.array_equal(evaluation["indices"], teacher_eval.indices):
                        raise RuntimeError(f"evaluation row mismatch for {fold}/{seed}/{output_method}/{subject}")
                    if set(map(str, evaluation["subjects"])) != {str(subject)}:
                        raise RuntimeError(f"subject isolation failure for {fold}/{seed}/{output_method}/{subject}")
                    combined = evaluation["protected_logits"] + evaluation["adaptive_logits"]
                    subject_metrics = metric_rows(
                        output_method,
                        fold,
                        seed,
                        evaluation["labels"],
                        combined,
                        evaluation["subjects"],
                    )
                    if len(subject_metrics) != 1:
                        raise RuntimeError("subject-at-a-time evaluation produced non-unit metric rows")
                    rows.extend(subject_metrics)
                    labels = evaluation["labels"]
                    combined_prediction = combined.argmax(axis=1)
                    adaptive_prediction = evaluation["adaptive_logits"].argmax(axis=1)
                    protected_prediction = evaluation["protected_logits"].argmax(axis=1)
                    combined_ba = float(balanced_accuracy_score(labels, combined_prediction))
                    adaptive_only_ba = float(balanced_accuracy_score(labels, adaptive_prediction))
                    protected_only_ba = float(balanced_accuracy_score(labels, protected_prediction))
                    rmse, corr = core.functional_agreement(
                        evaluation["protected_logits"], teacher_target
                    )
                    mechanism_rows.append(
                        {
                            "method": output_method,
                            "fold": fold,
                            "seed": seed,
                            "subject_id": subject,
                            "combined_BA": combined_ba,
                            "protected_only_BA": protected_only_ba,
                            "adaptive_only_BA": adaptive_only_ba,
                            "protected_branch_erasure_harm_BA": combined_ba - adaptive_only_ba,
                            "adaptive_branch_erasure_harm_BA": combined_ba - protected_only_ba,
                            "protected_D_finite": core.exact_d_finite(
                                np.zeros_like(evaluation["protected_logits"]),
                                evaluation["protected_logits"],
                            ),
                            "adaptive_D_finite": core.exact_d_finite(
                                np.zeros_like(evaluation["adaptive_logits"]),
                                evaluation["adaptive_logits"],
                            ),
                            "functional_teacher_target": "PUD_PROJECTED_TEACHER_LOGITS",
                            "functional_teacher_RMSE": rmse,
                            "functional_teacher_correlation": corr,
                            "target_history_labels_used": False,
                            "internal_holdout_used": False,
                            "WBCIC_outer_used": False,
                        }
                    )
                    if output_method == "PUD_SOURCE_ONLY":
                        actual_ba = float(subject_metrics[0]["BA"])
                        actual_subject = str(subject_metrics[0]["subject_id"])
                        if actual_subject != str(subject):
                            raise RuntimeError("PUD source subject identity mismatch")
                        for adapted_method in ("A6_PUD_ALL_ADAPT", "A10_FULL_PUD_FREEZE"):
                            expected = authoritative.loc[
                                authoritative.method.eq(adapted_method)
                                & authoritative.subject_id.eq(str(subject)),
                                "source_model_noadapt_BA",
                            ]
                            if len(expected) != 1:
                                raise RuntimeError(
                                    f"missing authoritative source-model value for {fold}/{seed}/{subject}/{adapted_method}"
                                )
                            expected_ba = float(expected.iloc[0])
                            validation_rows.append(
                                {
                                    "fold": fold,
                                    "seed": seed,
                                    "subject_id": str(subject),
                                    "authoritative_method": adapted_method,
                                    "PUD_source_only_BA_replayed": actual_ba,
                                    "authoritative_source_model_noadapt_BA": expected_ba,
                                    "absolute_error": abs(actual_ba - expected_ba),
                                }
                            )

            teacher_state_after = tensor_state_sha256(teacher)
            teacher_file_after = sha256_file(teacher_path)
            integrity_rows.append(
                {
                    "stage": "source_only",
                    "fold": fold,
                    "seed": seed,
                    "method": "B1_STRONG_EEGNET_TEACHER_TARGET_ONLY",
                    "checkpoint": str(teacher_path),
                    "checkpoint_sha256_before": teacher_file_before,
                    "checkpoint_sha256_after": teacher_file_after,
                    "checkpoint_unchanged": teacher_file_before == teacher_file_after,
                    "state_sha256_before": teacher_state_before,
                    "state_sha256_after": teacher_state_after,
                    "parameters_and_buffers_unchanged": teacher_state_before == teacher_state_after,
                    "model_eval_mode": not teacher.training,
                }
            )
            for output_method, (model, checkpoint, file_before, state_before) in restored.items():
                state_after = tensor_state_sha256(model)
                file_after = sha256_file(checkpoint)
                integrity_rows.append(
                    {
                        "stage": "source_only",
                        "fold": fold,
                        "seed": seed,
                        "method": output_method,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256_before": file_before,
                        "checkpoint_sha256_after": file_after,
                        "checkpoint_unchanged": file_before == file_after,
                        "state_sha256_before": state_before,
                        "state_sha256_after": state_after,
                        "parameters_and_buffers_unchanged": state_before == state_after,
                        "model_eval_mode": not model.training,
                    }
                )
                del model

            adapted = authoritative.loc[
                authoritative.method.eq("A10_FULL_PUD_FREEZE"),
                ["fold", "seed", "subject_id", "BA", "macro_f1", "n_trials"],
            ].copy()
            adapted.insert(0, "method", "PUD_AFTER_ADAPT")
            adapted["evaluation_session"] = 2
            adapted["target_history_labels_used"] = True
            adapted["internal_holdout_used"] = False
            adapted["WBCIC_outer_used"] = False
            adapted_rows.extend(adapted.to_dict("records"))
            del teacher
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[source-only] fold={fold} seed={seed}", flush=True)

    source_rows = pd.DataFrame(rows)
    mechanism = pd.DataFrame(mechanism_rows)
    integrity = pd.DataFrame(integrity_rows)
    validation = pd.DataFrame(validation_rows)
    max_source_error = float(validation.absolute_error.max())
    evaluation_pass = bool(
        len(source_rows) == 480
        and len(mechanism) == 480
        and len(validation) == 240
        and max_source_error <= TOL
        and integrity.checkpoint_unchanged.all()
        and integrity.parameters_and_buffers_unchanged.all()
        and integrity.model_eval_mode.all()
    )
    payload = {
        "status": "SOURCE_ONLY_EVALUATION_COMPLETE" if evaluation_pass else "SOURCE_ONLY_EVALUATION_FAIL",
        "pass": evaluation_pass,
        "input_fingerprint": audit["input_fingerprint"],
        "replay_validation_passed_before_PUD_evaluation": True,
        "evaluation_batch_geometry": "one outcome subject (100 trials) per inference call, matching final-v1",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "source_subject_method_rows": len(source_rows),
        "mechanism_rows": len(mechanism),
        "PUD_source_authoritative_validation_rows": len(validation),
        "PUD_source_max_abs_BA_error": max_source_error,
        "all_checkpoint_files_unchanged": bool(integrity.checkpoint_unchanged.all()),
        "all_parameters_and_buffers_unchanged": bool(integrity.parameters_and_buffers_unchanged.all()),
        "adaptation_function_called": False,
        "optimizer_steps": 0,
        "target_history_labels_materialized": False,
        "data_audit": data_audit,
        "runtime_s": time.time() - started,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    write_csv(RESULTS / "source_only_raw.csv", source_rows)
    write_csv(RESULTS / "mechanism_raw.csv", mechanism)
    write_csv(RESULTS / "source_checkpoint_validation.csv", validation)
    write_csv(RESULTS / "adapted_authoritative_raw.csv", pd.DataFrame(adapted_rows))
    write_csv(RESULTS / "evaluation_integrity_source_only.csv", integrity)
    write_json(RUNTIME / "SOURCE_ONLY_EVALUATION.json", payload)
    if not evaluation_pass:
        raise RuntimeError(f"source-only evaluation integrity failure: {payload}")
    print("SOURCE_ONLY_EVALUATION_COMPLETE", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "replay", "evaluate"), required=True)
    parser.add_argument("--source-experiment", type=Path, default=DEFAULT_SOURCE_EXPERIMENT)
    args = parser.parse_args()
    if args.stage == "audit":
        audit_inputs(args.source_experiment)
    elif args.stage == "replay":
        run_replay(args.source_experiment)
    else:
        run_source_only_evaluation(args.source_experiment)


if __name__ == "__main__":
    main()
