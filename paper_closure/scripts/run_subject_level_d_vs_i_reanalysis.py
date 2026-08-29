"""Locked subject-level D-versus-I validity reanalysis.

The ``manifest`` subcommand hashes the retained historical artifacts without
loading scientific outcomes.  The ``run`` subcommand refuses to execute until
the protocol, implementation, and manifest are committed and clean, verifies
every input hash, reconstructs per-subject consequences, and performs the
predeclared subject-clustered analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


PROTOCOL_ID = "PERSIST_EEG_SUBJECT_LEVEL_D_VS_I_REANALYSIS_V1"
CANONICAL_REPOSITORY_PATHS: Mapping[str, str] = {
    "protocol": "paper_closure/protocol/SUBJECT_LEVEL_D_VS_I_REANALYSIS_LOCK.json",
    "manifest": "paper_closure/protocol/SUBJECT_LEVEL_D_VS_I_INPUT_MANIFEST.csv",
    "implementation": "paper_closure/scripts/run_subject_level_d_vs_i_reanalysis.py",
    "test": "paper_closure/scripts/test_subject_level_d_vs_i_reanalysis.py",
    "rationale": "paper_closure/statistics/SUBJECT_LEVEL_D_VS_I_REANALYSIS_RATIONALE.md",
    "output_directory": "paper_closure/statistics/subject_level_d_vs_i",
}
REQUIRED_OUTPUT_FILES: tuple[str, ...] = (
    "paper_closure/statistics/subject_level_d_vs_i/subject_observations.csv",
    "paper_closure/statistics/subject_level_d_vs_i/subject_summary.csv",
    "paper_closure/statistics/subject_level_d_vs_i/subject_backbone_summary.csv",
    "paper_closure/statistics/subject_level_d_vs_i/SUBJECT_LEVEL_D_VS_I_SUMMARY.json",
    "paper_closure/statistics/subject_level_d_vs_i/SUBJECT_LEVEL_D_VS_I_REPORT.md",
    "paper_closure/statistics/subject_level_d_vs_i/VALIDATION.json",
)
DATASET_GATE_LABELS: tuple[str, ...] = (
    "SUPPORTED_CONDITIONAL",
    "PARTIAL",
    "NOT_SUPPORTED",
    "REVERSED",
    "POINT_CI_DIRECTION_CONFLICT",
)
CROSS_DATASET_TERMINALS: tuple[str, ...] = (
    "CROSS_DATASET_SUPPORTED_CONDITIONAL",
    "CROSS_DATASET_PARTIAL",
    "CROSS_DATASET_NOT_SUPPORTED",
    "CROSS_DATASET_POINT_CI_DIRECTION_CONFLICT",
)
RIDGE_ALPHA = 1.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2_026_082_901
CI_ALPHA = 0.05
STANDARDIZATION_MIN_STD = 1e-8
CONFIGURATIONS: tuple[tuple[str, float], ...] = (
    ("ERM", 0.0),
    ("DANN", 0.01),
    ("DANN", 0.10),
    ("DANN", 1.00),
    ("CORAL", 0.01),
    ("CORAL", 0.10),
    ("CORAL", 1.00),
    ("MMD", 0.01),
    ("MMD", 0.10),
    ("MMD", 1.00),
)
MODELS: Mapping[str, tuple[str, ...]] = {
    "M0": ("persistence", "geometry_strength", "rank_feature"),
    "MI": ("persistence", "geometry_strength", "rank_feature", "identity_score"),
    "MD": ("persistence", "geometry_strength", "rank_feature", "D_finite"),
    "MID": (
        "persistence",
        "geometry_strength",
        "rank_feature",
        "identity_score",
        "D_finite",
    ),
}
CELL_COLUMNS = [
    "dataset",
    "backbone",
    "fold",
    "seed",
    "method",
    "lambda",
    "direction_id",
]
PREDICTOR_COLUMNS = [
    "persistence",
    "geometry_strength",
    "rank_feature",
    "identity_score",
    "D_finite",
]
MANIFEST_COLUMNS = [
    "protocol_id",
    "source_alias",
    "role",
    "backbone",
    "fold",
    "seed",
    "method",
    "lambda",
    "relative_path",
    "bytes",
    "sha256",
]
AGGREGATE_ATOL = 5e-8
AGGREGATE_RTOL = 5e-6
NUMERICAL_EQUIVALENCE_ATOL = 1e-10
OUTCOME_INDEX_KEYS = (
    "outcome_subjects",
    "outcome_sessions",
    "outcome_labels",
    "outcome_indices",
)
SOURCE_INDEX_KEYS = (
    "source_subjects",
    "source_sessions",
    "source_labels",
    "source_indices",
)
PRIMARY_ANALYSES = {
    "OPENBMI_STRESS": "OPENBMI_EQUAL_FAMILY_PRIMARY",
    "WBCIC_REPLICATION": "WBCIC_ERM_PRIMARY",
}
BOOTSTRAP_ANALYSES = frozenset(PRIMARY_ANALYSES.values())


@dataclass(frozen=True)
class SourceRoots:
    stress: Path
    wbcic: Path

    def for_alias(self, alias: str) -> Path:
        if alias == "OPENBMI_STRESS_HISTORICAL":
            return self.stress
        if alias == "WBCIC_REPLICATION_HISTORICAL":
            return self.wbcic
        raise KeyError(f"unknown source alias: {alias}")


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    dataset: str
    methods: tuple[str, ...]
    training_weight_mode: str
    subject_aggregation_mode: str
    role: str


ANALYSES: tuple[AnalysisSpec, ...] = (
    AnalysisSpec(
        "OPENBMI_EQUAL_FAMILY_PRIMARY",
        "OPENBMI_STRESS",
        ("ERM", "DANN", "CORAL", "MMD"),
        "EQUAL_METHOD_FAMILY",
        "EQUAL_METHOD_FAMILY",
        "PRIMARY_OPENBMI",
    ),
    AnalysisSpec(
        "OPENBMI_EQUAL_CONFIG_SECONDARY",
        "OPENBMI_STRESS",
        ("ERM", "DANN", "CORAL", "MMD"),
        "EQUAL_CONFIGURATION",
        "POOLED_CONFIGURATION_GRID",
        "MANDATORY_GRID_WEIGHTING_SENSITIVITY",
    ),
    AnalysisSpec(
        "OPENBMI_ERM_ONLY_SENSITIVITY",
        "OPENBMI_STRESS",
        ("ERM",),
        "UNIFORM",
        "POOLED_CONFIGURATION_GRID",
        "MANDATORY_DIRECT_WBCIC_SCOPE_SENSITIVITY",
    ),
    AnalysisSpec(
        "WBCIC_ERM_PRIMARY",
        "WBCIC_REPLICATION",
        ("ERM",),
        "UNIFORM",
        "POOLED_CONFIGURATION_GRID",
        "PRIMARY_WBCIC",
    ),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.all(np.isfinite(numeric.to_numpy(dtype=np.float64))):
        raise RuntimeError(f"refusing to serialize non-finite numeric CSV values: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_text(path: Path) -> str:
    return sha256_file(path)


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def config_slug(method: str, lam: float) -> str:
    return f"{method.lower()}__lambda-{lam:.2f}"


def subject_sort(values: Iterable[str]) -> list[str]:
    return sorted(map(str, values), key=lambda item: (int(item) if item.isdigit() else 10**9, item))


def implementation_binding() -> dict[str, Any]:
    """Machine-readable scientific constants that must match the protocol lock."""

    return {
        "canonical_repository_paths": dict(CANONICAL_REPOSITORY_PATHS),
        "required_output_files": list(REQUIRED_OUTPUT_FILES),
        "dataset_gate_labels": list(DATASET_GATE_LABELS),
        "cross_dataset_terminals": list(CROSS_DATASET_TERMINALS),
        "configurations": [
            {"method": method, "lambda": float(lam)} for method, lam in CONFIGURATIONS
        ],
        "models": {name: list(columns) for name, columns in MODELS.items()},
        "analyses": [
            {
                "name": spec.name,
                "dataset": spec.dataset,
                "methods": list(spec.methods),
                "training_weight_mode": spec.training_weight_mode,
                "subject_aggregation_mode": spec.subject_aggregation_mode,
                "role": spec.role,
            }
            for spec in ANALYSES
        ],
        "primary_analyses": PRIMARY_ANALYSES,
        "bootstrap_analyses": sorted(BOOTSTRAP_ANALYSES),
        "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_alpha": CI_ALPHA,
        "standardization_min_std": STANDARDIZATION_MIN_STD,
        "aggregate_atol": AGGREGATE_ATOL,
        "aggregate_rtol": AGGREGATE_RTOL,
        "numerical_equivalence_atol": NUMERICAL_EQUIVALENCE_ATOL,
        "expected_reconstructed_rows": {
            "OPENBMI_STRESS": 19_200,
            "WBCIC_REPLICATION": 984,
        },
        "expected_analysis_rows": {
            "OPENBMI_EQUAL_FAMILY_PRIMARY": 19_200,
            "OPENBMI_EQUAL_CONFIG_SECONDARY": 19_200,
            "OPENBMI_ERM_ONLY_SENSITIVITY": 1_920,
            "WBCIC_ERM_PRIMARY": 984,
        },
        "expected_split_counts": {
            "OPENBMI_EQUAL_FAMILY_PRIMARY": 240,
            "OPENBMI_EQUAL_CONFIG_SECONDARY": 240,
            "OPENBMI_ERM_ONLY_SENSITIVITY": 240,
            "WBCIC_ERM_PRIMARY": 123,
        },
        "expected_manifest_artifact_count": 1_388,
        "expected_manifest_counts_by_alias": {
            "OPENBMI_STRESS_HISTORICAL": 1_264,
            "WBCIC_REPLICATION_HISTORICAL": 124,
        },
        "expected_manifest_counts_by_role": {
            "direction_table": 315,
            "embedding_archive": 315,
            "evaluation_guard": 315,
            "frozen_direction_basis": 15,
            "frozen_global_protocol": 2,
            "frozen_head_checkpoint": 315,
            "historical_aggregate_code": 2,
            "historical_common_code": 2,
            "historical_unit_runner": 2,
            "lambda_selection_freeze": 45,
            "source_freeze_guard": 15,
            "unit_protocol": 45,
        },
    }


def verify_protocol_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    observed = protocol.get("implementation_binding")
    expected = implementation_binding()
    if observed != expected:
        raise RuntimeError(
            "protocol implementation binding differs from executable constants; "
            f"expected={json.dumps(expected, sort_keys=True)} "
            f"observed={json.dumps(observed, sort_keys=True)}"
        )
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("protocol id mismatch")
    if protocol.get("status") != "FROZEN_BEFORE_SUBJECT_LEVEL_OUTCOME_RECONSTRUCTION":
        raise RuntimeError("protocol is not in the required frozen state")
    path_fields = {
        "rationale_path": "rationale",
        "implementation_path": "implementation",
        "test_path": "test",
        "input_manifest_path": "manifest",
    }
    for protocol_field, canonical_key in path_fields.items():
        if protocol.get(protocol_field) != CANONICAL_REPOSITORY_PATHS[canonical_key]:
            raise RuntimeError(f"protocol canonical path changed: {protocol_field}")
    if protocol.get("required_outputs") != list(REQUIRED_OUTPUT_FILES):
        raise RuntimeError("protocol required output file set changed")
    gates = protocol.get("decision_gates", {})
    required_gate_keys = {
        "dataset_supported_conditional",
        "dataset_partial",
        "dataset_not_supported",
        "dataset_reversed",
        "dataset_point_ci_direction_conflict",
        "cross_dataset_point_ci_direction_conflict",
    }
    if not required_gate_keys.issubset(gates):
        raise RuntimeError("protocol decision gates are not exhaustive")
    return {"status": "PASS", "implementation_binding": expected}


def implementation_repository() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_paths(repository: Path) -> dict[str, Path]:
    root = repository.resolve()
    return {
        key: (root / Path(relative)).resolve()
        for key, relative in CANONICAL_REPOSITORY_PATHS.items()
    }


def verify_canonical_invocation(
    repository: Path,
    *,
    protocol: Path,
    manifest: Path | None = None,
    output_directory: Path | None = None,
    manifest_output: Path | None = None,
) -> dict[str, Any]:
    """Bind both manifest creation and result execution to one canonical repository surface."""

    root = repository.resolve()
    if root != implementation_repository():
        raise RuntimeError(
            f"repository argument is not the repository containing this implementation: {root}"
        )
    expected = canonical_paths(root)
    observed: dict[str, Path] = {"protocol": protocol.resolve()}
    if manifest is not None:
        observed["manifest"] = manifest.resolve()
    if output_directory is not None:
        observed["output_directory"] = output_directory.resolve()
    if manifest_output is not None:
        if manifest is not None:
            raise RuntimeError("manifest input and manifest output cannot be supplied together")
        observed["manifest"] = manifest_output.resolve()
    for key, actual in observed.items():
        if actual != expected[key]:
            raise RuntimeError(f"noncanonical {key} path refused: {actual}; expected {expected[key]}")
    return {
        "status": "PASS",
        "repository_paths": dict(CANONICAL_REPOSITORY_PATHS),
    }


def artifact_specs(roots: SourceRoots) -> list[dict[str, Any]]:
    """Return the exact, outcome-blind artifact list required by the lock."""

    specs: list[dict[str, Any]] = []

    def add(
        alias: str,
        role: str,
        path: Path,
        *,
        backbone: str = "",
        fold: int | str = "",
        seed: int | str = "",
        method: str = "",
        lam: float | str = "",
    ) -> None:
        root = roots.for_alias(alias)
        specs.append(
            {
                "protocol_id": PROTOCOL_ID,
                "source_alias": alias,
                "role": role,
                "backbone": backbone,
                "fold": fold,
                "seed": seed,
                "method": method,
                "lambda": lam,
                "relative_path": path.relative_to(root).as_posix(),
                "_path": path,
            }
        )

    stress_alias = "OPENBMI_STRESS_HISTORICAL"
    add(stress_alias, "frozen_global_protocol", roots.stress / "STRESS_TEST_PROTOCOL_FROZEN.json")
    add(stress_alias, "historical_common_code", roots.stress / "code" / "common.py")
    add(stress_alias, "historical_aggregate_code", roots.stress / "code" / "aggregate.py")
    add(stress_alias, "historical_unit_runner", roots.stress / "code" / "run_stress.py")
    for backbone in ("eegnet", "eegconformer"):
        for fold in range(5):
            for seed in range(3):
                unit = roots.stress / "runtime" / "runs" / backbone / f"fold-{fold}" / f"seed-{seed}"
                add(stress_alias, "unit_protocol", unit / "UNIT_PROTOCOL.json", backbone=backbone, fold=fold, seed=seed)
                add(
                    stress_alias,
                    "lambda_selection_freeze",
                    unit / "LAMBDA_SELECTION_FROZEN.json",
                    backbone=backbone,
                    fold=fold,
                    seed=seed,
                )
                for method, lam in CONFIGURATIONS:
                    slug = config_slug(method, lam)
                    evaluation = unit / "evaluation" / slug
                    metadata = dict(backbone=backbone, fold=fold, seed=seed, method=method, lam=lam)
                    add(stress_alias, "frozen_head_checkpoint", unit / "checkpoints" / f"{slug}.pt", **metadata)
                    add(stress_alias, "embedding_archive", evaluation / "embeddings.npz", **metadata)
                    add(stress_alias, "direction_table", evaluation / "directions.csv", **metadata)
                    add(stress_alias, "evaluation_guard", evaluation / "EVALUATION_COMPLETE.json", **metadata)

    wbcic_alias = "WBCIC_REPLICATION_HISTORICAL"
    add(wbcic_alias, "frozen_global_protocol", roots.wbcic / "WBCIC_REPLICATION_PROTOCOL_FROZEN.json")
    add(wbcic_alias, "historical_common_code", roots.wbcic / "code" / "common.py")
    add(wbcic_alias, "historical_aggregate_code", roots.wbcic / "code" / "aggregate.py")
    add(wbcic_alias, "historical_unit_runner", roots.wbcic / "code" / "run_unit.py")
    for fold in range(5):
        for seed in range(3):
            backbone = "eegnet"
            method, lam = "ERM", 0.0
            slug = config_slug(method, lam)
            unit = roots.wbcic / "runtime" / "runs" / backbone / f"fold-{fold}" / f"seed-{seed}"
            evaluation = unit / "evaluation" / slug
            metadata = dict(backbone=backbone, fold=fold, seed=seed, method=method, lam=lam)
            add(wbcic_alias, "unit_protocol", unit / "UNIT_PROTOCOL.json", backbone=backbone, fold=fold, seed=seed)
            add(
                wbcic_alias,
                "lambda_selection_freeze",
                unit / "LAMBDA_SELECTION_FROZEN.json",
                backbone=backbone,
                fold=fold,
                seed=seed,
            )
            add(
                wbcic_alias,
                "source_freeze_guard",
                unit / "SOURCE_FREEZE_COMPLETE.json",
                backbone=backbone,
                fold=fold,
                seed=seed,
            )
            add(wbcic_alias, "frozen_direction_basis", unit / "source_freeze" / "erm_persistence_basis.npz", **metadata)
            add(wbcic_alias, "frozen_head_checkpoint", unit / "checkpoints" / f"{slug}.pt", **metadata)
            add(wbcic_alias, "embedding_archive", evaluation / "embeddings.npz", **metadata)
            add(wbcic_alias, "direction_table", evaluation / "directions.csv", **metadata)
            add(wbcic_alias, "evaluation_guard", evaluation / "EVALUATION_COMPLETE.json", **metadata)

    return sorted(specs, key=lambda row: (row["source_alias"], row["relative_path"], row["role"]))


def write_manifest_atomic(output: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite an existing pre-outcome manifest: {output}")
    if staging.exists():
        raise RuntimeError(f"prior manifest staging file exists and requires audit: {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = sha256_file(staging)
    staging.rename(output)
    return digest


def create_manifest(roots: SourceRoots, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite an existing pre-outcome manifest: {output}")
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        raise RuntimeError(f"prior manifest staging file exists and requires audit: {staging}")
    specs = artifact_specs(roots)
    binding = implementation_binding()
    if len(specs) != binding["expected_manifest_artifact_count"]:
        raise RuntimeError("artifact enumeration count differs from the protocol binding")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = spec.pop("_path")
        if not path.is_file():
            raise FileNotFoundError(f"required historical input missing: {path}")
        if "invalidated_smoke" in path.as_posix().lower():
            raise RuntimeError(f"forbidden invalidated runtime entered manifest: {path}")
        rows.append({**spec, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest_frame = pd.DataFrame(rows)
    by_alias = manifest_frame.groupby("source_alias").size().to_dict()
    by_role = manifest_frame.groupby("role").size().to_dict()
    normalized_alias = {str(key): int(value) for key, value in by_alias.items()}
    normalized_role = {str(key): int(value) for key, value in by_role.items()}
    if normalized_alias != binding["expected_manifest_counts_by_alias"]:
        raise RuntimeError(f"artifact enumeration alias counts changed: {normalized_alias}")
    if normalized_role != binding["expected_manifest_counts_by_role"]:
        raise RuntimeError(f"artifact enumeration role counts changed: {normalized_role}")
    manifest_sha256 = write_manifest_atomic(output, rows)
    return {
        "protocol_id": PROTOCOL_ID,
        "manifest_path": CANONICAL_REPOSITORY_PATHS["manifest"],
        "artifact_count": len(rows),
        "artifact_count_by_alias": normalized_alias,
        "artifact_count_by_role": normalized_role,
        "manifest_sha256": manifest_sha256,
        "scientific_outcomes_loaded": False,
    }


def verify_manifest(
    roots: SourceRoots, manifest: Path, *, include_hash_index: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[tuple[str, str], str]]:
    recorded = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if list(recorded.columns) != MANIFEST_COLUMNS:
        raise RuntimeError(f"manifest columns changed: {list(recorded.columns)}")
    expected = artifact_specs(roots)
    binding = implementation_binding()
    if len(expected) != binding["expected_manifest_artifact_count"]:
        raise RuntimeError("executable artifact enumeration count differs from the protocol binding")
    metadata_columns = [column for column in MANIFEST_COLUMNS if column not in {"bytes", "sha256"}]
    expected_metadata = [
        tuple(str(row[column]) for column in metadata_columns) for row in expected
    ]
    recorded_metadata = [
        tuple(str(getattr(row, column)) for column in metadata_columns)
        for row in recorded.itertuples(index=False)
    ]
    if len(recorded) != len(expected):
        raise RuntimeError(f"manifest row count {len(recorded)} != expected {len(expected)}")
    if len(set(recorded_metadata)) != len(recorded_metadata):
        raise RuntimeError("manifest contains duplicate metadata rows")
    if set(recorded_metadata) != set(expected_metadata):
        missing = sorted(set(expected_metadata) - set(recorded_metadata))
        extra = sorted(set(recorded_metadata) - set(expected_metadata))
        raise RuntimeError(f"manifest metadata mismatch; missing={missing[:3]} extra={extra[:3]}")
    if set(recorded.protocol_id) != {PROTOCOL_ID}:
        raise RuntimeError("manifest protocol id changed")
    counts_by_alias = {str(key): int(value) for key, value in recorded.groupby("source_alias").size().to_dict().items()}
    counts_by_role = {str(key): int(value) for key, value in recorded.groupby("role").size().to_dict().items()}
    if counts_by_alias != binding["expected_manifest_counts_by_alias"]:
        raise RuntimeError(f"manifest alias counts changed: {counts_by_alias}")
    if counts_by_role != binding["expected_manifest_counts_by_role"]:
        raise RuntimeError(f"manifest role counts changed: {counts_by_role}")
    failures: list[str] = []
    for row in recorded.itertuples(index=False):
        relative = Path(row.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe_path:{row.source_alias}:{row.relative_path}")
            continue
        path = roots.for_alias(row.source_alias) / relative
        if not path.is_file():
            failures.append(f"missing:{row.source_alias}:{row.relative_path}")
            continue
        if not row.bytes.isdigit() or int(row.bytes) <= 0:
            failures.append(f"invalid_bytes:{row.source_alias}:{row.relative_path}")
            continue
        if len(row.sha256) != 64 or any(character not in "0123456789abcdef" for character in row.sha256):
            failures.append(f"invalid_sha256:{row.source_alias}:{row.relative_path}")
            continue
        if str(path.stat().st_size) != row.bytes:
            failures.append(f"size:{row.source_alias}:{row.relative_path}")
            continue
        if sha256_file(path) != row.sha256:
            failures.append(f"sha256:{row.source_alias}:{row.relative_path}")
    if failures:
        raise RuntimeError(f"input-manifest verification failed: {failures[:10]}")
    summary = {
        "status": "PASS",
        "artifact_count": int(len(recorded)),
        "artifact_count_by_alias": counts_by_alias,
        "artifact_count_by_role": counts_by_role,
        "manifest_sha256": sha256_file(manifest),
        "metadata_bound_to_executable_scope": True,
        "duplicate_rows": 0,
        "failures": [],
    }
    if include_hash_index:
        index = {
            (str(row.source_alias), str(row.relative_path)): str(row.sha256)
            for row in recorded.itertuples(index=False)
        }
        return summary, index
    return summary


@contextmanager
def load_manifest_verified_npz(
    path: Path,
    *,
    source_alias: str,
    source_root: Path,
    manifest_hashes: Mapping[tuple[str, str], str],
    allow_pickle: bool,
) -> Iterator[Mapping[str, np.ndarray]]:
    """Hash and load the same open file handle, closing the manifest-to-pickle TOCTOU gap."""

    relative = path.resolve().relative_to(source_root.resolve()).as_posix()
    expected = manifest_hashes.get((source_alias, relative))
    if expected is None:
        raise RuntimeError(f"NPZ is outside the verified manifest scope: {source_alias}:{relative}")
    with path.open("rb") as stream:
        observed = sha256_stream(stream)
        if observed != expected:
            raise RuntimeError(f"NPZ changed after manifest verification: {source_alias}:{relative}")
        stream.seek(0)
        with np.load(stream, allow_pickle=allow_pickle) as archive:
            yield archive


def git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def verify_git_lock(repository: Path, paths: Sequence[Path]) -> dict[str, Any]:
    relative = [path.resolve().relative_to(repository.resolve()).as_posix() for path in paths]
    for item in relative:
        git_output(repository, "ls-files", "--error-unmatch", "--", item)
    dirty = git_output(repository, "status", "--porcelain=v1", "--", *relative)
    if dirty:
        raise RuntimeError(f"locked files are not clean: {dirty}")
    commits = {item: git_output(repository, "log", "-1", "--format=%H", "--", item) for item in relative}
    if any(not value for value in commits.values()) or len(set(commits.values())) != 1:
        raise RuntimeError(f"protocol, implementation, and manifest were not frozen in one commit: {commits}")
    return {
        "status": "PASS",
        "lock_commit": next(iter(commits.values())),
        "head_at_execution": git_output(repository, "rev-parse", "HEAD"),
        "locked_file_commits": commits,
    }


def numpy_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64).copy()
    value -= value.max(axis=1, keepdims=True)
    log_probability = value - np.log(np.exp(value).sum(axis=1, keepdims=True))
    return -log_probability[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)]


def exact_d_finite(clean_logits: np.ndarray, erased_logits: np.ndarray) -> float:
    delta = np.asarray(erased_logits, dtype=np.float64) - np.asarray(clean_logits, dtype=np.float64)
    centered = delta - delta.mean(axis=-1, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(np.square(centered), axis=-1))))


def unit_protocol_guard(
    path: Path,
    *,
    dataset: str,
    expected_backbone: str,
    expected_fold: int,
    expected_seed: int,
) -> dict[str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in {
        "backbone": expected_backbone,
        "fold": expected_fold,
        "seed": expected_seed,
    }.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"unit protocol {key} mismatch: {path}")
    if payload.get("outcome_labels_used") is not False:
        raise RuntimeError(f"unit protocol outcome purity flag changed: {path}")
    expected_scope = {
        "OPENBMI_STRESS": "authorized_40_subject_cache_only",
        "WBCIC_REPLICATION": "authorized_41_subject_WBCIC_development_cache_only",
    }[dataset]
    if payload.get("data_scope") != expected_scope:
        raise RuntimeError(f"unit protocol data scope changed: {path}")
    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, dict):
        raise RuntimeError(f"unit protocol roles missing: {path}")
    roles = {str(key): tuple(subject_sort(value)) for key, value in raw_roles.items()}
    if dataset == "OPENBMI_STRESS":
        required = {"outcome", "inner_validation", "inner_train", "source"}
        if not required.issubset(roles):
            raise RuntimeError(f"OpenBMI unit roles incomplete: {path}")
        outcome, validation, train, source = (
            set(roles["outcome"]),
            set(roles["inner_validation"]),
            set(roles["inner_train"]),
            set(roles["source"]),
        )
        if (len(outcome), len(validation), len(train), len(source)) != (8, 8, 24, 32):
            raise RuntimeError(f"OpenBMI unit role cardinality changed: {path}")
        if source != validation | train or outcome & source or validation & train:
            raise RuntimeError(f"OpenBMI unit roles overlap or are not exhaustive: {path}")
    else:
        required = {"outcome", "validation_discovery", "model_fit", "source"}
        if not required.issubset(roles):
            raise RuntimeError(f"WBCIC unit roles incomplete: {path}")
        outcome, validation, train, source = (
            set(roles["outcome"]),
            set(roles["validation_discovery"]),
            set(roles["model_fit"]),
            set(roles["source"]),
        )
        if len(outcome | validation | train) != 41 or source != train:
            raise RuntimeError(f"WBCIC unit roles are not the 41-subject development partition: {path}")
        if outcome & validation or outcome & train or validation & train:
            raise RuntimeError(f"WBCIC unit roles overlap: {path}")
        if len(outcome) not in (8, 9) or len(validation) not in (8, 9) or len(train) not in (24, 25):
            raise RuntimeError(f"WBCIC unit role cardinality changed: {path}")
    return roles


def global_protocol_roles(path: Path, *, dataset: str) -> dict[int, dict[str, tuple[str, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_schema = {
        "OPENBMI_STRESS": "PERSIST_EEG_SUBJECT_INVARIANCE_STRESS_TEST_V1",
        "WBCIC_REPLICATION": "PERSIST_EEG_WBCIC_INDEPENDENT_REPLICATION_V1",
    }[dataset]
    if payload.get("schema") != expected_schema:
        raise RuntimeError(f"global historical protocol schema changed: {path}")
    if payload.get("frozen_before_training") is not True or payload.get("frozen_before_outcome_evaluation") is not True:
        raise RuntimeError(f"global historical protocol was not frozen before outcomes: {path}")
    dataset_payload = payload.get("dataset", {})
    expected_count = 40 if dataset == "OPENBMI_STRESS" else 41
    pool = set(subject_sort(dataset_payload.get("subject_pool", [])))
    if int(dataset_payload.get("subject_count", -1)) != expected_count or len(pool) != expected_count:
        raise RuntimeError(f"global historical subject pool changed: {path}")
    if list(map(int, payload.get("training", {}).get("seeds", []))) != [0, 1, 2]:
        raise RuntimeError(f"global historical seed scope changed: {path}")
    roles_by_fold: dict[int, dict[str, tuple[str, ...]]] = {}
    for raw in dataset_payload.get("folds", []):
        fold = int(raw["fold"])
        outcome = set(map(str, raw["outcome"]))
        if dataset == "OPENBMI_STRESS":
            validation = set(map(str, raw["inner_validation"]))
            train = pool - outcome - validation
            source = pool - outcome
            roles = {
                "outcome": tuple(subject_sort(outcome)),
                "inner_validation": tuple(subject_sort(validation)),
                "inner_train": tuple(subject_sort(train)),
                "source": tuple(subject_sort(source)),
            }
        else:
            validation = set(map(str, raw["validation_discovery"]))
            train = set(map(str, raw["model_fit"]))
            if outcome | validation | train != pool:
                raise RuntimeError(f"WBCIC global fold is not exhaustive: {path} fold={fold}")
            roles = {
                "outcome": tuple(subject_sort(outcome)),
                "validation_discovery": tuple(subject_sort(validation)),
                "model_fit": tuple(subject_sort(train)),
                "source": tuple(subject_sort(train)),
            }
        roles_by_fold[fold] = roles
    if set(roles_by_fold) != set(range(5)):
        raise RuntimeError(f"global historical fold set changed: {path}")
    outcome_partition = [subject for fold in range(5) for subject in roles_by_fold[fold]["outcome"]]
    if len(outcome_partition) != expected_count or set(outcome_partition) != pool:
        raise RuntimeError(f"global historical outcomes do not partition the subject pool: {path}")
    if dataset == "OPENBMI_STRESS":
        historical_grid = []
        for method in ("ERM", "DANN", "CORAL", "MMD"):
            historical_grid.extend(
                (method, float(lam)) for lam in payload.get("methods", {}).get(method, {}).get("lambda_grid", [])
            )
        if tuple(historical_grid) != CONFIGURATIONS:
            raise RuntimeError(f"OpenBMI global formal configuration grid changed: {path}")
        if set(payload.get("backbones", {})) != {"eegnet", "eegconformer"}:
            raise RuntimeError(f"OpenBMI global backbone scope changed: {path}")
        if payload.get("dataset", {}).get("restricted_membership_enumerated") is not False:
            raise RuntimeError(f"OpenBMI restricted membership flag changed: {path}")
    else:
        if payload.get("dataset", {}).get("fold_role_rule") != (
            "outcome=F_k; validation/discovery=F_(k+1 mod 5); model-fit=remaining three folds"
        ):
            raise RuntimeError(f"WBCIC cyclic fold role rule changed: {path}")
        for fold in range(5):
            if set(roles_by_fold[fold]["validation_discovery"]) != set(
                roles_by_fold[(fold + 1) % 5]["outcome"]
            ):
                raise RuntimeError(f"WBCIC cyclic validation/discovery rule failed at fold {fold}")
        if payload.get("secondary_direction_audit", {}).get("method") != "ERM only":
            raise RuntimeError(f"WBCIC direction-audit method scope changed: {path}")
        if int(payload.get("secondary_direction_audit", {}).get("candidate_count", -1)) != 8:
            raise RuntimeError(f"WBCIC direction count changed in global protocol: {path}")
        restricted = payload.get("restricted_data_policy", {})
        if restricted.get("sealed_WBCIC_outer_access_permitted") is not False:
            raise RuntimeError(f"WBCIC outer policy changed: {path}")
    return roles_by_fold


def embedding_partition_guard(
    embedded: Mapping[str, np.ndarray],
    *,
    dataset: str,
    roles: Mapping[str, Sequence[str]],
    context: str,
) -> dict[str, np.ndarray]:
    required = {
        "source_features",
        "source_logits",
        *SOURCE_INDEX_KEYS,
        "outcome_features",
        "outcome_logits",
        *OUTCOME_INDEX_KEYS,
    }
    observed_keys = set(embedded.keys())
    if observed_keys != required:
        missing = sorted(required - observed_keys)
        extra = sorted(observed_keys - required)
        raise RuntimeError(f"embedding archive key set changed at {context}: missing={missing} extra={extra}")
    for prefix, keys in (("source", SOURCE_INDEX_KEYS), ("outcome", OUTCOME_INDEX_KEYS)):
        features = np.asarray(embedded[f"{prefix}_features"])
        logits = np.asarray(embedded[f"{prefix}_logits"])
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise RuntimeError(f"{prefix} feature shape/finiteness changed at {context}")
        if logits.shape != (len(features), 2) or not np.all(np.isfinite(logits)):
            raise RuntimeError(f"{prefix} logit shape/finiteness changed at {context}")
        length = len(features)
        for key in keys:
            value = np.asarray(embedded[key])
            if value.ndim != 1 or len(value) != length:
                raise RuntimeError(f"embedding row alignment changed for {key} at {context}")
        indices = np.asarray(embedded[f"{prefix}_indices"])
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"duplicate {prefix} indices at {context}")
    source_indices = set(np.asarray(embedded["source_indices"]).tolist())
    outcome_indices = set(np.asarray(embedded["outcome_indices"]).tolist())
    if source_indices & outcome_indices:
        raise RuntimeError(f"source/outcome row overlap at {context}")
    source_subjects = set(np.asarray(embedded["source_subjects"]).astype(str))
    outcome_subjects = set(np.asarray(embedded["outcome_subjects"]).astype(str))
    if source_subjects != set(map(str, roles["source"])):
        raise RuntimeError(f"source subject set differs from frozen roles at {context}")
    if outcome_subjects != set(map(str, roles["outcome"])):
        raise RuntimeError(f"outcome subject set differs from frozen roles at {context}")
    if source_subjects & outcome_subjects:
        raise RuntimeError(f"source/outcome biological-subject overlap at {context}")
    expected_sessions = {
        "OPENBMI_STRESS": ({1, 2}, {2}),
        "WBCIC_REPLICATION": ({0, 1}, {2}),
    }[dataset]
    if set(map(int, np.unique(embedded["source_sessions"]))) != expected_sessions[0]:
        raise RuntimeError(f"source session scope changed at {context}")
    if set(map(int, np.unique(embedded["outcome_sessions"]))) != expected_sessions[1]:
        raise RuntimeError(f"outcome session scope changed at {context}")
    if set(map(int, np.unique(embedded["source_labels"]))) != {0, 1}:
        raise RuntimeError(f"source label scope changed at {context}")
    if set(map(int, np.unique(embedded["outcome_labels"]))) != {0, 1}:
        raise RuntimeError(f"outcome label scope changed at {context}")
    if dataset == "OPENBMI_STRESS":
        if len(source_indices) != 6_400 or len(outcome_indices) != 800:
            raise RuntimeError(f"OpenBMI source/outcome row cardinality changed at {context}")
    minimum_cell_count = 50 if dataset == "OPENBMI_STRESS" else 20
    for prefix in ("source", "outcome"):
        cell = pd.DataFrame(
            {
                "subject": np.asarray(embedded[f"{prefix}_subjects"]).astype(str),
                "session": np.asarray(embedded[f"{prefix}_sessions"], dtype=np.int64),
                "label": np.asarray(embedded[f"{prefix}_labels"], dtype=np.int64),
            }
        )
        expected_cells = cell.subject.nunique() * cell.session.nunique() * 2
        cell = cell.groupby(["subject", "session", "label"]).size()
        if len(cell) != expected_cells:
            raise RuntimeError(f"{prefix} subject/session cells do not contain both classes at {context}")
        if dataset == "OPENBMI_STRESS":
            if not (cell == minimum_cell_count).all():
                raise RuntimeError(f"OpenBMI subject/session/class trial count changed at {context}")
        elif (cell < minimum_cell_count).any():
            raise RuntimeError(f"WBCIC subject/session/class cell has fewer than 20 trials at {context}")
    return {
        key: np.asarray(embedded[key]).copy()
        for key in (*SOURCE_INDEX_KEYS, *OUTCOME_INDEX_KEYS)
    }


def assert_same_index_arrays(
    reference: Mapping[str, np.ndarray], current: Mapping[str, np.ndarray], *, context: str
) -> None:
    if set(reference) != set(current):
        raise RuntimeError(f"index-array key set changed at {context}")
    for key in reference:
        left = np.asarray(reference[key])
        right = np.asarray(current[key])
        if left.dtype.kind in {"O", "U", "S"} or right.dtype.kind in {"O", "U", "S"}:
            equal = np.array_equal(left.astype(str), right.astype(str))
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise RuntimeError(f"formal configuration index array differs for {key} at {context}")


def erase_direction(features: np.ndarray, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return value - ((value - center) @ vector)[:, None] * vector[None, :]


def openbmi_persistent_directions(
    features: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, count: int = 8
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    value = np.asarray(features, dtype=np.float64)
    subject_text = subjects.astype(str)
    center = value.mean(axis=0)
    ordered = subject_sort(np.unique(subject_text))
    means1 = np.stack([value[(subject_text == subject) & (sessions == 1)].mean(axis=0) for subject in ordered])
    means2 = np.stack([value[(subject_text == subject) & (sessions == 2)].mean(axis=0) for subject in ordered])
    subject_geometry = np.concatenate((means1 - center, means2 - center), axis=0)
    _, _, vt = np.linalg.svd(subject_geometry, full_matrices=False)
    pool = vt[: min(24, len(vt))].T
    rows: list[dict[str, float]] = []
    for index in range(pool.shape[1]):
        direction = pool[:, index]
        projection1 = (means1 - center) @ direction
        projection2 = (means2 - center) @ direction
        persistence = (
            0.0
            if np.std(projection1) < 1e-12 or np.std(projection2) < 1e-12
            else float(np.corrcoef(projection1, projection2)[0, 1])
        )
        geometry = float(np.sqrt(np.mean(np.square((value - center) @ direction))))
        rows.append({"pool_index": index, "persistence": persistence, "geometry_strength": geometry})
    order = sorted(
        range(len(rows)),
        key=lambda index: (-rows[index]["persistence"], -rows[index]["geometry_strength"], index),
    )[:count]
    return center.astype(np.float64), pool[:, order].astype(np.float64), [rows[index] for index in order]


def load_head(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required only to read the frozen head checkpoints") from error
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    return (
        state["head.weight"].detach().cpu().numpy().astype(np.float64),
        state["head.bias"].detach().cpu().numpy().astype(np.float64),
    )


def evaluation_guard(
    path: Path,
    *,
    dataset: str,
    expected_backbone: str,
    expected_method: str,
    expected_lambda: float,
    expected_fold: int,
    expected_seed: int,
    expected_outcome_subjects: Sequence[str],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("pass") is not True or payload.get("selection_frozen_before_outcome_evaluation") is not True:
        raise RuntimeError(f"historical evaluation guard failed: {path}")
    if (
        payload.get("backbone") != expected_backbone
        or payload.get("method") != expected_method
        or not np.isclose(float(payload.get("lambda")), expected_lambda, atol=0.0, rtol=0.0)
        or int(payload["fold"]) != expected_fold
        or int(payload["seed"]) != expected_seed
    ):
        raise RuntimeError(f"historical evaluation guard configuration mismatch: {path}")
    if tuple(subject_sort(payload.get("outcome_subjects", []))) != tuple(subject_sort(expected_outcome_subjects)):
        raise RuntimeError(f"historical evaluation outcome-subject scope mismatch: {path}")
    selection_path = path.parents[2] / "LAMBDA_SELECTION_FROZEN.json"
    if payload.get("selection_file_sha256") != sha256_file(selection_path):
        raise RuntimeError(f"historical evaluation is not linked to the frozen selection file: {path}")
    if payload.get("restricted_data_accessed") is not False:
        raise RuntimeError(f"restricted data flag is not false: {path}")
    if dataset == "OPENBMI_STRESS" and payload.get("WBCIC_accessed") is not False:
        raise RuntimeError(f"OpenBMI guard WBCIC-access flag is not false: {path}")
    if dataset == "WBCIC_REPLICATION" and payload.get("sealed_WBCIC_outer_accessed") is not False:
        raise RuntimeError(f"WBCIC outer-access flag is not false: {path}")
    return payload


def direction_table_guard(
    table: pd.DataFrame,
    *,
    dataset: str,
    backbone: str,
    fold: int,
    seed: int,
    method: str,
    lam: float,
) -> None:
    required = {
        "backbone",
        "fold",
        "seed",
        "method",
        "lambda",
        "direction_id",
        "direction_source_only",
        "outcome_used_to_define_direction",
        "D_finite_definition",
        "identity_score",
        "identity_full",
        "identity_erased",
        "outcome_subject_count",
        "rank",
        "source_pool_index",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"historical direction-table columns missing: {missing}")
    if (
        set(table.backbone.astype(str)) != {backbone}
        or set(map(int, table.fold)) != {fold}
        or set(map(int, table.seed)) != {seed}
        or set(table.method.astype(str)) != {method}
        or not np.allclose(table["lambda"].to_numpy(np.float64), lam, atol=0.0, rtol=0.0)
    ):
        raise RuntimeError(f"historical direction-table configuration mismatch: {dataset} f{fold} s{seed}")
    if not table.direction_source_only.map(lambda value: str(value).lower() == "true").all():
        raise RuntimeError(f"direction source-purity flag failed: {dataset} f{fold} s{seed}")
    if not table.outcome_used_to_define_direction.map(lambda value: str(value).lower() == "false").all():
        raise RuntimeError(f"outcome-defined direction entered analysis: {dataset} f{fold} s{seed}")
    if set(table.D_finite_definition.astype(str)) != {"exact_exp3_centered_logit_RMS"}:
        raise RuntimeError(f"D_finite definition changed: {dataset} f{fold} s{seed}")
    if not np.allclose(
        table.identity_score.to_numpy(np.float64),
        table.identity_full.to_numpy(np.float64) - table.identity_erased.to_numpy(np.float64),
        atol=1e-10,
        rtol=1e-9,
    ):
        raise RuntimeError(f"identity-score arithmetic changed: {dataset} f{fold} s{seed}")
    if set(map(int, table["rank"])) != {1}:
        raise RuntimeError(f"one-direction intervention rank changed: {dataset} f{fold} s{seed}")
    if dataset == "WBCIC_REPLICATION":
        if "direction_rank" not in table or list(map(int, table.sort_values("direction_id").direction_rank)) != list(range(1, 9)):
            raise RuntimeError(f"WBCIC direction ranks changed: fold={fold} seed={seed}")
        if list(map(int, table.sort_values("direction_id").source_pool_index)) != list(range(8)):
            raise RuntimeError(f"WBCIC source-pool indices changed: fold={fold} seed={seed}")


def wbcic_source_freeze_guard(
    path: Path,
    *,
    roles: Mapping[str, Sequence[str]],
    backbone: str,
    fold: int,
    seed: int,
    basis_path: Path,
    checkpoint_path: Path,
    center: np.ndarray,
    basis: np.ndarray,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "WBCIC_REPLICATION_RUN_SOURCE_FREEZE_V1":
        raise RuntimeError(f"WBCIC source-freeze schema changed: {path}")
    if payload.get("pass") is not True:
        raise RuntimeError(f"WBCIC source freeze did not pass: {path}")
    if payload.get("backbone") != backbone or int(payload.get("fold", -1)) != fold or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"WBCIC source-freeze configuration mismatch: {path}")
    if tuple(subject_sort(payload.get("model_fit_subjects", []))) != tuple(roles["model_fit"]):
        raise RuntimeError(f"WBCIC source-freeze model-fit subjects changed: {path}")
    if tuple(subject_sort(payload.get("validation_discovery_subjects", []))) != tuple(
        roles["validation_discovery"]
    ):
        raise RuntimeError(f"WBCIC source-freeze validation subjects changed: {path}")
    required_false = [
        "outcome_S3_labels_used",
        "sealed_WBCIC_outer_accessed",
        "OpenBMI_holdout_accessed",
    ]
    if payload.get("outcome_subjects_not_loaded") is not True or any(
        payload.get(key) is not False for key in required_false
    ):
        raise RuntimeError(f"WBCIC source-freeze purity flags failed: {path}")
    if int(payload.get("direction_count", -1)) != 8:
        raise RuntimeError(f"WBCIC source-freeze direction count changed: {path}")
    if payload.get("persistence_basis_file_sha256") != sha256_file(basis_path):
        raise RuntimeError(f"WBCIC source-freeze basis-file hash mismatch: {path}")
    if payload.get("persistence_basis_array_sha256") != sha256_array(basis):
        raise RuntimeError(f"WBCIC source-freeze basis-array hash mismatch: {path}")
    if payload.get("persistence_center_array_sha256") != sha256_array(center):
        raise RuntimeError(f"WBCIC source-freeze center-array hash mismatch: {path}")
    selection_path = path.parent / "LAMBDA_SELECTION_FROZEN.json"
    if payload.get("selection_file_sha256") != sha256_file(selection_path):
        raise RuntimeError(f"WBCIC source-freeze selection-file hash mismatch: {path}")
    checkpoints = payload.get("checkpoints", [])
    if int(payload.get("checkpoint_count", -1)) != len(CONFIGURATIONS) or len(checkpoints) != len(CONFIGURATIONS):
        raise RuntimeError(f"WBCIC source-freeze checkpoint ledger cardinality changed: {path}")
    try:
        checkpoint_configurations = [
            (str(row.get("method")), float(row.get("lambda"))) for row in checkpoints
        ]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"WBCIC source-freeze checkpoint ledger is malformed: {path}") from error
    if len(set(checkpoint_configurations)) != len(CONFIGURATIONS) or set(
        checkpoint_configurations
    ) != set(CONFIGURATIONS):
        raise RuntimeError(f"WBCIC source-freeze checkpoint configuration grid changed: {path}")
    for row in checkpoints:
        digest = str(row.get("checkpoint_sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"WBCIC source-freeze checkpoint hash is malformed: {path}")
    erm = [
        row for row in checkpoints
        if row.get("method") == "ERM" and np.isclose(float(row.get("lambda")), 0.0, atol=0.0, rtol=0.0)
    ]
    if len(erm) != 1 or erm[0].get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise RuntimeError(f"WBCIC source-freeze ERM checkpoint hash mismatch: {path}")
    return {"status": "PASS", "checkpoint_count": len(checkpoints), "direction_count": 8}


def effect_rows(
    *,
    dataset: str,
    backbone: str,
    fold: int,
    seed: int,
    method: str,
    lam: float,
    direction_table: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    weight: np.ndarray,
    bias: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    legacy_uses_stored_clean_logits: bool,
    legacy_d_uses_stored_source_logits: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(direction_table) != 8:
        raise RuntimeError(f"direction cardinality changed: {dataset} f{fold} s{seed} {method} {lam}")
    table = direction_table.sort_values("direction_id").reset_index(drop=True)
    if list(map(int, table.direction_id)) != list(range(8)):
        raise RuntimeError(f"direction ids changed: {dataset} f{fold} s{seed} {method} {lam}")
    if np.asarray(basis).ndim != 2 or np.asarray(basis).shape[1] != 8:
        raise RuntimeError(f"direction basis shape changed: {dataset} f{fold} s{seed} {method} {lam}")
    features = np.asarray(embeddings["outcome_features"], dtype=np.float64)
    labels = np.asarray(embeddings["outcome_labels"], dtype=np.int64)
    subjects = np.asarray(embeddings["outcome_subjects"]).astype(str)
    source_features = np.asarray(embeddings["source_features"], dtype=np.float64)
    if features.ndim != 2 or source_features.ndim != 2 or features.shape[1] != source_features.shape[1]:
        raise RuntimeError(f"source/outcome feature dimensions changed: {dataset} f{fold} s{seed}")
    feature_dimension = features.shape[1]
    if (
        weight.shape != (2, feature_dimension)
        or bias.shape != (2,)
        or not np.all(np.isfinite(weight))
        or not np.all(np.isfinite(bias))
    ):
        raise RuntimeError(f"frozen head shape changed: {dataset} {backbone} f{fold} s{seed}")
    if (
        np.asarray(center).shape != (feature_dimension,)
        or np.asarray(basis).shape != (feature_dimension, 8)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(basis))
    ):
        raise RuntimeError(f"center/basis feature dimension changed: {dataset} {backbone} f{fold} s{seed}")
    outcome_subject_count = len(np.unique(subjects))
    if set(map(int, table.outcome_subject_count)) != {outcome_subject_count}:
        raise RuntimeError(f"direction-table outcome-subject count changed: {dataset} f{fold} s{seed}")
    corrected_clean_logits = features @ weight.T + bias
    corrected_clean_source_logits = source_features @ weight.T + bias
    legacy_clean_logits = (
        np.asarray(embeddings["outcome_logits"], dtype=np.float64)
        if legacy_uses_stored_clean_logits
        else corrected_clean_logits
    )
    legacy_clean_source_logits = (
        np.asarray(embeddings["source_logits"], dtype=np.float64)
        if legacy_d_uses_stored_source_logits
        else corrected_clean_source_logits
    )
    corrected_clean_ce = numpy_cross_entropy(corrected_clean_logits, labels)
    legacy_clean_ce = numpy_cross_entropy(legacy_clean_logits, labels)
    rows: list[dict[str, Any]] = []
    maximum_aggregate_difference = 0.0
    maximum_numerical_repair_difference = 0.0
    maximum_historical_d_difference = 0.0
    maximum_d_numerical_repair_difference = 0.0
    for direction_position, direction_row in enumerate(table.itertuples(index=False)):
        direction = basis[:, direction_position]
        erased = erase_direction(features, center, direction)
        erased_logits = erased @ weight.T + bias
        erased_source = erase_direction(source_features, center, direction)
        erased_source_logits = erased_source @ weight.T + bias
        erased_ce = numpy_cross_entropy(erased_logits, labels)
        legacy_trial_effect = erased_ce - legacy_clean_ce
        corrected_trial_effect = erased_ce - corrected_clean_ce
        historical = float(direction_row.outcome_CE_effect)
        legacy_reconstructed = float(np.mean(legacy_trial_effect))
        corrected_reconstructed = float(np.mean(corrected_trial_effect))
        maximum_aggregate_difference = max(maximum_aggregate_difference, abs(legacy_reconstructed - historical))
        maximum_numerical_repair_difference = max(
            maximum_numerical_repair_difference, abs(corrected_reconstructed - legacy_reconstructed)
        )
        if not np.isclose(legacy_reconstructed, historical, atol=AGGREGATE_ATOL, rtol=AGGREGATE_RTOL):
            raise RuntimeError(
                f"historical aggregate mismatch {dataset} {backbone} f{fold} s{seed} "
                f"{method} {lam} d{direction_position}: reconstructed={legacy_reconstructed} historical={historical}"
            )
        historical_d = float(direction_row.D_finite)
        legacy_d = exact_d_finite(legacy_clean_source_logits, erased_source_logits)
        corrected_d = exact_d_finite(corrected_clean_source_logits, erased_source_logits)
        maximum_historical_d_difference = max(maximum_historical_d_difference, abs(legacy_d - historical_d))
        maximum_d_numerical_repair_difference = max(
            maximum_d_numerical_repair_difference, abs(corrected_d - legacy_d)
        )
        if not np.isclose(legacy_d, historical_d, atol=AGGREGATE_ATOL, rtol=AGGREGATE_RTOL):
            raise RuntimeError(
                f"historical D_finite mismatch {dataset} {backbone} f{fold} s{seed} "
                f"{method} {lam} d{direction_position}: reconstructed={legacy_d} historical={historical_d}"
            )
        rank_feature = float(
            getattr(direction_row, "rank")
            if dataset == "OPENBMI_STRESS"
            else getattr(direction_row, "direction_rank")
        )
        for subject in subject_sort(np.unique(subjects)):
            mask = subjects == subject
            rows.append(
                {
                    "dataset": dataset,
                    "subject_id_internal": subject,
                    "backbone": backbone,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "lambda": lam,
                    "direction_id": int(direction_row.direction_id),
                    "persistence": float(direction_row.persistence),
                    "geometry_strength": float(direction_row.geometry_strength),
                    "rank_feature": rank_feature,
                    "identity_score": float(direction_row.identity_score),
                    "D_finite": corrected_d,
                    "subject_CE_effect": float(np.mean(corrected_trial_effect[mask])),
                    "subject_trial_count": int(np.sum(mask)),
                }
            )
    return rows, {
        "max_historical_aggregate_abs_difference": maximum_aggregate_difference,
        "max_legacy_to_float64_aggregate_abs_difference": maximum_numerical_repair_difference,
        "max_historical_D_finite_abs_difference": maximum_historical_d_difference,
        "max_legacy_to_float64_D_finite_abs_difference": maximum_d_numerical_repair_difference,
    }


def reconstruct_openbmi(
    stress_root: Path,
    manifest_hashes: Mapping[tuple[str, str], str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    maximum_aggregate_difference = 0.0
    maximum_numerical_repair_difference = 0.0
    maximum_historical_d_difference = 0.0
    maximum_d_numerical_repair_difference = 0.0
    reference_indices: dict[int, dict[str, np.ndarray]] = {}
    reference_roles: dict[int, dict[str, tuple[str, ...]]] = {}
    frozen_roles = global_protocol_roles(
        stress_root / "STRESS_TEST_PROTOCOL_FROZEN.json", dataset="OPENBMI_STRESS"
    )
    for backbone in ("eegnet", "eegconformer"):
        for fold in range(5):
            for seed in range(3):
                unit = stress_root / "runtime" / "runs" / backbone / f"fold-{fold}" / f"seed-{seed}"
                roles = unit_protocol_guard(
                    unit / "UNIT_PROTOCOL.json",
                    dataset="OPENBMI_STRESS",
                    expected_backbone=backbone,
                    expected_fold=fold,
                    expected_seed=seed,
                )
                if roles != frozen_roles[fold]:
                    raise RuntimeError(f"OpenBMI unit roles differ from the hashed global protocol for fold {fold}")
                if fold in reference_roles and roles != reference_roles[fold]:
                    raise RuntimeError(f"OpenBMI frozen roles differ across backbone/seed for fold {fold}")
                reference_roles.setdefault(fold, roles)
                for method, lam in CONFIGURATIONS:
                    slug = config_slug(method, lam)
                    evaluation = unit / "evaluation" / slug
                    evaluation_metadata = evaluation_guard(
                        evaluation / "EVALUATION_COMPLETE.json",
                        dataset="OPENBMI_STRESS",
                        expected_backbone=backbone,
                        expected_method=method,
                        expected_lambda=lam,
                        expected_fold=fold,
                        expected_seed=seed,
                        expected_outcome_subjects=roles["outcome"],
                    )
                    table = pd.read_csv(evaluation / "directions.csv")
                    direction_table_guard(
                        table,
                        dataset="OPENBMI_STRESS",
                        backbone=backbone,
                        fold=fold,
                        seed=seed,
                        method=method,
                        lam=lam,
                    )
                    # Historical subject arrays were saved by pandas with object dtype.
                    # The archive is allowed to unpickle only after exact manifest-hash verification.
                    with load_manifest_verified_npz(
                        evaluation / "embeddings.npz",
                        source_alias="OPENBMI_STRESS_HISTORICAL",
                        source_root=stress_root,
                        manifest_hashes=manifest_hashes,
                        allow_pickle=True,
                    ) as embedded:
                        arrays = embedding_partition_guard(
                            embedded,
                            dataset="OPENBMI_STRESS",
                            roles=roles,
                            context=f"OpenBMI/{backbone}/fold-{fold}/seed-{seed}/{slug}",
                        )
                        if int(evaluation_metadata.get("source_rows", -1)) != len(embedded["source_indices"]):
                            raise RuntimeError("OpenBMI evaluation guard source-row count mismatch")
                        if int(evaluation_metadata.get("outcome_rows", -1)) != len(embedded["outcome_indices"]):
                            raise RuntimeError("OpenBMI evaluation guard outcome-row count mismatch")
                        if fold in reference_indices:
                            assert_same_index_arrays(
                                reference_indices[fold],
                                arrays,
                                context=f"OpenBMI/{backbone}/fold-{fold}/seed-{seed}/{slug}",
                            )
                        else:
                            reference_indices[fold] = arrays
                        center, basis, meta = openbmi_persistent_directions(
                            embedded["source_features"],
                            embedded["source_subjects"],
                            embedded["source_sessions"],
                            count=8,
                        )
                        ordered = table.sort_values("direction_id").reset_index(drop=True)
                        for position, row in enumerate(ordered.itertuples(index=False)):
                            expected = meta[position]
                            if int(row.source_pool_index) != int(expected["pool_index"]):
                                raise RuntimeError("OpenBMI direction pool index reconstruction mismatch")
                            if not np.isclose(float(row.persistence), expected["persistence"], atol=1e-8, rtol=1e-7):
                                raise RuntimeError("OpenBMI direction persistence reconstruction mismatch")
                            if not np.isclose(
                                float(row.geometry_strength), expected["geometry_strength"], atol=1e-8, rtol=1e-7
                            ):
                                raise RuntimeError("OpenBMI direction geometry reconstruction mismatch")
                        weight, bias = load_head(unit / "checkpoints" / f"{slug}.pt")
                        rows, integrity = effect_rows(
                            dataset="OPENBMI_STRESS",
                            backbone=backbone,
                            fold=fold,
                            seed=seed,
                            method=method,
                            lam=lam,
                            direction_table=table,
                            embeddings=embedded,
                            weight=weight,
                            bias=bias,
                            center=center,
                            basis=basis,
                            legacy_uses_stored_clean_logits=True,
                            legacy_d_uses_stored_source_logits=True,
                        )
                    all_rows.extend(rows)
                    maximum_aggregate_difference = max(
                        maximum_aggregate_difference, integrity["max_historical_aggregate_abs_difference"]
                    )
                    maximum_numerical_repair_difference = max(
                        maximum_numerical_repair_difference,
                        integrity["max_legacy_to_float64_aggregate_abs_difference"],
                    )
                    maximum_historical_d_difference = max(
                        maximum_historical_d_difference,
                        integrity["max_historical_D_finite_abs_difference"],
                    )
                    maximum_d_numerical_repair_difference = max(
                        maximum_d_numerical_repair_difference,
                        integrity["max_legacy_to_float64_D_finite_abs_difference"],
                    )
    outcome_subjects = [subject for fold in sorted(reference_roles) for subject in reference_roles[fold]["outcome"]]
    if len(outcome_subjects) != 40 or len(set(outcome_subjects)) != 40:
        raise RuntimeError("OpenBMI outcome folds are not a one-time partition of 40 biological subjects")
    return pd.DataFrame(all_rows), {
        "max_historical_aggregate_abs_difference": maximum_aggregate_difference,
        "max_legacy_to_float64_aggregate_abs_difference": maximum_numerical_repair_difference,
        "max_historical_D_finite_abs_difference": maximum_historical_d_difference,
        "max_legacy_to_float64_D_finite_abs_difference": maximum_d_numerical_repair_difference,
        "analysis_logit_semantics": "matched_float64_intact_and_erased",
        "D_finite_logit_semantics": "matched_float64_intact_and_erased_source_logits",
        "source_outcome_subject_disjoint": True,
        "formal_configuration_index_arrays_identical_within_fold": True,
        "outcome_fold_partition_subjects": 40,
    }


def reconstruct_wbcic(
    wbcic_root: Path,
    manifest_hashes: Mapping[tuple[str, str], str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    maximum_aggregate_difference = 0.0
    maximum_numerical_repair_difference = 0.0
    maximum_historical_d_difference = 0.0
    maximum_d_numerical_repair_difference = 0.0
    reference_indices: dict[int, dict[str, np.ndarray]] = {}
    reference_roles: dict[int, dict[str, tuple[str, ...]]] = {}
    frozen_roles = global_protocol_roles(
        wbcic_root / "WBCIC_REPLICATION_PROTOCOL_FROZEN.json", dataset="WBCIC_REPLICATION"
    )
    method, lam, backbone = "ERM", 0.0, "eegnet"
    slug = config_slug(method, lam)
    for fold in range(5):
        for seed in range(3):
            unit = wbcic_root / "runtime" / "runs" / backbone / f"fold-{fold}" / f"seed-{seed}"
            evaluation = unit / "evaluation" / slug
            roles = unit_protocol_guard(
                unit / "UNIT_PROTOCOL.json",
                dataset="WBCIC_REPLICATION",
                expected_backbone=backbone,
                expected_fold=fold,
                expected_seed=seed,
            )
            if roles != frozen_roles[fold]:
                raise RuntimeError(f"WBCIC unit roles differ from the hashed global protocol for fold {fold}")
            if fold in reference_roles and roles != reference_roles[fold]:
                raise RuntimeError(f"WBCIC frozen roles differ across seeds for fold {fold}")
            reference_roles.setdefault(fold, roles)
            evaluation_metadata = evaluation_guard(
                evaluation / "EVALUATION_COMPLETE.json",
                dataset="WBCIC_REPLICATION",
                expected_backbone=backbone,
                expected_method=method,
                expected_lambda=lam,
                expected_fold=fold,
                expected_seed=seed,
                expected_outcome_subjects=roles["outcome"],
            )
            table = pd.read_csv(evaluation / "directions.csv")
            direction_table_guard(
                table,
                dataset="WBCIC_REPLICATION",
                backbone=backbone,
                fold=fold,
                seed=seed,
                method=method,
                lam=lam,
            )
            basis_path = unit / "source_freeze" / "erm_persistence_basis.npz"
            checkpoint_path = unit / "checkpoints" / f"{slug}.pt"
            with load_manifest_verified_npz(
                basis_path,
                source_alias="WBCIC_REPLICATION_HISTORICAL",
                source_root=wbcic_root,
                manifest_hashes=manifest_hashes,
                allow_pickle=False,
            ) as frozen:
                if set(frozen.files) != {
                    "center",
                    "basis",
                    "direction_meta_json",
                    "model_fit_subjects",
                    "sessions",
                }:
                    raise RuntimeError(f"WBCIC frozen basis key set changed: fold={fold} seed={seed}")
                center = np.asarray(frozen["center"], dtype=np.float64)
                basis = np.asarray(frozen["basis"], dtype=np.float64)
                direction_meta = json.loads(str(frozen["direction_meta_json"].item()))
                if tuple(subject_sort(frozen["model_fit_subjects"].astype(str))) != tuple(roles["model_fit"]):
                    raise RuntimeError(f"WBCIC frozen basis subject scope changed: fold={fold} seed={seed}")
                if list(map(int, frozen["sessions"])) != [0, 1]:
                    raise RuntimeError(f"WBCIC frozen basis session scope changed: fold={fold} seed={seed}")
            if basis.shape[1] != 8:
                raise RuntimeError(f"WBCIC frozen direction count changed: {basis.shape}")
            if np.linalg.norm(basis.T @ basis - np.eye(8), ord="fro") > 1e-8:
                raise RuntimeError(f"WBCIC frozen basis is not orthonormal: fold={fold} seed={seed}")
            ordered = table.sort_values("direction_id").reset_index(drop=True)
            if len(direction_meta) != 8 or list(map(int, ordered.direction_id)) != list(range(8)):
                raise RuntimeError(f"WBCIC direction metadata cardinality changed: fold={fold} seed={seed}")
            for position, row in enumerate(ordered.itertuples(index=False)):
                meta = direction_meta[position]
                if int(row.direction_rank) != position + 1 or int(row.source_pool_index) != int(meta["pool_index"]):
                    raise RuntimeError(f"WBCIC basis/table direction order mismatch: fold={fold} seed={seed}")
                if not np.isclose(float(row.persistence), float(meta["persistence"]), atol=1e-8, rtol=1e-7):
                    raise RuntimeError(f"WBCIC basis/table persistence mismatch: fold={fold} seed={seed}")
                if not np.isclose(
                    float(row.geometry_strength), float(meta["geometry_strength"]), atol=1e-8, rtol=1e-7
                ):
                    raise RuntimeError(f"WBCIC basis/table geometry mismatch: fold={fold} seed={seed}")
            wbcic_source_freeze_guard(
                unit / "SOURCE_FREEZE_COMPLETE.json",
                roles=roles,
                backbone=backbone,
                fold=fold,
                seed=seed,
                basis_path=basis_path,
                checkpoint_path=checkpoint_path,
                center=center,
                basis=basis,
            )
            weight, bias = load_head(checkpoint_path)
            # Historical subject arrays were saved by pandas with object dtype.
            # The archive is allowed to unpickle only after exact manifest-hash verification.
            with load_manifest_verified_npz(
                evaluation / "embeddings.npz",
                source_alias="WBCIC_REPLICATION_HISTORICAL",
                source_root=wbcic_root,
                manifest_hashes=manifest_hashes,
                allow_pickle=True,
            ) as embedded:
                arrays = embedding_partition_guard(
                    embedded,
                    dataset="WBCIC_REPLICATION",
                    roles=roles,
                    context=f"WBCIC/{backbone}/fold-{fold}/seed-{seed}/{slug}",
                )
                if int(evaluation_metadata.get("source_rows", -1)) != len(embedded["source_indices"]):
                    raise RuntimeError("WBCIC evaluation guard source-row count mismatch")
                if int(evaluation_metadata.get("outcome_rows", -1)) != len(embedded["outcome_indices"]):
                    raise RuntimeError("WBCIC evaluation guard outcome-row count mismatch")
                recomputed_center = np.asarray(embedded["source_features"], dtype=np.float64).mean(axis=0)
                if not np.allclose(center, recomputed_center, atol=1e-12, rtol=1e-12):
                    raise RuntimeError(f"WBCIC frozen basis center differs from ERM source mean: fold={fold} seed={seed}")
                if fold in reference_indices:
                    assert_same_index_arrays(
                        reference_indices[fold],
                        arrays,
                        context=f"WBCIC/{backbone}/fold-{fold}/seed-{seed}/{slug}",
                    )
                else:
                    reference_indices[fold] = arrays
                rows, integrity = effect_rows(
                    dataset="WBCIC_REPLICATION",
                    backbone=backbone,
                    fold=fold,
                    seed=seed,
                    method=method,
                    lam=lam,
                    direction_table=table,
                    embeddings=embedded,
                    weight=weight,
                    bias=bias,
                    center=center,
                    basis=basis,
                    legacy_uses_stored_clean_logits=False,
                    legacy_d_uses_stored_source_logits=False,
                )
            all_rows.extend(rows)
            maximum_aggregate_difference = max(
                maximum_aggregate_difference, integrity["max_historical_aggregate_abs_difference"]
            )
            maximum_numerical_repair_difference = max(
                maximum_numerical_repair_difference,
                integrity["max_legacy_to_float64_aggregate_abs_difference"],
            )
            maximum_historical_d_difference = max(
                maximum_historical_d_difference,
                integrity["max_historical_D_finite_abs_difference"],
            )
            maximum_d_numerical_repair_difference = max(
                maximum_d_numerical_repair_difference,
                integrity["max_legacy_to_float64_D_finite_abs_difference"],
            )
    outcome_subjects = [subject for fold in sorted(reference_roles) for subject in reference_roles[fold]["outcome"]]
    if len(outcome_subjects) != 41 or len(set(outcome_subjects)) != 41:
        raise RuntimeError("WBCIC outcome folds are not a one-time partition of 41 biological subjects")
    return pd.DataFrame(all_rows), {
        "max_historical_aggregate_abs_difference": maximum_aggregate_difference,
        "max_legacy_to_float64_aggregate_abs_difference": maximum_numerical_repair_difference,
        "max_historical_D_finite_abs_difference": maximum_historical_d_difference,
        "max_legacy_to_float64_D_finite_abs_difference": maximum_d_numerical_repair_difference,
        "analysis_logit_semantics": "matched_float64_intact_and_erased",
        "D_finite_logit_semantics": "matched_float64_intact_and_erased_source_logits",
        "source_outcome_subject_disjoint": True,
        "formal_configuration_index_arrays_identical_within_fold": True,
        "outcome_fold_partition_subjects": 41,
    }


def validate_reconstructed_grid(frame: pd.DataFrame) -> dict[str, Any]:
    """Reject offsetting omissions/duplicates before any fitted prediction is made."""

    key_columns = [*CELL_COLUMNS, "subject_id_internal"]
    required_columns = {
        *key_columns,
        *PREDICTOR_COLUMNS,
        "subject_CE_effect",
        "subject_trial_count",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"reconstructed table columns missing: {missing_columns}")
    if frame[list(required_columns)].isna().any().any():
        raise RuntimeError("reconstructed formal grid contains missing values")
    if frame.duplicated(key_columns).any():
        duplicate = frame.loc[frame.duplicated(key_columns, keep=False), key_columns].head().to_dict("records")
        raise RuntimeError(f"duplicate reconstructed formal cells: {duplicate}")
    numeric = frame[[*PREDICTOR_COLUMNS, "subject_CE_effect", "subject_trial_count"]].to_numpy(np.float64)
    if not np.all(np.isfinite(numeric)) or np.any(frame.subject_trial_count.to_numpy(np.int64) <= 0):
        raise RuntimeError("reconstructed grid contains non-finite values or nonpositive trial counts")

    expected_dataset = {
        "OPENBMI_STRESS": {
            "subjects": 40,
            "backbones": ("eegnet", "eegconformer"),
            "configurations": CONFIGURATIONS,
            "rows": 19_200,
            "fold_sizes": [8, 8, 8, 8, 8],
        },
        "WBCIC_REPLICATION": {
            "subjects": 41,
            "backbones": ("eegnet",),
            "configurations": (("ERM", 0.0),),
            "rows": 984,
            "fold_sizes": [8, 8, 8, 8, 9],
        },
    }
    details: dict[str, Any] = {}
    for dataset, expected in expected_dataset.items():
        part = frame[frame.dataset == dataset].copy()
        if len(part) != expected["rows"]:
            raise RuntimeError(f"{dataset} reconstructed rows {len(part)} != {expected['rows']}")
        subjects = subject_sort(part.subject_id_internal.unique())
        if len(subjects) != expected["subjects"]:
            raise RuntimeError(f"{dataset} reconstructed subject count changed")
        subject_folds = part.groupby("subject_id_internal").fold.nunique()
        if not (subject_folds == 1).all():
            raise RuntimeError(f"{dataset} subject appears in multiple outcome folds")
        fold_subjects = {
            int(fold): subject_sort(group.subject_id_internal.unique())
            for fold, group in part.groupby("fold")
        }
        if sorted(map(len, fold_subjects.values())) != expected["fold_sizes"]:
            raise RuntimeError(f"{dataset} outcome-fold subject cardinalities changed")
        observed_configurations = {
            (str(method), round(float(lam), 12))
            for method, lam in part[["method", "lambda"]].drop_duplicates().itertuples(index=False, name=None)
        }
        expected_configurations = {
            (method, round(float(lam), 12)) for method, lam in expected["configurations"]
        }
        if observed_configurations != expected_configurations:
            raise RuntimeError(f"{dataset} formal configuration set changed")
        if set(map(str, part.backbone.unique())) != set(expected["backbones"]):
            raise RuntimeError(f"{dataset} backbone scope changed")
        actual_keys = {
            (
                str(subject),
                str(backbone),
                int(fold),
                int(seed),
                str(method),
                round(float(lam), 12),
                int(direction),
            )
            for subject, backbone, fold, seed, method, lam, direction in part[
                ["subject_id_internal", "backbone", "fold", "seed", "method", "lambda", "direction_id"]
            ].itertuples(index=False, name=None)
        }
        expected_keys = {
            (subject, backbone, fold, seed, method, round(float(lam), 12), direction)
            for fold, fold_values in fold_subjects.items()
            for subject in fold_values
            for backbone in expected["backbones"]
            for seed in range(3)
            for method, lam in expected["configurations"]
            for direction in range(8)
        }
        if actual_keys != expected_keys:
            missing = list(expected_keys - actual_keys)[:3]
            extra = list(actual_keys - expected_keys)[:3]
            raise RuntimeError(f"{dataset} Cartesian grid mismatch; missing={missing} extra={extra}")
        predictor_nunique = part.groupby(CELL_COLUMNS)[PREDICTOR_COLUMNS].nunique(dropna=False)
        if not (predictor_nunique == 1).all().all():
            raise RuntimeError(f"{dataset} predictor values vary across subjects within a frozen cell")
        trial_nunique = part.groupby(
            ["subject_id_internal", "backbone", "fold", "seed"]
        ).subject_trial_count.nunique()
        if not (trial_nunique == 1).all():
            raise RuntimeError(f"{dataset} subject trial counts vary across formal configurations")
        details[dataset] = {
            "rows": int(len(part)),
            "biological_subjects": len(subjects),
            "fold_subject_counts": {str(fold): len(values) for fold, values in sorted(fold_subjects.items())},
            "formal_configurations": len(expected_configurations),
            "directions_per_configuration": 8,
            "seeds": 3,
            "backbones": list(expected["backbones"]),
            "duplicate_cells": 0,
            "missing_cartesian_cells": 0,
            "predictors_constant_within_cell": True,
            "subject_trial_counts_constant_across_configurations": True,
        }
    if set(frame.dataset.unique()) != set(expected_dataset):
        raise RuntimeError(f"unexpected reconstructed datasets: {sorted(frame.dataset.unique())}")
    return {"status": "PASS", "datasets": details}


def ridge_operator(
    x_train: np.ndarray,
    x_test: np.ndarray,
    sample_weight: np.ndarray | None = None,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    weight = (
        np.ones(len(x_train), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weight.shape != (len(x_train),) or np.any(weight <= 0) or not np.all(np.isfinite(weight)):
        raise RuntimeError("invalid ridge sample weights")
    weight = weight / weight.mean()
    mean = np.average(x_train, axis=0, weights=weight)
    std = np.sqrt(np.average(np.square(x_train - mean), axis=0, weights=weight))
    std[std < STANDARDIZATION_MIN_STD] = 1.0
    train = np.c_[(x_train - mean) / std, np.ones(len(x_train))]
    test = np.c_[(x_test - mean) / std, np.ones(len(x_test))]
    penalty = np.eye(train.shape[1])
    penalty[-1, -1] = 0.0
    lhs = train.T @ (weight[:, None] * train) + alpha * penalty
    try:
        inverse_rhs = np.linalg.solve(lhs, train.T * weight[None, :])
    except np.linalg.LinAlgError:
        inverse_rhs = np.linalg.pinv(lhs) @ (train.T * weight[None, :])
    return test @ inverse_rhs


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sample_weight: np.ndarray | None = None,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    return ridge_operator(x_train, x_test, sample_weight=sample_weight, alpha=alpha) @ np.asarray(
        y_train, dtype=np.float64
    )


def analysis_filter(frame: pd.DataFrame, spec: AnalysisSpec) -> pd.DataFrame:
    return frame[(frame.dataset == spec.dataset) & frame.method.isin(spec.methods)].copy()


def training_cell_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode in {"UNIFORM", "EQUAL_CONFIGURATION"}:
        return np.ones(len(frame), dtype=np.float64)
    if mode != "EQUAL_METHOD_FAMILY":
        raise KeyError(f"unknown training weight mode: {mode}")
    configuration_count = (
        frame[["method", "lambda"]].drop_duplicates().groupby("method").size().to_dict()
    )
    return np.asarray([1.0 / float(configuration_count[str(method)]) for method in frame.method], dtype=np.float64)


def add_doubly_cross_fitted_predictions(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold one subject and one seed out within the clean fold and backbone."""

    predicted_parts: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    predictor_aggregation = {column: (column, "first") for column in PREDICTOR_COLUMNS}
    for spec in ANALYSES:
        scoped = analysis_filter(observations, spec)
        for fold in range(5):
            for backbone in sorted(scoped.backbone.unique()):
                block = scoped[(scoped.fold == fold) & (scoped.backbone == backbone)]
                if block.empty:
                    continue
                subjects = subject_sort(block.subject_id_internal.unique())
                seeds = sorted(map(int, block.seed.unique()))
                if seeds != [0, 1, 2]:
                    raise RuntimeError(f"seed scope changed for {spec.name} fold={fold} backbone={backbone}")
                for held_subject in subjects:
                    for held_seed in seeds:
                        training_observations = block[
                            (block.subject_id_internal.astype(str) != held_subject)
                            & (block.seed != held_seed)
                        ]
                        test = block[
                            (block.subject_id_internal.astype(str) == held_subject)
                            & (block.seed == held_seed)
                        ].copy()
                        if training_observations.empty or test.empty:
                            raise RuntimeError(
                                f"empty double-cross-fit split: {spec.name} fold={fold} "
                                f"backbone={backbone} subject={held_subject} seed={held_seed}"
                            )
                        train = training_observations.groupby(CELL_COLUMNS, as_index=False).agg(
                            **predictor_aggregation,
                            equal_peer_subject_CE_effect=("subject_CE_effect", "mean"),
                            peer_subject_count=("subject_id_internal", "nunique"),
                        )
                        if not (train.peer_subject_count == len(subjects) - 1).all():
                            raise RuntimeError("peer-subject training count changed within a split")
                        cell_weight = training_cell_weights(train, spec.training_weight_mode)
                        y_train = train.equal_peer_subject_CE_effect.to_numpy(np.float64)
                        for model, columns in MODELS.items():
                            test[f"prediction_{model}"] = ridge_predict(
                                train[list(columns)].to_numpy(np.float64),
                                y_train,
                                test[list(columns)].to_numpy(np.float64),
                                sample_weight=cell_weight,
                                alpha=RIDGE_ALPHA,
                            )
                            test[f"squared_error_{model}"] = np.square(
                                test[f"prediction_{model}"] - test.subject_CE_effect
                            )
                        test.insert(0, "analysis", spec.name)
                        test.insert(1, "analysis_role", spec.role)
                        test.insert(2, "subject_aggregation_mode", spec.subject_aggregation_mode)
                        test.insert(3, "training_weight_mode", spec.training_weight_mode)
                        predicted_parts.append(test)
                        split_rows.append(
                            {
                                "analysis": spec.name,
                                "dataset": spec.dataset,
                                "fold": fold,
                                "backbone": backbone,
                                "held_subject_internal": held_subject,
                                "held_seed": held_seed,
                                "peer_subject_count": len(subjects) - 1,
                                "training_direction_cells": len(train),
                                "test_direction_cells": len(test),
                                "other_historical_folds_used": False,
                                "other_backbones_used": False,
                                "held_subject_outcomes_used_for_fit": False,
                                "held_seed_outcomes_used_for_fit": False,
                            }
                        )
    predicted = pd.concat(predicted_parts, ignore_index=True)
    modeled_columns = [
        *(f"prediction_{model}" for model in MODELS),
        *(f"squared_error_{model}" for model in MODELS),
    ]
    if not np.all(np.isfinite(predicted[modeled_columns].to_numpy(np.float64))):
        raise RuntimeError("double-cross-fitted prediction table contains non-finite values")
    return predicted, pd.DataFrame(split_rows)


def pseudonymize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    mapping: dict[tuple[str, str], str] = {}
    for dataset in sorted(result.dataset.unique()):
        subjects = subject_sort(result.loc[result.dataset == dataset, "subject_id_internal"].unique())
        prefix = "OB" if dataset == "OPENBMI_STRESS" else "WB"
        mapping.update({(dataset, subject): f"{prefix}-S{index:03d}" for index, subject in enumerate(subjects, start=1)})
    result["subject_cluster"] = [mapping[(row.dataset, str(row.subject_id_internal))] for row in result.itertuples()]
    return result.drop(columns=["subject_id_internal"])


def subject_summaries(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregations = {
        f"RMSE_{model}": (f"squared_error_{model}", lambda values: float(np.sqrt(np.mean(values))))
        for model in MODELS
    }
    backbone_parts: list[pd.DataFrame] = []
    for spec in ANALYSES:
        part = observations[observations.analysis == spec.name]
        keys = ["analysis", "analysis_role", "dataset", "subject_cluster", "fold", "backbone"]
        if spec.subject_aggregation_mode == "EQUAL_METHOD_FAMILY":
            family = part.groupby([*keys, "method"], as_index=False).agg(**aggregations)
            by_backbone = family.groupby(keys, as_index=False).agg(
                **{f"RMSE_{model}": (f"RMSE_{model}", "mean") for model in MODELS},
                method_family_count=("method", "nunique"),
            )
        else:
            by_backbone = part.groupby(keys, as_index=False).agg(**aggregations)
            by_backbone["method_family_count"] = part.method.nunique()
        by_backbone["Delta_MI_minus_MD"] = by_backbone.RMSE_MI - by_backbone.RMSE_MD
        backbone_parts.append(by_backbone)
    backbone_summary = pd.concat(backbone_parts, ignore_index=True)
    summary = backbone_summary.groupby(
        ["analysis", "analysis_role", "dataset", "subject_cluster", "fold"], as_index=False
    ).agg(
        RMSE_M0=("RMSE_M0", "mean"),
        RMSE_MI=("RMSE_MI", "mean"),
        RMSE_MD=("RMSE_MD", "mean"),
        RMSE_MID=("RMSE_MID", "mean"),
        Delta_MI_minus_MD=("Delta_MI_minus_MD", "mean"),
        backbone_count=("backbone", "nunique"),
        repeated_backbone_rows=("backbone", "size"),
    )
    numeric_columns = [
        *(f"RMSE_{model}" for model in MODELS),
        "Delta_MI_minus_MD",
    ]
    if not np.all(np.isfinite(backbone_summary[numeric_columns].to_numpy(np.float64))) or not np.all(
        np.isfinite(summary[numeric_columns].to_numpy(np.float64))
    ):
        raise RuntimeError("subject summaries contain non-finite values")
    return (
        summary.sort_values(["analysis", "subject_cluster"]).reset_index(drop=True),
        backbone_summary.sort_values(["analysis", "backbone", "subject_cluster"]).reset_index(drop=True),
    )


def prepare_refit_blocks(observations: pd.DataFrame, spec: AnalysisSpec) -> list[dict[str, Any]]:
    scoped = analysis_filter(observations, spec)
    blocks: list[dict[str, Any]] = []
    cell_keys = ["method", "lambda", "direction_id"]
    for fold in range(5):
        for backbone in sorted(scoped.backbone.unique()):
            block_frame = scoped[(scoped.fold == fold) & (scoped.backbone == backbone)]
            if block_frame.empty:
                continue
            subjects = subject_sort(block_frame.subject_id_internal.unique())
            seeds = sorted(map(int, block_frame.seed.unique()))
            x_by_seed: dict[int, pd.DataFrame] = {}
            c_by_seed: dict[int, np.ndarray] = {}
            method_by_seed: dict[int, np.ndarray] = {}
            for seed in seeds:
                seed_frame = block_frame[block_frame.seed == seed]
                feature_table = (
                    seed_frame.groupby(cell_keys, as_index=False)
                    .agg(**{column: (column, "first") for column in PREDICTOR_COLUMNS})
                    .sort_values(cell_keys)
                    .reset_index(drop=True)
                )
                index = pd.MultiIndex.from_frame(feature_table[cell_keys])
                subject_effects: list[np.ndarray] = []
                for subject in subjects:
                    subject_frame = seed_frame[seed_frame.subject_id_internal.astype(str) == subject].set_index(cell_keys)
                    ordered = subject_frame.reindex(index)
                    if ordered.subject_CE_effect.isna().any():
                        raise RuntimeError(f"missing formal cell: {spec.name} f{fold} {backbone} s{seed} {subject}")
                    subject_effects.append(ordered.subject_CE_effect.to_numpy(np.float64))
                x_by_seed[seed] = feature_table
                c_by_seed[seed] = np.stack(subject_effects)
                method_by_seed[seed] = feature_table.method.astype(str).to_numpy()
            held_payload: dict[int, Any] = {}
            for held_seed in seeds:
                training_seeds = [seed for seed in seeds if seed != held_seed]
                train_frame = pd.concat([x_by_seed[seed] for seed in training_seeds], ignore_index=True)
                train_c = np.concatenate([c_by_seed[seed] for seed in training_seeds], axis=1)
                test_frame = x_by_seed[held_seed]
                contributions: dict[str, np.ndarray] = {}
                cell_weight = training_cell_weights(train_frame, spec.training_weight_mode)
                for model in ("MI", "MD"):
                    columns = MODELS[model]
                    operator = ridge_operator(
                        train_frame[list(columns)].to_numpy(np.float64),
                        test_frame[list(columns)].to_numpy(np.float64),
                        sample_weight=cell_weight,
                        alpha=RIDGE_ALPHA,
                    )
                    contributions[model] = train_c @ operator.T
                held_payload[held_seed] = {
                    "contributions": contributions,
                    "test_C": c_by_seed[held_seed],
                    "test_methods": method_by_seed[held_seed],
                }
            blocks.append(
                {
                    "analysis": spec.name,
                    "fold": fold,
                    "backbone": backbone,
                    "subjects": subjects,
                    "held": held_payload,
                    "aggregation_mode": spec.subject_aggregation_mode,
                }
            )
    return blocks


def refit_block_delta(block: Mapping[str, Any], multiplicity: np.ndarray) -> np.ndarray:
    multiplicity = np.asarray(multiplicity, dtype=np.float64)
    subject_count = len(block["subjects"])
    if multiplicity.shape != (subject_count,):
        raise RuntimeError("bootstrap multiplicity shape mismatch")
    total_weight = float(multiplicity.sum())
    denominator = total_weight - multiplicity
    if np.any((multiplicity > 0) & (denominator <= 0)):
        raise ZeroDivisionError("bootstrap draw has no peer subject for a held id")
    safe_denominator = np.where(denominator > 0, denominator, 1.0)
    if block["aggregation_mode"] == "EQUAL_METHOD_FAMILY":
        families = sorted({str(value) for payload in block["held"].values() for value in payload["test_methods"]})
    else:
        families = ["ALL"]
    squared_error = {
        model: np.zeros((subject_count, len(families)), dtype=np.float64) for model in ("MI", "MD")
    }
    cell_count = np.zeros(len(families), dtype=np.int64)
    for payload in block["held"].values():
        methods = payload["test_methods"]
        test_c = payload["test_C"]
        for family_index, family in enumerate(families):
            mask = np.ones(len(methods), dtype=bool) if family == "ALL" else methods == family
            cell_count[family_index] += int(np.sum(mask))
            for model in ("MI", "MD"):
                contribution = payload["contributions"][model]
                weighted_total = multiplicity @ contribution
                prediction = (
                    weighted_total[None, :] - multiplicity[:, None] * contribution
                ) / safe_denominator[:, None]
                squared_error[model][:, family_index] += np.square(prediction[:, mask] - test_c[:, mask]).sum(axis=1)
    rmse = {
        model: np.sqrt(squared_error[model] / cell_count[None, :]) for model in ("MI", "MD")
    }
    delta = (rmse["MI"] - rmse["MD"]).mean(axis=1)
    if not np.all(np.isfinite(delta)):
        raise RuntimeError("bootstrap ridge refit produced a non-finite subject delta")
    return delta


def refitted_subject_bootstrap(
    observations: pd.DataFrame,
    spec: AnalysisSpec,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    blocks = prepare_refit_blocks(observations, spec)
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for block in blocks:
        by_fold.setdefault(int(block["fold"]), []).append(block)
    rng = np.random.default_rng(seed)
    overall = np.empty(draws, dtype=np.float64)
    backbones = sorted({str(block["backbone"]) for block in blocks})
    by_backbone = {backbone: np.empty(draws, dtype=np.float64) for backbone in backbones}
    accepted = 0
    rejected = 0
    while accepted < draws:
        fold_weights: dict[int, np.ndarray] = {}
        valid = True
        for fold, fold_blocks in by_fold.items():
            count = len(fold_blocks[0]["subjects"])
            weight = rng.multinomial(count, np.full(count, 1.0 / count))
            if np.any((weight > 0) & ((weight.sum() - weight) <= 0)):
                valid = False
                break
            fold_weights[fold] = weight.astype(np.float64)
        if not valid:
            rejected += 1
            continue
        total_weight = 0.0
        overall_weighted_sum = 0.0
        backbone_weighted_sum = {backbone: 0.0 for backbone in backbones}
        backbone_total_weight = {backbone: 0.0 for backbone in backbones}
        for fold, fold_blocks in by_fold.items():
            weight = fold_weights[fold]
            reference_subjects = fold_blocks[0]["subjects"]
            delta_by_backbone: dict[str, np.ndarray] = {}
            for block in fold_blocks:
                if block["subjects"] != reference_subjects:
                    raise RuntimeError("OpenBMI backbone subject order differs inside a fold")
                delta = refit_block_delta(block, weight)
                backbone = str(block["backbone"])
                delta_by_backbone[backbone] = delta
                backbone_weighted_sum[backbone] += float(weight @ delta)
                backbone_total_weight[backbone] += float(weight.sum())
            subject_delta = np.stack(list(delta_by_backbone.values())).mean(axis=0)
            overall_weighted_sum += float(weight @ subject_delta)
            total_weight += float(weight.sum())
        overall[accepted] = overall_weighted_sum / total_weight
        for backbone in backbones:
            by_backbone[backbone][accepted] = (
                backbone_weighted_sum[backbone] / backbone_total_weight[backbone]
            )
        if not np.isfinite(overall[accepted]) or any(
            not np.isfinite(by_backbone[backbone][accepted]) for backbone in backbones
        ):
            raise RuntimeError("bootstrap produced a non-finite draw")
        accepted += 1
    uniform_points: dict[str, float] = {}
    uniform_overall_sum = 0.0
    uniform_total = 0.0
    uniform_backbone_sum = {backbone: 0.0 for backbone in backbones}
    uniform_backbone_total = {backbone: 0.0 for backbone in backbones}
    for fold, fold_blocks in by_fold.items():
        count = len(fold_blocks[0]["subjects"])
        weight = np.ones(count, dtype=np.float64)
        deltas = []
        for block in fold_blocks:
            delta = refit_block_delta(block, weight)
            deltas.append(delta)
            backbone = str(block["backbone"])
            uniform_backbone_sum[backbone] += float(delta.sum())
            uniform_backbone_total[backbone] += float(len(delta))
        uniform_overall_sum += float(np.stack(deltas).mean(axis=0).sum())
        uniform_total += float(count)
    uniform_points["OVERALL"] = uniform_overall_sum / uniform_total
    for backbone in backbones:
        uniform_points[backbone] = uniform_backbone_sum[backbone] / uniform_backbone_total[backbone]
    if not np.all(np.isfinite(overall)) or any(
        not np.all(np.isfinite(values)) for values in by_backbone.values()
    ) or not all(np.isfinite(value) for value in uniform_points.values()):
        raise RuntimeError("bootstrap output contains non-finite values")
    return {
        "draws": overall,
        "backbone_draws": by_backbone,
        "valid_draws": draws,
        "rejected_degenerate_draws": rejected,
        "uniform_refit_points": uniform_points,
    }


def verify_uniform_refit_points(
    subjects: pd.DataFrame,
    backbone_subjects: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    *,
    analysis: str,
) -> dict[str, Any]:
    explicit_overall = float(subjects.Delta_MI_minus_MD.mean())
    vectorized_overall = float(bootstrap["uniform_refit_points"]["OVERALL"])
    differences = {"OVERALL": abs(explicit_overall - vectorized_overall)}
    if differences["OVERALL"] > NUMERICAL_EQUIVALENCE_ATOL:
        raise RuntimeError(
            f"{analysis} uniform-refit overall point differs from explicit double cross-fit: "
            f"{vectorized_overall} versus {explicit_overall}"
        )
    for backbone, part in backbone_subjects.groupby("backbone"):
        explicit = float(part.Delta_MI_minus_MD.mean())
        vectorized = float(bootstrap["uniform_refit_points"][str(backbone)])
        differences[str(backbone)] = abs(explicit - vectorized)
        if differences[str(backbone)] > NUMERICAL_EQUIVALENCE_ATOL:
            raise RuntimeError(
                f"{analysis}/{backbone} uniform-refit point differs from explicit double cross-fit: "
                f"{vectorized} versus {explicit}"
            )
    return {
        "status": "PASS",
        "absolute_differences": differences,
        "tolerance": NUMERICAL_EQUIVALENCE_ATOL,
    }


def interval(values: np.ndarray) -> list[float]:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 1 or len(value) == 0 or not np.all(np.isfinite(value)):
        raise RuntimeError("confidence-interval input must be a nonempty finite vector")
    tail = CI_ALPHA / 2.0
    return [float(np.quantile(value, tail)), float(np.quantile(value, 1.0 - tail))]


def classify_dataset_gate(observed: float, ci95: Sequence[float]) -> str:
    value = float(observed)
    lower, upper = map(float, ci95)
    if not np.all(np.isfinite([value, lower, upper])) or lower > upper:
        raise RuntimeError("dataset gate received an invalid point estimate or interval")
    if value > 0 and upper < 0:
        return "POINT_CI_DIRECTION_CONFLICT"
    if value <= 0 and lower > 0:
        return "POINT_CI_DIRECTION_CONFLICT"
    if upper < 0:
        return "REVERSED"
    if value > 0 and lower > 0:
        return "SUPPORTED_CONDITIONAL"
    if lower <= 0 <= upper:
        return "PARTIAL" if value > 0 else "NOT_SUPPORTED"
    raise RuntimeError("dataset gate reached an uncovered point/interval relation")


def classify_cross_dataset_terminal(
    primary_gates: Sequence[str],
    *,
    points_positive: bool,
    openbmi_both_backbone_points_positive: bool,
) -> str:
    unknown = set(primary_gates) - set(DATASET_GATE_LABELS)
    if unknown:
        raise RuntimeError(f"unknown dataset gate entered cross-dataset terminal: {sorted(unknown)}")
    if "POINT_CI_DIRECTION_CONFLICT" in primary_gates:
        return "CROSS_DATASET_POINT_CI_DIRECTION_CONFLICT"
    if all(gate == "SUPPORTED_CONDITIONAL" for gate in primary_gates) and openbmi_both_backbone_points_positive:
        return "CROSS_DATASET_SUPPORTED_CONDITIONAL"
    if points_positive:
        return "CROSS_DATASET_PARTIAL"
    return "CROSS_DATASET_NOT_SUPPORTED"


def summarize_analysis(
    subjects: pd.DataFrame,
    backbone_subjects: pd.DataFrame,
    *,
    bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    delta = subjects.Delta_MI_minus_MD.to_numpy(np.float64)
    summary_numeric = subjects[[f"RMSE_{model}" for model in MODELS] + ["Delta_MI_minus_MD"]].to_numpy(
        np.float64
    )
    if len(delta) == 0 or not np.all(np.isfinite(summary_numeric)):
        raise RuntimeError("analysis summary input contains no subjects or non-finite values")
    observed = float(np.mean(delta))
    result: dict[str, Any] = {
        "biological_subject_count": int(len(delta)),
        "mean_subject_RMSE_M0": float(subjects.RMSE_M0.mean()),
        "mean_subject_RMSE_MI": float(subjects.RMSE_MI.mean()),
        "mean_subject_RMSE_MD": float(subjects.RMSE_MD.mean()),
        "mean_subject_RMSE_MID": float(subjects.RMSE_MID.mean()),
        "mean_subject_Delta_MI_minus_MD": observed,
        "median_subject_Delta_MI_minus_MD": float(np.median(delta)),
        "subjects_favoring_MD": int(np.sum(delta > 0)),
        "subjects_tied": int(np.sum(delta == 0)),
        "subjects_favoring_MI": int(np.sum(delta < 0)),
        "sign_flip_test": "NOT_PERFORMED_EXCHANGEABILITY_NOT_ESTABLISHED",
        "backbones": {},
    }
    for backbone, part in backbone_subjects.groupby("backbone"):
        result["backbones"][str(backbone)] = {
            "mean_subject_Delta_MI_minus_MD": float(part.Delta_MI_minus_MD.mean()),
            "subjects_favoring_MD": int((part.Delta_MI_minus_MD > 0).sum()),
            "biological_subject_count": int(part.subject_cluster.nunique()),
        }
    if bootstrap is None:
        result["refitted_subject_bootstrap_CI95"] = None
        result["bootstrap_draws"] = 0
        result["gate"] = "POINT_ONLY_MANDATORY_SENSITIVITY"
        return result
    ci95 = interval(np.asarray(bootstrap["draws"], dtype=np.float64))
    result["refitted_subject_bootstrap_CI95"] = ci95
    result["bootstrap_draws"] = int(bootstrap["valid_draws"])
    result["rejected_degenerate_bootstrap_draws"] = int(bootstrap["rejected_degenerate_draws"])
    gate = classify_dataset_gate(observed, ci95)
    result["gate"] = gate
    for backbone, draws in bootstrap["backbone_draws"].items():
        result["backbones"][str(backbone)]["refitted_subject_bootstrap_CI95"] = interval(
            np.asarray(draws, dtype=np.float64)
        )
    return result


def validate_scope(
    reconstructed: pd.DataFrame, observations: pd.DataFrame, subjects: pd.DataFrame
) -> dict[str, Any]:
    expected_subjects = {"OPENBMI_STRESS": 40, "WBCIC_REPLICATION": 41}
    binding = implementation_binding()
    expected_reconstructed_rows = binding["expected_reconstructed_rows"]
    expected_analysis_rows = binding["expected_analysis_rows"]
    details: dict[str, Any] = {}
    for dataset in expected_subjects:
        part = observations[observations.dataset == dataset]
        subject_part = subjects[subjects.dataset == dataset]
        reconstructed_part = reconstructed[reconstructed.dataset == dataset]
        if len(reconstructed_part) != expected_reconstructed_rows[dataset]:
            raise RuntimeError(
                f"{dataset} reconstructed rows {len(reconstructed_part)} != {expected_reconstructed_rows[dataset]}"
            )
        subject_counts = subject_part.groupby("analysis").subject_cluster.nunique()
        if subject_counts.empty or not (subject_counts == expected_subjects[dataset]).all():
            raise RuntimeError(f"{dataset} subject count changed")
        folds_per_subject = subject_part.groupby("subject_cluster").fold.nunique()
        if not (folds_per_subject == 1).all():
            raise RuntimeError(f"{dataset} subject appears in multiple outcome folds")
        if set(part.fold.unique()) != set(range(5)):
            raise RuntimeError(f"{dataset} fold set changed: {sorted(part.fold.unique())}")
        expected_backbones = 2 if dataset == "OPENBMI_STRESS" else 1
        if not (subject_part.backbone_count == expected_backbones).all():
            raise RuntimeError(f"{dataset} backbone aggregation changed")
        details[dataset] = {
            "reconstructed_rows": int(len(reconstructed_part)),
            "predicted_rows": int(len(part)),
            "biological_subjects": int(subject_part.subject_cluster.nunique()),
            "biological_subjects_by_analysis": {
                str(key): int(value) for key, value in subject_counts.to_dict().items()
            },
            "folds": sorted(map(int, part.fold.unique())),
            "backbones_per_subject": expected_backbones,
        }
    for analysis, expected in expected_analysis_rows.items():
        actual = int((observations.analysis == analysis).sum())
        if actual != expected:
            raise RuntimeError(f"{analysis} predicted rows {actual} != {expected}")
    if set(observations.analysis.unique()) != set(expected_analysis_rows):
        raise RuntimeError(f"unexpected analysis variants: {sorted(observations.analysis.unique())}")
    return {
        "status": "PASS",
        "datasets": details,
        "analysis_rows": {
            analysis: int((observations.analysis == analysis).sum())
            for analysis in expected_analysis_rows
        },
    }


def report_markdown(summary: Mapping[str, Any], lock: Mapping[str, Any]) -> str:
    lines = [
        "# Subject-level Decision Dependence versus Identity reanalysis",
        "",
        f"Protocol: `{PROTOCOL_ID}`  ",
        f"Pre-outcome lock commit: `{lock['lock_commit']}`  ",
        "Statistical unit: **biological subject**",
        "",
        "This report is the binding post-hoc subject-level validity repair. Seeds, folds, configurations,",
        "directions, trials, runs, and backbone rows are repeated measurements, not independent subjects.",
        "",
        "## Results",
        "",
        "| Dataset | Primary analysis | Subjects | mean RMSE I | mean RMSE D | mean subject Δ(I−D) | refitted subject-bootstrap 95% CI | D better (subjects) | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in ("OPENBMI_STRESS", "WBCIC_REPLICATION"):
        row = summary["datasets"][dataset]
        lines.append(
            f"| {dataset} | `{row['primary_analysis']}` | {row['biological_subject_count']} | "
            f"{row['mean_subject_RMSE_MI']:.8f} | "
            f"{row['mean_subject_RMSE_MD']:.8f} | {row['mean_subject_Delta_MI_minus_MD']:.8f} | "
            f"[{row['refitted_subject_bootstrap_CI95'][0]:.8f}, "
            f"{row['refitted_subject_bootstrap_CI95'][1]:.8f}] | "
            f"{row['subjects_favoring_MD']}/{row['biological_subject_count']} | "
            f"`{row['gate']}` |"
        )
    lines.extend(
        [
            "",
            f"Cross-dataset terminal: **`{summary['cross_dataset_terminal']}`**.",
            "",
            "Δ(I−D) is computed within each subject as RMSE(Identity model) minus RMSE(Decision model);",
            "positive values favor Decision Dependence. OpenBMI backbone-specific deltas are averaged",
            "within subject before any inference.",
            "",
            "The OpenBMI primary is the predeclared equal-family full-grid estimand; the WBCIC primary",
            "is the frozen ERM bank. They are two dataset-specific estimands, not an exactly matched",
            "cross-dataset intervention-bank replication. The OpenBMI ERM-only result is reported below",
            "as the mandatory direct-scope sensitivity.",
            "",
            "### Backbone wording gate",
            "",
            f"OpenBMI primary backbone point estimates are both positive: "
            f"**{str(summary['openbmi_both_backbone_points_positive']).upper()}**.",
            "The phrase 'across two OpenBMI backbones' is permitted only when this value is TRUE.",
            "",
            "### Mandatory analyses and sensitivities",
            "",
            "| Analysis | Role | Subjects | mean subject Δ(I−D) | 95% CI | Gate |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for analysis in [spec.name for spec in ANALYSES]:
        row = summary["analyses"][analysis]
        ci = row["refitted_subject_bootstrap_CI95"]
        ci_text = "NOT COMPUTED (point-only sensitivity)" if ci is None else f"[{ci[0]:.8f}, {ci[1]:.8f}]"
        lines.append(
            f"| `{analysis}` | `{row['analysis_role']}` | {row['biological_subject_count']} | "
            f"{row['mean_subject_Delta_MI_minus_MD']:.8f} | {ci_text} | `{row['gate']}` |"
        )
    lines.extend(
        [
            "",
            "## Scope and interpretation",
            "",
            "- This uses already-observed development outcomes only and is not an independent confirmation.",
            "- OpenBMI internal-14, the OpenBMI policy holdout, and WBCIC outer-10 were not accessed.",
            "- Exp3 remains exact but run-level/algorithmic because its per-subject runtime was not retained.",
            "- No sign-flip p-value was computed because exchangeability is not established for the overlapping",
            "  peer-subject fits.",
            "- `POINT_CI_DIRECTION_CONFLICT` is a predeclared conservative terminal for a point estimate and",
            "  percentile interval lying entirely in opposite directions; it permits no directional claim.",
            "- The narrow estimand is whether source-side Decision Dependence better predicts held-subject",
            "  intervention consequence for held algorithmic runs within the frozen intervention bank.",
            "- It does not establish a causal mechanism, unseen-intervention transfer, deployment utility,",
            "  learning-algorithm population performance, or that Decision Dependence guarantees utility.",
            "- The predeclared outcome sign is binding; no alternative feature, fold, alpha, or aggregation",
            "  was searched after reconstruction.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_split_audit(split_audit: pd.DataFrame) -> dict[str, Any]:
    expected_counts = implementation_binding()["expected_split_counts"]
    details: dict[str, Any] = {}
    for spec in ANALYSES:
        part = split_audit[split_audit.analysis == spec.name]
        if len(part) != expected_counts[spec.name]:
            raise RuntimeError(f"{spec.name} split count {len(part)} != {expected_counts[spec.name]}")
        configuration_count = sum(method in spec.methods for method, _ in CONFIGURATIONS)
        if spec.dataset == "WBCIC_REPLICATION":
            configuration_count = 1
        expected_training_cells = 2 * configuration_count * 8
        expected_test_cells = configuration_count * 8
        if set(map(int, part.training_direction_cells.unique())) != {expected_training_cells}:
            raise RuntimeError(f"{spec.name} training-cell scope changed")
        if set(map(int, part.test_direction_cells.unique())) != {expected_test_cells}:
            raise RuntimeError(f"{spec.name} test-cell scope changed")
        forbidden_true = [
            "other_historical_folds_used",
            "other_backbones_used",
            "held_subject_outcomes_used_for_fit",
            "held_seed_outcomes_used_for_fit",
        ]
        if any(bool(part[column].any()) for column in forbidden_true):
            raise RuntimeError(f"{spec.name} double-cross-fit purity flag failed")
        details[spec.name] = {
            "dataset": spec.dataset,
            "split_count": int(len(part)),
            "training_direction_cells": expected_training_cells,
            "test_direction_cells": expected_test_cells,
            "peer_subject_count_min": int(part.peer_subject_count.min()),
            "peer_subject_count_max": int(part.peer_subject_count.max()),
            "other_historical_folds_used": False,
            "other_backbones_used": False,
            "held_subject_outcomes_used_for_fit": False,
            "held_seed_outcomes_used_for_fit": False,
        }
    if set(split_audit.analysis.unique()) != set(expected_counts):
        raise RuntimeError(f"unexpected split-audit analyses: {sorted(split_audit.analysis.unique())}")
    return {"status": "PASS", "analyses": details}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    protocol_path = args.protocol.resolve()
    manifest_path = args.manifest.resolve()
    canonical_invocation = verify_canonical_invocation(
        repository,
        protocol=protocol_path,
        manifest=manifest_path,
        output_directory=args.output_dir,
    )
    implementation_path = Path(__file__).resolve()
    canonical = canonical_paths(repository)
    rationale_path = canonical["rationale"]
    test_path = canonical["test"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_binding = verify_protocol_binding(protocol)
    lock = verify_git_lock(
        repository,
        [protocol_path, manifest_path, implementation_path, rationale_path, test_path],
    )
    roots = SourceRoots(args.stress_root.resolve(), args.wbcic_root.resolve())
    manifest_result = verify_manifest(roots, manifest_path, include_hash_index=True)
    if not isinstance(manifest_result, tuple):
        raise RuntimeError("internal manifest verification did not return its hash index")
    manifest_verification, manifest_hashes = manifest_result
    output = canonical["output_directory"]
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite one-shot result directory: {output}")
    if staging.exists():
        raise RuntimeError(f"prior one-shot staging directory exists and requires audit: {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    write_json(
        staging / "EXECUTION_STATE.json",
        {
            "status": "STARTED",
            "protocol_id": PROTOCOL_ID,
            "pre_outcome_lock_commit": lock["lock_commit"],
            "scientific_outcomes_loaded_after_this_marker": True,
        },
    )
    try:
        openbmi, openbmi_integrity = reconstruct_openbmi(roots.stress, manifest_hashes)
        wbcic, wbcic_integrity = reconstruct_wbcic(roots.wbcic, manifest_hashes)
        reconstructed = pd.concat([openbmi, wbcic], ignore_index=True)
        reconstructed_grid = validate_reconstructed_grid(reconstructed)
        predicted, split_audit = add_doubly_cross_fitted_predictions(reconstructed)
        split_validation = validate_split_audit(split_audit)
        public_observations = pseudonymize(predicted)
        subject, backbone_subject = subject_summaries(public_observations)
        scope = validate_scope(reconstructed, public_observations, subject)

        analyses: dict[str, Any] = {}
        refit_equivalence: dict[str, Any] = {}
        bootstrap_seed = BOOTSTRAP_SEED
        bootstrap_draws = BOOTSTRAP_DRAWS
        for spec in ANALYSES:
            subject_part = subject[subject.analysis == spec.name]
            backbone_part = backbone_subject[backbone_subject.analysis == spec.name]
            bootstrap: Mapping[str, Any] | None = None
            if spec.name in BOOTSTRAP_ANALYSES:
                bootstrap = refitted_subject_bootstrap(
                    reconstructed,
                    spec,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                )
                refit_equivalence[spec.name] = verify_uniform_refit_points(
                    subject_part,
                    backbone_part,
                    bootstrap,
                    analysis=spec.name,
                )
            result = summarize_analysis(subject_part, backbone_part, bootstrap=bootstrap)
            result.update(
                {
                    "analysis": spec.name,
                    "analysis_role": spec.role,
                    "dataset": spec.dataset,
                    "training_weight_mode": spec.training_weight_mode,
                    "subject_aggregation_mode": spec.subject_aggregation_mode,
                }
            )
            analyses[spec.name] = result

        datasets: dict[str, Any] = {}
        for dataset, primary_name in PRIMARY_ANALYSES.items():
            primary = dict(analyses[primary_name])
            primary["primary_analysis"] = primary_name
            primary["mandatory_sensitivity_analyses"] = [
                spec.name for spec in ANALYSES if spec.dataset == dataset and spec.name != primary_name
            ]
            datasets[dataset] = primary

        openbmi_backbones = datasets["OPENBMI_STRESS"]["backbones"]
        if set(openbmi_backbones) != {"eegnet", "eegconformer"}:
            raise RuntimeError(f"OpenBMI primary backbone result set changed: {sorted(openbmi_backbones)}")
        openbmi_both_positive = all(
            row["mean_subject_Delta_MI_minus_MD"] > 0 for row in openbmi_backbones.values()
        )
        primary_gates = [datasets[dataset]["gate"] for dataset in PRIMARY_ANALYSES]
        points_positive = all(
            datasets[dataset]["mean_subject_Delta_MI_minus_MD"] > 0 for dataset in PRIMARY_ANALYSES
        )
        terminal = classify_cross_dataset_terminal(
            primary_gates,
            points_positive=points_positive,
            openbmi_both_backbone_points_positive=openbmi_both_positive,
        )
        summary = {
            "protocol_id": PROTOCOL_ID,
            "pre_outcome_lock_commit": lock["lock_commit"],
            "statistical_unit": "biological subject",
            "estimand_scope": (
                "held-subject intervention-consequence prediction for held algorithmic runs "
                "within the frozen intervention bank"
            ),
            "analyses": analyses,
            "datasets": datasets,
            "openbmi_both_backbone_points_positive": openbmi_both_positive,
            "cross_dataset_terminal": terminal,
            "cross_dataset_estimand_note": (
                "OpenBMI equal-family full-grid and WBCIC ERM are predeclared dataset-specific "
                "estimands; OpenBMI ERM-only is the mandatory direct-scope sensitivity."
            ),
            "sign_flip_test": "NOT_PERFORMED_EXCHANGEABILITY_NOT_ESTABLISHED",
            "exp3_status": "RUN_LEVEL_ALGORITHMIC_ONLY_SUBJECT_RECONSTRUCTION_UNAVAILABLE",
            "sealed_or_outer_accessed": False,
        }
        validation = {
            "status": "PASS",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": sha256_text(protocol_path),
            "implementation_sha256": sha256_text(implementation_path),
            "rationale_sha256": sha256_text(rationale_path),
            "test_sha256": sha256_text(test_path),
            "protocol_binding": protocol_binding,
            "canonical_invocation": canonical_invocation,
            "git_lock": lock,
            "input_manifest": manifest_verification,
            "reconstructed_grid": reconstructed_grid,
            "scope": scope,
            "reconstruction_integrity": {
                "OPENBMI_STRESS": openbmi_integrity,
                "WBCIC_REPLICATION": wbcic_integrity,
                "aggregate_atol": AGGREGATE_ATOL,
                "aggregate_rtol": AGGREGATE_RTOL,
            },
            "double_cross_fit_splits": split_validation,
            "uniform_refit_equivalence": refit_equivalence,
            "bootstrap": {
                "analyses": sorted(BOOTSTRAP_ANALYSES),
                "draws_per_primary_analysis": bootstrap_draws,
                "seed": bootstrap_seed,
                "unit": "fold-stratified biological-subject cluster",
                "complete_ridge_refit_each_draw": True,
                "sign_flip_test_performed": False,
            },
            "no_sealed_or_outer_access": True,
            "atomic_one_shot_publication": True,
        }

        observation_columns = [
            "analysis",
            "analysis_role",
            "training_weight_mode",
            "subject_aggregation_mode",
            "dataset",
            "subject_cluster",
            "backbone",
            "fold",
            "seed",
            "method",
            "lambda",
            "direction_id",
            "subject_trial_count",
            *PREDICTOR_COLUMNS,
            "subject_CE_effect",
            *(f"prediction_{model}" for model in MODELS),
            *(f"squared_error_{model}" for model in MODELS),
        ]
        write_csv(staging / "subject_observations.csv", public_observations[observation_columns])
        write_csv(staging / "subject_summary.csv", subject)
        write_csv(staging / "subject_backbone_summary.csv", backbone_subject)
        write_json(staging / "SUBJECT_LEVEL_D_VS_I_SUMMARY.json", summary)
        write_json(staging / "VALIDATION.json", validation)
        (staging / "SUBJECT_LEVEL_D_VS_I_REPORT.md").write_text(
            report_markdown(summary, lock), encoding="utf-8", newline="\n"
        )
        write_json(
            staging / "EXECUTION_STATE.json",
            {
                "status": "COMPLETE_VALIDATION_PASS",
                "protocol_id": PROTOCOL_ID,
                "pre_outcome_lock_commit": lock["lock_commit"],
                "cross_dataset_terminal": terminal,
            },
        )
        staging.rename(output)
        return summary
    except Exception as error:
        write_json(
            staging / "EXECUTION_STATE.json",
            {
                "status": "FAILED_REQUIRES_AUDIT_BEFORE_ANY_RERUN",
                "protocol_id": PROTOCOL_ID,
                "pre_outcome_lock_commit": lock["lock_commit"],
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest", help="hash inputs without loading scientific outcomes")
    manifest.add_argument("--stress-root", type=Path, required=True)
    manifest.add_argument("--wbcic-root", type=Path, required=True)
    manifest.add_argument("--protocol", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="execute the committed locked reanalysis")
    run.add_argument("--repository", type=Path, required=True)
    run.add_argument("--stress-root", type=Path, required=True)
    run.add_argument("--wbcic-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "manifest":
        repository = implementation_repository()
        verify_canonical_invocation(
            repository,
            protocol=args.protocol,
            manifest_output=args.output,
        )
        protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
        verify_protocol_binding(protocol)
        roots = SourceRoots(args.stress_root.resolve(), args.wbcic_root.resolve())
        print(
            json.dumps(
                create_manifest(roots, args.output.resolve()),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    summary = execute(args)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
