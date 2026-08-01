#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from _common import parser
from hsc_tta.utils import require_cpu


REQUIRED_MANIFESTS = {
    "subjects.parquet",
    "recordings.parquet",
    "channels.parquet",
    "annotations.parquet",
    "exclusions.parquet",
    "eegmmidb_download_manifest.parquet",
    "hmc_download_manifest.parquet",
    "cap_download_manifest.parquet",
    "dataset_audit.json",
}
REQUIRED_CACHE_FIELDS = {
    "signal",
    "label",
    "window_start",
    "window_end",
    "channel_names",
    "channel_mask",
    "sampling_rate",
    "recording_id",
    "run_id",
    "quality_flags",
}


def _as_indices(value: object) -> list[int]:
    if isinstance(value, np.ndarray):
        return value.astype(int).tolist()
    return [int(item) for item in value]


def main() -> int:
    args = parser("Validate full CPU artifacts and leakage invariants").parse_args()
    require_cpu(args.device)
    root = Path("/root/autodl-tmp/hsc_tta_eeg")
    failures: list[str] = []
    manifest_dir = root / "data/manifests"
    missing_manifests = sorted(REQUIRED_MANIFESTS - {p.name for p in manifest_dir.iterdir()})
    failures.extend(f"missing_manifest:{name}" for name in missing_manifests)

    subjects = pd.read_parquet(manifest_dir / "subjects.parquet")
    summary: dict[str, object] = {
        "missing_manifests": missing_manifests,
        "datasets": {},
        "leakage_failures": [],
    }
    cache_by_dataset: dict[str, dict[str, Path]] = {}

    for dataset in ("eegmmidb", "hmc", "cap"):
        eligible = set(subjects.loc[(subjects.dataset == dataset) & subjects.eligible, "subject_id"])
        cache_map: dict[str, Path] = {}
        total_windows = 0
        for cache in sorted((root / f"data/processed/{dataset}").glob("*.h5")):
            try:
                with h5py.File(cache, "r") as handle:
                    metadata = json.loads(handle.attrs["metadata_json"])
                    sid = metadata["subject_id"]
                    cache_map[sid] = cache
                    if not bool(handle.attrs.get("complete", False)):
                        failures.append(f"incomplete_cache:{cache}")
                    missing_fields = REQUIRED_CACHE_FIELDS - set(handle.keys())
                    failures.extend(f"missing_cache_field:{cache}:{name}" for name in sorted(missing_fields))
                    n_windows = len(handle["label"])
                    total_windows += n_windows
                    if "quality_flags" in handle and handle["quality_flags"].shape != (n_windows, 3):
                        failures.append(f"quality_shape:{cache}:{handle['quality_flags'].shape}")
                    for field in ("signal", "window_start", "window_end", "recording_id", "run_id"):
                        if field in handle and len(handle[field]) != n_windows:
                            failures.append(f"cache_length:{cache}:{field}")
            except Exception as exc:
                failures.append(f"cache_read:{cache}:{type(exc).__name__}:{exc}")
        if set(cache_map) != eligible:
            failures.append(
                f"cache_subject_mismatch:{dataset}:missing={sorted(eligible-set(cache_map))}:extra={sorted(set(cache_map)-eligible)}"
            )
        cache_by_dataset[dataset] = cache_map
        summary["datasets"][dataset] = {
            "eligible_subjects": len(eligible),
            "complete_caches": len(cache_map),
            "total_windows": total_windows,
            "seeds": {},
        }

    for dataset in ("eegmmidb", "hmc", "cap"):
        eligible = set(cache_by_dataset[dataset])
        for seed in range(5):
            split_path = root / f"data/splits/{dataset}/seed_{seed}.json"
            episode_path = root / f"data/episodes/{dataset}/seed_{seed}.parquet"
            if not split_path.exists() or not episode_path.exists():
                failures.append(f"missing_split_or_episode:{dataset}:{seed}")
                continue
            roles = json.loads(split_path.read_text(encoding="utf-8"))["roles"]
            role_sets = {role: set(ids) for role, ids in roles.items()}
            role_union = set().union(*role_sets.values())
            if role_union != eligible:
                failures.append(f"split_coverage:{dataset}:{seed}")
            role_names = list(role_sets)
            for i, left in enumerate(role_names):
                for right in role_names[i + 1 :]:
                    overlap = role_sets[left] & role_sets[right]
                    if overlap:
                        failures.append(f"subject_leakage:{dataset}:{seed}:{left}:{right}:{sorted(overlap)}")
            subject_role = {sid: role for role, ids in roles.items() for sid in ids}
            episodes = pd.read_parquet(episode_path)
            if set(episodes.subject_id) != eligible or len(episodes) != len(eligible):
                failures.append(f"episode_coverage:{dataset}:{seed}")
            excluded = 0
            n_context: list[int] = []
            n_future: list[int] = []
            for row in episodes.itertuples():
                context = _as_indices(row.context_indices)
                future = _as_indices(row.future_indices)
                if set(context) & set(future):
                    failures.append(f"uv_overlap:{dataset}:{seed}:{row.subject_id}")
                if row.split_role != subject_role.get(row.subject_id):
                    failures.append(f"episode_role:{dataset}:{seed}:{row.subject_id}")
                n_context.append(int(row.n_context))
                n_future.append(int(row.n_future))
                if pd.notna(row.exclusion_reason):
                    excluded += 1
                with h5py.File(cache_by_dataset[dataset][row.subject_id], "r") as handle:
                    if context and max(context) >= len(handle["label"]):
                        failures.append(f"context_oob:{dataset}:{seed}:{row.subject_id}")
                    if future and max(future) >= len(handle["label"]):
                        failures.append(f"future_oob:{dataset}:{seed}:{row.subject_id}")
                    if dataset == "eegmmidb":
                        runs = handle["run_id"][:]
                        if set(runs[context]) - {4, 6} or set(runs[future]) - {8, 10, 12, 14}:
                            failures.append(f"mi_run_leakage:{seed}:{row.subject_id}")
                    elif context:
                        starts = handle["window_start"][:]
                        first = starts[context[0]]
                        boundary = first + 90 * 60
                        if np.any(starts[context] >= boundary) or (future and np.any(starts[future] < boundary)):
                            failures.append(f"sleep_time_leakage:{dataset}:{seed}:{row.subject_id}")
            summary["datasets"][dataset]["seeds"][str(seed)] = {
                "role_counts": {role: len(ids) for role, ids in roles.items()},
                "episodes": len(episodes),
                "excluded_episodes": excluded,
                "context_min": min(n_context) if n_context else None,
                "context_max": max(n_context) if n_context else None,
                "future_min": min(n_future) if n_future else None,
                "future_max": max(n_future) if n_future else None,
            }

    summary["leakage_failures"] = [item for item in failures if "leakage" in item or "overlap" in item]
    summary["failures"] = failures
    summary["valid"] = not failures
    output = root / "outputs/cpu_validation/validation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
