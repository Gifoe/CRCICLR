#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from hsc_tta.protocols import choose_common_central_channel, scan_sleep_channel_availability
from hsc_tta.splits import make_internal_subject_split
from hsc_tta.utils import load_yaml, require_cpu


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
REPO = ROOT / "repo"


def _indices(value: object) -> list[int]:
    if isinstance(value, np.ndarray):
        return value.astype(int).tolist()
    return [int(item) for item in value]


def _cache_path(dataset: str, subject_id: str) -> Path:
    safe = subject_id.replace(":", "_")
    return ROOT / f"data/processed/{dataset}/{safe}.h5"


def main() -> int:
    require_cpu("cpu")
    failures: list[str] = []
    summaries: list[dict[str, object]] = []
    method = load_yaml(REPO / "configs/method/hsc_tta.yaml")
    lambdas = np.asarray(method["lambdas"], float)
    if len(lambdas) != 21 or not np.isclose(lambdas[-1], 1.0) or np.any(np.diff(lambdas) <= 0):
        failures.append("invalid_lambda_grid_or_sentinel")
    for dataset in ("hmc", "cap", "eegmmidb"):
        for seed in range(5):
            split_payload = json.loads(
                (ROOT / f"data/splits/{dataset}/seed_{seed}.json").read_text(encoding="utf-8")
            )
            role_map = {
                subject_id: role
                for role, ids in split_payload["roles"].items()
                for subject_id in ids
            }
            if dataset == "cap":
                role_counts = {role: len(ids) for role, ids in split_payload["roles"].items()}
                if len(role_map) != 99:
                    failures.append(f"formal_cap_cohort_not_99:{seed}:{len(role_map)}")
                if role_counts != {"target_site_calibration": 25, "external_final_test": 74}:
                    failures.append(f"formal_cap_roles_not_25_74:{seed}:{role_counts}")
                formal_ids = set(
                    json.loads((REPO / "CHANNEL_PROTOCOL.json").read_text(encoding="utf-8"))["cap_subject_ids"]
                )
                if set(role_map) != formal_ids:
                    failures.append(f"formal_cap_subjects_not_c4_protocol:{seed}")
            episodes = pd.read_parquet(
                ROOT / f"data/episodes_main120/{dataset}/seed_{seed}.parquet"
            )
            if set(episodes.subject_id) != set(role_map):
                failures.append(f"episode_subject_coverage:{dataset}:{seed}")
            excluded = int(episodes.exclusion_reason.notna().sum())
            if excluded:
                failures.append(f"episode_exclusions:{dataset}:{seed}:{excluded}")
            for row in episodes.itertuples(index=False):
                context = _indices(row.context_indices)
                future = _indices(row.future_indices)
                future_full = _indices(row.future_full_indices)
                if set(context) & set(future):
                    failures.append(f"uv_overlap:{dataset}:{seed}:{row.subject_id}")
                if future != future_full[: len(future)]:
                    failures.append(f"main_not_prefix_of_full:{dataset}:{seed}:{row.subject_id}")
                if row.split_role != role_map.get(row.subject_id):
                    failures.append(f"episode_role:{dataset}:{seed}:{row.subject_id}")
                if dataset in {"hmc", "cap"}:
                    if len(future) != 240:
                        failures.append(f"sleep_future_not_240:{dataset}:{seed}:{row.subject_id}")
                    if context and future and max(context) >= min(future):
                        failures.append(f"sleep_boundary:{dataset}:{seed}:{row.subject_id}")
                elif seed == 0:
                    with h5py.File(_cache_path(dataset, row.subject_id), "r") as handle:
                        run_ids = handle["run_id"][:]
                    if set(run_ids[context]) != {4, 6} or set(run_ids[future]) != {8, 10, 12, 14}:
                        failures.append(f"mi_protocol:{row.subject_id}")
            internal_path = ROOT / f"data/splits_internal/{dataset}/seed_{seed}.json"
            internal = json.loads(internal_path.read_text(encoding="utf-8"))
            expected = make_internal_subject_split(split_payload["roles"], dataset, seed)
            if internal != expected:
                failures.append(f"internal_split_not_deterministic:{dataset}:{seed}")
            summaries.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "subjects": len(episodes),
                    "excluded": excluded,
                    "context_min": int(episodes.n_context.min()),
                    "context_max": int(episodes.n_context.max()),
                    "future_min": int(episodes.n_future.min()),
                    "future_max": int(episodes.n_future.max()),
                }
            )
    frozen_channel = json.loads((REPO / "CHANNEL_PROTOCOL.json").read_text(encoding="utf-8"))
    recomputed = choose_common_central_channel(
        scan_sleep_channel_availability(ROOT / "data/processed")
    )
    if frozen_channel != recomputed:
        failures.append("channel_protocol_not_reproducible")
    payload = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "summaries": summaries,
        "channel_protocol_hash": frozen_channel.get("protocol_hash"),
        "selected_channel": frozen_channel.get("selected_channel"),
        "lambda_grid": lambdas.tolist(),
    }
    output = ROOT / "outputs/cpu_critical_index_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
