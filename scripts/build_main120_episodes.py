#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd

from _common import parser
from hsc_tta.episodes import build_mi_episode, build_sleep_main120_episode
from hsc_tta.utils import load_yaml, require_cpu


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")


def main() -> int:
    args = parser("Build immutable V_s-main120 episodes without replacing full-night episodes").parse_args()
    require_cpu(args.device)
    cfg = load_yaml(args.config)
    dataset = cfg["dataset"]
    split_path = ROOT / f"data/splits/{dataset}/seed_{args.seed}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["roles"]
    roles = {subject_id: role for role, ids in split.items() for subject_id in ids}
    caches = sorted((ROOT / f"data/processed/{dataset}").glob("*.h5"))
    if args.limit_subjects:
        caches = caches[: args.limit_subjects]
    if args.dry_run:
        print(f"dataset={dataset} seed={args.seed} caches={len(caches)}")
        return 0
    rows: list[dict[str, object]] = []
    for cache in caches:
        with h5py.File(cache, "r") as handle:
            metadata = json.loads(handle.attrs["metadata_json"])
            subject_id = metadata["subject_id"]
            if cfg["task"] == "sleep_staging":
                episode = build_sleep_main120_episode(
                    handle["window_start"][:],
                    pd.notna(handle["label"][:]),
                    context_minutes=int(cfg["context_minutes"]),
                    future_epochs=240,
                )
            else:
                episode = build_mi_episode(
                    handle["run_id"][:],
                    tuple(cfg["context_runs"]),
                    tuple(cfg["future_runs"]),
                )
                episode["future_full_indices"] = list(episode["future_indices"])
                episode["n_future_full"] = episode["n_future"]
                episode["protocol"] = "mi_fixed_official_runs"
            episode.update(
                {
                    "dataset": dataset,
                    "seed": args.seed,
                    "split_role": roles.get(subject_id, "unassigned"),
                    "subject_id": subject_id,
                    "episode_id": f"{dataset}:{args.seed}:{subject_id}:main120",
                }
            )
            rows.append(episode)
    output = ROOT / f"data/episodes_main120/{dataset}/seed_{args.seed}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"episodes={len(rows)} path={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
