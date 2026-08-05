from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import atomic_parquet, canonical_hash, sha256_file

SCHEMA_VERSION = "contextual-risk-episode-v1"


def build_contextual_episodes(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root)
    manifest_rows = []
    for dataset in ("hmc", "eegmmidb", "cap"):
        for seed in range(5):
            source = root / "data/episodes_v3" / dataset / f"seed_{seed}.parquet"
            frame = pd.read_parquet(source)
            rows = []
            for row in frame.itertuples(index=False):
                context = np.unique(np.r_[row.adapt_indices, row.probe_indices]).astype(np.int64)
                future = np.asarray(row.future_indices, dtype=np.int64)
                if len(np.intersect1d(context, future)):
                    raise RuntimeError(f"overlap: {dataset}/{seed}/{row.subject_id}")
                if len(context) and len(future) and int(context.max()) >= int(future.min()):
                    raise RuntimeError(f"context is not strictly before Future: {row.subject_id}")
                if not np.all(context[:-1] < context[1:]) or not np.all(future[:-1] < future[1:]):
                    raise RuntimeError(f"non-temporal indices: {row.subject_id}")
                episode_hash = canonical_hash({
                    "schema": SCHEMA_VERSION, "dataset": dataset, "seed": seed,
                    "subject_id": row.subject_id, "context": context.tolist(),
                    "future": future.tolist(), "source_episode_hash": row.original_episode_sha256,
                })
                rows.append({
                    "dataset": dataset, "seed": seed, "subject_id": row.subject_id,
                    "context_indices": context, "future_indices": future,
                    "n_context": len(context), "n_future": len(future),
                    "split_unit": row.split_unit, "episode_hash": episode_hash,
                    "source_v3_file_sha256": sha256_file(source), "schema_version": SCHEMA_VERSION,
                })
            target = root / "data/episodes_contextual_risk" / dataset / f"seed_{seed}.parquet"
            atomic_parquet(pd.DataFrame(rows), target)
            target_hash = sha256_file(target)
            for row in rows:
                manifest_rows.append({
                    "dataset": dataset, "seed": seed, "subject_id": row["subject_id"],
                    "file": str(target), "file_sha256": target_hash,
                    "episode_hash": row["episode_hash"], "n_context": row["n_context"],
                    "n_future": row["n_future"], "split_unit": row["split_unit"],
                    "schema_version": SCHEMA_VERSION,
                })
    return pd.DataFrame(manifest_rows)
