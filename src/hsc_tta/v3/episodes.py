from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeProtocol:
    adapt_probe_ratio: str = "1:1"
    candidate_ratios: tuple[str, ...] = ("1:2", "1:1", "2:1")
    sleep_window_seconds: float = 30.0
    min_adapt: int = 10
    min_probe: int = 10

    @property
    def fraction(self) -> float:
        left, right = (int(x) for x in self.adapt_probe_ratio.split(":"))
        if left <= 0 or right <= 0:
            raise ValueError("A:P ratio must be positive")
        return left / (left + right)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def validate_three_way(adapt: np.ndarray, probe: np.ndarray, future: np.ndarray) -> None:
    a, p, v = map(lambda x: np.asarray(x, dtype=int), (adapt, probe, future))
    if min(len(a), len(p), len(v)) == 0:
        raise ValueError("A, P, and V must all be nonempty")
    if set(a) & set(p) or set(a) & set(v) or set(p) & set(v):
        raise ValueError("A/P/V overlap")
    if np.any(np.diff(a) < 0) or np.any(np.diff(p) < 0) or np.any(np.diff(v) < 0):
        raise ValueError("indices must preserve temporal order")
    if a[-1] >= p[0] or p[-1] >= v[0]:
        raise ValueError("A, P, V must be strictly chronological")


def split_context(context: np.ndarray, future: np.ndarray, protocol: EpisodeProtocol,
                  *, run_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    context = np.asarray(context, dtype=int); future = np.asarray(future, dtype=int)
    if run_ids is None:
        boundary = int(round(len(context) * protocol.fraction))
        boundary = min(max(boundary, protocol.min_adapt), len(context) - protocol.min_probe)
        adapt, probe = context[:boundary], context[boundary:]
        metadata = {"split_unit": "window", "adapt_runs": [], "probe_runs": []}
    else:
        runs = np.asarray(run_ids, dtype=int)
        context_runs = runs[context]
        ordered = list(dict.fromkeys(context_runs.tolist()))
        if len(ordered) < 2:
            raise ValueError("MI A/P split needs at least two chronological runs")
        target = len(context) * protocol.fraction
        choices = []
        for cut in range(1, len(ordered)):
            a_runs, p_runs = ordered[:cut], ordered[cut:]
            a = context[np.isin(context_runs, a_runs)]; p = context[np.isin(context_runs, p_runs)]
            choices.append((abs(len(a) - target), cut, a, p, a_runs, p_runs))
        _, _, adapt, probe, a_runs, p_runs = min(choices, key=lambda row: (row[0], row[1]))
        metadata = {"split_unit": "run", "adapt_runs": a_runs, "probe_runs": p_runs}
    if len(adapt) < protocol.min_adapt or len(probe) < protocol.min_probe:
        raise ValueError("insufficient A or P samples")
    validate_three_way(adapt, probe, future)
    metadata["effective_adapt_fraction"] = len(adapt) / len(context)
    return adapt, probe, metadata


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v3_episodes(root: str | Path, protocol: EpisodeProtocol, datasets: tuple[str, ...] = ("hmc", "eegmmidb", "cap")) -> pd.DataFrame:
    root = Path(root); output = root / "data/episodes_v3"; rows = []
    for dataset in datasets:
        for seed in range(5):
            original = root / "data/episodes_main120" / dataset / f"seed_{seed}.parquet"
            frame = pd.read_parquet(original); converted = []
            for row in frame.itertuples(index=False):
                context, future = np.asarray(row.context_indices, int), np.asarray(row.future_indices, int)
                run_ids = None
                if dataset == "eegmmidb":
                    raw = root / "data/processed/eegmmidb" / f"eegmmidb_{str(row.subject_id).split(':',1)[1]}.h5"
                    with h5py.File(raw, "r") as handle: run_ids = handle["run_id"][...]
                adapt, probe, metadata = split_context(context, future, protocol, run_ids=run_ids)
                converted.append({"dataset": dataset, "seed": seed, "subject_id": row.subject_id,
                    "split_role": row.split_role, "episode_id": f"{row.episode_id}:v3:{protocol.adapt_probe_ratio}",
                    "adapt_indices": adapt, "probe_indices": probe, "future_indices": future,
                    "n_adapt": len(adapt), "n_probe": len(probe), "n_future": len(future),
                    "adapt_duration_seconds": len(adapt) * (protocol.sleep_window_seconds if dataset != "eegmmidb" else 4.1),
                    "probe_duration_seconds": len(probe) * (protocol.sleep_window_seconds if dataset != "eegmmidb" else 4.1),
                    "future_duration_seconds": len(future) * (protocol.sleep_window_seconds if dataset != "eegmmidb" else 4.1),
                    "original_episode_sha256": _sha(original), "protocol_config_hash": protocol.config_hash,
                    **metadata})
            target = output / dataset / f"seed_{seed}.parquet"; target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_suffix(".parquet.part"); pd.DataFrame(converted).to_parquet(part, index=False); os.replace(part, target)
            rows.extend({"dataset": dataset, "seed": seed, "subject_id": x["subject_id"], "file": str(target),
                         "file_sha256": _sha(target), "protocol_config_hash": protocol.config_hash,
                         "n_adapt": x["n_adapt"], "n_probe": x["n_probe"], "n_future": x["n_future"],
                         "split_unit": x["split_unit"]} for x in converted)
    manifest = pd.DataFrame(rows); out = root / "outputs/v3_probecert/episodes"; out.mkdir(parents=True, exist_ok=True)
    part = out / "EPISODE_MANIFEST.parquet.part"; manifest.to_parquet(part, index=False); os.replace(part, out / "EPISODE_MANIFEST.parquet")
    return manifest
