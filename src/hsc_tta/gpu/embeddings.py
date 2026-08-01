from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import torch

from hsc_tta.backbones import CBraModInputAdapter, FrozenCBraMod


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_from_path(dataset: str, path: Path) -> str:
    stem = path.stem
    prefix = f"{dataset}_"
    if not stem.startswith(prefix):
        raise ValueError(f"unexpected cache filename: {path}")
    return f"{dataset}:{stem[len(prefix):]}"


def load_embedding(path: str | Path) -> dict[str, np.ndarray | str]:
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise RuntimeError(f"incomplete embedding: {path}")
        result: dict[str, np.ndarray | str] = {name: handle[name][...] for name in handle.keys()}
        result["subject_id"] = str(handle.attrs["subject_id"])
        return result


def _episode_roles(episode_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_parquet(episode_file)
    roles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in frame.itertuples(index=False):
        roles[str(row.subject_id)] = (np.asarray(row.context_indices, int), np.asarray(row.future_indices, int))
    return roles


def _write_subject(path: Path, payload: dict[str, np.ndarray], attrs: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with h5py.File(temporary, "w") as handle:
        for name, value in payload.items():
            array = np.asarray(value)
            kwargs = {} if array.ndim == 0 else {"compression": "gzip", "compression_opts": 1, "chunks": True}
            handle.create_dataset(name, data=array, **kwargs)
        for name, value in attrs.items():
            handle.attrs[name] = value
        handle.attrs["complete"] = True
    os.replace(temporary, path)


def extract_subject(
    dataset: str,
    source: Path,
    destination: Path,
    backbone: FrozenCBraMod,
    adapter: CBraModInputAdapter,
    episode_roles: tuple[np.ndarray, np.ndarray],
    checkpoint_sha256: str,
    backbone_revision: str,
    *,
    device: str,
    batch_size: int,
    resume: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    subject_id = subject_from_path(dataset, source)
    if resume and destination.exists():
        try:
            with h5py.File(destination, "r") as old:
                if (bool(old.attrs.get("complete", False)) and old.attrs.get("adapter_config_hash") == adapter.config_hash
                        and old.attrs.get("checkpoint_sha256") == checkpoint_sha256):
                    return {"dataset": dataset, "subject_id": subject_id, "status": "resumed",
                            "n_windows": int(old["embedding"].shape[0]), "embedding_dim": 200,
                            "dtype": str(old["embedding"].dtype), "file_path": str(destination),
                            "file_size": destination.stat().st_size, "sha256": file_sha256(destination),
                            "runtime_seconds": 0.0, "peak_vram_mb": 0.0, "failure_reason": ""}
        except OSError:
            pass
    torch.cuda.reset_peak_memory_stats()
    with h5py.File(source, "r") as handle:
        n = int(handle["signal"].shape[0])
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["channel_names"][...]]
        rate = float(handle["sampling_rate"][()])
        output: list[np.ndarray] = []
        effective = int(batch_size if dataset != "eegmmidb" else min(batch_size, 16))
        start = 0
        while start < n:
            stop = min(n, start + effective)
            try:
                batch = adapter.adapt(dataset, handle["signal"][start:stop], names, rate, device=device)
                output.append(backbone(batch.tensor).cpu().numpy().astype(np.float16))
                start = stop
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if effective <= 1:
                    raise
                effective = max(1, effective // 2)
        embeddings = np.concatenate(output, axis=0)
        if embeddings.shape != (n, 200) or not np.isfinite(embeddings).all():
            raise RuntimeError("embedding/index alignment or finiteness failure")
        context, future = episode_roles
        role = np.zeros(n, dtype=np.int8)
        role[context] = 1
        role[future] = 2
        payload = {
            "embedding": embeddings,
            "window_id": np.arange(n, dtype=np.int64),
            "original_index": np.arange(n, dtype=np.int64),
            "label": handle["label"][...].astype(np.int16),
            "episode_role_main": role,
            "recording_id": handle["recording_id"][...],
            "run_id": handle["run_id"][...].astype(np.int16),
            "channel_mask": np.ones(len(names) if dataset == "eegmmidb" else 1, dtype=bool),
            "input_valid_mask": np.ones((n, 4 if dataset == "eegmmidb" else 30), dtype=bool),
            "quality_flags": handle["quality_flags"][...].astype(np.float32),
        }
        source_config = str(handle.attrs.get("preprocessing_config_hash", ""))
    source_hash = file_sha256(source)
    _write_subject(destination, payload, {
        "dataset": dataset, "subject_id": subject_id, "backbone_revision": backbone_revision,
        "checkpoint_sha256": checkpoint_sha256, "adapter_config_hash": adapter.config_hash,
        "source_cache_hash": source_hash, "source_preprocessing_config_hash": source_config,
        "pooling_rule": adapter.config["mi_pooling" if dataset == "eegmmidb" else "sleep_pooling"],
    })
    return {"dataset": dataset, "subject_id": subject_id, "status": "complete", "n_windows": n,
            "embedding_dim": 200, "dtype": "float16", "file_path": str(destination),
            "file_size": destination.stat().st_size, "sha256": file_sha256(destination),
            "runtime_seconds": time.perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20, "failure_reason": ""}


def extract_all_embeddings(root: str | Path, backbone: FrozenCBraMod, adapter: CBraModInputAdapter,
                           checkpoint_sha256: str, backbone_revision: str, *, datasets: list[str],
                           device: str = "cuda", batch_size: int = 64, resume: bool = True) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        roles = _episode_roles(root / "data" / "episodes_main120" / dataset / "seed_0.parquet")
        sources = sorted((root / "data" / "processed" / dataset).glob("*.h5"))
        if dataset == "cap":
            sources = [p for p in sources if subject_from_path(dataset, p) in roles]
        for source in sources:
            subject_id = subject_from_path(dataset, source)
            destination = root / "outputs" / "full_experiment" / "embeddings" / dataset / f"{subject_id.split(':',1)[1]}.h5"
            try:
                row = extract_subject(dataset, source, destination, backbone, adapter, roles[subject_id],
                                      checkpoint_sha256, backbone_revision, device=device,
                                      batch_size=batch_size, resume=resume)
            except Exception as error:
                row = {"dataset": dataset, "subject_id": subject_id, "status": "failed", "n_windows": 0,
                       "embedding_dim": 200, "dtype": "", "file_path": str(destination), "file_size": 0,
                       "sha256": "", "runtime_seconds": 0.0, "peak_vram_mb": 0.0,
                       "failure_reason": f"{type(error).__name__}: {error}"}
            rows.append(row)
            manifest = pd.DataFrame(rows)
            output = root / "outputs" / "full_experiment" / "embeddings" / "embedding_manifest.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".parquet.part")
            manifest.to_parquet(temporary, index=False)
            os.replace(temporary, output)
            gc.collect()
            torch.cuda.empty_cache()
    failures = [row for row in rows if row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"embedding extraction failed for {len(failures)} subjects")
    backbone.verify_frozen()
    return pd.DataFrame(rows)
