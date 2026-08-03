from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pandas as pd
import torch

from hsc_tta.backbones import CBraModInputAdapter, FrozenCBraModTokens
from hsc_tta.gpu.embeddings import subject_from_path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_roles(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_parquet(path)
    return {str(row.subject_id): (np.asarray(row.context_indices, int), np.asarray(row.future_indices, int))
            for row in frame.itertuples(index=False)}


def extract_token_subject(dataset: str, source: Path, destination: Path, backbone: FrozenCBraModTokens,
                          adapter: CBraModInputAdapter, episode: tuple[np.ndarray, np.ndarray], *,
                          checkpoint_sha256: str, backbone_commit: str, device: str = "cuda",
                          batch_size: int = 32, resume: bool = True) -> dict[str, object]:
    started = time.perf_counter()
    subject = subject_from_path(dataset, source)
    if resume and destination.is_file():
        try:
            with h5py.File(destination, "r") as old:
                if (bool(old.attrs.get("complete", False)) and old.attrs.get("checkpoint_sha256") == checkpoint_sha256
                        and old.attrs.get("adapter_config_hash") == adapter.config_hash):
                    return {"dataset": dataset, "subject_id": subject, "status": "resumed",
                            "n_windows": int(old["token_embeddings"].shape[0]),
                            "token_shape": str(tuple(old["token_embeddings"].shape[1:])),
                            "file_path": str(destination), "file_size": destination.stat().st_size,
                            "sha256": sha256(destination), "runtime_seconds": 0.0, "peak_vram_mb": 0.0}
        except OSError:
            pass
    probe = destination.parent
    while not probe.exists():
        if probe == probe.parent:
            raise RuntimeError("cannot resolve disk-usage probe path")
        probe = probe.parent
    if shutil.disk_usage(probe).free < 60 * 2**30:
        raise RuntimeError("disk safety gate: less than 60 GiB free")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".h5.part")
    torch.cuda.reset_peak_memory_stats()
    try:
        with h5py.File(source, "r") as raw, h5py.File(temporary, "w") as out:
            n = int(raw["signal"].shape[0])
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in raw["channel_names"][...]]
            rate = float(raw["sampling_rate"][()])
            channels, patches = (64, 4) if dataset == "eegmmidb" else (1, 30)
            token_ds = out.create_dataset("token_embeddings", shape=(n, channels, patches, 200),
                                          dtype=np.float16, chunks=(1, channels, patches, 200), compression="lzf")
            valid_ds = out.create_dataset("valid_token_mask", shape=(n, channels, patches), dtype=np.bool_,
                                          chunks=(1, channels, patches), compression="lzf")
            effective = min(batch_size, 12) if dataset == "eegmmidb" else batch_size
            start = 0
            while start < n:
                stop = min(n, start + effective)
                try:
                    adapted = adapter.adapt(dataset, raw["signal"][start:stop], names, rate, device=device)
                    tokens = backbone(adapted.tensor).cpu().numpy().astype(np.float16)
                    if tokens.shape != (stop - start, channels, patches, 200):
                        raise RuntimeError(f"token/window alignment failure: {tokens.shape}")
                    token_ds[start:stop] = tokens
                    valid_ds[start:stop] = adapted.input_valid_mask
                    start = stop
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if effective == 1:
                        raise
                    effective = max(1, effective // 2)
            context, future = episode
            role = np.zeros(n, np.int8); role[context] = 1; role[future] = 2
            out.create_dataset("window_indices", data=np.arange(n, dtype=np.int64))
            out.create_dataset("labels", data=raw["label"][...].astype(np.int16))
            out.create_dataset("episode_role_main", data=role)
            out.create_dataset("channel_indices", data=np.arange(channels, dtype=np.int16))
            out.create_dataset("patch_indices", data=np.arange(patches, dtype=np.int16))
            out.attrs.update({"complete": True, "dataset": dataset, "subject_id": subject,
                "token_shape": f"{channels}x{patches}x200", "checkpoint_sha256": checkpoint_sha256,
                "backbone_commit": backbone_commit, "backbone_parameter_hash": backbone.frozen_hash,
                "adapter_config_hash": adapter.config_hash, "raw_source_hash": sha256(source),
                "channel_names_json": __import__("json").dumps(names if dataset == "eegmmidb" else [adapter.config["sleep_channels"][dataset]]),
                "normalization_uses_future_statistics": False})
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists(): temporary.unlink()
        raise
    backbone.verify_frozen()
    return {"dataset": dataset, "subject_id": subject, "status": "complete", "n_windows": n,
            "token_shape": f"{channels}x{patches}x200", "file_path": str(destination),
            "file_size": destination.stat().st_size, "sha256": sha256(destination),
            "runtime_seconds": time.perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}


def extract_all_token_embeddings(root: str | Path, *, datasets: list[str], device: str = "cuda",
                                 batch_size: int = 32, resume: bool = True) -> pd.DataFrame:
    root = Path(root)
    checkpoint = root / "checkpoints" / "cbramod" / "pretrained_weights.pth"
    checkpoint_hash = sha256(checkpoint)
    backbone = FrozenCBraModTokens(root / "external" / "CBraMod", checkpoint).to(device).eval()
    adapter = CBraModInputAdapter()
    rows: list[dict[str, object]] = []
    manifest_path = root / "outputs" / "v2_joint_certified" / "source_models" / "TOKEN_EMBEDDING_MANIFEST.parquet"
    for dataset in datasets:
        roles = _episode_roles(root / "data" / "episodes_main120" / dataset / "seed_0.parquet")
        sources = sorted((root / "data" / "processed" / dataset).glob("*.h5"))
        if dataset == "cap": sources = [p for p in sources if subject_from_path(dataset, p) in roles]
        for source in sources:
            subject = subject_from_path(dataset, source)
            destination = root / "data" / "embeddings_tokens_v2" / dataset / f"{subject.split(':',1)[1]}.h5"
            row = extract_token_subject(dataset, source, destination, backbone, adapter, roles[subject],
                                        checkpoint_sha256=checkpoint_hash,
                                        backbone_commit="0ff6be918985689e7df679bc731ffb70e6c6224f",
                                        device=device, batch_size=batch_size, resume=resume)
            rows.append(row)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            part = manifest_path.with_suffix(".parquet.part")
            pd.DataFrame(rows).to_parquet(part, index=False); os.replace(part, manifest_path)
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)
